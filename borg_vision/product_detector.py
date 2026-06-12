"""High-level, GUI-free detector API for ROS/automation consumers.

Typical usage:

    detector = ProductDetector(ProductDetectionConfig(), mxid=None)
    detector.load_model()
    detector.open()
    detector.warmup()
    result = detector.detect_after_barcode(timeout_s=30.0)
    if result is not None:
        detector.save_debug(result)
    detector.close()

`should_abort` callables are polled once per camera frame during warmup and
barcode scanning, so a ROS action server can wire goal cancellation into them.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from .barcode import detect_barcode
from .camera import Frames, OakCamera
from .config import ProductDetectionConfig
from .detection import run_sam_package_depth_type
from .visualization import (
    draw_result,
    make_json_result,
    save_accepted_masks,
    save_binary_mask,
)


@dataclass
class ProductDetectionResult:
    """Mirrors the final_output dict of the original script (units: mm / deg,
    camera frame: x right, y down, z = depth forward)."""

    barcode_type: Optional[str]
    barcode_data: Optional[str]
    package_type: str                              # "box" | "polymailer"
    package_type_confidence_percent: float
    mask_source: str
    top_face_depth_mm: Optional[float]
    package_depth_mm: Optional[float]
    length_mm: Optional[float]
    width_mm: Optional[float]
    angle_deg: Optional[float]
    center_x_mm: Optional[float]
    center_y_mm: Optional[float]
    product_inside_found: bool
    product_inside_center_x_mm: Optional[float]
    product_inside_center_y_mm: Optional[float]
    product_inside_center_pixel_u: Optional[int]
    product_inside_center_pixel_v: Optional[int]
    product_inside_depth_mm: Optional[float]

    # Non-serialized attachments
    raw: dict = field(default=None, repr=False)        # full run_sam_package_depth_type dict
    frames: Frames = field(default=None, repr=False)   # frames the detection ran on

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(**raw["final_output"], raw=raw, frames=frames)

    def to_json_dict(self, cfg, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


class ProductDetector:
    def __init__(self, cfg=None, mxid=None, torch_device=None):
        self.cfg = cfg if cfg is not None else ProductDetectionConfig()
        self.camera = OakCamera(self.cfg, mxid=mxid)

        self._torch_device = torch_device
        self._mask_generator = None

    @property
    def model_loaded(self):
        return self._mask_generator is not None

    def load_model(self):
        if self._mask_generator is not None:
            return

        device_name = self._torch_device
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading SAM 2 on device: {device_name}")

        sam2_model = build_sam2(
            self.cfg.model_cfg,
            self.cfg.checkpoint,
            device=device_name,
        )

        self._mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=self.cfg.sam_points_per_side,
            pred_iou_thresh=self.cfg.sam_pred_iou_thresh,
            stability_score_thresh=self.cfg.sam_stability_score_thresh,
            min_mask_region_area=self.cfg.sam_min_mask_region_area,
        )

    def open(self):
        self.camera.open()
        return self

    def close(self):
        self.camera.close()

    def warmup(self, seconds=None, should_abort=None):
        """Pump frames for the configured warmup period. Returns False if aborted."""
        if seconds is None:
            seconds = self.cfg.warmup_seconds

        start = time.time()

        while time.time() - start < seconds:
            if should_abort is not None and should_abort():
                return False
            self.camera.get_frames()

        return True

    def scan_barcode(self, timeout_s, should_abort=None, on_frame=None):
        """Grab frames until a barcode is decoded.

        Returns (barcode_dict, Frames) on success, None on timeout or abort.
        `on_frame(frames, barcode_or_none)` is called once per frame if given.
        """
        start = time.time()

        while time.time() - start < timeout_s:
            if should_abort is not None and should_abort():
                return None

            frames = self.camera.get_frames()
            barcode = detect_barcode(self.cfg, frames.rgb)

            if on_frame is not None:
                on_frame(frames, barcode)

            if barcode is not None:
                return barcode, frames

        return None

    def detect(self, frames, barcode=None):
        """Run the full SAM2 package detection on the given frames.

        Returns ProductDetectionResult or None when no valid package mask
        was found.
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        raw = run_sam_package_depth_type(
            self.cfg,
            frames.rgb,
            frames.depth_class_aligned,
            frames.depth_measure_aligned,
            self._mask_generator,
            self.camera.intrinsics,
            barcode,
        )

        if raw is None:
            return None

        return ProductDetectionResult.from_raw(raw, frames)

    def detect_after_barcode(self, timeout_s, should_abort=None, on_frame=None):
        """Convenience: scan_barcode then detect.

        Returns the ProductDetectionResult, or None on barcode
        timeout/abort or when detection finds no valid package mask.
        """
        scan = self.scan_barcode(timeout_s, should_abort=should_abort, on_frame=on_frame)

        if scan is None:
            return None

        barcode, frames = scan
        return self.detect(frames, barcode)

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Save the same artifact set as the original script.

        Returns a dict of saved paths (keys: dir, raw_rgb, result, mask,
        product_mask, heatmap, json).
        """
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) if out_dir is not None else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = draw_result(
            cfg,
            result.frames.rgb,
            result.frames.depth_class_aligned,
            result.raw,
        )

        raw_path = save_dir / f"raw_rgb_{timestamp}.jpg"
        result_path = save_dir / f"package_final_{timestamp}.png"
        mask_path = save_dir / f"package_mask_{timestamp}.png"
        product_mask_path = save_dir / f"product_inside_mask_{timestamp}.png"
        heatmap_path = save_dir / f"package_depth_heatmap_{timestamp}.png"
        json_path = save_dir / f"package_final_{timestamp}.json"

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(mask_path, result.raw["mask"])
        save_binary_mask(product_mask_path, result.raw["product_inside"]["mask"])
        cv2.imwrite(str(heatmap_path), result.raw["depth_heatmap"])

        if cfg.debug_save_all_accepted_masks:
            save_accepted_masks(save_dir, timestamp, result.raw)

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "mask": mask_path,
            "product_mask": product_mask_path,
            "heatmap": heatmap_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }

    def __enter__(self):
        self.load_model()
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
