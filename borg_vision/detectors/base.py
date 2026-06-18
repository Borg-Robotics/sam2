"""Shared, GUI-free detector base for all OAK-D + SAM2 detection modes.

`BaseDetector` owns everything the modes have in common: the OAK-D camera, the
SAM2 automatic-mask-generator, and the warmup / barcode-scan / cancellation
lifecycle. Each mode subclasses it and implements:

  - `detect(frames, barcode=None) -> BaseResult`   (the mode's logic)
  - `_draw_result(result) -> np.ndarray`           (annotated overlay image)
  - `_serialize(result) -> dict`                   (JSON of measurements)
  - `_debug_artifacts(result) -> dict`             (extra mask/heatmap files)

`save_debug` is a template method here so every mode writes the same uniform
artifact set (raw RGB + annotated image + JSON, plus whatever extra files the
mode declares). `should_abort` callables are polled once per camera frame
during warmup and barcode scanning so a ROS action server can wire goal
cancellation into them.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from ..barcode import detect_barcode
from ..camera import OakCamera
from ..config import BaseConfig


class BaseDetector:
    #: Default config class for the mode; subclasses override.
    config_class = BaseConfig
    #: Whether detection is gated on first decoding a barcode.
    requires_barcode = False

    def __init__(self, cfg=None, mxid=None, torch_device=None, camera=None):
        self.cfg = cfg if cfg is not None else self.config_class()

        # When `camera` is provided (multi-mode-per-camera setups), several
        # detectors share one OAK-D device. The shared camera is owned by the
        # caller, so this detector must NOT open or close it -- open()/close()
        # become no-ops and the owner manages the device lifecycle once.
        if camera is not None:
            self.camera = camera
            self._owns_camera = False
        else:
            self.camera = OakCamera(self.cfg, mxid=mxid)
            self._owns_camera = True

        self._torch_device = torch_device
        self._mask_generator = None

    # ----------------------------------------------------------- lifecycle

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
        # No-op for a shared camera; the owner opens it once.
        if self._owns_camera:
            self.camera.open()
        return self

    def close(self):
        # No-op for a shared camera; the owner closes it once.
        if self._owns_camera:
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

    # -------------------------------------------------------------- barcode

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

    # --------------------------------------------------------------- detect

    def detect(self, frames, barcode=None):
        """Run the mode's detection on the given frames.

        Subclasses must implement this and return a BaseResult subclass (or
        None when no valid mask was found).
        """
        raise NotImplementedError

    def detect_after_barcode(self, timeout_s, should_abort=None, on_frame=None):
        """Convenience: for barcode-gated modes, scan_barcode then detect;
        for non-gated modes, grab one frame and detect immediately.

        Returns the mode's result, or None on barcode timeout/abort or when
        detection finds no valid mask.
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        if self.requires_barcode:
            scan = self.scan_barcode(
                timeout_s, should_abort=should_abort, on_frame=on_frame
            )
            if scan is None:
                return None
            barcode, frames = scan
            return self.detect(frames, barcode)

        if should_abort is not None and should_abort():
            return None
        frames = self.camera.get_frames()
        if on_frame is not None:
            on_frame(frames, None)
        return self.detect(frames, None)

    # ---------------------------------------------------------- visualization

    def draw_result(self, result):
        """Return the annotated overlay image (BGR) for the result.

        Available for every mode so a consumer can render without saving to
        disk (e.g. publish a debug image). Subclasses implement `_draw_result`.
        """
        return self._draw_result(result)

    def _draw_result(self, result):
        raise NotImplementedError

    def _serialize(self, result):
        """Return the JSON-able dict of measurements for the result."""
        raise NotImplementedError

    def _debug_artifacts(self, result):
        """Return {filename_suffix: image_or_mask} extra files to save.

        Keys are filename stems (without timestamp/extension); values are
        either uint8 images (saved as-is) or boolean/uint8 masks (saved via
        save_binary_mask when the value is a mask). Default: none.
        """
        return {}

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Save the uniform artifact set for this mode.

        Always writes: raw RGB, the annotated result image, and a JSON of the
        measurements. Modes add mask/heatmap files via `_debug_artifacts`.
        Returns a dict of saved paths (always includes: dir, raw_rgb, result,
        json, result_bgr; plus any extra artifact keys).
        """
        cfg = self.cfg
        mode = getattr(cfg, "mode_name", self.__class__.__name__.lower())

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) if out_dir is not None else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / f"raw_rgb_{timestamp}.jpg"
        result_path = save_dir / f"{mode}_final_{timestamp}.png"
        json_path = save_dir / f"{mode}_final_{timestamp}.json"

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)

        paths = {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "result_bgr": result_bgr,
        }

        from ..visualization import save_binary_mask

        for key, value in self._debug_artifacts(result).items():
            path = save_dir / f"{key}_{timestamp}.png"
            if getattr(value, "dtype", None) is not None and value.ndim == 2:
                save_binary_mask(path, value)
            else:
                cv2.imwrite(str(path), value)
            paths[key] = path

        with open(json_path, "w") as f:
            json.dump(self._serialize(result, timestamp=timestamp), f, indent=2)
        paths["json"] = json_path

        return paths

    # ------------------------------------------------------- context manager

    def __enter__(self):
        self.load_model()
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
