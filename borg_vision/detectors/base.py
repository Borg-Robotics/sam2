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
import numpy as np
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from ..barcode import detect_barcode
from ..camera import OakCamera
from ..config import BaseConfig


def artifact_name(stem, ext, timestamp, organized):
    """Build a debug-artifact filename.

    When ``organized`` is True the caller saves into a per-detection folder that
    already encodes the timestamp (``<camera>/<mode>/<timestamp>/``), so the
    per-file timestamp suffix is dropped for clean names (``raw_rgb.jpg``).
    Otherwise the legacy flat naming (``raw_rgb_<timestamp>.jpg``) is kept.
    """
    if organized:
        return f"{stem}.{ext}"
    return f"{stem}_{timestamp}.{ext}"


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

        self._warm_up_inference(device_name)

    def _warm_up_inference(self, device_name):
        """Run one throwaway inference so the FIRST real detection is not slow.

        Building the model does not initialise CUDA: the context, kernel
        autotuning and allocator caches all happen on the first forward pass, and
        that lands on whichever detection happens to be first. Measured on
        camera_2/polymailer 2026-08-24:

            first  detection: run_sam2_polymailer 1278 ms
            second detection: run_sam2_polymailer  565 ms

        ~700 ms of pure cold start, paid by the first package of a run. This
        spends it at load time instead, on a synthetic ROI-sized image.

        Best effort: a failure here costs only the warm-up, so it is logged and
        swallowed rather than blocking the node from coming up.
        """
        if device_name != "cuda":
            return

        try:
            height = max(int(self.cfg.roi_y2 - self.cfg.roi_y1), 64)
            width = max(int(self.cfg.roi_x2 - self.cfg.roi_x1), 64)

            # Mid-grey rather than zeros: a flat black frame can be rejected
            # before the heavy path runs, which would defeat the point.
            dummy = np.full((height, width, 3), 128, dtype=np.uint8)

            start = time.time()

            with torch.inference_mode():
                self._mask_generator.generate(dummy)

            print(
                f"SAM 2 warm-up inference: {(time.time() - start) * 1000:.0f}ms "
                f"({width}x{height})"
            )
        except Exception as exc:  # noqa: BLE001 - warm-up must never be fatal
            print(f"SAM 2 warm-up inference skipped: {exc}")

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

        # Cleared per call so save_rejection() can never write a stale frame
        # from an earlier failure. See save_rejection.
        self.last_frames = None

        if self.requires_barcode:
            scan = self.scan_barcode(
                timeout_s, should_abort=should_abort, on_frame=on_frame
            )
            if scan is None:
                return None
            barcode, frames = scan
            self.last_frames = frames
            return self.detect(frames, barcode)

        if should_abort is not None and should_abort():
            return None
        frames = self.camera.get_frames()
        self.last_frames = frames
        if on_frame is not None:
            on_frame(frames, None)
        return self.detect(frames, None)

    def save_rejection(self, out_dir, reason=""):
        """Write the frame a FAILED detection was made on.

        The node returns early when detect() gives None, so the normal
        save_debug() path never runs and a rejection leaves nothing on disk --
        exactly when the image is most worth looking at. This writes the raw RGB
        and both depth ROIs into a <timestamp>_rejected folder, kept separate so
        a failure is never mistaken for a good capture.

        Returns the dict of written paths, or None when there is no frame.
        """
        frames = getattr(self, "last_frames", None)

        if frames is None:
            return None

        cfg = self.cfg
        save_dir = Path(out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        written = {"dir": save_dir, "reason": reason}

        raw_path = save_dir / "raw_rgb.jpg"
        cv2.imwrite(str(raw_path), frames.rgb)
        written["raw_rgb"] = raw_path

        for attr, name in (("depth_measure_aligned", "depth_roi_mm"),
                           ("depth_class_aligned", "depth_class_roi_mm")):
            depth = getattr(frames, attr, None)
            if depth is None:
                continue
            roi = depth[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2]
            path = save_dir / (name + ".png")
            cv2.imwrite(str(path), roi.astype(np.uint16))
            written[name] = path

        info = {
            "rejected": True,
            "reason": reason,
            "roi": [cfg.roi_x1, cfg.roi_y1, cfg.roi_x2, cfg.roi_y2],
        }
        json_path = save_dir / "rejection.json"
        json_path.write_text(json.dumps(info, indent=2))
        written["rejection"] = json_path

        return written

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

        organized = out_dir is not None
        save_dir = Path(out_dir) if organized else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / artifact_name("raw_rgb", "jpg", timestamp, organized)
        result_path = save_dir / artifact_name(f"{mode}_final", "png", timestamp, organized)
        json_path = save_dir / artifact_name(f"{mode}_final", "json", timestamp, organized)

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
            path = save_dir / artifact_name(key, "png", timestamp, organized)
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
