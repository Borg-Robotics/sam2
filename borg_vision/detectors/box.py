"""Cardboard-box measurement mode (no barcode gate).

BoxDetector runs SAM2 on the ROI, picks the best cardboard-box mask by HSV +
geometry scoring, and reports its face depth, dimensions and center. Behaviour
is a verbatim port of run_box()/box_detection_final.py; only the shared
lifecycle now lives in BaseDetector.

Box measures from the measurement stereo stream (640x400 aligned to RGB), the
same stream the package mode uses, so frames.depth_measure_aligned is used.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from ..config import BoxConfig
from ..detection.box import run_sam2_cardboard_box
from ..results import BoxResult
from ..visualization import save_binary_mask
from ..visualization.box import draw_result, make_json_result
from .base import BaseDetector


class BoxDetector(BaseDetector):
    config_class = BoxConfig
    requires_barcode = False

    def detect(self, frames, barcode=None):
        """Run SAM2 cardboard-box detection + measurement on the given frames.

        Returns BoxResult or None when no valid box mask was found.
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        raw = run_sam2_cardboard_box(
            self.cfg,
            frames.rgb,
            frames.depth_measure_aligned,
            self._mask_generator,
            self.camera.intrinsics,
        )

        if raw is None:
            return None

        return BoxResult.from_raw(raw, frames)

    # ---------------------------------------------------------- visualization

    def _draw_result(self, result):
        return draw_result(
            self.cfg,
            result.frames.rgb,
            result.frames.depth_measure_aligned,
            result.raw,
        )

    def _serialize(self, result, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(self.cfg, result.raw, timestamp)

    def _debug_artifacts(self, result):
        return {"cardboard_box_mask": result.raw["box"]["mask"]}

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Same artifact set as the original box script (keys: dir, raw_rgb,
        result, mask, json, result_bgr)."""
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) if out_dir is not None else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / f"raw_rgb_{timestamp}.jpg"
        result_path = save_dir / f"cardboard_box_measurements_{timestamp}.png"
        mask_path = save_dir / f"cardboard_box_mask_{timestamp}.png"
        json_path = save_dir / f"cardboard_box_measurements_{timestamp}.json"

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(mask_path, result.raw["box"]["mask"])

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "mask": mask_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }
