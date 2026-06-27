"""Polymailer measurement mode (no barcode gate).

PolymailerDetector runs SAM2 on the ROI, picks the best polymailer mask by HSV
+ geometry scoring, measures its face depth/dimensions, and detects any product
bulge inside. Behaviour is a verbatim port of run_polymailer()/
polymailer_final.py; only the shared lifecycle now lives in BaseDetector.

Polymailer uses a single full-resolution stereo, so frames.depth_class_aligned
is used.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from ..config import PolymailerConfig
from ..detection.polymailer import run_sam2_polymailer
from ..results import PolymailerResult
from ..visualization import save_binary_mask
from ..visualization.polymailer import draw_result, make_json_result
from .base import BaseDetector, artifact_name


class PolymailerDetector(BaseDetector):
    config_class = PolymailerConfig
    requires_barcode = False

    def detect(self, frames, barcode=None):
        """Run SAM2 polymailer detection + measurement on the given frames.

        Returns PolymailerResult or None when no valid polymailer mask was found.
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        raw = run_sam2_polymailer(
            self.cfg,
            frames.rgb,
            frames.depth_class_aligned,
            self._mask_generator,
            self.camera.intrinsics,
        )

        if raw is None:
            return None

        return PolymailerResult.from_raw(raw, frames)

    # ---------------------------------------------------------- visualization

    def _draw_result(self, result):
        return draw_result(
            self.cfg,
            result.frames.rgb,
            result.frames.depth_class_aligned,
            result.raw,
        )

    def _serialize(self, result, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(self.cfg, result.raw, timestamp)

    def _debug_artifacts(self, result):
        return {
            "polymailer_mask": result.raw["poly"]["mask"],
            "product_bulge_mask": result.raw["product"]["mask"],
        }

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Same artifact set as the original polymailer script (keys: dir,
        raw_rgb, result, poly_mask, product_mask, json, result_bgr)."""
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        organized = out_dir is not None
        save_dir = Path(out_dir) if organized else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / artifact_name("raw_rgb", "jpg", timestamp, organized)
        result_path = save_dir / artifact_name("polymailer_output", "png", timestamp, organized)
        poly_mask_path = save_dir / artifact_name("polymailer_mask", "png", timestamp, organized)
        product_mask_path = save_dir / artifact_name("product_bulge_mask", "png", timestamp, organized)
        json_path = save_dir / artifact_name("polymailer_output", "json", timestamp, organized)

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(poly_mask_path, result.raw["poly"]["mask"])
        save_binary_mask(product_mask_path, result.raw["product"]["mask"])

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "poly_mask": poly_mask_path,
            "product_mask": product_mask_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }
