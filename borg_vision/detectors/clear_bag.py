"""Clear-bag measurement mode (no barcode gate).

ClearBagDetector runs SAM2 on the ROI, picks the best visible product mask
(contrast/texture scoring), then detects the clear bag around it from depth and
measures it. Behaviour is a verbatim port of run_clear_bag()/clear_bag_final.py;
only the shared lifecycle now lives in BaseDetector.

Clear bag uses a single full-resolution stereo, so frames.depth_class_aligned
is used.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from ..config import ClearBagConfig
from ..detection.clear_bag import run_sam2_product_and_bag
from ..results import ClearBagResult
from ..visualization import save_binary_mask
from ..visualization.clear_bag import draw_result, make_json_result
from .base import BaseDetector, artifact_name


class ClearBagDetector(BaseDetector):
    config_class = ClearBagConfig
    requires_barcode = False

    def detect(self, frames, barcode=None):
        """Run SAM2 product + clear-bag detection on the given frames.

        Returns ClearBagResult or None when no valid product mask was found.
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        raw = run_sam2_product_and_bag(
            self.cfg,
            frames.rgb,
            frames.depth_class_aligned,
            self._mask_generator,
            self.camera.intrinsics,
        )

        if raw is None:
            return None

        return ClearBagResult.from_raw(raw, frames)

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
            "clear_bag_product_mask": result.raw["mask"],
            "clear_bag_mask": result.raw["bag"]["bag_mask"],
        }

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Same artifact set as the original clear-bag script (keys: dir,
        raw_rgb, result, product_mask, bag_mask, heatmap, json, result_bgr)."""
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        organized = out_dir is not None
        save_dir = Path(out_dir) if organized else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / artifact_name("raw_rgb", "jpg", timestamp, organized)
        result_path = save_dir / artifact_name("clear_bag_final_output", "png", timestamp, organized)
        product_mask_path = save_dir / artifact_name("clear_bag_product_mask", "png", timestamp, organized)
        bag_mask_path = save_dir / artifact_name("clear_bag_mask", "png", timestamp, organized)
        heatmap_path = save_dir / artifact_name("clear_bag_depth_heatmap", "png", timestamp, organized)
        json_path = save_dir / artifact_name("clear_bag_final_output", "json", timestamp, organized)

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(product_mask_path, result.raw["mask"])
        save_binary_mask(bag_mask_path, result.raw["bag"]["bag_mask"])
        cv2.imwrite(str(heatmap_path), result.raw["depth_heatmap"])

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "product_mask": product_mask_path,
            "bag_mask": bag_mask_path,
            "heatmap": heatmap_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }
