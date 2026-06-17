"""Package/product detection mode (barcode-gated).

PackageDetector is the refactor of the original ProductDetector: it runs the
barcode-gated SAM2 package pipeline (box vs. polymailer classification, depth
plane fitting, product-inside detection). Behaviour and saved-artifact names
are identical to the pre-refactor detector; only the shared lifecycle now lives
in BaseDetector.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from ..config import PackageConfig
from ..detection import run_sam_package_depth_type
from ..results import PackageResult
from ..visualization import (
    draw_result,
    make_json_result,
    save_accepted_masks,
    save_binary_mask,
)
from .base import BaseDetector


class PackageDetector(BaseDetector):
    config_class = PackageConfig
    requires_barcode = True

    def detect(self, frames, barcode=None):
        """Run the full SAM2 package detection on the given frames.

        Returns PackageResult or None when no valid package mask was found.
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

        return PackageResult.from_raw(raw, frames)

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

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Save the same artifact set, with the same filenames, as the original
        script (keys: dir, raw_rgb, result, mask, product_mask, heatmap, json,
        result_bgr)."""
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) if out_dir is not None else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

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
