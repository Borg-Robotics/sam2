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
import numpy as np

from ..config import PackageConfig
from ..detection import run_sam_package_depth_type
from ..results import PackageResult
from ..visualization import (
    draw_result,
    make_json_result,
    save_accepted_masks,
    save_binary_mask,
)
from .base import BaseDetector, artifact_name


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
        """Save the original script's artifact set (keys: dir, raw_rgb, result,
        mask, product_mask, heatmap, json, result_bgr) plus the two raw depth
        arrays (depth_measure_aligned, depth_class_aligned) needed to retune
        product-inside detection offline."""
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        organized = out_dir is not None
        save_dir = Path(out_dir) if organized else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        raw_path = save_dir / artifact_name("raw_rgb", "jpg", timestamp, organized)
        result_path = save_dir / artifact_name("package_final", "png", timestamp, organized)
        mask_path = save_dir / artifact_name("package_mask", "png", timestamp, organized)
        product_mask_path = save_dir / artifact_name("product_inside_mask", "png", timestamp, organized)
        product_core_path = save_dir / artifact_name("product_core_mask", "png", timestamp, organized)
        heatmap_path = save_dir / artifact_name("package_depth_heatmap", "png", timestamp, organized)
        json_path = save_dir / artifact_name("package_final", "json", timestamp, organized)

        # Both depth streams, unquantised. The heatmap is a JET-colourmapped,
        # min-max-normalised, alpha-blended view, so it cannot be used to
        # re-derive millimetres -- retuning product-inside detection offline
        # needs the actual arrays. Measurement depth is what the product-inside
        # estimate runs on; classification depth is the higher-resolution
        # stream, kept so the two can be compared on the same capture.
        depth_measure_path = save_dir / artifact_name(
            "depth_measure_aligned", "npy", timestamp, organized
        )
        depth_class_path = save_dir / artifact_name(
            "depth_class_aligned", "npy", timestamp, organized
        )

        cv2.imwrite(str(raw_path), result.frames.rgb)
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(mask_path, result.raw["mask"])
        save_binary_mask(product_mask_path, result.raw["product_inside"]["mask"])
        save_binary_mask(product_core_path, result.raw["product_inside"]["core_mask"])
        cv2.imwrite(str(heatmap_path), result.raw["depth_heatmap"])
        np.save(str(depth_measure_path), result.frames.depth_measure_aligned)
        np.save(str(depth_class_path), result.frames.depth_class_aligned)

        if cfg.debug_save_all_accepted_masks:
            save_accepted_masks(save_dir, timestamp, result.raw, organized)

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "result": result_path,
            "mask": mask_path,
            "product_mask": product_mask_path,
            "product_core_mask": product_core_path,
            "heatmap": heatmap_path,
            "depth_measure_aligned": depth_measure_path,
            "depth_class_aligned": depth_class_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }
