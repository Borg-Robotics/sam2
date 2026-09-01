"""Object segmentation + center-depth mode (no barcode gate).

ObjectDetector runs SAM2 on the ROI, picks the best generic object mask and
reports its center pixel and center depth. Behaviour is a verbatim port of
run_object()/object_detection_final.py; only the shared lifecycle now lives in
BaseDetector.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import ObjectConfig
from ..detection import run_sam2_object_segmentation
from ..results import ObjectResult
from ..visualization.object import draw_result, make_json_result
from .base import BaseDetector, artifact_name


class ObjectDetector(BaseDetector):
    config_class = ObjectConfig
    requires_barcode = False

    def detect(self, frames, barcode=None):
        """Run SAM2 object segmentation + top-face depth on the given frames.

        Returns ObjectResult or None when no valid object mask was found.
        Depth is measured from the stream chosen by
        cfg.object_use_measurement_depth (see _measure_depth).
        """
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        # Under BORG_TIME_FRAMES=1, report the detection cost so it can be
        # compared against the [frames] line from get_frames.
        import os
        import time

        _timed = os.environ.get("BORG_TIME_FRAMES") == "1"
        _t0 = time.time() if _timed else None

        raw = run_sam2_object_segmentation(
            self.cfg,
            frames.rgb,
            self._measure_depth(frames),
            self._mask_generator,
            self.camera.intrinsics,
        )

        if _timed:
            print(
                f"[detect] run_sam2_object_segmentation "
                f"{(time.time() - _t0) * 1000:.0f}ms",
                flush=True,
            )

        if raw is None:
            return None

        return ObjectResult.from_raw(raw, frames)

    def _measure_depth(self, frames):
        """The depth stream all measuring runs on (segmentation is RGB-only).

        The measurement stereo sees small and glossy objects the
        classification stream is near-blind to; on a camera without a
        dedicated measurement stream the two are the same array. Debug
        artifacts save this same stream so replays see what detection saw."""
        if self.cfg.object_use_measurement_depth:
            return frames.depth_measure_aligned
        return frames.depth_class_aligned

    # ---------------------------------------------------------- visualization

    def _draw_result(self, result):
        return draw_result(
            self.cfg,
            result.frames.rgb,
            self._measure_depth(result.frames),
            result.raw,
        )

    def _serialize(self, result, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(self.cfg, result.raw, timestamp)

    def _debug_artifacts(self, result):
        # The object mask (saved via save_binary_mask by the base template).
        return {"object_mask": result.raw["object"]["mask"]}

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Same artifact set as the original object script (keys: dir, raw_rgb,
        depth_aligned, result, mask, json, result_bgr).

        Note: the pre-alignment raw depth is not retained by the library camera,
        so only the aligned depth is dumped as .npy.
        """
        cfg = self.cfg

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        organized = out_dir is not None
        save_dir = Path(out_dir) if organized else Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_bgr = self._draw_result(result)

        from ..visualization import save_binary_mask

        raw_path = save_dir / artifact_name("raw_rgb", "jpg", timestamp, organized)
        depth_aligned_path = save_dir / artifact_name("depth_aligned", "npy", timestamp, organized)
        result_path = save_dir / artifact_name("object_segment_depth", "png", timestamp, organized)
        mask_path = save_dir / artifact_name("object_mask", "png", timestamp, organized)
        json_path = save_dir / artifact_name("object_segment_depth", "json", timestamp, organized)

        cv2.imwrite(str(raw_path), result.frames.rgb)
        np.save(str(depth_aligned_path), self._measure_depth(result.frames))
        cv2.imwrite(str(result_path), result_bgr)
        save_binary_mask(mask_path, result.raw["object"]["mask"])

        import json

        with open(json_path, "w") as f:
            json.dump(make_json_result(cfg, result.raw, timestamp), f, indent=2)

        return {
            "dir": save_dir,
            "raw_rgb": raw_path,
            "depth_aligned": depth_aligned_path,
            "result": result_path,
            "mask": mask_path,
            "json": json_path,
            "result_bgr": result_bgr,
        }
