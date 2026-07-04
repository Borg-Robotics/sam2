"""Per-mode detection result dataclasses.

`BaseResult` carries the attachments every mode shares (the raw detection dict
and the Frames the detection ran on); each mode subclasses it with its own
measurement fields. `PackageResult` mirrors the final_output dict of the
original product_detection_final.py script, and `ProductDetectionResult` is a
backwards-compatible alias so existing consumers keep importing it unchanged.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .camera import Frames


@dataclass
class BaseResult:
    """Common attachments shared by every mode's result.

    Subclasses add the mode-specific measurement fields and are built from the
    mode's raw detection dict via `from_raw`.
    """

    # Non-serialized attachments
    raw: dict = field(default=None, repr=False)        # full detection dict
    frames: Frames = field(default=None, repr=False)   # frames the detection ran on


@dataclass
class PackageResult(BaseResult):
    """Mirrors the final_output dict of the original script (units: mm / deg,
    camera frame: x right, y down, z = depth forward)."""

    barcode_type: Optional[str] = None
    barcode_data: Optional[str] = None
    package_type: str = ""                              # "box" | "polymailer"
    package_type_confidence_percent: float = 0.0
    mask_source: str = ""
    top_face_depth_mm: Optional[float] = None
    package_depth_mm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None
    product_inside_found: bool = False
    product_inside_center_x_mm: Optional[float] = None
    product_inside_center_y_mm: Optional[float] = None
    product_inside_center_pixel_u: Optional[int] = None
    product_inside_center_pixel_v: Optional[int] = None
    product_inside_depth_mm: Optional[float] = None

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(raw=raw, frames=frames, **raw["final_output"])

    def to_json_dict(self, cfg, timestamp=None):
        from .visualization import make_json_result

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


@dataclass
class ObjectResult(BaseResult):
    """Generic object segmentation + center depth (units: mm, pixels in the
    full RGB frame)."""

    center_pixel_u: Optional[int] = None
    center_pixel_v: Optional[int] = None
    distance_mm: Optional[float] = None
    depth_count: int = 0

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(raw=raw, frames=frames, **raw["final_output"])

    def to_json_dict(self, cfg, timestamp=None):
        from .visualization.object import make_json_result

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


@dataclass
class BoxResult(BaseResult):
    """Cardboard-box measurement (units: mm / deg, camera frame: x right,
    y down, z = depth forward)."""

    box_face_depth_mm: Optional[float] = None
    box_depth_mm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(raw=raw, frames=frames, **raw["final_output"])

    def to_json_dict(self, cfg, timestamp=None):
        from .visualization.box import make_json_result

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


@dataclass
class PolymailerReleaseResult(BaseResult):
    """Latched outcome of the polymailer product-release monitor (units: mm,
    camera frame: x right, y down, z = depth forward).

    Built when the monitor latches FULL PRODUCT OUT; `raw` carries the latched
    candidate dict (mask + metrics) and `frames` the frame at latch time.
    """

    released: bool = False
    product_center_x_mm: Optional[float] = None
    product_center_y_mm: Optional[float] = None
    product_depth_mm: Optional[float] = None
    height_above_base_mm: Optional[float] = None
    phase_history: List = field(default_factory=list)

    @classmethod
    def from_candidate(cls, cfg, candidate, frames, intrinsics, phase_history):
        import numpy as np

        from .detection.package import pixel_to_camera_xy_mm

        height_mm = candidate.get("height_above_base_mm")
        depth_mm = None

        # Robust product depth: median valid depth inside the latched mask,
        # falling back to base depth minus the measured height.
        mask = candidate.get("mask")
        if mask is not None and frames is not None:
            depth_roi = frames.depth_class_aligned[
                cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2
            ]
            values = depth_roi[
                (mask == 1)
                & (depth_roi > cfg.min_valid_depth_mm)
                & (depth_roi < cfg.max_valid_depth_mm)
            ]
            if values.size > 0:
                depth_mm = float(np.median(values.astype(np.float32)))

        if depth_mm is None and height_mm is not None:
            depth_mm = float(cfg.base_depth_mm) - float(height_mm)

        center = candidate.get("center")
        x_mm = y_mm = None
        if center is not None and depth_mm is not None:
            center_full = (
                float(center[0]) + cfg.roi_x1,
                float(center[1]) + cfg.roi_y1,
            )
            x_mm, y_mm = pixel_to_camera_xy_mm(center_full, depth_mm, intrinsics)

        return cls(
            raw={"candidate": candidate},
            frames=frames,
            released=True,
            product_center_x_mm=x_mm,
            product_center_y_mm=y_mm,
            product_depth_mm=depth_mm,
            height_above_base_mm=(
                float(height_mm) if height_mm is not None else None
            ),
            phase_history=list(phase_history),
        )


@dataclass
class PolymailerResult(BaseResult):
    """Polymailer measurement + product bulge inside (units: mm / deg, camera
    frame: x right, y down, z = depth forward)."""

    polymailer_face_depth_mm: Optional[float] = None
    polymailer_depth_mm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None
    product_inside_found: bool = False
    product_inside_center_x_mm: Optional[float] = None
    product_inside_center_y_mm: Optional[float] = None
    product_inside_center_face_depth: Optional[float] = None

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(raw=raw, frames=frames, **raw["final_output"])

    def to_json_dict(self, cfg, timestamp=None):
        from .visualization.polymailer import make_json_result

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


@dataclass
class ClearBagResult(BaseResult):
    """Clear-bag measurement + visible product inside (units: mm / deg, camera
    frame: x right, y down, z = depth forward)."""

    product_face_depth_mm: Optional[float] = None
    clearbag_depth_mm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    center_x_mm: Optional[float] = None
    center_y_mm: Optional[float] = None
    product_inside_center_x_mm: Optional[float] = None
    product_inside_center_y_mm: Optional[float] = None

    @classmethod
    def from_raw(cls, raw, frames):
        return cls(raw=raw, frames=frames, **raw["final_output"])

    def to_json_dict(self, cfg, timestamp=None):
        from .visualization.clear_bag import make_json_result

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return make_json_result(cfg, self.raw, timestamp)


@dataclass
class InspectionResult:
    """Product-inspection verdict from the OpenAI vision model.

    Unlike the segmentation results above, inspection has no raw detection dict
    or Frames -- it carries the captured/optimized image paths and the verdict.
    """

    result: str = ""                                # "good" | "damaged" | "manual_review"
    confidence: float = 0.0                         # 0.0 - 1.0
    summary: str = ""
    reasons: List[str] = field(default_factory=list)
    observed_defects: List[str] = field(default_factory=list)
    product_name: str = ""
    request_id: str = ""
    image_paths: List[str] = field(default_factory=list)         # captured originals
    optimized_image_paths: List[str] = field(default_factory=list)
    usage: Optional[dict] = field(default=None, repr=False)
    raw_response_id: Optional[str] = None

    @classmethod
    def from_run(cls, run, product_name, request_id, image_paths):
        """Build from the dict returned by inspection.run_inspection."""
        return cls(
            result=run["result"],
            confidence=run["confidence"],
            summary=run["summary"],
            reasons=run["reasons"],
            observed_defects=run["observed_defects"],
            product_name=product_name,
            request_id=request_id,
            image_paths=[str(p) for p in image_paths],
            optimized_image_paths=run["optimized_image_paths"],
            usage=run["usage"],
            raw_response_id=run["raw_response_id"],
        )

    def to_json_dict(self):
        return {
            "request_id": self.request_id,
            "product_name": self.product_name,
            "result": self.result,
            "confidence": self.confidence,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "observed_defects": list(self.observed_defects),
            "image_paths": list(self.image_paths),
            "optimized_image_paths": list(self.optimized_image_paths),
            "usage": self.usage,
            "raw_response_id": self.raw_response_id,
        }


# Backwards-compatible alias for the pre-refactor single-mode result name.
ProductDetectionResult = PackageResult
