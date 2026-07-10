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
        # image_paths/optimized_image_paths stay on the dataclass but are left
        # out of the report JSON: the images already live in the session dir.
        return {
            "request_id": self.request_id,
            "product_name": self.product_name,
            "result": self.result,
            "confidence": self.confidence,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "observed_defects": list(self.observed_defects),
            "usage": self.usage,
            "raw_response_id": self.raw_response_id,
        }


# Backwards-compatible alias for the pre-refactor single-mode result name.
ProductDetectionResult = PackageResult
