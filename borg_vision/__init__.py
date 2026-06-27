"""Borg Robotics vision library built on the SAM2 fork.

Importable, GUI-free detection pipeline for OAK-D Pro cameras. Multiple
detection modes (package/product, and — as they are ported — object, box,
clear_bag, polymailer) share one OAK-D + SAM2 base and are selected via
`get_detector(mode)`.

The heavy imports (torch, depthai) happen when a detector module is imported,
so importing borg_vision pulls them in via the convenience exports below.

Backwards compatibility: `ProductDetectionConfig`, `ProductDetector` and
`ProductDetectionResult` remain importable from the package root and are
aliases of the `Package*` classes.
"""

from .config import (
    BaseConfig,
    BoxConfig,
    ClearBagConfig,
    InspectionConfig,
    ObjectConfig,
    PackageConfig,
    PolymailerConfig,
    ProductDetectionConfig,
)
from .detectors import (
    BaseDetector,
    BoxDetector,
    ClearBagDetector,
    InspectionDetector,
    ObjectDetector,
    PackageDetector,
    PolymailerDetector,
    ProductDetector,
)
from .registry import (
    available_modes,
    config_class_for,
    get_detector,
    get_inspection_detector,
    resolve_mode,
)
from .results import (
    BaseResult,
    BoxResult,
    ClearBagResult,
    InspectionResult,
    ObjectResult,
    PackageResult,
    PolymailerResult,
    ProductDetectionResult,
)

__all__ = [
    # factory / registry
    "get_detector",
    "get_inspection_detector",
    "available_modes",
    "resolve_mode",
    "config_class_for",
    # base classes
    "BaseConfig",
    "BaseDetector",
    "BaseResult",
    # package mode
    "PackageConfig",
    "PackageDetector",
    "PackageResult",
    # object mode
    "ObjectConfig",
    "ObjectDetector",
    "ObjectResult",
    # box mode
    "BoxConfig",
    "BoxDetector",
    "BoxResult",
    # polymailer mode
    "PolymailerConfig",
    "PolymailerDetector",
    "PolymailerResult",
    # clear_bag mode
    "ClearBagConfig",
    "ClearBagDetector",
    "ClearBagResult",
    # inspection mode (dual-camera + OpenAI)
    "InspectionConfig",
    "InspectionDetector",
    "InspectionResult",
    # backwards-compatible aliases
    "ProductDetectionConfig",
    "ProductDetector",
    "ProductDetectionResult",
]
