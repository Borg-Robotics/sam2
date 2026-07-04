"""Per-mode detector classes.

BaseDetector holds the shared OAK-D + SAM2 lifecycle; each mode subclasses it.
`ProductDetector` is a backwards-compatible alias of `PackageDetector`.
"""

from .base import BaseDetector
from .box import BoxDetector
from .clear_bag import ClearBagDetector
from .inspection import InspectionDetector
from .object import ObjectDetector
from .package import PackageDetector
from .polymailer import PolymailerDetector
from .polymailer_release import (
    MonitorUpdate,
    PolymailerReleaseDetector,
    ReleasePhase,
)

# Backwards-compatible alias for the pre-refactor single-mode detector name.
ProductDetector = PackageDetector

__all__ = [
    "BaseDetector",
    "PackageDetector",
    "ObjectDetector",
    "BoxDetector",
    "PolymailerDetector",
    "PolymailerReleaseDetector",
    "ReleasePhase",
    "MonitorUpdate",
    "ClearBagDetector",
    "InspectionDetector",
    "ProductDetector",
]
