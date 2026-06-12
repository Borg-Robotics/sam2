"""Borg Robotics vision library built on the SAM2 fork.

Importable, GUI-free detection pipeline for OAK-D Pro cameras. The heavy
imports (torch, depthai) happen lazily via the submodules, so importing
borg_vision itself is cheap only if you import from the submodules directly;
the package-level convenience imports below pull everything in.
"""

from .config import ProductDetectionConfig
from .product_detector import ProductDetectionResult, ProductDetector

__all__ = [
    "ProductDetectionConfig",
    "ProductDetector",
    "ProductDetectionResult",
]
