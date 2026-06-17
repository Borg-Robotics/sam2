"""Per-mode detector classes.

BaseDetector holds the shared OAK-D + SAM2 lifecycle; each mode subclasses it.
`ProductDetector` is a backwards-compatible alias of `PackageDetector`.
"""

from .base import BaseDetector
from .object import ObjectDetector
from .package import PackageDetector

# Backwards-compatible alias for the pre-refactor single-mode detector name.
ProductDetector = PackageDetector

__all__ = [
    "BaseDetector",
    "PackageDetector",
    "ObjectDetector",
    "ProductDetector",
]
