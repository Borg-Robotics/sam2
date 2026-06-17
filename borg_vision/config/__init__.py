"""Per-mode configuration dataclasses.

BaseConfig holds shared camera/SAM2/ROI/depth fields; each mode subclasses it.
`ProductDetectionConfig` is kept as a backwards-compatible alias of
`PackageConfig` so existing consumers (the sam2_vision ROS2 node, the original
product_detection CLI) keep importing it unchanged.
"""

from .base import BaseConfig
from .object import ObjectConfig
from .package import PackageConfig

# Backwards-compatible alias for the pre-refactor single-mode config name.
ProductDetectionConfig = PackageConfig

__all__ = [
    "BaseConfig",
    "PackageConfig",
    "ObjectConfig",
    "ProductDetectionConfig",
]
