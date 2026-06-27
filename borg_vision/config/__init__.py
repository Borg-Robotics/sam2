"""Per-mode configuration dataclasses.

BaseConfig holds shared camera/SAM2/ROI/depth fields; each mode subclasses it.
`ProductDetectionConfig` is kept as a backwards-compatible alias of
`PackageConfig` so existing consumers (the sam2_vision ROS2 node, the original
product_detection CLI) keep importing it unchanged.
"""

from .base import BaseConfig
from .box import BoxConfig
from .clear_bag import ClearBagConfig
from .inspection import InspectionConfig
from .object import ObjectConfig
from .package import PackageConfig
from .polymailer import PolymailerConfig

# Backwards-compatible alias for the pre-refactor single-mode config name.
ProductDetectionConfig = PackageConfig

__all__ = [
    "BaseConfig",
    "PackageConfig",
    "ObjectConfig",
    "BoxConfig",
    "PolymailerConfig",
    "ClearBagConfig",
    "InspectionConfig",
    "ProductDetectionConfig",
]
