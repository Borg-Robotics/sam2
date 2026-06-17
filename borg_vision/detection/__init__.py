"""Per-mode pure detection logic.

Each mode's segmentation/scoring/measurement functions live in detection/<mode>.py
and take the mode's config as the first argument. Entry points are re-exported
here for convenience.
"""

from .box import run_sam2_cardboard_box
from .clear_bag import run_sam2_product_and_bag
from .object import run_sam2_object_segmentation
from .package import run_sam_package_depth_type
from .polymailer import run_sam2_polymailer

__all__ = [
    "run_sam_package_depth_type",
    "run_sam2_object_segmentation",
    "run_sam2_cardboard_box",
    "run_sam2_polymailer",
    "run_sam2_product_and_bag",
]
