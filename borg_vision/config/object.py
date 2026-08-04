"""Configuration for the object segmentation + center-depth mode.

ObjectConfig == BaseConfig plus the object-segmentation scoring/cleanup fields.
Every field corresponds 1:1 to a module-level constant of the original
run_object()/object_detection_final.py, with the original value as default.

Differences from the package mode captured as defaults: narrower locked ROI
(330,60)-(940,700), negative depth x-shift (-40), and a single full-resolution
stereo stream (needs_measurement_stereo = False) reused for the center depth.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ObjectConfig(BaseConfig):
    save_dir: str = "object_segmentation_depth_results"

    # Single full-res stereo; center depth comes from the classification depth.
    needs_measurement_stereo: bool = False

    # Distance to the base surface, used for height_mm. Object mode inherited
    # 695.0 from BaseConfig, but the tray this rig measures against sits at
    # 705.0 (measured median over the surface, and what box/polymailer/
    # clear_bag already use); the old value under-reported every height by 10mm.
    base_depth_mm: float = 705.0

    # Locked ROI (from the polymailer/object script).
    roi_x1: int = 330
    roi_y1: int = 30
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (note the negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    # Center-depth sampling.
    center_depth_radius_px: int = 35
    min_center_depth_count: int = 30

    # Object mask gating & scoring.
    min_area_ratio: float = 0.008
    max_area_ratio: float = 0.35
    target_area_ratio: float = 0.08

    min_rectangularity: float = 0.18
    max_aspect_ratio: float = 7.0

    reject_masks_touching_roi_border: bool = True
    roi_border_margin_px: int = 12

    use_mask_cleanup: bool = True
    use_convex_hull: bool = True
    mask_close_kernel_px: int = 25
    mask_close_iterations: int = 2
    # Cap on how much mask cleanup (close + convex hull) may grow a mask. At
    # 2.8 the hull was free to bridge an object to whatever it sits on -- a box
    # on a pedestal came back as one 1.48x-inflated mask, which then outscored
    # the correct one because area_score rewards being closer to
    # target_area_ratio. 1.2 still allows the hull to fill small holes in a
    # mask but rejects it once it balloons.
    max_cleaned_area_growth: float = 1.2

    # Scoring weights.
    center_score_weight: float = 1.4
    area_score_weight: float = 1.2
    rect_score_weight: float = 1.0
    sam_iou_score_weight: float = 0.5
    sam_stability_score_weight: float = 0.5

    debug_print_masks: bool = False
