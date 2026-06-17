"""Configuration for the clear-bag (visible product + bag) measurement mode.

ClearBagConfig == BaseConfig plus the visible-product scoring fields and the
depth-based bag-detection/expansion fields. Every field corresponds 1:1 to a
module-level constant of the original run_clear_bag()/clear_bag_final.py, with
the original value as default.

Differences captured as defaults: narrower locked ROI (330,60)-(940,700),
negative depth x-shift (-40), base depth 705 mm, and a single full-resolution
stereo (needs_measurement_stereo = False).
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ClearBagConfig(BaseConfig):
    save_dir: str = "clear_bag_final_results"

    # Single full-res stereo; depth comes from the classification stream.
    needs_measurement_stereo: bool = False

    base_depth_mm: float = 705.0
    poly_size_scale: float = 1.04

    # Locked ROI.
    roi_x1: int = 330
    roi_y1: int = 60
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    # ----- visible product mask gating & scoring -----------------------
    min_product_area_ratio: float = 0.003
    max_product_area_ratio: float = 0.40
    target_product_area_ratio: float = 0.08

    min_product_rectangularity: float = 0.08
    max_product_aspect_ratio: float = 10.0

    hard_reject_roller_like: bool = True
    roller_strip_min_width_ratio: float = 0.70
    roller_strip_max_height_ratio: float = 0.25

    use_mask_cleanup: bool = True
    mask_close_kernel_px: int = 17
    mask_close_iterations: int = 1
    max_cleaned_area_growth: float = 2.5
    safe_erode_radius_px: int = 12

    ring_dilate_px: int = 25
    good_contrast: float = 35.0
    bad_contrast: float = 5.0
    good_texture: float = 28.0
    bad_texture: float = 4.0

    area_score_weight: float = 1.2
    center_score_weight: float = 1.0
    contrast_score_weight: float = 2.4
    texture_score_weight: float = 1.0
    rect_score_weight: float = 0.5
    safe_area_score_weight: float = 0.4
    sam_iou_score_weight: float = 0.4
    sam_stability_score_weight: float = 0.4

    center_top_face_radius_px: int = 35
    min_center_top_face_depth_count: int = 30

    # ----- depth-based bag detection -----------------------------------
    bag_closer_than_base_min_mm: float = 25
    bag_closer_than_base_max_mm: float = 170

    bag_seed_dilate_px: int = 135
    bag_component_keep_near_product_px: int = 190

    bag_close_kernel_px: int = 19
    bag_dilate_kernel_px: int = 7
    bag_erode_kernel_px: int = 5

    bag_min_area_ratio: float = 0.025
    bag_max_area_ratio: float = 0.70

    bag_use_median_blur_depth: bool = True
    bag_depth_median_blur_ksize: int = 5

    # ----- bag corner expansion ----------------------------------------
    bag_corner_expand_enable: bool = True
    bag_corner_expand_dilate_px: int = 9
    bag_edge_activity_smooth_px: int = 21
    bag_edge_min_activity_px: int = 12
    bag_edge_activity_fraction: float = 0.12
    bag_corner_window_px: int = 45
    bag_corner_min_pixels: int = 25
    bag_expand_padding_x_px: int = 0
    bag_expand_padding_y_px: int = 0

    heatmap_max_closer_than_base_mm: float = 180.0

    debug_print_masks: bool = False
