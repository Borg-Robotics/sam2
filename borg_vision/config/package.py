"""Configuration for the package/product detection mode.

PackageConfig == BaseConfig (shared camera/SAM2/ROI/depth fields) plus the
package-specific scoring, box/polymailer classification, and product-inside
fields. Every field corresponds 1:1 to a module-level constant of the original
product_detection_final.py script (lowercased), with the original value as the
default. Behaviour is identical to the pre-refactor ProductDetectionConfig.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class PackageConfig(BaseConfig):
    save_dir: str = "package_barcode_detection_results"

    base_depth_mm: float = 695.0
    package_size_scale: float = 1.04

    barcode_decode_upscale: float = 2.0
    barcode_draw_enable: bool = True

    # ----- package (product-like) mask gating & scoring ----------------
    min_package_area_ratio: float = 0.003
    max_package_area_ratio: float = 0.70
    target_package_area_ratio: float = 0.18

    min_package_rectangularity: float = 0.08
    max_package_aspect_ratio: float = 10.0
    min_package_center_score: float = 0.18

    package_mask_close_kernel_px: int = 19
    package_mask_close_iterations: int = 1
    package_max_cleaned_area_growth: float = 2.8

    safe_erode_radius_px: int = 12

    hard_reject_roller_like: bool = True
    roller_strip_min_width_ratio: float = 0.72
    roller_strip_max_height_ratio: float = 0.22

    reject_edge_strips: bool = True
    edge_strip_margin_px: int = 24
    edge_strip_max_width_ratio: float = 0.16
    edge_strip_max_height_ratio: float = 0.16

    ring_dilate_px: int = 25

    good_contrast: float = 35.0
    bad_contrast: float = 5.0
    good_texture: float = 28.0
    bad_texture: float = 4.0

    area_score_weight: float = 1.25
    center_score_weight: float = 1.20
    contrast_score_weight: float = 1.30
    texture_score_weight: float = 0.75
    rect_score_weight: float = 0.85
    bbox_size_score_weight: float = 0.65
    safe_area_score_weight: float = 0.45
    sam_iou_score_weight: float = 0.45
    sam_stability_score_weight: float = 0.45

    # ----- cardboard-box mask gating & scoring -------------------------
    min_box_area_ratio: float = 0.04
    max_box_area_ratio: float = 0.75
    target_box_area_ratio: float = 0.45

    min_box_rectangularity: float = 0.35
    max_box_aspect_ratio: float = 4.0

    min_box_value: float = 70
    min_box_color_score: float = 0.25

    box_mask_close_kernel_px: int = 21
    box_mask_close_iterations: int = 1
    box_max_cleaned_area_growth: float = 2.8

    box_flat_std_good_mm: float = 6.0
    box_flat_std_bad_mm: float = 18.0

    box_residual_range_good_mm: float = 20.0
    box_residual_range_bad_mm: float = 55.0

    box_rectangularity_good: float = 0.90
    box_rectangularity_bad: float = 0.60

    box_score_threshold: float = 0.60
    box_score_poly_signature_penalty: float = 0.65

    # ----- polymailer depth-signature classification -------------------
    poly_center_edge_soft_mm: float = 14.0
    poly_center_edge_hard_mm: float = 24.0

    center_edge_override_min_valid_fraction: float = 0.20
    center_edge_override_min_valid_points: int = 20000

    sparse_depth_box_max_valid_fraction: float = 0.08
    sparse_depth_box_min_rectangularity: float = 0.88
    sparse_depth_box_min_area_ratio: float = 0.12

    center_edge_kernel_px: int = 70
    max_plane_points: int = 8000

    poly_signature_min_valid_fraction: float = 0.18
    poly_signature_min_valid_points: int = 15000
    poly_signature_min_area_ratio: float = 0.16

    poly_signature_override_threshold: float = 0.30

    poly_center_closer_soft_mm: float = 4.0
    poly_center_closer_hard_mm: float = 16.0

    poly_full_depth_range_soft_mm: float = 7.0
    poly_full_depth_range_hard_mm: float = 24.0

    poly_edge_depth_range_soft_mm: float = 7.0
    poly_edge_depth_range_hard_mm: float = 26.0

    poly_side_spread_soft_mm: float = 4.0
    poly_side_spread_hard_mm: float = 20.0

    poly_multi_signal_min_count: int = 3
    poly_multi_signal_confidence: float = 0.82

    min_surface_depth_count: int = 30

    # ----- product-inside (polymailer) detection -----------------------
    product_inside_enable: bool = True
    product_inside_edge_erode_px: int = 45
    product_inside_min_closer_than_poly_mm: float = 8.0
    product_inside_max_closer_than_poly_mm: float = 90.0
    product_inside_close_kernel_px: int = 17
    product_inside_dilate_px: int = 5
    product_inside_min_area_ratio_of_package: float = 0.015
    product_inside_max_area_ratio_of_package: float = 0.65
    product_inside_min_valid_pixels: int = 80

    heatmap_max_closer_than_base_mm: float = 180.0

    debug_print_masks: bool = False
    debug_save_all_accepted_masks: bool = False
