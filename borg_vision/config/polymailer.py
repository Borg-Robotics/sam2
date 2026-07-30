"""Configuration for the polymailer (padded envelope) measurement mode.

PolymailerConfig == BaseConfig plus the polymailer HSV-scoring, mask-cleanup,
dimension and product-bulge fields. Every field corresponds 1:1 to a
module-level constant of the original run_polymailer()/polymailer_final.py,
with the original value as default.

Differences captured as defaults: narrower locked ROI (330,60)-(940,700),
negative depth x-shift (-40), base depth 705 mm, and a single full-resolution
stereo (needs_measurement_stereo = False) used for both face depth and the
product bulge.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class PolymailerConfig(BaseConfig):
    save_dir: str = "polymailer_segment_depth_heatmap_results"

    # Single full-res stereo; depth comes from the classification stream.
    needs_measurement_stereo: bool = False

    base_depth_mm: float = 705.0

    # Locked ROI.
    roi_x1: int = 330
    roi_y1: int = 30
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    center_depth_radius_px: int = 35
    min_center_depth_count: int = 30

    # Dimension measurement.
    poly_measure_erode_px: int = 8
    poly_size_scale: float = 1.04

    # Polymailer mask gating & scoring.
    min_poly_area_ratio: float = 0.04
    max_poly_area_ratio: float = 0.75
    target_poly_area_ratio: float = 0.55

    min_poly_rectangularity: float = 0.35
    max_poly_aspect_ratio: float = 3.5

    min_poly_value: float = 80
    min_poly_color_score: float = 0.30

    # Polymailer scoring weights.
    poly_color_score_weight: float = 3.0
    poly_rect_score_weight: float = 1.5
    poly_area_score_weight: float = 1.2
    poly_center_score_weight: float = 1.0

    # Mask cleanup (close kernel + max area growth).
    poly_mask_close_kernel_px: int = 21
    poly_mask_close_iterations: int = 1
    poly_max_cleaned_area_growth: float = 2.8

    # Product bulge detection inside the polymailer.
    poly_inner_erode_px: int = 22
    poly_surface_depth_percentile: int = 75

    poly_bulge_min_mm: float = 7
    poly_bulge_max_mm: float = 140

    poly_bulge_open_kernel_px: int = 7
    poly_bulge_close_kernel_px: int = 28
    poly_bulge_dilate_px: int = 5

    min_product_area_ratio_of_poly: float = 0.010
    max_product_area_ratio_of_poly: float = 0.65

    debug_print_masks: bool = False
