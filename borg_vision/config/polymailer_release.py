"""Configuration for the polymailer product-release monitoring mode.

PolymailerReleaseConfig == BaseConfig plus every tunable of the original
polymailer_product_release.py prototype ("moving-slit product-exit test").
Every field corresponds 1:1 to a module-level constant of that script, with
the original value as default, grouped by the script's comment blocks. The
fields stay flat so from_dict()/YAML `overrides:` partial loading works.

Camera-level differences captured as defaults (matching PolymailerConfig):
locked ROI (330,60)-(940,700), negative depth x-shift (-40), base depth
705 mm, single full-resolution stereo (needs_measurement_stereo = False).

Fields that did NOT exist in the prototype (added for the monitor lifecycle):
`baseline_stable_updates` (replaces the interactive SPACE key: the baseline
locks after this many consecutive stable bag selections) and
`tracker_lost_grace_sec` (how long the tracker may stay lost before the
monitor reports failure).
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class PolymailerReleaseConfig(BaseConfig):
    save_dir: str = "polymailer_release_results"

    # Single full-res stereo; depth comes from the classification stream.
    needs_measurement_stereo: bool = False

    warmup_seconds: float = 3.0
    base_depth_mm: float = 705.0

    # Locked ROI (same station mount as the polymailer mode).
    roi_x1: int = 330
    roi_y1: int = 60
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    # ----- SAM (base fields carry the FULL-ROI generator) ---------------
    sam_points_per_side: int = 16              # FULL_SAM_POINTS_PER_SIDE
    slit_sam_points_per_side: int = 12
    sam_pred_iou_thresh: float = 0.76
    sam_stability_score_thresh: float = 0.80
    sam_min_mask_region_area: int = 300
    full_sam_interval_frames: int = 2
    slit_sam_interval_frames: int = 1
    use_cuda_autocast: bool = True

    # ----- monitor lifecycle (new; no prototype equivalent) -------------
    default_slit_side: str = "top"
    baseline_stable_updates: int = 2
    baseline_stable_min_iou: float = 0.60
    tracker_lost_grace_sec: float = 2.0

    # ----- arming: do not decide immediately after baseline -------------
    exit_arm_delay_seconds: float = 0.60
    exit_arm_min_slit_motion_px: float = 18.0

    # ----- initial polymailer selection ----------------------------------
    min_poly_area_ratio: float = 0.04
    max_poly_area_ratio: float = 0.75
    target_poly_area_ratio: float = 0.55
    min_poly_rectangularity: float = 0.35
    max_poly_aspect_ratio: float = 3.5
    min_poly_value: float = 80
    min_poly_color_score: float = 0.30

    # ----- optical-flow tracking of the moving polymailer and slit ------
    flow_max_corners: int = 300
    flow_quality_level: float = 0.01
    flow_min_distance: int = 7
    flow_block_size: int = 7
    flow_win_size: tuple = (31, 31)
    flow_max_level: int = 4
    flow_min_good_points: int = 12
    flow_reseed_point_count: int = 55
    flow_reseed_interval_frames: int = 12
    flow_forward_backward_max_error: float = 1.8
    flow_ransac_reproj_threshold: float = 3.0

    # Robust moving-slit tracking (similarity transforms only, slit-local
    # point weighting).
    flow_local_slit_band_inward_px: float = 150.0
    flow_local_slit_band_outward_px: float = 28.0
    flow_local_slit_side_margin_px: float = 65.0
    flow_min_local_slit_points: int = 8
    flow_min_affine_inlier_ratio: float = 0.45
    flow_min_frame_scale: float = 0.82
    flow_max_frame_scale: float = 1.22
    flow_max_frame_rotation_deg: float = 24.0
    flow_max_frame_translation_px: float = 85.0
    flow_max_slit_length_ratio: float = 1.28
    flow_min_slit_length_ratio: float = 0.78
    flow_slit_snap_min_parallel: float = 0.68
    flow_slit_snap_max_midpoint_distance_px: float = 105.0
    flow_slit_snap_blend: float = 0.72

    # ----- dynamic slit corridor -----------------------------------------
    slit_side_margin_px: int = 75
    slit_outward_distance_px: int = 220
    slit_inward_distance_px: int = 85
    slit_contact_outward_px: int = 75
    slit_contact_inward_px: int = 55
    slit_crop_padding_px: int = 18
    slit_crop_min_size_px: int = 160

    # Dynamic yellow-corridor growth (monotonic during one cycle).
    dynamic_corridor_enable: bool = True
    dynamic_corridor_grow_only: bool = True
    dynamic_corridor_pull_factor: float = 1.35
    dynamic_corridor_total_outward_factor: float = 0.20
    dynamic_corridor_lateral_factor: float = 1.05
    dynamic_corridor_total_side_factor: float = 0.12
    dynamic_corridor_total_inward_factor: float = 0.08
    dynamic_corridor_max_extra_outward_px: float = 360.0
    dynamic_corridor_max_extra_side_px: float = 190.0
    dynamic_corridor_max_extra_inward_px: float = 80.0
    dynamic_corridor_min_pose_scale: float = 1.0
    dynamic_corridor_max_pose_scale: float = 1.45

    # Rate-limited corridor growth (px/s) to remove visible jitter.
    dynamic_corridor_outward_growth_px_per_second: float = 260.0
    dynamic_corridor_side_growth_px_per_second: float = 190.0
    dynamic_corridor_inward_growth_px_per_second: float = 110.0
    dynamic_corridor_pose_growth_per_second: float = 0.55

    # Low-pass smoothing for the corridor polygon.
    dynamic_corridor_polygon_smooth_alpha: float = 0.34
    dynamic_corridor_fast_motion_alpha: float = 0.70
    dynamic_corridor_fast_motion_threshold_px: float = 48.0

    # Quantized crop boundaries keep the SAM input tensor size stable.
    dynamic_corridor_crop_quantize_px: int = 8

    # ----- emerging-product rules ----------------------------------------
    emerging_min_area_px: int = 2200
    emerging_min_area_ratio_of_poly: float = 0.012
    emerging_max_area_ratio_of_poly: float = 0.62
    emerging_min_bbox_width_px: int = 34
    emerging_min_bbox_height_px: int = 34
    emerging_min_rectangularity: float = 0.12
    emerging_min_corridor_overlap: float = 0.08
    emerging_min_contact_overlap: float = 0.008
    emerging_min_outward_fraction: float = 0.12
    emerging_min_outward_pixels: int = 500
    emerging_min_outside_poly_fraction: float = 0.12
    emerging_max_poly_overlap: float = 0.82
    emerging_min_baseline_change_fraction: float = 0.12
    emerging_max_roi_edge_touch_fraction: float = 0.35
    emerging_duplicate_iou: float = 0.80
    emerging_bag_color_reject_similarity: float = 0.92

    # Stronger rejection for black fixture/base-side segments.
    product_crop_edge_border_px: int = 10
    emerging_max_crop_edge_touch_fraction: float = 0.10
    released_max_crop_edge_touch_fraction: float = 0.16
    emerging_min_slit_span_fraction: float = 0.28
    emerging_slit_span_margin_px: float = 55.0

    baseline_static_reject_candidate_containment: float = 0.55

    base_side_color_edge_reject_similarity: float = 0.72
    base_side_long_aspect_ratio: float = 3.4
    base_side_low_rectangularity: float = 0.52
    base_side_strong_crossing_min_poly_overlap: float = 0.035
    base_side_strong_crossing_min_change: float = 0.30

    # A candidate must cross the slit line or be large immediately outside it.
    emerging_line_band_px: float = 24.0
    emerging_min_line_band_fraction: float = 0.018
    emerging_min_inward_fraction_for_crossing: float = 0.025
    emerging_outside_only_min_area_ratio_of_poly: float = 0.020
    emerging_outside_only_max_line_distance_px: float = 38.0

    # ----- reject static objects already present at baseline ------------
    baseline_static_min_mask_area_px: int = 500
    baseline_static_max_mask_area_ratio: float = 0.70
    baseline_static_reject_iou: float = 0.28
    baseline_static_reject_center_distance_px: float = 28.0
    baseline_static_reject_area_ratio_low: float = 0.58
    baseline_static_reject_area_ratio_high: float = 1.72
    baseline_visible_min_fraction: float = 0.22
    baseline_unchanged_pixel_threshold: float = 18.0
    baseline_unchanged_reject_fraction: float = 0.68
    baseline_base_color_reject_similarity: float = 0.86
    baseline_base_color_bypass_max_similarity: float = 0.72

    # ----- depth rejection for black-base fragments ----------------------
    emerging_depth_ring_inner_px: int = 8
    emerging_depth_ring_outer_px: int = 34
    emerging_min_depth_sample_count: int = 60
    emerging_min_height_above_base_mm: float = 8.0
    emerging_strong_visual_bypass_area_ratio: float = 0.030
    emerging_strong_visual_bypass_change_fraction: float = 0.45

    baseline_diff_threshold: int = 22
    baseline_diff_open_kernel_px: int = 5
    baseline_diff_close_kernel_px: int = 11

    # ----- exit confirmation ---------------------------------------------
    product_confirm_required_sam_updates: int = 2
    product_match_max_center_distance_px: int = 85
    product_match_min_iou: float = 0.08
    product_match_min_area_ratio: float = 0.45
    product_match_max_area_ratio: float = 2.20

    # ----- full-release verification --------------------------------------
    full_release_verify_crop_padding_px: int = 150
    full_release_verify_crop_min_size_px: int = 240
    full_release_reference_min_area_ratio: float = 0.35
    full_release_reference_max_area_ratio: float = 4.00
    full_release_reference_max_center_distance_px: float = 165.0
    full_release_reference_min_iou: float = 0.025
    full_release_max_poly_overlap: float = 0.035
    full_release_max_inward_fraction: float = 0.045
    full_release_min_outward_fraction: float = 0.72
    full_release_max_contact_overlap: float = 0.025
    full_release_min_line_distance_px: float = 14.0

    # The trailing edge must clear the moving slit and the product core must
    # have a real gap from the polymailer core.
    full_release_min_raw_trailing_clearance_px: float = 5.0
    full_release_min_core_trailing_clearance_px: float = 10.0
    full_release_min_core_poly_gap_px: float = 7.0
    full_release_max_core_line_band_fraction: float = 0.08
    full_release_min_confirm_delay_seconds: float = 0.75

    full_release_min_height_above_base_mm: float = 5.0
    full_release_max_center_step_px: float = 18.0
    full_release_min_stable_area_ratio: float = 0.80
    full_release_max_stable_area_ratio: float = 1.25
    full_release_required_stable_updates: int = 3
    full_release_timeout_seconds: float = 6.0
    full_release_visual_table_min_change: float = 0.58
    full_release_visual_table_max_base_color_similarity: float = 0.66

    # Eroded core masks tolerate thin SAM/flow outline overlap.
    full_release_product_core_erode_px: int = 6
    full_release_poly_core_erode_px: int = 10
    full_release_min_core_area_px: int = 180
    full_release_max_raw_poly_overlap_with_clear_core: float = 0.12
    full_release_max_core_poly_overlap: float = 0.025
    full_release_max_core_inward_fraction: float = 0.070
    full_release_min_core_outward_fraction: float = 0.72

    # ----- periodic SAM re-anchoring of the tracked bag mask -------------
    poly_refresh_enable: bool = True
    poly_refresh_interval_seconds: float = 0.75
    poly_refresh_crop_padding_px: int = 55
    poly_refresh_crop_min_size_px: int = 260
    poly_refresh_min_area_ratio_to_predicted: float = 0.24
    poly_refresh_max_area_ratio_to_predicted: float = 1.55
    poly_refresh_min_color_similarity: float = 0.58
    poly_refresh_min_candidate_inside_predicted: float = 0.34
    poly_refresh_min_iou: float = 0.12
    poly_refresh_max_center_distance_px: float = 145.0
    poly_refresh_slit_band_width_px: int = 34
    poly_refresh_min_slit_band_pixels: int = 45
    poly_refresh_max_crop_edge_touch_fraction: float = 0.38
    poly_refresh_print_every: int = 5

    overlay_alpha: float = 0.30

    _TUPLE_FIELDS = BaseConfig._TUPLE_FIELDS + ("flow_win_size",)
