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

    # ----- dedicated exact-box path on the same package-ROI masks -------
    dedicated_box_sam_enable: bool = True

    # ----- full-box override (prefer a complete cardboard mask) ---------
    box_full_mask_override_enable: bool = True
    box_full_mask_min_package_containment: float = 0.72
    box_full_mask_min_area_growth: float = 1.30
    box_full_mask_strong_color_score: float = 0.45
    box_full_mask_strong_rectangularity: float = 0.75

    # ----- split-mask merge repair (rejoin box halves) -----------------
    box_merge_split_masks_enable: bool = True
    box_merge_max_candidates: int = 14
    box_merge_min_horizontal_overlap: float = 0.55
    box_merge_max_center_x_diff_ratio: float = 0.35
    box_merge_max_width_ratio: float = 1.80
    box_merge_max_vertical_gap_px: int = 90
    box_merge_max_pair_iou: float = 0.65
    box_merge_min_area_growth_over_largest: float = 1.20
    box_merge_min_result_rectangularity: float = 0.72
    box_merge_score_bonus: float = 0.85

    # ----- rotated split-mask merge (angled boxes split along a seam) ---
    # Fallback pairing rule for build_merged_cardboard_box_candidate: when a
    # pair fails the upright (axis-aligned) relationship, accept it anyway if
    # the two fragments share a rotation angle and sit close together.
    box_merge_rotated_enable: bool = True
    box_merge_rotated_max_center_distance_ratio: float = 1.35
    box_merge_rotated_max_angle_diff_deg: float = 22.0
    box_merge_rotated_min_completed_fill_ratio: float = 0.42
    box_merge_rotated_max_completed_area_ratio: float = 0.78

    # ----- multi-face merge (angled box showing top + side face) -------
    # Joins a box-rule mask to a touching package-rule mask via their convex
    # hull, for angled boxes where SAM segments each visible face separately.
    box_multiface_merge_enable: bool = True
    box_multiface_max_candidates: int = 16
    box_multiface_touch_dilate_px: int = 28
    box_multiface_max_pair_iou: float = 0.20
    box_multiface_min_second_area_ratio: float = 0.025
    box_multiface_min_area_growth: float = 1.25
    box_multiface_min_union_fill_ratio: float = 0.58
    box_multiface_max_hull_area_ratio: float = 0.72
    box_multiface_score_bonus: float = 1.10

    # ----- segmentation box-type override / thin-polymailer veto -------
    box_type_segmentation_override_enable: bool = True
    box_type_segmentation_min_score: float = 5.25
    box_type_segmentation_min_color_score: float = 0.65
    box_type_segmentation_min_rectangularity: float = 0.82
    box_type_segmentation_min_area_ratio: float = 0.08
    box_type_segmentation_min_confidence: float = 0.90

    box_type_polymailer_veto_enable: bool = True
    box_type_polymailer_veto_max_package_depth_mm: float = 65.0
    box_type_polymailer_hard_thin_max_depth_mm: float = 45.0
    box_type_polymailer_hard_thin_min_poly_score: float = 0.55
    box_type_polymailer_veto_min_poly_score: float = 0.68
    box_type_polymailer_veto_min_signal_count: int = 1

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
    # The mailer drapes over the product, so depth shows a smooth DOME, not a
    # slab with edges. Amplitude varies with mailer stock (a padded mailer
    # spreads the same product into a ~5 mm dome where kraft gives ~10 mm), so
    # every geometric knob below is RELATIVE -- a fraction of the dome's own
    # height or of the package's own size -- and adapts per frame. The only
    # absolute values left are the sanity rails at the bottom.
    product_inside_enable: bool = True

    # Border band excluded from the search, as a fraction of the package span
    # (sqrt of its area). Keeps the drooping mailer edge out of the plane fit.
    #
    # Kept small on purpose: at 0.12 this band was cutting into the product
    # itself on 2 of 3 ground-truthed captures, clipping the footprint before
    # the dome had finished falling off and dragging the reported centre away
    # from the clipped side by 13-20 mm.
    product_inside_edge_erode_frac: float = 0.05

    # Low-pass applied to the elevation field, as a fraction of package span.
    # Wrinkles are high-frequency, the product dome is not; this is what lets
    # the threshold sit far below the wrinkle amplitude.
    product_inside_smooth_frac: float = 0.035

    # Reference surface: robust plane fit that iteratively trims the raised
    # side, so the product cannot drag its own reference upward.
    product_inside_plane_iterations: int = 5
    product_inside_plane_trim_sigma: float = 1.5
    product_inside_max_plane_points: int = 20000

    # Dome peak is read at this percentile (not the max) so a few hot pixels
    # cannot set the scale.
    product_inside_peak_percentile: float = 99.5

    # Baseline is read low rather than at the median: a product covering a big
    # share of the mailer pulls the median up into its own dome, which shrinks
    # the measured height and makes large products look like weak ones.
    product_inside_baseline_percentile: float = 25.0

    # Footprint = everything at least this fraction of the dome height. 0.5 is
    # the half-maximum contour: scale-free, so it lands in the same relative
    # place on a 5 mm padded dome and a 10 mm kraft dome.
    product_inside_height_fraction: float = 0.5

    # The measurement stereo drops out in streaks over low-texture kraft. A
    # pixel with no reading of its own is still usable when enough of the
    # smoothing kernel around it landed on valid depth; this is that minimum
    # (as a fraction of the kernel). Raise it to trust interpolation less.
    product_inside_min_blur_support: float = 0.25

    # Closing kernel that rejoins a dome fragmented by those dropout streaks,
    # as a fraction of the smoothing scale (so it too follows package size).
    product_inside_close_frac: float = 1.0

    # Trims thin tapering appendages off the footprint, as a fraction of the
    # blob's own span. Where the mailer runs off the product it keeps sloping,
    # and that ramp can hold above the half-max cut for a long way while staying
    # narrow -- a spur that is drape, not product. Judged on width rather than
    # height because the ramp reaches the same elevations the genuine far side
    # of the dome does; only its cross-section gives it away. 0 disables.
    product_inside_trim_frac: float = 0.18

    # ----- product CORE (the reported centre) --------------------------
    # The footprint above is the whole raised region, drape shoulders included;
    # it is deliberately generous because a mailer's creases and slope make the
    # true product outline unsegmentable. Its centroid is therefore pulled
    # around by however the sheet happens to fall. The core is the crown of the
    # dome -- the part most likely to be over the product itself -- and is what
    # the reported centre and depth are taken from.
    #
    # The core keeps the part of the footprint that is NOT sloping. Where the
    # mailer runs off the product it descends continuously, and that slope makes
    # up most of the footprint's area -- so a centroid over the whole footprint,
    # or over any contour of it, is really measuring the drape.
    #
    # Cut is relative to the slope found in this frame's own footprint (a
    # fraction of its 90th-percentile gradient), so nothing here is a fixed
    # millimetre or a fixed direction. Checked against three hand-outlined
    # captures it roughly halved the centre error, 59 mm -> 28 mm.
    product_inside_core_max_slope_frac: float = 0.25

    # Opening applied to the core, as a fraction of its own span.
    product_inside_core_open_frac: float = 0.10


    # If the crown comes out smaller than this share of the footprint the dome
    # has no usable top, and the centre falls back to the footprint.
    product_inside_core_min_frac_of_blob: float = 0.02

    # Detection gate, expressed in multiples of the frame's own measured depth
    # noise rather than in mm.
    product_inside_min_peak_noise_multiple: float = 3.0

    # Sanity rails (absolute, deliberately wide -- these reject nonsense, they
    # do not tune the estimate).
    product_inside_min_area_ratio_of_package: float = 0.02
    product_inside_max_area_ratio_of_package: float = 0.80
    product_inside_min_valid_pixels: int = 80
    product_inside_max_peak_mm: float = 90.0

    heatmap_max_closer_than_base_mm: float = 180.0

    debug_print_masks: bool = False
    debug_save_all_accepted_masks: bool = False
