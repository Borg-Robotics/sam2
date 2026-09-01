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

    # Package targets are large (a mailer or box filling much of the ROI), so
    # the base 24x24 SAM prompt grid is overkill. 16x16 halves segmentation
    # time and reproduced the 24x24 results exactly on the 2026-09-01 A/B
    # battery (5 captures, box + polymailer: same type, same mask source,
    # sizes within 1 mm, product-inside centres within 2 px). Other modes
    # keep the base default.
    sam_points_per_side: int = 18

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
    # A merge partner must itself look like a piece of cardboard box. The
    # 2026-09-01 rotated-box failure glued the COMPLETE box (color 0.82,
    # rect 0.96, already the exact-rules winner) to a roller/carpet strip
    # (color 0.51, rect 0.62); rectangle completion made the union a perfect
    # rect and the inflated score beat the correct candidate by 1.3. Real
    # split-box halves measure color ~0.8 and rect ~0.9, so these floors cost
    # nothing on genuine splits.
    box_merge_min_fragment_color_score: float = 0.60
    box_merge_min_fragment_rectangularity: float = 0.70

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
    # The multiface hull's second fragment is a PACKAGE candidate, which has
    # no color gate of its own -- a carpet patch next to the box qualifies
    # geometrically. Require cardboard color on it (and the box fragment is
    # held to the box_merge fragment floors above).
    box_multiface_min_second_color_score: float = 0.60

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
    # The mailer drapes and tents over the product, so depth shows a smooth
    # DOME whose skirt extends well past the product -- and trapped air can
    # hold the film HIGHER than the product itself, so no height threshold can
    # outline it. The detector instead finds the flat PLANAR patch where the
    # film rests on the rigid product's top (the seed), fits that plane, and
    # grows the footprint across everything close to the plane. Length knobs
    # are RELATIVE (fractions of the package span or the dome height); the
    # plane tolerances are millimetres because they describe how far film can
    # physically sit from a rigid top it is touching.
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

    # The measurement stereo drops out in streaks over low-texture kraft. A
    # pixel with no reading of its own is still usable when enough of the
    # smoothing kernel around it landed on valid depth; this is that minimum
    # (as a fraction of the kernel). Raise it to trust interpolation less.
    product_inside_min_blur_support: float = 0.25

    # ----- seed: where the product's top is looked for ------------------
    # Height CANNOT outline the product (the drape reaches every height the
    # product does; trapped air even holds the film higher than the product),
    # so height only gates where SEEDS may start: at least this fraction of
    # the dome height up.
    product_inside_seed_min_height_frac: float = 0.35

    # Within that elevated zone, the seed is its flattest part: pixels whose
    # smoothed-elevation slope is under this percentile of the zone's own
    # slopes. Film resting on a rigid product is flat; drape is sloped.
    product_inside_seed_slope_percentile: float = 30.0

    # A seed component must be at least this fraction of the package area;
    # of those, the largest is tried first (up to seed_max_candidates).
    product_inside_seed_min_area_frac: float = 0.005
    product_inside_seed_max_candidates: int = 3

    # ----- plane growth: the footprint --------------------------------
    # A tilted plane is fitted to the seed (the product's top) and the seed
    # grows across everything close to that plane. ASYMMETRIC on purpose:
    # drape leaves the product plane UPWARD (tent, billow), so above-plane is
    # tight; film sags BELOW the plane off edges and corners, so below-plane
    # is looser. Loosening the above-tolerance is what re-admits the drape.
    product_inside_plane_tol_above_mm: float = 0.6
    product_inside_plane_tol_below_mm: float = 1.2

    # Closing applied to the on-plane zone before growth (bridges dropout
    # streaks and small contact gaps), as a fraction of the smoothing scale.
    product_inside_grow_close_frac: float = 0.75

    # Growth is geodesically capped at this multiple of the seed's own span
    # (sqrt of its area): an on-plane band that merely osculates the curved
    # drape would otherwise run across the whole mailer.
    product_inside_grow_dist_seed_spans: float = 1.5

    # Trims thin tapering appendages off the grown footprint, as a fraction of
    # the blob's own span. Where the drape happens to lie in the product's
    # plane it forms a narrow tongue -- drape, not product; only its
    # cross-section gives it away. 0 disables.
    product_inside_trim_frac: float = 0.10

    # ----- peak candidate ----------------------------------------------
    # A second candidate seeded from the dome's very top. When the drape forms
    # a long level crest, the flat path rides it -- but the product still owns
    # the dome's peak (nothing rests ON a drape). The band is how far below
    # the p99.5 elevation the seed may reach; the growth cap is tighter than
    # the flat path's because the peak plane necessarily skims the crest
    # crown. The two candidates compete on edge-drop x solidity x aspect;
    # aspect is referenced to score_aspect_ref so a genuinely elongated
    # product (2:1) is not penalised.
    product_inside_peak_band_mm: float = 0.4
    product_inside_peak_grow_dist_spans: float = 0.7
    product_inside_score_aspect_ref: float = 2.0

    # ----- edge-drop gate: does anything rigid end here? ----------------
    # Around a rigid product the film must FALL off the plane. The drop is
    # measured in a ring around the footprint, offset by edge_band_frac of the
    # package span (the smoothing spreads the cliff), and a ring pixel counts
    # when it is below the plane by max(edge_drop_mm_min, edge_drop_dome_frac
    # of the dome height). Less than min_edge_drop_frac of the ring dropping
    # means no rigid edges: found=False rather than a guessed centre.
    product_inside_edge_band_frac: float = 0.035
    product_inside_edge_drop_mm_min: float = 2.0
    product_inside_edge_drop_dome_frac: float = 0.15
    product_inside_min_edge_drop_frac: float = 0.35

    # ----- footprint shape gates ---------------------------------------
    # A rigid product's footprint is compact. Solidity (area over convex hull
    # area) is the primary gate -- it is rotation-invariant. Rect fill (area
    # over rotated-bbox area) is only a backstop: a rotated product in an
    # axis-aligned minAreaRect legitimately fills as little as half of it.
    product_inside_min_solidity: float = 0.72
    product_inside_min_rect_fill: float = 0.40

    # Detection gate, expressed in multiples of the frame's own measured depth
    # noise rather than in mm.
    product_inside_min_peak_noise_multiple: float = 3.0

    # Sanity rails (absolute, deliberately wide -- these reject nonsense, they
    # do not tune the estimate). The max area ratio doubles as the drape
    # sanity gate: a "product" covering nearly half the mailer is the drape.
    product_inside_min_area_ratio_of_package: float = 0.02
    product_inside_max_area_ratio_of_package: float = 0.45
    product_inside_min_valid_pixels: int = 80
    product_inside_max_peak_mm: float = 90.0

    # ----- product-inside suction-cup fallback grasp points ----------------
    # Ranked alternatives near the product_inside centre for when the cup
    # fails to seal on the primary point. Scored by the object-mode grasp
    # scorer (score_object_grasp_candidates), which reads these obj_grasp_*
    # and center-depth fields off whatever config it is handed -- the names
    # therefore MUST match ObjectConfig's. The primary grasp stays the
    # product_inside centre; the fallbacks are simply the best-scoring other
    # spots (operator's call 2026-09-01: no full-cup spacing rule), kept only
    # far enough from the centre and each other to be different film at all.
    obj_grasp_cup_diameter_mm: float = 30.0
    obj_grasp_edge_margin_mm: float = 5.0
    obj_grasp_max_offset_mm: float = 55.0
    obj_grasp_grid_step_px: int = 6
    obj_grasp_min_valid_px: int = 30
    # Lower than object mode's 0.60: the box fallbacks score on the sparse
    # measurement stream (~15% coverage over a cardboard top on camera_1),
    # and a cup-sized disc still holds hundreds of valid points at that
    # density -- plenty for the plane fit. Mailer film is ~97% covered in
    # the stream the product-inside fallbacks use, so this floor is inert
    # there.
    obj_grasp_min_valid_frac: float = 0.10
    obj_grasp_min_depth_levels: int = 3
    # More than the wire needs: the centre-separation filter thins this list
    # down to product_inside_grasp_retry_count afterwards.
    obj_grasp_max_candidates: int = 6
    # Spacing between kept candidates, in cup radii (1.0 = half a cup: the
    # next candidate's centre clears the previous cup's rim). Object mode
    # uses 2.0; the operator wants product fallbacks purely score-ranked.
    obj_grasp_min_sep_radii: float = 1.0
    obj_grasp_w_rough: float = 0.35
    obj_grasp_w_tilt: float = 0.15
    obj_grasp_w_centre: float = 0.45
    obj_grasp_w_edge: float = 0.05
    # Center-depth sampling for candidate z (same names as ObjectConfig).
    center_depth_radius_px: int = 35
    min_center_depth_count: int = 30
    # How many fallbacks to report after the separation filter.
    product_inside_grasp_retry_count: int = 2
    # A fallback must sit at least this far from the primary point -- not a
    # spacing preference, just "a different spot at all" (half a cup, so the
    # retry's centre is off the failed cup's footprint).
    product_inside_grasp_min_offset_mm: float = 15.0

    heatmap_max_closer_than_base_mm: float = 180.0

    debug_print_masks: bool = False
    debug_save_all_accepted_masks: bool = False
