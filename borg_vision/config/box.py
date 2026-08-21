"""Configuration for the cardboard-box measurement mode.

BoxConfig == BaseConfig plus the cardboard-box HSV-scoring, mask-cleanup and
depth fields. Every field corresponds 1:1 to a module-level constant of the
original run_box()/box_detection_final.py, with the original value as default.

Differences from the package mode captured as defaults: narrower locked ROI
(330,60)-(940,700), negative depth x-shift (-40), and base depth 705 mm. Box
measures from the (640x400) measurement stereo, so needs_measurement_stereo
stays True.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class BoxConfig(BaseConfig):
    save_dir: str = "cardboard_box_depth_results"

    # Box measures from the dedicated measurement stereo stream (640x400).
    needs_measurement_stereo: bool = True

    base_depth_mm: float = 705.0

    # Multiplier on the measured box WIDTH and LENGTH (not height -- that comes
    # from depth, not from the mask). The raw pinhole result reads ~3% small:
    # a 203 x 210 mm box reported 197.2 x 203.3.
    #
    # Box-only, deliberately separate from the package/polymailer *_size_scale
    # (1.04) so tuning one cannot move the others.
    box_size_scale: float = 1.03

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

    min_surface_depth_count: int = 30

    # Minimum height a mask must stand above the base plane to be accepted as a
    # box, i.e. base_depth_mm - box_face_depth_mm >= this. Rejects masks lying ON
    # the table: the box grasp mechanism's tray scores as a large flat rectangle
    # and wins whenever the box's own colour evidence weakens (a slight rotation
    # under the clamp shadows part of the top face and dulls it). Two captures on
    # 2026-08-20 (11-34-20 and 14-43-29) both reported the tray at a face depth
    # of 706-707 mm against this 705 mm base -- at or BELOW the table -- as
    # 389 x 309 and 389 x 232 mm boxes with success: true.
    #
    # Depth is the right gate because it is exactly what rotation does NOT
    # degrade: in both failing frames the box remained a clean, well-separated
    # depth plateau while its RGB score collapsed.
    #
    # This makes a bad detection FAIL rather than measure the wrong object; it
    # does not make the box get found. A depth-aware fallback to the next-best
    # mask is the follow-on fix.
    min_box_height_mm: float = 15.0

    # Fraction of a mask's valid depth samples that must lie within
    # +/- box_face_depth_tolerance_mm of its median for the mask to be accepted.
    #
    # min_box_height_mm alone is not enough: a mask covering the tray AND the box
    # averages the two into a plausible height. Capture 2026-08-20_15-12-02 did
    # exactly this -- 357 x 295 mm of tray+box reported a 674 mm face depth
    # (between the ~600 mm box top and the ~706 mm tray), clearing the 15 mm bar
    # at 31 mm.
    #
    # A real box top is a single flat plateau, so nearly all of its samples sit
    # at one depth; a mixed mask is bimodal. This catches the mixed case that the
    # height gate cannot, while the height gate catches the pure-tray case that
    # this cannot (a tray-only mask is just as uniform as a box). Both are needed.
    #
    # 0.85 is provisional -- set from the depth profiles implied by the reported
    # face depths, NOT measured from saved depth arrays (only the colour-mapped
    # PNG is saved, so the raw values could not be recovered). The rejection
    # message prints the observed fraction: tune this against real numbers from a
    # few runs before trusting it, and raise it toward ~0.9 if mixed masks still
    # pass or lower it if good boxes are rejected.
    min_box_face_depth_uniformity: float = 0.85
    box_face_depth_tolerance_mm: float = 12.0

    # ----- depth-plateau mask completion --------------------------------
    # Grow an accepted mask into every connected pixel sitting at the same depth
    # as its own face, then re-measure from the completed mask.
    #
    # SAM2 segments in RGB, so a tonal seam across the box top -- a lighting
    # change, a tape line, a print band -- splits the face and the mask covers
    # only part of it. Captures 2026-08-21_10-06-57 and 10-07-21 both reported
    # the SAME box as 200 x 124 mm that had measured 206 x 203 mm minutes
    # earlier: the mask stopped mid-face. Nothing already in the pipeline caught
    # it -- a partial box top is still one flat plateau at the right height, so
    # it passes both min_box_height_mm and min_box_face_depth_uniformity, and
    # box_merge_split_masks_enable did not fire because SAM2 produced no second
    # fragment to pair with.
    #
    # Depth resolves it unambiguously: the box top is a single continuous
    # plateau in every one of these frames, and the part the mask missed is the
    # same depth as the part it caught. This is a REPAIR, not a gate -- it fixes
    # the measurement rather than rejecting the frame.
    #
    # Bounded by box_plateau_max_area_growth so it can only complete a face, not
    # run away onto a same-height neighbour; growth beyond that is treated as
    # evidence the plateau is not the box and the original mask is kept.
    box_plateau_complete_enable: bool = True
    box_plateau_tolerance_mm: float = 12.0
    box_plateau_max_area_growth: float = 2.5
    box_plateau_min_area_growth: float = 1.05
    box_plateau_close_kernel_px: int = 7

    # Cardboard-box mask gating & scoring.
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

    # Box scoring weights (standalone box mode; weighted color + geometry).
    box_color_score_weight: float = 2.7
    box_rect_score_weight: float = 1.6
    box_area_score_weight: float = 1.2
    box_center_score_weight: float = 1.0
    box_sam_iou_score_weight: float = 0.4
    box_sam_stability_score_weight: float = 0.4

    # Full-box override: prefer a larger mask that contains a partial box mask.
    box_full_mask_override_enable: bool = True
    box_full_mask_min_partial_containment: float = 0.72
    box_full_mask_min_area_growth: float = 1.25
    box_full_mask_max_score_drop: float = 0.80
    box_full_mask_min_color_score: float = 0.25
    box_full_mask_min_rectangularity: float = 0.35

    # Split-mask repair: rejoin two box halves split by a seam.
    box_merge_split_masks_enable: bool = True
    box_merge_max_candidates: int = 14
    box_merge_min_axis_overlap: float = 0.55
    box_merge_max_center_diff_ratio: float = 0.35
    box_merge_max_size_ratio: float = 1.80
    box_merge_max_gap_px: int = 90
    box_merge_max_pair_iou: float = 0.65
    box_merge_min_area_growth_over_largest: float = 1.20
    box_merge_min_result_rectangularity: float = 0.72
    box_merge_score_bonus: float = 0.85

    # ----- rotated split-mask merge (angled boxes split along a seam) ---
    # Fallback pairing rule: when a pair fails both the top/bottom and
    # left/right relationships, accept it if the two fragments share a
    # rotation angle and sit close together.
    box_merge_rotated_enable: bool = True
    box_merge_rotated_max_center_distance_ratio: float = 1.35
    box_merge_rotated_max_angle_diff_deg: float = 22.0
    box_merge_rotated_min_completed_fill_ratio: float = 0.42
    box_merge_rotated_max_completed_area_ratio: float = 0.78

    # ----- multi-face merge (angled box showing top + side face) -------
    # Joins a box-rule mask to a touching secondary-face mask via their
    # convex hull, for angled boxes where SAM segments each face separately.
    #
    # DEFAULT OFF (2026-08-10). On the return station this path absorbed the box
    # grasp mechanism's tray as a "secondary face" and reported it instead of the
    # box -- 307 x 257 mm rather than the true 200 x 196 mm. It wins because it
    # adds box_multiface_score_bonus (1.10, the largest bonus of any construction
    # path) and because secondary_face_min_rectangularity is only 0.08, so a large
    # rectangle sitting against the box qualifies as a face of it.
    #
    # SAM2 is not at fault: on a failing frame its generator produced 62 masks
    # whose best-scoring candidate WAS the cardboard (score 6.554, area 0.268,
    # IoU 0.984 vs the true box). The tray mask is absent from those raw outputs
    # -- this path constructs it.
    #
    # Cost of the default: an angled box showing two faces now measures only its
    # top face. Re-enable per camera via the vision_cameras.yaml mode overrides
    # where that geometry actually occurs, or tighten
    # secondary_face_min_rectangularity first.
    box_multiface_merge_enable: bool = False
    box_multiface_max_candidates: int = 16
    box_multiface_touch_dilate_px: int = 28
    box_multiface_max_pair_iou: float = 0.20
    box_multiface_min_second_area_ratio: float = 0.025
    box_multiface_min_area_growth: float = 1.25
    box_multiface_min_union_fill_ratio: float = 0.58
    box_multiface_max_hull_area_ratio: float = 0.72
    box_multiface_score_bonus: float = 1.10

    # ----- secondary-face gating ---------------------------------------
    # A box's second visible face is often too dark / off-colour to pass the
    # cardboard-box rules, so multi-face merging scores its partner masks
    # with these looser geometry-only gates instead.
    secondary_face_max_area_ratio: float = 0.70
    secondary_face_min_rectangularity: float = 0.08
    secondary_face_max_aspect_ratio: float = 10.0
    secondary_face_min_center_score: float = 0.12

    debug_print_masks: bool = False
