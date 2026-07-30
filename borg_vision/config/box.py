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

    min_surface_depth_count: int = 30

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

    # Rotated-pair fallback for the split-mask repair: an angled box splits into
    # fragments whose bboxes fail every axis-aligned test, so fall back to
    # comparing min-area-rect long axes. The two completed-rectangle gates apply
    # to this path only -- without them a rotated pair could bridge unrelated
    # blobs, which the axis-aligned geometry already rules out.
    box_merge_rotated_enable: bool = True
    box_merge_rotated_max_center_distance_ratio: float = 1.35
    box_merge_rotated_max_angle_diff_deg: float = 22.0
    box_merge_rotated_min_completed_fill_ratio: float = 0.42
    box_merge_rotated_max_completed_area_ratio: float = 0.78

    # Multi-face merge: an angled box shows two faces at different brightness,
    # so SAM segments each separately and the top face alone scores as a small
    # box. Join a scoring box face with an adjacent second face and keep the
    # convex hull of the union (not a min-area rectangle).
    box_multiface_merge_enable: bool = True
    box_multiface_max_candidates: int = 16
    box_multiface_touch_dilate_px: int = 28
    box_multiface_max_pair_iou: float = 0.20
    box_multiface_min_second_area_ratio: float = 0.025
    box_multiface_min_area_growth: float = 1.25
    box_multiface_min_union_fill_ratio: float = 0.58
    box_multiface_max_hull_area_ratio: float = 0.72
    box_multiface_score_bonus: float = 1.10

    # Loose secondary-face pool feeding the multi-face merge. Box mode only:
    # package mode reuses its own package candidates instead. No HSV /
    # min_box_value gate here on purpose -- a shaded side face is not brown.
    secondary_face_max_area_ratio: float = 0.70
    secondary_face_min_rectangularity: float = 0.08
    secondary_face_max_aspect_ratio: float = 10.0
    secondary_face_min_center_score: float = 0.12

    debug_print_masks: bool = False
