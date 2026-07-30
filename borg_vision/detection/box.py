"""Box-mode detection logic: cardboard-box HSV mask scoring + depth measurement.

Ported verbatim from run_box()/box_detection_final.py with the module-level
constants replaced by fields of a BoxConfig passed as the first argument.
Generic, config-free helpers (mask cleanup, rotated box, pixel->camera, robust
median depth) are reused from detection.package so the two modes share one
implementation.
"""

import cv2
import numpy as np
import torch

from .package import (
    clean_box_mask,
    complete_min_area_rectangle,
    convex_hull_mask,
    get_mask_center,
    get_rotated_box_from_mask,
    mask_iou,
    mask_overlap_fraction,
    masks_are_near,
    normalized_rect_angle,
    pixel_to_camera_xy_mm,
    robust_median_depth,
)


def estimate_box_depth_mm(cfg, box_face_depth_mm):
    if box_face_depth_mm is None:
        return None

    box_depth = cfg.base_depth_mm - box_face_depth_mm

    if box_depth < 0:
        box_depth = 0.0

    return float(box_depth)


def estimate_box_dimensions_mm(cfg, box, depth_mm, intrinsics):
    empty = {
        "rotated_width_px": None,
        "rotated_height_px": None,
        "angle_deg": None,
        "width_mm": None,
        "height_mm": None,
        "length_mm": None,
        "short_side_mm": None,
        "points_roi": None,
        "points_full": None,
    }

    if depth_mm is None or intrinsics is None:
        return empty

    rotated = get_rotated_box_from_mask(box["mask"])

    if rotated is None:
        return empty

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    width_mm = (w_px * depth_mm) / fx
    height_mm = (h_px * depth_mm) / fy

    length_mm = max(width_mm, height_mm)
    short_side_mm = min(width_mm, height_mm)

    points_roi = rotated["points_roi"]
    points_full = points_roi.copy()
    points_full[:, 0] += cfg.roi_x1
    points_full[:, 1] += cfg.roi_y1

    return {
        "rotated_width_px": float(w_px),
        "rotated_height_px": float(h_px),
        "angle_deg": float(rotated["angle_deg"]),
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "length_mm": float(length_mm),
        "short_side_mm": float(short_side_mm),
        "points_roi": points_roi.tolist(),
        "points_full": points_full.tolist(),
    }


def box_face_depth_mm(cfg, depth_roi, mask):
    valid_sample = (
        (mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    values = depth_roi[valid_sample].astype(np.float32)

    distance_mm, depth_count, mean_depth, min_depth = robust_median_depth(cfg, values)

    return {
        "distance_mm": distance_mm,
        "depth_count": depth_count,
        "sample_count_total": int(values.size),
        "mean_depth_mm": mean_depth,
        "min_depth_mm": min_depth,
    }


def axis_overlap_fraction(start_a, size_a, start_b, size_b):
    left = max(start_a, start_b)
    right = min(start_a + size_a, start_b + size_b)
    overlap = max(0, right - left)
    return float(overlap / max(min(size_a, size_b), 1))


def axis_gap_px(start_a, size_a, start_b, size_b):
    end_a = start_a + size_a
    end_b = start_b + size_b

    if end_a < start_b:
        return int(start_b - end_a)

    if end_b < start_a:
        return int(start_a - end_b)

    return 0


def score_cardboard_box_candidate(
    cfg,
    mask,
    roi_rgb,
    index=None,
    sam_iou=0.0,
    sam_stability=0.0,
    source="single_sam_mask",
    color_score_floor=None,
):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2.0
    roi_cy = h / 2.0

    mask = mask.astype(np.uint8)
    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < cfg.min_box_area_ratio:
        return None

    if area_ratio > cfg.max_box_area_ratio:
        return None

    x, y, bw, bh = cv2.boundingRect(mask)

    if bw <= 0 or bh <= 0:
        return None

    rectangularity = area / max(bw * bh, 1)
    aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

    if rectangularity < cfg.min_box_rectangularity:
        return None

    if aspect_ratio > cfg.max_box_aspect_ratio:
        return None

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    masked_hsv = hsv[mask == 1]

    if masked_hsv.size == 0:
        return None

    mean_h = float(np.mean(masked_hsv[:, 0]))
    mean_s = float(np.mean(masked_hsv[:, 1]))
    mean_v = float(np.mean(masked_hsv[:, 2]))

    if mean_v < cfg.min_box_value:
        return None

    hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
    sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
    val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)

    measured_color_score = (
        0.50 * hue_score
        + 0.25 * sat_score
        + 0.25 * val_score
    )

    # A merged mask spans the box's darker second face, which drags the mean
    # colour down. The contributing fragment's own colour score is the better
    # evidence, so callers may floor the merged result with it.
    color_score = measured_color_score

    if color_score_floor is not None:
        color_score = max(color_score, float(color_score_floor))

    if color_score < cfg.min_box_color_score:
        return None

    center_roi = get_mask_center(mask)

    if center_roi is None:
        return None

    cx, cy = center_roi
    dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    area_score = 1.0 - min(
        abs(area_ratio - cfg.target_box_area_ratio) / cfg.target_box_area_ratio,
        1.0,
    )

    score = (
        cfg.box_color_score_weight * color_score
        + cfg.box_rect_score_weight * rectangularity
        + cfg.box_area_score_weight * area_score
        + cfg.box_center_score_weight * center_score
        + cfg.box_sam_iou_score_weight * sam_iou
        + cfg.box_sam_stability_score_weight * sam_stability
    )

    return {
        "source": source,
        "index": index,
        "score": float(score),
        "mask": mask,
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "color_score": float(color_score),
        "measured_color_score": float(measured_color_score),
        "mean_hsv": (mean_h, mean_s, mean_v),
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
        "merged_from_indices": None,
        "selection_override": None,
    }


def select_complete_single_box_candidate(cfg, box_candidates):
    if len(box_candidates) == 0:
        return None

    selected = max(box_candidates, key=lambda item: item["score"])

    if not cfg.box_full_mask_override_enable:
        return selected

    changed = True

    while changed:
        changed = False
        selected_area = int(selected["mask"].sum())
        best_replacement = None

        for candidate in box_candidates:
            if candidate is selected:
                continue

            candidate_area = int(candidate["mask"].sum())

            if candidate_area <= selected_area:
                continue

            containment = mask_overlap_fraction(
                selected["mask"],
                candidate["mask"],
            )
            area_growth = candidate_area / max(selected_area, 1)
            score_drop = selected["score"] - candidate["score"]

            valid_complete_mask = (
                containment >= cfg.box_full_mask_min_partial_containment
                and area_growth >= cfg.box_full_mask_min_area_growth
                and score_drop <= cfg.box_full_mask_max_score_drop
                and candidate["color_score"] >= cfg.box_full_mask_min_color_score
                and candidate["rectangularity"] >= cfg.box_full_mask_min_rectangularity
            )

            if not valid_complete_mask:
                continue

            replacement_rank = (
                candidate_area,
                candidate["score"],
                containment,
            )

            if (
                best_replacement is None
                or replacement_rank > best_replacement[0]
            ):
                best_replacement = (
                    replacement_rank,
                    candidate,
                    containment,
                    area_growth,
                )

        if best_replacement is not None:
            _, replacement, containment, area_growth = best_replacement
            replacement = dict(replacement)
            replacement["selection_override"] = "larger_containing_box_mask"
            replacement["contained_partial_fraction"] = float(containment)
            replacement["area_growth_over_partial"] = float(area_growth)
            selected = replacement
            changed = True

    return selected


def pair_fragment_compatibility(cfg, first, second):
    first_bbox = first["bbox"]
    second_bbox = second["bbox"]

    fx, fy, fw, fh = first_bbox
    sx, sy, sw, sh = second_bbox

    pair_iou = mask_iou(first["mask"], second["mask"])

    if pair_iou > cfg.box_merge_max_pair_iou:
        return None

    # Top/bottom split: strong X overlap, similar widths and aligned X centers.
    x_overlap = axis_overlap_fraction(fx, fw, sx, sw)
    width_ratio = max(fw, sw) / max(min(fw, sw), 1)
    center_x_diff_ratio = abs((fx + fw / 2.0) - (sx + sw / 2.0)) / max(max(fw, sw), 1)
    y_gap = axis_gap_px(fy, fh, sy, sh)

    vertical_split_ok = (
        x_overlap >= cfg.box_merge_min_axis_overlap
        and width_ratio <= cfg.box_merge_max_size_ratio
        and center_x_diff_ratio <= cfg.box_merge_max_center_diff_ratio
        and y_gap <= cfg.box_merge_max_gap_px
    )

    # Left/right split: strong Y overlap, similar heights and aligned Y centers.
    y_overlap = axis_overlap_fraction(fy, fh, sy, sh)
    height_ratio = max(fh, sh) / max(min(fh, sh), 1)
    center_y_diff_ratio = abs((fy + fh / 2.0) - (sy + sh / 2.0)) / max(max(fh, sh), 1)
    x_gap = axis_gap_px(fx, fw, sx, sw)

    horizontal_split_ok = (
        y_overlap >= cfg.box_merge_min_axis_overlap
        and height_ratio <= cfg.box_merge_max_size_ratio
        and center_y_diff_ratio <= cfg.box_merge_max_center_diff_ratio
        and x_gap <= cfg.box_merge_max_gap_px
    )

    # Rotation-independent relationship. SAM often splits an angled cardboard
    # box along its seam; in image coordinates those pieces line up on neither
    # axis, even though together they form one clean rotated rectangle.
    if not vertical_split_ok and not horizontal_split_ok:
        if not cfg.box_merge_rotated_enable:
            return None

        first_rotated = get_rotated_box_from_mask(first["mask"])
        second_rotated = get_rotated_box_from_mask(second["mask"])

        if first_rotated is None or second_rotated is None:
            return None

        angle_a = normalized_rect_angle(first_rotated)
        angle_b = normalized_rect_angle(second_rotated)
        angle_diff = abs(angle_a - angle_b)
        angle_diff = min(angle_diff, 180.0 - angle_diff)

        center_distance = float(np.hypot(
            (fx + fw / 2.0) - (sx + sw / 2.0),
            (fy + fh / 2.0) - (sy + sh / 2.0),
        ))
        characteristic_size = max(
            first_rotated["width_px"],
            first_rotated["height_px"],
            second_rotated["width_px"],
            second_rotated["height_px"],
            1.0,
        )
        center_distance_ratio = center_distance / characteristic_size

        rotated_ok = (
            angle_diff <= cfg.box_merge_rotated_max_angle_diff_deg
            and center_distance_ratio
            <= cfg.box_merge_rotated_max_center_distance_ratio
        )

        if not rotated_ok:
            return None

        return {
            "pair_iou": float(pair_iou),
            "split_axis": "rotated",
            "axis_overlap": float(max(x_overlap, y_overlap)),
            "gap_px": int(min(x_gap, y_gap)),
            "geometry_mode": "rotated",
            "angle_diff_deg": float(angle_diff),
            "center_distance_ratio": float(center_distance_ratio),
        }

    if vertical_split_ok and horizontal_split_ok:
        if x_overlap >= y_overlap:
            split_axis = "top_bottom"
            axis_overlap = x_overlap
            gap_px = y_gap
        else:
            split_axis = "left_right"
            axis_overlap = y_overlap
            gap_px = x_gap
    elif vertical_split_ok:
        split_axis = "top_bottom"
        axis_overlap = x_overlap
        gap_px = y_gap
    else:
        split_axis = "left_right"
        axis_overlap = y_overlap
        gap_px = x_gap

    return {
        "pair_iou": float(pair_iou),
        "split_axis": split_axis,
        "axis_overlap": float(axis_overlap),
        "gap_px": int(gap_px),
        "geometry_mode": "axis_aligned",
    }


def build_merged_cardboard_box_candidate(cfg, box_candidates, roi_rgb):
    if not cfg.box_merge_split_masks_enable:
        return None

    if len(box_candidates) < 2:
        return None

    candidates = sorted(
        box_candidates,
        key=lambda item: item.get("area", int(item["mask"].sum())),
        reverse=True,
    )[:cfg.box_merge_max_candidates]

    best_merged = None

    for i in range(len(candidates)):
        first = candidates[i]
        first_area = int(first["mask"].sum())

        for j in range(i + 1, len(candidates)):
            second = candidates[j]
            second_area = int(second["mask"].sum())

            compatibility = pair_fragment_compatibility(cfg, first, second)

            if compatibility is None:
                continue

            union_mask = np.logical_or(
                first["mask"].astype(bool),
                second["mask"].astype(bool),
            ).astype(np.uint8)

            completed_mask = complete_min_area_rectangle(union_mask)

            if completed_mask is None:
                continue

            largest_fragment_area = max(first_area, second_area)
            completed_area = int(completed_mask.sum())
            area_growth = completed_area / max(largest_fragment_area, 1)
            union_area = int(union_mask.sum())
            completed_fill_ratio = union_area / max(completed_area, 1)
            completed_area_ratio = completed_area / max(
                completed_mask.shape[0] * completed_mask.shape[1],
                1,
            )

            if area_growth < cfg.box_merge_min_area_growth_over_largest:
                continue

            # A rotated pair is a weaker signal than an axis-aligned one, so
            # the completed rectangle must actually be filled by the fragments
            # and must not swallow the whole ROI.
            rotated_pair = compatibility["geometry_mode"] == "rotated"

            if rotated_pair and (
                completed_fill_ratio < cfg.box_merge_rotated_min_completed_fill_ratio
                or completed_area_ratio > cfg.box_merge_rotated_max_completed_area_ratio
            ):
                continue

            merged_result = score_cardboard_box_candidate(
                cfg,
                completed_mask,
                roi_rgb,
                index=None,
                sam_iou=max(first["sam_iou"], second["sam_iou"]),
                sam_stability=max(
                    first["sam_stability"],
                    second["sam_stability"],
                ),
                source="cardboard_box_merged_masks",
                color_score_floor=max(
                    first.get("color_score", 0.0),
                    second.get("color_score", 0.0),
                ),
            )

            if merged_result is None:
                continue

            if (
                merged_result["rectangularity"]
                < cfg.box_merge_min_result_rectangularity
            ):
                continue

            merged_result["merged_from_indices"] = [
                int(first["index"]),
                int(second["index"]),
            ]
            merged_result["merged_pair_iou"] = compatibility["pair_iou"]
            merged_result["merged_split_axis"] = compatibility["split_axis"]
            merged_result["merged_axis_overlap"] = compatibility["axis_overlap"]
            merged_result["merged_gap_px"] = compatibility["gap_px"]
            merged_result["merged_geometry_mode"] = compatibility["geometry_mode"]
            merged_result["merged_completed_fill_ratio"] = float(
                completed_fill_ratio
            )
            merged_result["merged_area_growth"] = float(area_growth)
            merged_result["score_before_merge_bonus"] = float(
                merged_result["score"]
            )
            merged_result["score"] = float(
                merged_result["score"] + cfg.box_merge_score_bonus
            )
            merged_result["selection_override"] = "merged_split_box_masks"

            if (
                best_merged is None
                or merged_result["score"] > best_merged["score"]
            ):
                best_merged = merged_result

    return best_merged


def build_secondary_face_candidates(cfg, masks, roi_rgb):
    """Score every SAM mask with loose, geometry-only gates.

    The second visible face of an angled box is usually too dark or off-colour
    to survive the cardboard-box rules, so it never becomes a box candidate and
    the box is measured as one face. These looser candidates exist purely as
    merge partners for build_multiface_cardboard_box_candidate.
    """
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2
    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    candidates = []

    for i, item in enumerate(masks):
        mask = clean_box_mask(cfg, item["segmentation"].astype(np.uint8))
        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.box_multiface_min_second_area_ratio:
            continue
        if area_ratio > cfg.secondary_face_max_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)
        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.secondary_face_min_rectangularity:
            continue
        if aspect_ratio > cfg.secondary_face_max_aspect_ratio:
            continue

        center_roi = get_mask_center(mask)
        if center_roi is None:
            continue

        dist = np.hypot(center_roi[0] - roi_cx, center_roi[1] - roi_cy)
        center_score = 1.0 - min(dist / max(max_dist, 1.0), 1.0)

        if center_score < cfg.secondary_face_min_center_score:
            continue

        candidates.append({
            "index": i,
            "mask": mask,
            "area": area,
            "area_ratio": float(area_ratio),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_roi": center_roi,
            "center_score": float(center_score),
            "sam_iou": float(item.get("predicted_iou", 0.0)),
            "sam_stability": float(item.get("stability_score", 0.0)),
        })

    return candidates


def build_multiface_cardboard_box_candidate(
    cfg,
    box_candidates,
    secondary_candidates,
    roi_rgb,
):
    if not cfg.box_multiface_merge_enable:
        return None

    if not box_candidates or not secondary_candidates:
        return None

    boxes = sorted(
        box_candidates,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )[:cfg.box_multiface_max_candidates]
    secondary = sorted(
        secondary_candidates,
        key=lambda item: item.get("area", int(item["mask"].sum())),
        reverse=True,
    )[:cfg.box_multiface_max_candidates]

    roi_area = roi_rgb.shape[0] * roi_rgb.shape[1]
    best = None

    for box in boxes:
        box_mask = box["mask"].astype(np.uint8)
        box_area = int(box_mask.sum())

        for face in secondary:
            if face.get("index") == box.get("index"):
                continue

            face_mask = face["mask"].astype(np.uint8)
            face_area = int(face_mask.sum())
            face_area_ratio = face_area / max(roi_area, 1)

            if face_area_ratio < cfg.box_multiface_min_second_area_ratio:
                continue

            pair_iou = mask_iou(box_mask, face_mask)
            if pair_iou > cfg.box_multiface_max_pair_iou:
                continue

            if not masks_are_near(
                box_mask,
                face_mask,
                cfg.box_multiface_touch_dilate_px,
            ):
                continue

            union_mask = np.logical_or(box_mask, face_mask).astype(np.uint8)
            hull_mask = convex_hull_mask(union_mask)
            if hull_mask is None:
                continue

            hull_area = int(hull_mask.sum())
            union_area = int(union_mask.sum())
            largest_area = max(box_area, face_area)
            area_growth = hull_area / max(largest_area, 1)
            union_fill_ratio = union_area / max(hull_area, 1)
            hull_area_ratio = hull_area / max(roi_area, 1)

            if area_growth < cfg.box_multiface_min_area_growth:
                continue
            if union_fill_ratio < cfg.box_multiface_min_union_fill_ratio:
                continue
            if hull_area_ratio > cfg.box_multiface_max_hull_area_ratio:
                continue

            merged = score_cardboard_box_candidate(
                cfg,
                hull_mask,
                roi_rgb,
                index=None,
                sam_iou=max(box.get("sam_iou", 0.0), face.get("sam_iou", 0.0)),
                sam_stability=max(
                    box.get("sam_stability", 0.0),
                    face.get("sam_stability", 0.0),
                ),
                source="cardboard_box_multiface_merge",
                color_score_floor=box.get("color_score", 0.0),
            )

            if merged is None:
                continue

            merged["selection_override"] = "angled_box_multiface_merge"
            merged["color_score"] = box.get("color_score", merged["color_score"])
            merged["merged_from_indices"] = [
                int(box["index"]),
                int(face["index"]),
            ]
            merged["merged_pair_iou"] = float(pair_iou)
            merged["merged_union_fill_ratio"] = float(union_fill_ratio)
            merged["merged_area_growth"] = float(area_growth)
            merged["score_before_merge_bonus"] = float(merged["score"])
            merged["score"] = float(
                merged["score"] + cfg.box_multiface_score_bonus
            )

            if best is None or merged["score"] > best["score"]:
                best = merged

    return best


def choose_cardboard_box_mask(cfg, masks, roi_rgb):
    box_candidates = []

    for i, sam_mask in enumerate(masks):
        raw_mask = sam_mask["segmentation"].astype(np.uint8)
        cleaned_mask = clean_box_mask(cfg, raw_mask)

        candidate = score_cardboard_box_candidate(
            cfg,
            cleaned_mask,
            roi_rgb,
            index=i,
            sam_iou=float(sam_mask.get("predicted_iou", 0.0)),
            sam_stability=float(sam_mask.get("stability_score", 0.0)),
            source="single_sam_mask",
        )

        if candidate is None:
            continue

        box_candidates.append(candidate)

        if cfg.debug_print_masks:
            print(
                f"mask={i:03d} "
                f"score={candidate['score']:.3f} "
                f"area={candidate['area_ratio']:.3f} "
                f"rect={candidate['rectangularity']:.3f} "
                f"color={candidate['color_score']:.3f}"
            )

    if len(box_candidates) == 0:
        return None

    best_single = select_complete_single_box_candidate(cfg, box_candidates)
    best_merged = build_merged_cardboard_box_candidate(
        cfg,
        box_candidates,
        roi_rgb,
    )

    selected = best_single

    if best_merged is not None:
        print()
        print("MERGED SPLIT BOX CANDIDATE FOUND")
        print(f"  merged SAM indices: {best_merged['merged_from_indices']}")
        print(f"  split axis:         {best_merged['merged_split_axis']}")
        print(f"  geometry mode:      {best_merged['merged_geometry_mode']}")
        print(f"  axis overlap:       {best_merged['merged_axis_overlap']:.3f}")
        print(f"  gap px:             {best_merged['merged_gap_px']}")
        print(
            "  completed fill:     "
            f"{best_merged['merged_completed_fill_ratio']:.3f}"
        )
        print(f"  area growth:        {best_merged['merged_area_growth']:.3f}")
        print(f"  merged score:       {best_merged['score']:.3f}")

        if selected is None or best_merged["score"] > selected["score"]:
            selected = best_merged

    multiface = build_multiface_cardboard_box_candidate(
        cfg,
        box_candidates,
        build_secondary_face_candidates(cfg, masks, roi_rgb),
        roi_rgb,
    )

    if multiface is not None:
        print()
        print("ANGLED MULTI-FACE BOX CANDIDATE FOUND")
        print(f"  merged SAM indices: {multiface['merged_from_indices']}")
        print(
            "  union fill ratio:   "
            f"{multiface['merged_union_fill_ratio']:.3f}"
        )
        print(f"  area growth:        {multiface['merged_area_growth']:.3f}")
        print(f"  merged score:       {multiface['score']:.3f}")

        if selected is None or multiface["score"] > selected["score"]:
            selected = multiface

    return selected


def run_sam2_cardboard_box(cfg, frame_bgr, depth_measure_aligned, mask_generator, intrinsics):
    """Run SAM2 on the ROI, pick the best cardboard-box mask, and measure its
    face depth + dimensions. Returns a result dict or None when no valid box
    mask is found. `depth_measure_aligned` is the measurement-stereo depth
    aligned to RGB."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi_scaled = depth_measure_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    box = choose_cardboard_box_mask(cfg, masks, roi_rgb)

    if box is None:
        return None

    center_full = (
        cfg.roi_x1 + box["center_roi"][0],
        cfg.roi_y1 + box["center_roi"][1],
    )

    scaled_depth_result = box_face_depth_mm(cfg, depth_roi_scaled, box["mask"])

    box_face_depth = scaled_depth_result["distance_mm"]
    box_depth = estimate_box_depth_mm(cfg, box_face_depth)

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        box_face_depth,
        intrinsics,
    )

    dimensions = estimate_box_dimensions_mm(
        cfg,
        box,
        box_face_depth,
        intrinsics,
    )

    return {
        "roi_rgb": roi_rgb,
        "box": box,
        "center_full": center_full,
        "box_face_depth_mm": box_face_depth,
        "box_depth_mm": box_depth,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "dimensions": dimensions,
        "scaled_depth_result": scaled_depth_result,
        "final_output": {
            "box_face_depth_mm": box_face_depth,
            "box_depth_mm": box_depth,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["short_side_mm"],
            "angle_deg": dimensions["angle_deg"],
            "center_x_mm": center_x_mm,
            "center_y_mm": center_y_mm,
        },
    }
