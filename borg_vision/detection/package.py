"""Package-mode detection logic: mask scoring, classification, dimensions,
depth stats.

All functions are ports of the original product_detection_final.py with the
module-level constants replaced by fields of a PackageConfig passed as the
first argument. Function bodies are otherwise unchanged.
"""

import warnings

import cv2
import numpy as np
import torch

from ..utils import _long_axis_angle_deg, fmt3, score_from_bad_good, score_from_range
from ..visualization import make_package_depth_heatmap


# Mask sources whose strong RGB segmentation evidence is allowed to override
# the depth-based package type. Fixed set literal (not a config field).
BOX_TYPE_SEGMENTATION_OVERRIDE_SOURCES = {
    "package_roi_exact_box_rules",
    "cardboard_box_merged_masks",
    "cardboard_box_multiface_merge",
}


def pixel_to_camera_xy_mm(center_full, depth_mm, intrinsics):
    if center_full is None or depth_mm is None or intrinsics is None:
        return None, None

    u, v = center_full

    x_mm = (u - intrinsics["cx"]) * depth_mm / intrinsics["fx"]
    y_mm = (v - intrinsics["cy"]) * depth_mm / intrinsics["fy"]

    return float(x_mm), float(y_mm)


def close_mask(mask, kernel_px, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )


def erode_mask(mask, radius_px):
    kernel_size = radius_px * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilate_mask(mask, radius_px):
    kernel_size = radius_px * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    return (labels == largest_label).astype(np.uint8)


def clean_package_mask(cfg, mask):
    mask = mask.astype(np.uint8)
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(
        mask,
        cfg.package_mask_close_kernel_px,
        cfg.package_mask_close_iterations,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.package_max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask


def clean_box_mask(cfg, mask):
    mask = mask.astype(np.uint8)
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(
        mask,
        cfg.box_mask_close_kernel_px,
        cfg.box_mask_close_iterations,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.box_max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def get_safe_center(cfg, mask):
    safe_mask = erode_mask(mask, cfg.safe_erode_radius_px)
    safe_component = largest_component(safe_mask)

    if safe_component is None:
        safe_component = largest_component(mask)

    if safe_component is None:
        return None, safe_mask

    center = get_mask_center(safe_component)

    if center is None:
        return None, safe_component

    return center, safe_component.astype(np.uint8)


def get_rotated_box_from_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    (cx, cy), (w_px, h_px), angle = rect

    return {
        "center_roi": (float(cx), float(cy)),
        "width_px": float(w_px),
        "height_px": float(h_px),
        "angle_deg": _long_axis_angle_deg(w_px, h_px, angle),
        "points_roi": box_points,
    }


def is_roller_like_horizontal_strip(cfg, mask):
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))

    width_ratio = bw / max(w, 1)
    height_ratio = bh / max(h, 1)

    return (
        width_ratio >= cfg.roller_strip_min_width_ratio
        and height_ratio <= cfg.roller_strip_max_height_ratio
    )


def is_edge_strip(cfg, mask):
    roi_h, roi_w = mask.shape[:2]
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    if w <= 0 or h <= 0:
        return True

    bbox_width_ratio = w / max(roi_w, 1)
    bbox_height_ratio = h / max(roi_h, 1)

    touches_left = x <= cfg.edge_strip_margin_px
    touches_right = (x + w) >= (roi_w - cfg.edge_strip_margin_px)
    touches_top = y <= cfg.edge_strip_margin_px
    touches_bottom = (y + h) >= (roi_h - cfg.edge_strip_margin_px)

    vertical_edge_strip = (
        (touches_left or touches_right)
        and bbox_width_ratio <= cfg.edge_strip_max_width_ratio
    )

    horizontal_edge_strip = (
        (touches_top or touches_bottom)
        and bbox_height_ratio <= cfg.edge_strip_max_height_ratio
    )

    return bool(vertical_edge_strip or horizontal_edge_strip)


def compute_rgb_contrast_score(cfg, roi_rgb, mask):
    if int(mask.sum()) < 20:
        return 0.0, 0.0

    dilated = dilate_mask(mask, cfg.ring_dilate_px)
    ring = dilated.copy()
    ring[mask == 1] = 0

    if int(ring.sum()) < 20:
        return 0.0, 0.0

    mask_pixels = roi_rgb[mask == 1].astype(np.float32)
    ring_pixels = roi_rgb[ring == 1].astype(np.float32)

    if mask_pixels.size == 0 or ring_pixels.size == 0:
        return 0.0, 0.0

    mask_mean = np.mean(mask_pixels, axis=0)
    ring_mean = np.mean(ring_pixels, axis=0)

    contrast = float(np.mean(np.abs(mask_mean - ring_mean)))
    contrast_score = score_from_bad_good(contrast, cfg.bad_contrast, cfg.good_contrast)

    return contrast_score, contrast


def compute_texture_score(cfg, roi_rgb, mask):
    if int(mask.sum()) < 20:
        return 0.0, 0.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    values = gray[mask == 1].astype(np.float32)

    if values.size < 20:
        return 0.0, 0.0

    texture = float(np.std(values))
    texture_score = score_from_bad_good(texture, cfg.bad_texture, cfg.good_texture)

    return texture_score, texture


def score_package_product_mask(cfg, mask, roi_rgb, sam_iou, sam_stability):
    mask = mask.astype(np.uint8)

    if cfg.hard_reject_roller_like and is_roller_like_horizontal_strip(cfg, mask):
        return None, "roller_like"

    if cfg.reject_edge_strips and is_edge_strip(cfg, mask):
        return None, "edge_strip"

    roi_h, roi_w = mask.shape[:2]
    roi_area = roi_h * roi_w

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < cfg.min_package_area_ratio:
        return None, "too_small"

    if area_ratio > cfg.max_package_area_ratio:
        return None, "too_large"

    x, y, w, h = cv2.boundingRect(mask)

    if w <= 0 or h <= 0:
        return None, "bad_bbox"

    bbox_area = w * h
    rectangularity = area / max(bbox_area, 1)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    if rectangularity < cfg.min_package_rectangularity:
        return None, "low_rectangularity"

    if aspect_ratio > cfg.max_package_aspect_ratio:
        return None, "bad_aspect"

    center_roi, safe_mask = get_safe_center(cfg, mask)

    if center_roi is None:
        return None, "no_center"

    roi_cx = roi_w / 2
    roi_cy = roi_h / 2

    dist = np.sqrt(
        (center_roi[0] - roi_cx) ** 2
        + (center_roi[1] - roi_cy) ** 2
    )

    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    if center_score < cfg.min_package_center_score:
        return None, "center_too_far"

    bbox_width_ratio = w / max(roi_w, 1)
    bbox_height_ratio = h / max(roi_h, 1)

    area_score = 1.0 - min(
        abs(area_ratio - cfg.target_package_area_ratio) / cfg.target_package_area_ratio,
        1.0,
    )

    bbox_size_score = min(
        min(bbox_width_ratio / 0.45, bbox_height_ratio / 0.45),
        1.0,
    )

    safe_area_ratio = int(safe_mask.sum()) / max(roi_area, 1)
    safe_area_score = min(safe_area_ratio / 0.03, 1.0)

    contrast_score, contrast_value = compute_rgb_contrast_score(cfg, roi_rgb, mask)
    texture_score, texture_value = compute_texture_score(cfg, roi_rgb, mask)

    score = (
        cfg.area_score_weight * area_score
        + cfg.center_score_weight * center_score
        + cfg.contrast_score_weight * contrast_score
        + cfg.texture_score_weight * texture_score
        + cfg.rect_score_weight * rectangularity
        + cfg.bbox_size_score_weight * bbox_size_score
        + cfg.safe_area_score_weight * safe_area_score
        + cfg.sam_iou_score_weight * sam_iou
        + cfg.sam_stability_score_weight * sam_stability
    )

    rotated = get_rotated_box_from_mask(mask)

    return {
        "source": "package_product_rules",
        "index": None,
        "score": float(score),
        "mask": mask,
        "safe_mask": safe_mask,
        "bbox": (int(x), int(y), int(w), int(h)),
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "bbox_width_ratio": float(bbox_width_ratio),
        "bbox_height_ratio": float(bbox_height_ratio),
        "bbox_size_score": float(bbox_size_score),
        "safe_area_ratio": float(safe_area_ratio),
        "safe_area_score": float(safe_area_score),
        "contrast_score": float(contrast_score),
        "contrast_value": float(contrast_value),
        "texture_score": float(texture_score),
        "texture_value": float(texture_value),
        "color_score": None,
        "mean_hsv": None,
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
        "rotated": rotated,
    }, "ok"


def score_cardboard_box_mask(cfg, mask, roi_rgb, sam_iou, sam_stability):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    mask = mask.astype(np.uint8)

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < cfg.min_box_area_ratio:
        return None, "box_too_small"

    if area_ratio > cfg.max_box_area_ratio:
        return None, "box_too_large"

    x, y, bw, bh = cv2.boundingRect(mask)

    if bw <= 0 or bh <= 0:
        return None, "box_bad_bbox"

    rectangularity = area / max(bw * bh, 1)
    aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

    if rectangularity < cfg.min_box_rectangularity:
        return None, "box_low_rectangularity"

    if aspect_ratio > cfg.max_box_aspect_ratio:
        return None, "box_bad_aspect"

    masked_hsv = hsv[mask == 1]

    if masked_hsv.size == 0:
        return None, "box_no_hsv"

    mean_h = float(np.mean(masked_hsv[:, 0]))
    mean_s = float(np.mean(masked_hsv[:, 1]))
    mean_v = float(np.mean(masked_hsv[:, 2]))

    if mean_v < cfg.min_box_value:
        return None, "box_too_dark"

    hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
    sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
    val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)

    color_score = (
        0.50 * hue_score
        + 0.25 * sat_score
        + 0.25 * val_score
    )

    if color_score < cfg.min_box_color_score:
        return None, "box_low_color_score"

    center_roi = get_mask_center(mask)

    if center_roi is None:
        return None, "box_no_center"

    cx, cy = center_roi

    dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    area_score = 1.0 - min(
        abs(area_ratio - cfg.target_box_area_ratio) / cfg.target_box_area_ratio,
        1.0,
    )

    score = (
        2.7 * color_score
        + 1.6 * rectangularity
        + 1.2 * area_score
        + 1.0 * center_score
        + 0.4 * sam_iou
        + 0.4 * sam_stability
    )

    rotated = get_rotated_box_from_mask(mask)

    return {
        "source": "cardboard_box_rules",
        "index": None,
        "score": float(score),
        "mask": mask,
        "safe_mask": mask,
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "bbox_width_ratio": float(bw / max(w, 1)),
        "bbox_height_ratio": float(bh / max(h, 1)),
        "bbox_size_score": None,
        "safe_area_ratio": None,
        "safe_area_score": None,
        "contrast_score": None,
        "contrast_value": None,
        "texture_score": None,
        "texture_value": None,
        "color_score": float(color_score),
        "mean_hsv": (mean_h, mean_s, mean_v),
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
        "rotated": rotated,
    }, "ok"


def mask_overlap_fraction(inner_mask, outer_mask):
    inner_mask = inner_mask.astype(bool)
    outer_mask = outer_mask.astype(bool)

    inner_area = int(inner_mask.sum())

    if inner_area <= 0:
        return 0.0

    overlap = int(np.logical_and(inner_mask, outer_mask).sum())

    return float(overlap / inner_area)


def mask_iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    union = int(np.logical_or(a, b).sum())

    if union <= 0:
        return 0.0

    intersection = int(np.logical_and(a, b).sum())
    return float(intersection / union)


def horizontal_overlap_fraction(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    left = max(ax, bx)
    right = min(ax + aw, bx + bw)
    overlap = max(0, right - left)

    return float(overlap / max(min(aw, bw), 1))


def vertical_gap_px(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    a_bottom = ay + ah
    b_bottom = by + bh

    if a_bottom < by:
        return int(by - a_bottom)

    if b_bottom < ay:
        return int(ay - b_bottom)

    return 0


def complete_min_area_rectangle(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    points = np.vstack(contours)

    if points.shape[0] < 4:
        return None

    rect = cv2.minAreaRect(points)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(np.round(box_points))

    completed = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(completed, box_points, 1)

    return completed


def convex_hull_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    points = np.vstack(contours)
    hull = cv2.convexHull(points)
    completed = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(completed, hull, 1)
    return completed


def masks_are_near(mask_a, mask_b, radius_px):
    dilated_a = dilate_mask(mask_a, radius_px)
    return bool(np.any((dilated_a == 1) & (mask_b == 1)))


def normalized_rect_angle(rotated):
    """Long-axis rotation angle of a min-area rect, wrapped to [-90, 90).

    The standalone scripts apply the long-axis correction (+90 deg when the
    rect's width is the short side) here, because their
    get_rotated_box_from_mask returns the raw cv2.minAreaRect angle. The
    library's version already applies that same correction via
    _long_axis_angle_deg, so re-applying it here would cancel it out and leave
    the raw angle -- making two perpendicular fragments compare as aligned.
    Only the wrap is needed.
    """
    angle = float(rotated["angle_deg"])

    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    return angle


def cardboard_color_score(roi_rgb, mask):
    """Cardboard-tone score of a mask, 0..1.

    Same HSV formula as score_cardboard_box_mask (keep the two in sync); split
    out so merge paths can judge a fragment that never went through the full
    box scorer."""
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    masked = hsv[mask.astype(bool)]
    if masked.size == 0:
        return 0.0
    mean_h = float(np.mean(masked[:, 0]))
    mean_s = float(np.mean(masked[:, 1]))
    mean_v = float(np.mean(masked[:, 2]))
    hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
    sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
    val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)
    return 0.50 * hue_score + 0.25 * sat_score + 0.25 * val_score


def build_multiface_cardboard_box_candidate(
    cfg,
    box_candidates,
    package_candidates,
    roi_rgb,
):
    if not cfg.box_multiface_merge_enable:
        return None

    if not box_candidates or not package_candidates:
        return None

    # Same fragment-quality bar as the split-mask merge: a hull anchored on a
    # weak "box" fragment can only invent geometry.
    box_candidates = [
        c for c in box_candidates
        if (c.get("color_score") or 0.0) >= cfg.box_merge_min_fragment_color_score
        and (c.get("rectangularity") or 0.0) >= cfg.box_merge_min_fragment_rectangularity
    ]
    if not box_candidates:
        return None

    boxes = sorted(
        box_candidates,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )[:cfg.box_multiface_max_candidates]
    packages = sorted(
        package_candidates,
        key=lambda item: item.get("area", int(item["mask"].sum())),
        reverse=True,
    )[:cfg.box_multiface_max_candidates]

    # The package-side fragment never went through the box color gate; a
    # carpet patch beside the box qualifies geometrically, so require the
    # second face to actually look like cardboard.
    packages = [
        p for p in packages
        if cardboard_color_score(roi_rgb, p["mask"])
        >= cfg.box_multiface_min_second_color_score
    ]
    if not packages:
        return None

    roi_area = roi_rgb.shape[0] * roi_rgb.shape[1]
    best = None

    for box in boxes:
        box_mask = box["mask"].astype(np.uint8)
        box_area = int(box_mask.sum())

        for package in packages:
            if package.get("index") == box.get("index"):
                continue

            package_mask = package["mask"].astype(np.uint8)
            package_area = int(package_mask.sum())
            package_area_ratio = package_area / max(roi_area, 1)

            if package_area_ratio < cfg.box_multiface_min_second_area_ratio:
                continue

            pair_iou = mask_iou(box_mask, package_mask)
            if pair_iou > cfg.box_multiface_max_pair_iou:
                continue

            if not masks_are_near(
                box_mask,
                package_mask,
                cfg.box_multiface_touch_dilate_px,
            ):
                continue

            union_mask = np.logical_or(box_mask, package_mask).astype(np.uint8)
            hull_mask = convex_hull_mask(union_mask)
            if hull_mask is None:
                continue

            hull_area = int(hull_mask.sum())
            union_area = int(union_mask.sum())
            largest_area = max(box_area, package_area)
            area_growth = hull_area / max(largest_area, 1)
            union_fill_ratio = union_area / max(hull_area, 1)
            hull_area_ratio = hull_area / max(roi_area, 1)

            if area_growth < cfg.box_multiface_min_area_growth:
                continue
            if union_fill_ratio < cfg.box_multiface_min_union_fill_ratio:
                continue
            if hull_area_ratio > cfg.box_multiface_max_hull_area_ratio:
                continue

            merged, reason = score_package_product_mask(
                cfg,
                hull_mask,
                roi_rgb,
                max(box.get("sam_iou", 0.0), package.get("sam_iou", 0.0)),
                max(
                    box.get("sam_stability", 0.0),
                    package.get("sam_stability", 0.0),
                ),
            )

            if merged is None:
                continue

            merged["source"] = "cardboard_box_multiface_merge"
            merged["selection_override"] = "angled_box_multiface_merge"
            merged["reason"] = reason
            merged["color_score"] = box.get("color_score")
            merged["merged_from_indices"] = [
                int(box["index"]),
                int(package["index"]),
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


def build_merged_cardboard_box_candidate(cfg, box_candidates, roi_rgb):
    if not cfg.box_merge_split_masks_enable:
        return None

    if len(box_candidates) < 2:
        return None

    # Only box-like fragments may merge: see box_merge_min_fragment_* in the
    # config for the failure this guards against (complete box + background
    # strip, rectangle-completed into an oversized "box").
    box_candidates = [
        c for c in box_candidates
        if (c.get("color_score") or 0.0) >= cfg.box_merge_min_fragment_color_score
        and (c.get("rectangularity") or 0.0) >= cfg.box_merge_min_fragment_rectangularity
    ]

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
        first_bbox = first["bbox"]
        first_area = int(first["mask"].sum())

        for j in range(i + 1, len(candidates)):
            second = candidates[j]
            second_bbox = second["bbox"]
            second_area = int(second["mask"].sum())

            pair_iou = mask_iou(first["mask"], second["mask"])

            # Near-duplicate masks do not represent two separate box halves.
            if pair_iou > cfg.box_merge_max_pair_iou:
                continue

            x_overlap = horizontal_overlap_fraction(first_bbox, second_bbox)

            fx, fy, fw, fh = first_bbox
            sx, sy, sw, sh = second_bbox

            width_ratio = max(fw, sw) / max(min(fw, sw), 1)

            first_cx = fx + fw / 2.0
            first_cy = fy + fh / 2.0
            second_cx = sx + sw / 2.0
            second_cy = sy + sh / 2.0
            center_x_diff_ratio = abs(first_cx - second_cx) / max(max(fw, sw), 1)
            gap_px = vertical_gap_px(first_bbox, second_bbox)

            # Original upright-box relationship.
            axis_aligned_pair = (
                x_overlap >= cfg.box_merge_min_horizontal_overlap
                and width_ratio <= cfg.box_merge_max_width_ratio
                and center_x_diff_ratio <= cfg.box_merge_max_center_x_diff_ratio
                and gap_px <= cfg.box_merge_max_vertical_gap_px
            )

            # Rotation-independent relationship. SAM often splits an angled
            # cardboard box along its seam. In image coordinates those pieces
            # may have little horizontal overlap, even though together they
            # form one clean rotated rectangle.
            rotated_pair = False

            if cfg.box_merge_rotated_enable and not axis_aligned_pair:
                first_rotated = get_rotated_box_from_mask(first["mask"])
                second_rotated = get_rotated_box_from_mask(second["mask"])

                if first_rotated is not None and second_rotated is not None:
                    angle_a = normalized_rect_angle(first_rotated)
                    angle_b = normalized_rect_angle(second_rotated)
                    angle_diff = abs(angle_a - angle_b)
                    angle_diff = min(angle_diff, 180.0 - angle_diff)

                    center_distance = float(np.hypot(
                        first_cx - second_cx,
                        first_cy - second_cy,
                    ))
                    characteristic_size = max(
                        first_rotated["width_px"],
                        first_rotated["height_px"],
                        second_rotated["width_px"],
                        second_rotated["height_px"],
                        1.0,
                    )
                    center_distance_ratio = center_distance / characteristic_size

                    rotated_pair = (
                        angle_diff <= cfg.box_merge_rotated_max_angle_diff_deg
                        and center_distance_ratio
                        <= cfg.box_merge_rotated_max_center_distance_ratio
                    )

            if not axis_aligned_pair and not rotated_pair:
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

            # A rotated pair is a weaker signal than an upright one, so the
            # completed rectangle must actually be filled by the fragments and
            # must not swallow the whole ROI.
            if rotated_pair and (
                completed_fill_ratio < cfg.box_merge_rotated_min_completed_fill_ratio
                or completed_area_ratio > cfg.box_merge_rotated_max_completed_area_ratio
            ):
                continue

            merged_result, merged_reason = score_cardboard_box_mask(
                cfg,
                completed_mask,
                roi_rgb,
                max(first.get("sam_iou", 0.0), second.get("sam_iou", 0.0)),
                max(
                    first.get("sam_stability", 0.0),
                    second.get("sam_stability", 0.0),
                ),
            )

            if merged_result is None:
                continue

            if (
                merged_result["rectangularity"]
                < cfg.box_merge_min_result_rectangularity
            ):
                continue

            merged_result["source"] = "cardboard_box_merged_masks"
            merged_result["selection_override"] = "merged_split_box_masks"
            merged_result["reason"] = merged_reason
            merged_result["merged_from_indices"] = [
                int(first["index"]),
                int(second["index"]),
            ]
            merged_result["merged_pair_iou"] = float(pair_iou)
            merged_result["merged_horizontal_overlap"] = float(x_overlap)
            merged_result["merged_vertical_gap_px"] = int(gap_px)
            merged_result["merged_geometry_mode"] = (
                "axis_aligned" if axis_aligned_pair else "rotated"
            )
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

            if (
                best_merged is None
                or merged_result["score"] > best_merged["score"]
            ):
                best_merged = merged_result

    return best_merged


def choose_best_package_mask_from_package_roi(cfg, masks, roi_rgb):
    best_package = None
    best_box = None
    package_candidates = []
    box_candidates = []
    accepted = []

    print(
        "\nChecking SAM 2 masks with package/product rules "
        "+ exact cardboard box rules..."
    )

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        # General package / polymailer scoring path.
        package_mask = clean_package_mask(cfg, raw_mask)
        package_result, package_reason = score_package_product_mask(
            cfg,
            package_mask,
            roi_rgb,
            sam_iou,
            sam_stability,
        )

        if package_result is not None:
            package_result["index"] = i
            package_result["reason"] = package_reason
            package_candidates.append(package_result)
            accepted.append(package_result)

            if (
                best_package is None
                or package_result["score"] > best_package["score"]
            ):
                best_package = package_result

        # Exact cardboard-box scoring path.
        box_mask = clean_box_mask(cfg, raw_mask)
        box_result, box_reason = score_cardboard_box_mask(
            cfg,
            box_mask,
            roi_rgb,
            sam_iou,
            sam_stability,
        )

        if box_result is not None:
            box_result["index"] = i
            box_result["reason"] = box_reason
            box_candidates.append(box_result)
            accepted.append(box_result)

            if best_box is None or box_result["score"] > best_box["score"]:
                best_box = box_result

        if cfg.debug_print_masks:
            if package_result is not None:
                print(
                    f"mask={i:03d} "
                    f"source=package "
                    f"score={package_result['score']:.3f} "
                    f"area={package_result['area_ratio']:.3f} "
                    f"rect={package_result['rectangularity']:.3f}"
                )

            if box_result is not None:
                print(
                    f"mask={i:03d} "
                    f"source=box "
                    f"score={box_result['score']:.3f} "
                    f"area={box_result['area_ratio']:.3f} "
                    f"rect={box_result['rectangularity']:.3f} "
                    f"color={box_result['color_score']:.3f}"
                )

    merged_box = build_merged_cardboard_box_candidate(
        cfg,
        box_candidates,
        roi_rgb,
    )

    if merged_box is not None:
        accepted.append(merged_box)

        print()
        print("MERGED SPLIT BOX CANDIDATE FOUND")
        print(
            "  merged SAM indices:       "
            f"{merged_box['merged_from_indices']}"
        )
        print(
            "  horizontal overlap:       "
            f"{merged_box['merged_horizontal_overlap']:.3f}"
        )
        print(
            "  vertical gap px:          "
            f"{merged_box['merged_vertical_gap_px']}"
        )
        print(
            "  geometry mode:            "
            f"{merged_box.get('merged_geometry_mode', 'axis_aligned')}"
        )
        print(
            "  completed fill ratio:     "
            f"{merged_box.get('merged_completed_fill_ratio', 0.0):.3f}"
        )
        print(
            "  area growth over fragment:"
            f" {merged_box['merged_area_growth']:.3f}"
        )
        print(
            "  merged box score:         "
            f"{merged_box['score']:.3f}"
        )

        if best_box is None or merged_box["score"] > best_box["score"]:
            best_box = merged_box

    multiface_box = build_multiface_cardboard_box_candidate(
        cfg,
        box_candidates,
        package_candidates,
        roi_rgb,
    )

    if multiface_box is not None:
        accepted.append(multiface_box)

        print()
        print("ANGLED MULTI-FACE BOX CANDIDATE FOUND")
        print(f"  merged SAM indices: {multiface_box['merged_from_indices']}")
        print(
            "  union fill ratio:  "
            f"{multiface_box['merged_union_fill_ratio']:.3f}"
        )
        print(
            "  area growth:       "
            f"{multiface_box['merged_area_growth']:.3f}"
        )
        print(
            "  merged score:      "
            f"{multiface_box['score']:.3f}"
        )

        if best_box is None or multiface_box["score"] > best_box["score"]:
            best_box = multiface_box

    if best_package is None and best_box is None:
        return None, accepted

    if best_package is None:
        best_box["selection_override"] = "only_box_candidate"
        print("Selected cardboard-box candidate: no general candidate.")
        return best_box, accepted

    if best_box is None:
        best_package["selection_override"] = "only_package_candidate"
        print("Selected general package candidate: no box candidate.")
        return best_package, accepted

    package_area = int(best_package["mask"].sum())
    box_area = int(best_box["mask"].sum())

    package_inside_box = mask_overlap_fraction(
        best_package["mask"],
        best_box["mask"],
    )

    box_inside_package = mask_overlap_fraction(
        best_box["mask"],
        best_package["mask"],
    )

    area_growth = box_area / max(package_area, 1)

    box_color_score = best_box.get("color_score")

    if box_color_score is None:
        box_color_score = 0.0

    box_rectangularity = best_box.get("rectangularity", 0.0)
    box_area_ratio = best_box.get("area_ratio", 0.0)

    # Main full-box override. The general candidate lies mostly inside the
    # cardboard candidate, while the cardboard candidate is substantially
    # larger and still passes the original box color and geometry rules.
    full_box_override = (
        cfg.box_full_mask_override_enable
        and package_inside_box >= cfg.box_full_mask_min_package_containment
        and area_growth >= cfg.box_full_mask_min_area_growth
        and box_color_score >= cfg.min_box_color_score
        and box_rectangularity >= cfg.min_box_rectangularity
        and box_area_ratio >= cfg.min_box_area_ratio
    )

    # Stronger-looking full cardboard masks may use a slightly smaller area
    # increase because their color and rectangularity provide extra evidence.
    strong_box_override = (
        cfg.box_full_mask_override_enable
        and package_inside_box >= 0.60
        and area_growth >= 1.15
        and box_color_score >= cfg.box_full_mask_strong_color_score
        and box_rectangularity >= cfg.box_full_mask_strong_rectangularity
    )

    # A very strong cardboard candidate can also win when it surrounds most
    # of the smaller package candidate and is clearly more complete.
    independent_box_override = (
        cfg.box_full_mask_override_enable
        and box_color_score >= 0.70
        and box_rectangularity >= 0.88
        and box_area >= package_area * 1.45
        and package_inside_box >= 0.55
    )

    if (
        full_box_override
        or strong_box_override
        or independent_box_override
    ):
        print()
        print("FULL BOX OVERRIDE APPLIED")
        print(f"  package mask area:       {package_area}")
        print(f"  cardboard mask area:     {box_area}")
        print(f"  box/package area growth: {area_growth:.3f}")
        print(f"  package inside box:      {package_inside_box:.3f}")
        print(f"  box inside package:      {box_inside_package:.3f}")
        print(f"  box color score:         {box_color_score:.3f}")
        print(f"  box rectangularity:      {box_rectangularity:.3f}")

        best_box["selection_override"] = "full_box_override"
        best_box["package_inside_box"] = package_inside_box
        best_box["box_inside_package"] = box_inside_package
        best_box["box_to_package_area_ratio"] = area_growth

        return best_box, accepted

    # Preserve the original score-based selection when no full-box evidence
    # is strong enough to justify an override.
    if best_box["score"] > best_package["score"]:
        best_box["selection_override"] = "box_score"
        best_box["package_inside_box"] = package_inside_box
        best_box["box_inside_package"] = box_inside_package
        best_box["box_to_package_area_ratio"] = area_growth
        return best_box, accepted

    best_package["selection_override"] = "package_score"
    best_package["package_inside_box"] = package_inside_box
    best_package["box_inside_package"] = box_inside_package
    best_package["box_to_package_area_ratio"] = area_growth
    return best_package, accepted


def choose_cardboard_box_mask_exact(cfg, masks, box_roi_rgb):
    """Exact SAM mask-selection path from box_detection_final.py."""
    h, w = box_roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(box_roi_rgb, cv2.COLOR_RGB2HSV)

    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_box_mask(cfg, raw_mask)

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.min_box_area_ratio:
            continue

        if area_ratio > cfg.max_box_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.min_box_rectangularity:
            continue

        if aspect_ratio > cfg.max_box_aspect_ratio:
            continue

        masked_hsv = hsv[mask == 1]

        if masked_hsv.size == 0:
            continue

        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))

        if mean_v < cfg.min_box_value:
            continue

        hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
        sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
        val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)

        color_score = (
            0.50 * hue_score
            + 0.25 * sat_score
            + 0.25 * val_score
        )

        if color_score < cfg.min_box_color_score:
            continue

        center_roi = get_mask_center(mask)

        if center_roi is None:
            continue

        cx, cy = center_roi

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_box_area_ratio) / cfg.target_box_area_ratio,
            1.0,
        )

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        score = (
            2.7 * color_score
            + 1.6 * rectangularity
            + 1.2 * area_score
            + 1.0 * center_score
            + 0.4 * sam_iou
            + 0.4 * sam_stability
        )

        candidate = {
            "index": i,
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
            "mean_hsv": (mean_h, mean_s, mean_v),
            "sam_iou": sam_iou,
            "sam_stability": sam_stability,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def map_dedicated_box_candidate_to_package_roi(cfg, box_candidate):
    """Normalize the exact box candidate already produced in the package ROI."""
    if box_candidate is None:
        return None

    package_h = cfg.roi_y2 - cfg.roi_y1
    package_w = cfg.roi_x2 - cfg.roi_x1

    mask = box_candidate["mask"].astype(np.uint8)

    if mask.shape[:2] != (package_h, package_w):
        print(
            "Exact box candidate shape does not match the package ROI: "
            f"candidate={mask.shape[:2]} expected={(package_h, package_w)}"
        )
        return None

    area = int(mask.sum())

    if area <= 0:
        return None

    center_roi = get_mask_center(mask)

    if center_roi is None:
        return None

    x, y, w, h = cv2.boundingRect(mask)
    rectangularity = area / max(w * h, 1)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))
    package_area = package_h * package_w

    candidate = dict(box_candidate)
    candidate.update(
        {
            "source": "package_roi_exact_box_rules",
            "selection_override": "package_roi_exact_box_rules",
            "mask": mask,
            "safe_mask": mask,
            "bbox": (int(x), int(y), int(w), int(h)),
            "center_roi": center_roi,
            "area": area,
            "area_ratio": float(area / max(package_area, 1)),
            "box_roi_area_ratio": float(box_candidate["area_ratio"]),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "bbox_width_ratio": float(w / max(package_w, 1)),
            "bbox_height_ratio": float(h / max(package_h, 1)),
            "bbox_size_score": None,
            "safe_area_ratio": None,
            "safe_area_score": None,
            "contrast_score": None,
            "contrast_value": None,
            "texture_score": None,
            "texture_value": None,
            "rotated": get_rotated_box_from_mask(mask),
            "exact_box_rules_roi": (
                cfg.roi_x1,
                cfg.roi_y1,
                cfg.roi_x2,
                cfg.roi_y2,
            ),
        }
    )

    return candidate


def choose_best_package_mask(cfg, masks, roi_rgb, dedicated_box_candidate=None):
    """
    Keep the complete existing package-ROI selection path, then compare it
    against the exact box-script candidate made from the same package ROI masks.
    """
    package_best, accepted = choose_best_package_mask_from_package_roi(
        cfg,
        masks,
        roi_rgb,
    )

    if dedicated_box_candidate is None:
        return package_best, accepted

    accepted.append(dedicated_box_candidate)

    if package_best is None:
        print("Selected exact-box-rules candidate from the package ROI: no general candidate.")
        return dedicated_box_candidate, accepted

    package_area = int(package_best["mask"].sum())
    dedicated_area = int(dedicated_box_candidate["mask"].sum())

    package_inside_dedicated = mask_overlap_fraction(
        package_best["mask"],
        dedicated_box_candidate["mask"],
    )

    dedicated_inside_package = mask_overlap_fraction(
        dedicated_box_candidate["mask"],
        package_best["mask"],
    )

    area_ratio = dedicated_area / max(package_area, 1)

    # Prefer the exact box-script segmentation when it is at least as complete
    # as the package candidate, or when it clearly contains a smaller half-box
    # candidate. Do not replace a larger complete package mask with a smaller
    # dedicated mask.
    similar_complete_mask = (
        dedicated_area >= package_area * 0.90
        and (
            package_inside_dedicated >= 0.60
            or dedicated_inside_package >= 0.60
        )
    )

    full_box_contains_package = (
        dedicated_area >= package_area * 1.10
        and package_inside_dedicated >= 0.50
    )

    if similar_complete_mask or full_box_contains_package:
        dedicated_box_candidate["package_inside_dedicated_box"] = (
            package_inside_dedicated
        )
        dedicated_box_candidate["dedicated_box_inside_package"] = (
            dedicated_inside_package
        )
        dedicated_box_candidate["dedicated_to_package_area_ratio"] = area_ratio

        print()
        print("EXACT BOX RULES SELECTED FROM PACKAGE ROI")
        print(f"  package mask area:              {package_area}")
        print(f"  exact box mask area:        {dedicated_area}")
        print(f"  exact/package area ratio:   {area_ratio:.3f}")
        print(f"  package inside exact box:   {package_inside_dedicated:.3f}")
        print(f"  exact box inside package:   {dedicated_inside_package:.3f}")
        print(f"  exact box color score:      {dedicated_box_candidate['color_score']:.3f}")
        print(f"  exact box rectangularity:   {dedicated_box_candidate['rectangularity']:.3f}")

        return dedicated_box_candidate, accepted

    print()
    print("Exact-box-rules candidate was smaller or inconsistent.")
    print("Keeping the package-ROI selection.")
    print(f"  package mask area:              {package_area}")
    print(f"  exact box mask area:        {dedicated_area}")
    print(f"  exact/package area ratio:   {area_ratio:.3f}")
    print(f"  package inside exact box:   {package_inside_dedicated:.3f}")
    print(f"  exact box inside package:   {dedicated_inside_package:.3f}")

    return package_best, accepted


def robust_depth_values(cfg, values):
    return values[
        (values > cfg.min_valid_depth_mm)
        & (values < cfg.max_valid_depth_mm)
    ].astype(np.float32)


def robust_median_depth(cfg, values):
    values = values.astype(np.float32)

    if values.size < cfg.min_surface_depth_count:
        return None, 0, None, None

    raw_median = float(np.median(values))
    abs_dev = np.abs(values - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = values[np.abs(values - raw_median) <= 12.0]
    else:
        filtered = values[abs_dev <= 3.5 * mad]

    if filtered.size < cfg.min_surface_depth_count:
        filtered = values

    return (
        float(np.median(filtered)),
        int(filtered.size),
        float(np.mean(filtered)),
        float(np.min(filtered)),
    )


def package_face_depth_mm(cfg, depth_roi, mask):
    valid_sample = (
        (mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    values = depth_roi[valid_sample].astype(np.float32)

    distance_mm, depth_count, mean_depth, min_depth = robust_median_depth(cfg, values)

    return {
        "raw_distance_mm": distance_mm,
        "distance_mm": (
            distance_mm + cfg.measurement_depth_offset_mm
            if distance_mm is not None
            else None
        ),
        "depth_count": depth_count,
        "sample_count_total": int(values.size),
        "mean_depth_mm": (
            mean_depth + cfg.measurement_depth_offset_mm
            if mean_depth is not None
            else None
        ),
        "min_depth_mm": (
            min_depth + cfg.measurement_depth_offset_mm
            if min_depth is not None
            else None
        ),
        "measurement_depth_offset_mm": cfg.measurement_depth_offset_mm,
    }


def robust_stats_from_values(cfg, values, total_count):
    valid = robust_depth_values(cfg, values)
    valid_fraction = valid.size / max(total_count, 1)

    if valid.size == 0:
        return {
            "valid_fraction": 0.0,
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
            "range_p90_p10": None,
            "p10": None,
            "p90": None,
        }

    raw_median = float(np.median(valid))
    abs_dev = np.abs(valid - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = valid[np.abs(valid - raw_median) <= 12.0]
    else:
        filtered = valid[abs_dev <= 3.5 * mad]

    if filtered.size == 0:
        filtered = valid

    p10 = float(np.percentile(valid, 10))
    p90 = float(np.percentile(valid, 90))

    return {
        "valid_fraction": float(valid_fraction),
        "median": float(np.median(filtered)),
        "mean": float(np.mean(filtered)),
        "std": float(np.std(filtered)),
        "min": float(np.min(filtered)),
        "max": float(np.max(filtered)),
        "count": int(filtered.size),
        "range_p90_p10": float(p90 - p10),
        "p10": p10,
        "p90": p90,
    }


def depth_stats_inside_mask(cfg, depth_roi, mask):
    selected_depth = depth_roi[mask == 1]
    return robust_stats_from_values(cfg, selected_depth, selected_depth.size)


def get_mask_rectangularity(mask):
    area = int(mask.sum())

    if area <= 0:
        return 0.0

    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    if w <= 0 or h <= 0:
        return 0.0

    return float(area / max(w * h, 1))


def fit_plane_depth_features(cfg, depth_roi, package_mask):
    mask = package_mask.astype(np.uint8)

    valid = (
        (mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    valid_points = int(valid.sum())
    mask_area = int(mask.sum())
    valid_fraction_on_mask = valid_points / max(mask_area, 1)

    empty = {
        "success": False,
        "valid_points": valid_points,
        "valid_fraction_on_mask": float(valid_fraction_on_mask),
        "plane_residual_std_mm": None,
        "plane_residual_range_mm": None,
        "center_edge_depth_delta_mm": None,
    }

    if valid_points < 100:
        return empty

    yy, xx = np.where(valid)
    zz = depth_roi[valid].astype(np.float32)

    if zz.size > cfg.max_plane_points:
        rng = np.random.default_rng(12345)
        idx = rng.choice(zz.size, size=cfg.max_plane_points, replace=False)
        xx_fit = xx[idx].astype(np.float32)
        yy_fit = yy[idx].astype(np.float32)
        zz_fit = zz[idx]
    else:
        xx_fit = xx.astype(np.float32)
        yy_fit = yy.astype(np.float32)
        zz_fit = zz

    a = np.column_stack(
        [
            xx_fit,
            yy_fit,
            np.ones_like(xx_fit),
        ]
    )

    try:
        coeffs, _, _, _ = np.linalg.lstsq(a, zz_fit, rcond=None)
        pred = a @ coeffs
        residuals = zz_fit - pred

        residual_std = float(np.std(residuals))
        residual_range = float(np.percentile(residuals, 90) - np.percentile(residuals, 10))

    except Exception:
        residual_std = None
        residual_range = None

    center = get_mask_center(mask)
    center_edge_delta = None

    if center is not None:
        cx, cy = center
        yy_all, xx_all = np.indices(mask.shape)

        center_circle = (
            ((xx_all - cx) ** 2 + (yy_all - cy) ** 2)
            <= cfg.center_edge_kernel_px ** 2
        )

        eroded = erode_mask(mask, cfg.center_edge_kernel_px // 2)
        edge = mask.copy()
        edge[eroded == 1] = 0

        center_values = depth_roi[
            (mask == 1)
            & center_circle
            & (depth_roi > cfg.min_valid_depth_mm)
            & (depth_roi < cfg.max_valid_depth_mm)
        ].astype(np.float32)

        edge_values = depth_roi[
            (edge == 1)
            & (depth_roi > cfg.min_valid_depth_mm)
            & (depth_roi < cfg.max_valid_depth_mm)
        ].astype(np.float32)

        if center_values.size >= 50 and edge_values.size >= 50:
            center_median = float(np.median(center_values))
            edge_median = float(np.median(edge_values))
            center_edge_delta = center_median - edge_median

    return {
        "success": True,
        "valid_points": valid_points,
        "valid_fraction_on_mask": float(valid_fraction_on_mask),
        "plane_residual_std_mm": residual_std,
        "plane_residual_range_mm": residual_range,
        "center_edge_depth_delta_mm": (
            float(center_edge_delta)
            if center_edge_delta is not None
            else None
        ),
    }


def compute_polymailer_depth_signature(cfg, depth_roi, package_mask):
    mask = package_mask.astype(np.uint8)
    mask_area = int(mask.sum())

    empty = {
        "success": False,
        "valid_points": 0,
        "valid_fraction_on_mask": 0.0,
        "full_range_mm": None,
        "edge_range_mm": None,
        "center_range_mm": None,
        "center_closer_than_edge_mm": None,
        "center_edge_signed_mm": None,
        "side_spread_mm": None,
        "score": 0.0,
        "center_closer_score": 0.0,
        "full_range_score": 0.0,
        "edge_range_score": 0.0,
        "side_spread_score": 0.0,
    }

    if mask_area <= 0:
        return empty

    full_stats = depth_stats_inside_mask(cfg, depth_roi, mask)
    valid_points = full_stats["count"]
    valid_fraction = full_stats["valid_fraction"]

    if valid_points < 50:
        return empty

    center = get_mask_center(mask)

    if center is None:
        return empty

    cx, cy = center
    yy, xx = np.indices(mask.shape)

    center_region = (
        (mask == 1)
        & (((xx - cx) ** 2 + (yy - cy) ** 2) <= cfg.center_edge_kernel_px ** 2)
    ).astype(np.uint8)

    eroded = erode_mask(mask, cfg.center_edge_kernel_px // 2)
    edge_region = mask.copy()
    edge_region[eroded == 1] = 0

    center_stats = depth_stats_inside_mask(cfg, depth_roi, center_region)
    edge_stats = depth_stats_inside_mask(cfg, depth_roi, edge_region)

    center_closer_than_edge_mm = None
    center_edge_signed_mm = None

    if center_stats["median"] is not None and edge_stats["median"] is not None:
        center_edge_signed_mm = center_stats["median"] - edge_stats["median"]
        center_closer_than_edge_mm = edge_stats["median"] - center_stats["median"]

    x, y, w, h = cv2.boundingRect(mask)

    thirds = [
        (x, y, max(1, w // 3), h),
        (x + max(1, w // 3), y, max(1, w // 3), h),
        (x + max(1, 2 * w // 3), y, max(1, w - 2 * (w // 3)), h),
        (x, y, w, max(1, h // 3)),
        (x, y + max(1, 2 * h // 3), w, max(1, h - 2 * (h // 3))),
    ]

    side_medians = []

    for tx, ty, tw, th in thirds:
        patch_mask = mask[ty:ty + th, tx:tx + tw]
        patch_depth = depth_roi[ty:ty + th, tx:tx + tw]

        patch_valid = patch_depth[
            (patch_mask == 1)
            & (patch_depth > cfg.min_valid_depth_mm)
            & (patch_depth < cfg.max_valid_depth_mm)
        ].astype(np.float32)

        if patch_valid.size >= 50:
            side_medians.append(float(np.median(patch_valid)))

    if len(side_medians) >= 2:
        side_spread_mm = float(max(side_medians) - min(side_medians))
    else:
        side_spread_mm = None

    center_closer_score = score_from_range(
        center_closer_than_edge_mm,
        cfg.poly_center_closer_hard_mm,
        cfg.poly_center_closer_soft_mm,
        reverse=True,
    )

    full_range_score = score_from_range(
        full_stats["range_p90_p10"],
        cfg.poly_full_depth_range_hard_mm,
        cfg.poly_full_depth_range_soft_mm,
        reverse=True,
    )

    edge_range_score = score_from_range(
        edge_stats["range_p90_p10"],
        cfg.poly_edge_depth_range_hard_mm,
        cfg.poly_edge_depth_range_soft_mm,
        reverse=True,
    )

    side_spread_score = score_from_range(
        side_spread_mm,
        cfg.poly_side_spread_hard_mm,
        cfg.poly_side_spread_soft_mm,
        reverse=True,
    )

    score = (
        0.35 * center_closer_score
        + 0.25 * full_range_score
        + 0.25 * edge_range_score
        + 0.15 * side_spread_score
    )

    score = float(np.clip(score, 0.0, 1.0))

    return {
        "success": True,
        "valid_points": int(valid_points),
        "valid_fraction_on_mask": float(valid_fraction),
        "full_range_mm": full_stats["range_p90_p10"],
        "edge_range_mm": edge_stats["range_p90_p10"],
        "center_range_mm": center_stats["range_p90_p10"],
        "center_closer_than_edge_mm": (
            float(center_closer_than_edge_mm)
            if center_closer_than_edge_mm is not None
            else None
        ),
        "center_edge_signed_mm": (
            float(center_edge_signed_mm)
            if center_edge_signed_mm is not None
            else None
        ),
        "side_spread_mm": (
            float(side_spread_mm)
            if side_spread_mm is not None
            else None
        ),
        "score": score,
        "center_closer_score": float(center_closer_score),
        "full_range_score": float(full_range_score),
        "edge_range_score": float(edge_range_score),
        "side_spread_score": float(side_spread_score),
    }


def count_polymailer_depth_signals(poly_signature):
    signals = 0

    center_closer = poly_signature.get("center_closer_than_edge_mm")
    full_range = poly_signature.get("full_range_mm")
    edge_range = poly_signature.get("edge_range_mm")
    side_spread = poly_signature.get("side_spread_mm")

    if center_closer is not None and center_closer >= 6.0:
        signals += 1

    if full_range is not None and full_range >= 14.0:
        signals += 1

    if edge_range is not None and edge_range >= 14.0:
        signals += 1

    if side_spread is not None and side_spread >= 10.0:
        signals += 1

    return int(signals)


def classify_package_type(cfg, depth_roi, package_mask, info, package_depth_mm=None):
    depth_features = fit_plane_depth_features(cfg, depth_roi, package_mask)
    poly_signature = compute_polymailer_depth_signature(cfg, depth_roi, package_mask)

    rectangularity = get_mask_rectangularity(package_mask)

    residual_std = depth_features["plane_residual_std_mm"]
    residual_range = depth_features["plane_residual_range_mm"]
    center_edge_delta = depth_features["center_edge_depth_delta_mm"]
    valid_points = depth_features["valid_points"]
    valid_fraction_on_mask = depth_features["valid_fraction_on_mask"]

    mask_area_ratio = info.get("area_ratio", 0.0)

    flatness_score = score_from_range(
        residual_std,
        cfg.box_flat_std_good_mm,
        cfg.box_flat_std_bad_mm,
        reverse=False,
    )

    residual_range_score = score_from_range(
        residual_range,
        cfg.box_residual_range_good_mm,
        cfg.box_residual_range_bad_mm,
        reverse=False,
    )

    rectangularity_score = score_from_range(
        rectangularity,
        cfg.box_rectangularity_good,
        cfg.box_rectangularity_bad,
        reverse=True,
    )

    if center_edge_delta is None:
        center_edge_abs = None
        center_edge_box_score = 0.6
    else:
        center_edge_abs = abs(float(center_edge_delta))

        center_edge_box_score = score_from_range(
            center_edge_abs,
            cfg.poly_center_edge_soft_mm,
            cfg.poly_center_edge_hard_mm,
            reverse=False,
        )

    center_edge_override_allowed = (
        center_edge_abs is not None
        and valid_fraction_on_mask >= cfg.center_edge_override_min_valid_fraction
        and valid_points >= cfg.center_edge_override_min_valid_points
    )

    sparse_rectangular_box = (
        valid_fraction_on_mask <= cfg.sparse_depth_box_max_valid_fraction
        and rectangularity >= cfg.sparse_depth_box_min_rectangularity
        and mask_area_ratio >= cfg.sparse_depth_box_min_area_ratio
    )

    poly_signal_count = count_polymailer_depth_signals(poly_signature)

    polymailer_signature_override_allowed = (
        poly_signature["success"]
        and poly_signature["valid_fraction_on_mask"] >= cfg.poly_signature_min_valid_fraction
        and poly_signature["valid_points"] >= cfg.poly_signature_min_valid_points
        and mask_area_ratio >= cfg.poly_signature_min_area_ratio
        and (
            poly_signature["score"] >= cfg.poly_signature_override_threshold
            or poly_signal_count >= cfg.poly_multi_signal_min_count
        )
    )

    raw_box_score = (
        0.35 * flatness_score
        + 0.25 * residual_range_score
        + 0.15 * rectangularity_score
        + 0.25 * center_edge_box_score
    )

    raw_box_score = float(np.clip(raw_box_score, 0.0, 1.0))

    box_score = raw_box_score * (
        1.0 - cfg.box_score_poly_signature_penalty * poly_signature["score"]
    )

    box_score = float(np.clip(box_score, 0.0, 1.0))

    if polymailer_signature_override_allowed:
        package_type = "polymailer"

        if poly_signal_count >= cfg.poly_multi_signal_min_count:
            confidence = cfg.poly_multi_signal_confidence
        else:
            confidence = 0.72 + 0.25 * poly_signature["score"]

        confidence = float(np.clip(confidence, 0.72, 0.95))

    elif sparse_rectangular_box:
        package_type = "box"
        confidence = 0.78

    elif (
        center_edge_override_allowed
        and center_edge_abs is not None
        and center_edge_abs >= cfg.poly_center_edge_hard_mm
    ):
        package_type = "polymailer"
        confidence = 0.75 + min((center_edge_abs - cfg.poly_center_edge_hard_mm) / 40.0, 0.20)
        confidence = float(np.clip(confidence, 0.75, 0.95))

    else:
        polymailer_score = 1.0 - box_score

        if box_score >= cfg.box_score_threshold:
            package_type = "box"
            confidence = box_score
        else:
            package_type = "polymailer"
            confidence = polymailer_score

        confidence = float(np.clip(confidence, 0.0, 1.0))

    segmentation_box_override = False
    segmentation_box_override_reason = None
    segmentation_box_override_vetoed = False
    segmentation_box_override_veto_reason = None

    # Preserve the depth classifier result before any RGB segmentation
    # override is considered. This is needed because brown polymailers can
    # pass the same HSV and rectangularity rules as cardboard boxes.
    depth_classifier_type = package_type
    depth_classifier_confidence = float(confidence)

    selected_source = info.get("source")
    selected_score = float(info.get("score", 0.0) or 0.0)
    selected_color_score = float(info.get("color_score", 0.0) or 0.0)
    selected_rectangularity = float(
        info.get("rectangularity", rectangularity) or 0.0
    )
    selected_area_ratio = float(info.get("area_ratio", mask_area_ratio) or 0.0)

    merged_box_source = selected_source in {
        "cardboard_box_merged_masks",
        "cardboard_box_multiface_merge",
    }

    strong_exact_box_source = (
        selected_source == "package_roi_exact_box_rules"
        and selected_score >= cfg.box_type_segmentation_min_score
        and selected_color_score >= cfg.box_type_segmentation_min_color_score
        and selected_rectangularity >= cfg.box_type_segmentation_min_rectangularity
        and selected_area_ratio >= cfg.box_type_segmentation_min_area_ratio
    )

    poly_score = float(poly_signature.get("score", 0.0) or 0.0)

    package_depth_value = (
        float(package_depth_mm)
        if package_depth_mm is not None
        else None
    )

    # Primary rule:
    # If the depth classifier says polymailer and the package is physically
    # thin, do not let the brown-cardboard RGB override change it to box.
    depth_classified_thin_polymailer = (
        package_depth_value is not None
        and depth_classifier_type == "polymailer"
        and package_depth_value <= cfg.box_type_polymailer_veto_max_package_depth_mm
    )

    # Secondary hard rule:
    # Very thin packages with a meaningful polymailer signature are also
    # protected even if the base depth classifier happens to flicker to box.
    hard_thin_polymailer = (
        package_depth_value is not None
        and package_depth_value <= cfg.box_type_polymailer_hard_thin_max_depth_mm
        and (
            poly_score >= cfg.box_type_polymailer_hard_thin_min_poly_score
            or poly_signal_count >= cfg.box_type_polymailer_veto_min_signal_count
        )
    )

    # Medium-thickness fallback. This still requires stronger polymailer
    # evidence, but uses OR instead of requiring both signals simultaneously.
    thin_with_supporting_poly_evidence = (
        package_depth_value is not None
        and package_depth_value <= cfg.box_type_polymailer_veto_max_package_depth_mm
        and (
            poly_score >= cfg.box_type_polymailer_veto_min_poly_score
            or poly_signal_count >= cfg.box_type_polymailer_veto_min_signal_count
        )
    )

    strong_thin_polymailer_veto = (
        cfg.box_type_polymailer_veto_enable
        and (
            depth_classified_thin_polymailer
            or hard_thin_polymailer
            or thin_with_supporting_poly_evidence
        )
    )

    if strong_thin_polymailer_veto:
        segmentation_box_override_vetoed = True
        segmentation_box_override_veto_reason = "strong_thin_polymailer_depth"

        print()
        print("SEGMENTATION BOX TYPE OVERRIDE VETOED")
        print(f"  source:              {selected_source}")
        print(f"  depth type:          {depth_classifier_type}")
        print(f"  depth confidence:    {depth_classifier_confidence:.3f}")
        print(f"  poly score:          {poly_score:.3f}")
        print(f"  poly signal count:   {poly_signal_count}")
        print(f"  package depth mm:    {package_depth_value:.3f}")
        print(f"  depth-thin rule:     {depth_classified_thin_polymailer}")
        print(f"  hard-thin rule:      {hard_thin_polymailer}")
        print(f"  supported-thin rule: {thin_with_supporting_poly_evidence}")
        print("  final type remains:  polymailer")

    # Depth can look polymailer-like on a cardboard box because the center seam,
    # tape, labels, and stereo holes create a strong center/edge depth signature.
    # A complete mask selected by the exact box rules is stronger type evidence,
    # except when the strong thin-polymailer veto above is active.
    if (
        cfg.box_type_segmentation_override_enable
        and not strong_thin_polymailer_veto
        and selected_source in BOX_TYPE_SEGMENTATION_OVERRIDE_SOURCES
        and (merged_box_source or strong_exact_box_source)
    ):
        package_type = "box"

        segmentation_confidence = (
            0.55
            + 0.25 * selected_color_score
            + 0.20 * selected_rectangularity
        )
        confidence = float(
            np.clip(
                max(
                    cfg.box_type_segmentation_min_confidence,
                    segmentation_confidence,
                ),
                0.0,
                0.97,
            )
        )

        segmentation_box_override = True

        if selected_source == "cardboard_box_multiface_merge":
            segmentation_box_override_reason = "angled_box_multiface_merge"
        elif selected_source == "cardboard_box_merged_masks":
            segmentation_box_override_reason = "merged_cardboard_fragments"
        else:
            segmentation_box_override_reason = "strong_exact_box_rules"

        print()
        print("SEGMENTATION BOX TYPE OVERRIDE APPLIED")
        print(f"  source:          {selected_source}")
        print(f"  reason:          {segmentation_box_override_reason}")
        print(f"  mask score:      {selected_score:.3f}")
        print(f"  color score:     {selected_color_score:.3f}")
        print(f"  rectangularity:  {selected_rectangularity:.3f}")
        print(f"  area ratio:      {selected_area_ratio:.3f}")
        print(f"  final confidence:{confidence:.3f}")

    return {
        "package_type": package_type,
        "package_type_confidence": float(confidence),
        "segmentation_box_override": segmentation_box_override,
        "segmentation_box_override_reason": segmentation_box_override_reason,
        "segmentation_box_override_vetoed": segmentation_box_override_vetoed,
        "segmentation_box_override_veto_reason": segmentation_box_override_veto_reason,
        "depth_classifier_type_before_override": depth_classifier_type,
        "depth_classifier_confidence_before_override": depth_classifier_confidence,
        "package_depth_mm_for_type": (
            float(package_depth_mm)
            if package_depth_mm is not None
            else None
        ),
        "box_score": float(box_score),
        "raw_box_score": float(raw_box_score),
        "flatness_score": float(flatness_score),
        "residual_range_score": float(residual_range_score),
        "rectangularity_score": float(rectangularity_score),
        "center_edge_box_score": float(center_edge_box_score),
        "poly_signal_count": int(poly_signal_count),
        "depth_features": depth_features,
        "polymailer_depth_signature": poly_signature,
    }


def estimate_package_dimensions_mm(cfg, info, depth_mm, intrinsics):
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

    rotated = get_rotated_box_from_mask(info["mask"])

    if rotated is None:
        return empty

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    width_mm = (w_px * depth_mm) / intrinsics["fx"]
    height_mm = (h_px * depth_mm) / intrinsics["fy"]

    width_mm *= cfg.package_size_scale
    height_mm *= cfg.package_size_scale

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


def masked_gaussian_blur(values, mask, sigma_px, return_support=False):
    """Gaussian blur that ignores invalid pixels instead of bleeding zeros in.

    The returned support is the fraction of the kernel that landed on valid
    pixels, so it doubles as a confidence map: a dropout smaller than the kernel
    still gets a well-supported interpolated value, a large one does not.
    """
    weight = mask.astype(np.float32)
    data = np.where(mask, values, 0.0).astype(np.float32)

    kernel_px = int(sigma_px * 6.0) | 1

    num = cv2.GaussianBlur(data, (kernel_px, kernel_px), sigma_px)
    den = cv2.GaussianBlur(weight, (kernel_px, kernel_px), sigma_px)

    blurred = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3)

    if return_support:
        return blurred, den

    return blurred


def fill_mask_holes(mask):
    """Fill every enclosed hole in a binary mask.

    Flood-fills the background inward from a border of zeros; whatever the flood
    never reaches is enclosed, and is added back to the mask.
    """
    mask = mask.astype(np.uint8)

    # Pad so the flood always has a border to start from, even for a blob that
    # touches the image edge.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)

    enclosed = (flood == 0)[1:-1, 1:-1]

    return (mask | enclosed).astype(np.uint8)


def fit_reference_plane(
    depth_roi,
    mask,
    max_points=20000,
    iterations=5,
    trim_sigma=1.5,
):
    """Robust plane through the mailer surface, trimming the raised side.

    A plain least-squares fit is dragged toward the product it is meant to
    measure against, which shrinks the very signal we want. Re-fitting while
    discarding points that sit well above the current plane keeps the reference
    on the flat mailer.

    Takes explicit tuning arguments rather than a cfg so BOTH the package and
    polymailer modes can share it -- their config classes name these fields
    differently, and duplicating the fit is how the two modes drift apart.
    """
    yy, xx = np.where(mask)
    zz = depth_roi[mask].astype(np.float64)

    if zz.size > max_points:
        rng = np.random.default_rng(12345)
        idx = rng.choice(zz.size, size=max_points, replace=False)
        xs, ys, zs = xx[idx], yy[idx], zz[idx]
    else:
        xs, ys, zs = xx, yy, zz

    keep = np.ones(zs.shape, dtype=bool)
    coeffs = None

    for _ in range(max(1, iterations)):
        if int(keep.sum()) < 32:
            break

        a = np.column_stack(
            [
                xs[keep].astype(np.float64),
                ys[keep].astype(np.float64),
                np.ones(int(keep.sum())),
            ]
        )

        coeffs, _, _, _ = np.linalg.lstsq(a, zs[keep], rcond=None)

        residual = zs - (coeffs[0] * xs + coeffs[1] * ys + coeffs[2])

        # The product is CLOSER to the camera, i.e. a negative depth residual.
        # Spread is measured on the lower (product-free) half so the product
        # cannot inflate the very scale used to reject it.
        low, high = np.percentile(residual, [2.0, 60.0])
        sigma = (high - low) / 1.2 if high > low else 1.0

        keep = residual > -trim_sigma * sigma

    if coeffs is None:
        return None

    grid_y, grid_x = np.indices(depth_roi.shape)

    return (coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2]).astype(np.float32)


def geodesic_grow(seed_u8, allowed_u8, max_dist_px, step=4):
    """Dilate seed within allowed, up to ~max_dist_px (approximate geodesic).

    Runs inside the seed's bounding window (padded by max_dist) -- growth
    cannot escape it, and the crop keeps the dilation loop cheap.
    """
    ys, xs = np.where(seed_u8 == 1)

    if ys.size == 0:
        return np.zeros_like(seed_u8)

    pad = int(np.ceil(max_dist_px)) + step + 1
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(seed_u8.shape[0], int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(seed_u8.shape[1], int(xs.max()) + pad + 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (step * 2 + 1, step * 2 + 1))
    allowed_win = allowed_u8[y1:y2, x1:x2]
    cur = (seed_u8[y1:y2, x1:x2] & allowed_win).astype(np.uint8)

    for _ in range(max(1, int(round(max_dist_px / step)))):
        nxt = cv2.dilate(cur, kernel) & allowed_win

        if int(nxt.sum()) == int(cur.sum()):
            break

        cur = nxt

    out = np.zeros_like(seed_u8)
    out[y1:y2, x1:x2] = cur

    return out


def detect_product_inside_polymailer(cfg, depth_roi, package_mask, center_full_depth_mm, intrinsics):
    """Locate the product inside a polymailer from its depth signature.

    The mailer is not rigid: it drapes and tents over whatever is inside, so
    depth shows a smooth dome whose sloped skirt extends well past the product,
    and trapped air can hold the film HIGHER than the product itself. No height
    threshold can separate product from drape (the drape reaches every height
    the product does), so the estimate is built around the one thing only a
    rigid product produces: a large PLANAR patch where the film rests on its
    top.

    Pipeline: fit the mailer reference surface and gate on the dome as before;
    seed from the flattest part of the elevated region; fit a tilted plane to
    that seed (the product's top); grow the seed across everything close to
    that plane (tight above -- drape hovers above the plane, loose below --
    film can sag off edges and corners), with the growth geodesically capped
    near the seed so a band osculating the curved drape cannot run away; trim
    drape tongues on width; merge film pockets bulging above the plane that
    are enclosed by the contact region (air trapped over the product's own
    middle); then report the rotated-rectangle fit of the result. Gates on
    edge drop (a rigid product's surroundings must fall off the plane),
    solidity, and area keep an unreadable or absent product reported as
    found=False rather than as a guessed centre.
    """
    empty = {
        "found": False,
        "center_roi": None,
        "center_full": None,
        "center_x_mm": None,
        "center_y_mm": None,
        "depth_mm": None,
        "area_px": 0,
        "area_ratio_of_package": 0.0,
        "mask": np.zeros_like(package_mask, dtype=np.uint8),
        "core_mask": np.zeros_like(package_mask, dtype=np.uint8),
        "core_area_px": 0,
        "core_found": False,
        "reason": "not_run",
        "peak_mm": None,
        "noise_mm": None,
        "peak_noise_multiple": None,
        "threshold_mm": None,
        "rect_w_mm": None,
        "rect_h_mm": None,
        "rect_angle_deg": None,
        "edge_drop_frac": None,
        "rect_fill": None,
        "solidity": None,
        "center_source": None,
        "candidate_score": None,
    }

    if not cfg.product_inside_enable:
        empty["reason"] = "disabled"
        return empty

    package_mask = package_mask.astype(np.uint8)
    package_area = int(package_mask.sum())

    if package_area <= 0:
        empty["reason"] = "empty_package_mask"
        return empty

    # Every pixel length below is derived from the package's own size, so the
    # same settings hold for a small mailer and a large one.
    package_span_px = float(np.sqrt(package_area))
    erode_px = max(1, int(round(package_span_px * cfg.product_inside_edge_erode_frac)))

    inner_mask = erode_mask(package_mask, erode_px)

    if int(inner_mask.sum()) < cfg.product_inside_min_valid_pixels:
        inner_mask = package_mask.copy()

    valid_inner = (
        (inner_mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    if int(valid_inner.sum()) < cfg.product_inside_min_valid_pixels:
        empty["reason"] = "not_enough_depth"
        return empty

    plane = fit_reference_plane(
        depth_roi,
        valid_inner,
        max_points=cfg.product_inside_max_plane_points,
        iterations=cfg.product_inside_plane_iterations,
        trim_sigma=cfg.product_inside_plane_trim_sigma,
    )

    if plane is None:
        empty["reason"] = "no_reference_plane"
        return empty

    # Positive = raised toward the camera relative to the fitted mailer surface.
    raw_elevation = np.where(
        valid_inner,
        plane - depth_roi.astype(np.float32),
        0.0,
    ).astype(np.float32)

    smooth_px = max(2.0, package_span_px * cfg.product_inside_smooth_frac)
    elevation, support = masked_gaussian_blur(
        raw_elevation,
        valid_inner,
        smooth_px,
        return_support=True,
    )

    # What the blur removed is per-pixel sensor noise; its spread is this
    # frame's own depth noise, which is what the detection gate is scaled to.
    high_pass = (raw_elevation - elevation)[valid_inner]
    noise_mm = float(1.4826 * np.median(np.abs(high_pass - np.median(high_pass))))
    noise_mm = max(noise_mm, 1e-3)

    inner_elevation = elevation[valid_inner]
    baseline_mm = float(
        np.percentile(inner_elevation, cfg.product_inside_baseline_percentile)
    )
    peak_mm = float(np.percentile(inner_elevation, cfg.product_inside_peak_percentile))
    dome_mm = peak_mm - baseline_mm

    empty["peak_mm"] = dome_mm
    empty["noise_mm"] = noise_mm
    empty["peak_noise_multiple"] = dome_mm / noise_mm

    if dome_mm > cfg.product_inside_max_peak_mm:
        empty["reason"] = "peak_out_of_range"
        return empty

    if dome_mm < cfg.product_inside_min_peak_noise_multiple * noise_mm:
        empty["reason"] = "no_dome_above_noise"
        return empty

    # The measurement stereo drops out in streaks over low-texture kraft; a
    # pixel is usable where enough of the smoothing kernel around it landed on
    # valid depth.
    well_supported = support >= cfg.product_inside_min_blur_support
    grow_zone = (inner_mask == 1) & well_supported

    grad_x = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=5) / 8.0
    grad_y = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=5) / 8.0
    slope = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    # ----- seed: the flattest part of the raised region ----------------
    # The film on the product's top is the flattest thing standing above the
    # mailer surface. Height alone CANNOT gate the footprint (drape billows
    # reach the same heights), so height only picks where seeds may start.
    seed_threshold_mm = baseline_mm + cfg.product_inside_seed_min_height_frac * dome_mm
    empty["threshold_mm"] = seed_threshold_mm

    elevated = grow_zone & valid_inner & (elevation >= seed_threshold_mm)

    if int(elevated.sum()) < cfg.product_inside_min_valid_pixels:
        empty["reason"] = "no_seed"
        return empty

    slope_thr = float(
        np.percentile(slope[elevated], cfg.product_inside_seed_slope_percentile)
    )
    seed = (elevated & (slope <= slope_thr)).astype(np.uint8)
    seed_open_px = max(2, int(round(smooth_px * 0.25)))
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (seed_open_px * 2 + 1, seed_open_px * 2 + 1)
        ),
    )

    n_seeds, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    min_seed_area = cfg.product_inside_seed_min_area_frac * package_area
    cand_ids = [i for i in range(1, n_seeds) if seed_stats[i, 4] >= min_seed_area]
    cand_ids = sorted(cand_ids, key=lambda i: -seed_stats[i, 4])
    cand_ids = cand_ids[: cfg.product_inside_seed_max_candidates]

    if not cand_ids:
        empty["reason"] = "no_seed"
        return empty

    grid_y, grid_x = np.indices(elevation.shape)

    def fit_patch_plane(mask_b):
        """Tilted plane through the smoothed elevation of a pixel set."""
        ys, xs = np.where(mask_b)

        if len(xs) > 12000:
            idx = np.random.default_rng(0).choice(len(xs), 12000, replace=False)
            xs, ys = xs[idx], ys[idx]

        e = elevation[ys, xs].astype(np.float64)
        a = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
        coeffs, _, _, _ = np.linalg.lstsq(a, e, rcond=None)

        residual = e - a @ coeffs
        keep = np.abs(residual - np.median(residual)) < 2.0 * max(float(np.std(residual)), 0.05)

        if int(keep.sum()) > 32:
            coeffs, _, _, _ = np.linalg.lstsq(a[keep], e[keep], rcond=None)

        return (coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2]).astype(np.float32)

    close_px = max(1, int(round(smooth_px * cfg.product_inside_grow_close_frac)))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    band_px = max(4, int(round(package_span_px * cfg.product_inside_edge_band_frac)))
    band_in_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)
    )
    band_out_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 6 + 1, band_px * 6 + 1)
    )
    drop_thr_mm = max(
        cfg.product_inside_edge_drop_mm_min,
        cfg.product_inside_edge_drop_dome_frac * dome_mm,
    )

    def build_candidate(patch_u8, dist_spans, source):
        """Grow one seed into a footprint candidate and measure its quality."""
        max_dist = dist_spans * float(np.sqrt(int(patch_u8.sum())))

        plane_fit = fit_patch_plane(patch_u8 == 1)
        region = None

        # Grow across everything near the seed's plane, refit on the grown
        # region once, and grow again. Tolerance is ASYMMETRIC: the drape
        # leaves the product plane upward (tent, billow), so above-plane is
        # tight; film sags below the plane off edges and corners, so
        # below-plane is looser. The geodesic cap keeps a band that happens to
        # osculate the curved drape from running across the mailer.
        prev_area = int(patch_u8.sum())

        for _ in range(2):
            on_plane = (
                grow_zone
                & valid_inner
                & (elevation - plane_fit <= cfg.product_inside_plane_tol_above_mm)
                & (plane_fit - elevation <= cfg.product_inside_plane_tol_below_mm)
            ).astype(np.uint8)
            on_plane = cv2.morphologyEx(on_plane, cv2.MORPH_CLOSE, close_kernel)

            region_u8 = geodesic_grow(patch_u8, on_plane, max_dist)

            if int(region_u8.sum()) == 0:
                return None

            region = region_u8
            plane_fit = fit_patch_plane(region == 1)

            # Converged: the refit will not move a region that barely changed.
            if abs(int(region.sum()) - prev_area) < 0.05 * prev_area:
                break

            prev_area = int(region.sum())

        # Width-based spur trim first: drape tongues must go before the
        # trapped-air test, or a pocket beside a tongue reads as enclosed. ALL
        # sizable pieces are kept (not just the largest): a thin contact ring
        # may be cut into limbs here and is rejoined by the pocket merge below.
        if cfg.product_inside_trim_frac > 0.0:
            trim_px = max(
                1, int(round(np.sqrt(int(region.sum())) * cfg.product_inside_trim_frac))
            )
            opened = cv2.morphologyEx(
                region,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (trim_px * 2 + 1, trim_px * 2 + 1)
                ),
            )

            if int(opened.sum()) > 0:
                n_pieces, piece_labels, piece_stats, _ = cv2.connectedComponentsWithStats(
                    opened, 8
                )
                keep_min = max(
                    cfg.product_inside_min_valid_pixels, int(0.05 * int(opened.sum()))
                )
                pieces = [
                    j for j in range(1, n_pieces) if piece_stats[j, 4] >= keep_min
                ]

                if pieces:
                    region = np.isin(piece_labels, pieces).astype(np.uint8)

        # Trapped air: the film can bulge ABOVE the product plane over the
        # product's own middle, leaving only a contact ring on the plane.
        # Merge above-plane blobs whose wide surrounding ring is mostly this
        # region -- a pocket enclosed by product contact is product; a billow
        # off to one side is not.
        above = (
            grow_zone
            & valid_inner
            & (elevation - plane_fit > cfg.product_inside_plane_tol_above_mm)
        ).astype(np.uint8)
        n_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(above, 8)
        air_ring_px = max(9, int(round(band_px * 1.5)))
        air_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (air_ring_px * 2 + 1, air_ring_px * 2 + 1)
        )

        for j in range(1, n_blobs):
            blob_area = int(blob_stats[j, 4])

            if blob_area < 100 or blob_area > 2 * int(region.sum()):
                continue

            # Work in the blob's padded bounding window; a full-frame dilate
            # per blob dominated the whole detector's runtime.
            bx, by, bw, bh = blob_stats[j, 0], blob_stats[j, 1], blob_stats[j, 2], blob_stats[j, 3]
            y1 = max(0, by - air_ring_px - 1)
            y2 = min(region.shape[0], by + bh + air_ring_px + 1)
            x1 = max(0, bx - air_ring_px - 1)
            x2 = min(region.shape[1], bx + bw + air_ring_px + 1)

            blob_win = (blob_labels[y1:y2, x1:x2] == j).astype(np.uint8)
            ring_win = (cv2.dilate(blob_win, air_kernel) == 1) & (blob_win == 0)
            n_ring = int(ring_win.sum())

            if n_ring == 0:
                continue

            if float((ring_win & (region[y1:y2, x1:x2] == 1)).sum()) / n_ring >= 0.8:
                region[y1:y2, x1:x2] |= blob_win

        region = fill_mask_holes(region)
        biggest = largest_component(region)

        if biggest is not None and int(biggest.sum()) > 0:
            region = fill_mask_holes(biggest)

        product_area = int(region.sum())

        if product_area < cfg.product_inside_min_valid_pixels:
            return None

        ys, xs = np.where(region == 1)
        points = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect
        rect_fill = product_area / max(rect_w * rect_h, 1.0)
        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
        hull = cv2.convexHull(points)
        solidity = product_area / max(float(cv2.contourArea(hull)), 1.0)

        # Ring OUTSIDE the region, offset one band width: the smoothing spreads
        # the fall-off, so the drop is measured past the immediate boundary.
        # Computed in the region's padded window -- the outer kernel is large
        # and a full-frame dilate with it is most of the candidate's cost.
        pad = band_px * 3 + 2
        ry1 = max(0, int(ys.min()) - pad)
        ry2 = min(region.shape[0], int(ys.max()) + pad + 1)
        rx1 = max(0, int(xs.min()) - pad)
        rx2 = min(region.shape[1], int(xs.max()) + pad + 1)
        region_win = region[ry1:ry2, rx1:rx2]
        ring = (
            (cv2.dilate(region_win, band_out_kernel) == 1)
            & (cv2.dilate(region_win, band_in_kernel) == 0)
            & valid_inner[ry1:ry2, rx1:rx2]
        )

        if int(ring.sum()) < 30:
            return None

        edge_drop_frac = float(
            ((plane_fit[ry1:ry2, rx1:rx2] - elevation[ry1:ry2, rx1:rx2])[ring] > drop_thr_mm).mean()
        )

        # Candidate quality: rigid edges all round, compact, not a drape band.
        # The aspect term is soft -- genuinely elongated products exist -- and
        # referenced to score_aspect_ref so a 2:1 product is not penalised.
        score = (
            edge_drop_frac
            * solidity
            * min(1.0, cfg.product_inside_score_aspect_ref / max(aspect, 1e-3)) ** 0.5
        )

        return {
            "patch": patch_u8,
            "region": region,
            "source": source,
            "area": product_area,
            "rect": rect,
            "fill": rect_fill,
            "solidity": solidity,
            "drop_frac": edge_drop_frac,
            "area_ratio": product_area / package_area,
            "score": score,
        }

    candidates = []

    # FLAT candidate: seeds tried largest-first, first that grows a usable
    # region wins -- the dominant flat patch is the product's contact area.
    for i in cand_ids:
        cand = build_candidate(
            (seed_labels == i).astype(np.uint8),
            cfg.product_inside_grow_dist_seed_spans,
            "flat",
        )

        if cand is not None:
            candidates.append(cand)
            break

    # PEAK candidate: the top of the dome. When the drape forms a long level
    # crest, the flat path rides it -- but the product still owns the dome's
    # peak (nothing rests ON a drape). Grown with a tighter cap because its
    # plane necessarily skims the crest crown. Skipped when a COMPACT flat
    # region already covers the peak (then both candidates describe the same
    # bump and the second growth pass would be pure cost) -- an elongated band
    # can cover the peak while centring far from it, so it never skips.
    peak_band = (
        grow_zone
        & valid_inner
        & (elevation >= peak_mm - cfg.product_inside_peak_band_mm)
    ).astype(np.uint8)
    peak_band = cv2.morphologyEx(
        peak_band,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    n_pk, pk_labels, pk_stats, _ = cv2.connectedComponentsWithStats(peak_band, 8)

    if n_pk > 1:
        j = max(range(1, n_pk), key=lambda j: pk_stats[j, 4])
        pk_patch = (pk_labels == j).astype(np.uint8)

        skip_peak = False

        if candidates:
            covered = float((candidates[0]["region"] & pk_patch).sum()) / max(
                int(pk_patch.sum()), 1
            )
            (_, (fw, fh), _) = candidates[0]["rect"]
            flat_aspect = max(fw, fh) / max(min(fw, fh), 1.0)
            skip_peak = (
                covered >= 0.6
                and flat_aspect <= cfg.product_inside_score_aspect_ref
            )

        if pk_stats[j, 4] >= cfg.product_inside_min_valid_pixels and not skip_peak:
            cand = build_candidate(
                pk_patch,
                cfg.product_inside_peak_grow_dist_spans,
                "peak",
            )

            if cand is not None:
                candidates.append(cand)

    best = max(candidates, key=lambda c: c["score"]) if candidates else None

    if best is None:
        empty["reason"] = "no_component"
        return empty

    (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = best["rect"]

    mm_per_px = None

    if center_full_depth_mm is not None and intrinsics is not None:
        mm_per_px = float(center_full_depth_mm) / float(intrinsics["fx"])

    empty["area_px"] = best["area"]
    empty["area_ratio_of_package"] = float(best["area_ratio"])
    empty["edge_drop_frac"] = float(best["drop_frac"])
    empty["rect_fill"] = float(best["fill"])
    empty["solidity"] = float(best["solidity"])
    empty["rect_angle_deg"] = float(rect_angle)
    empty["center_source"] = best["source"]
    empty["candidate_score"] = float(best["score"])

    if mm_per_px is not None:
        empty["rect_w_mm"] = float(rect_w * mm_per_px)
        empty["rect_h_mm"] = float(rect_h * mm_per_px)

    empty["mask"] = best["region"]

    # ----- sanity gates: an implausible footprint is reported as absent ----
    if best["drop_frac"] < cfg.product_inside_min_edge_drop_frac:
        empty["reason"] = "no_rigid_edges"
        return empty

    if (
        best["solidity"] < cfg.product_inside_min_solidity
        or best["fill"] < cfg.product_inside_min_rect_fill
    ):
        empty["reason"] = "footprint_not_compact"
        return empty

    if best["area_ratio"] < cfg.product_inside_min_area_ratio_of_package:
        empty["reason"] = "too_small"
        return empty

    if best["area_ratio"] > cfg.product_inside_max_area_ratio_of_package:
        empty["reason"] = "too_large"
        return empty

    center_roi = (int(round(rect_cx)), int(round(rect_cy)))
    center_full = (
        int(round(cfg.roi_x1 + rect_cx)),
        int(round(cfg.roi_y1 + rect_cy)),
    )

    # Depth comes from the contact patch: that is film resting ON the product,
    # so its median is the product's top and not the drape around it.
    core = best["patch"]
    supported = (
        (core == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )
    product_values = depth_roi[supported].astype(np.float32)

    if product_values.size >= cfg.min_surface_depth_count:
        product_depth_mm = float(np.median(product_values)) + cfg.measurement_depth_offset_mm
    else:
        product_depth_mm = center_full_depth_mm

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        product_depth_mm,
        intrinsics,
    )

    empty.update(
        found=True,
        reason="ok",
        center_roi=center_roi,
        center_full=center_full,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
        depth_mm=product_depth_mm,
        core_mask=core,
        core_area_px=int(core.sum()),
        core_found=True,
    )

    return empty


def run_sam_package_depth_type(
    cfg,
    frame_bgr,
    depth_class_aligned,
    depth_measure_aligned,
    mask_generator,
    intrinsics,
    barcode,
):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    depth_class_roi = depth_class_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_measure_roi = depth_measure_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    print("Running SAM 2 package segmentation on main ROI...")

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    print(f"Generated {len(masks)} masks inside main package ROI")

    dedicated_box_candidate = None
    dedicated_box_mask_count = 0

    if cfg.dedicated_box_sam_enable:
        print(
            "Applying exact box-script mask-selection rules to the SAME "
            "package ROI masks..."
        )

        # Reuse the masks generated from the package ROI. This does not run a
        # second ROI and does not change the package detector search area.
        dedicated_box_mask_count = len(masks)

        dedicated_box_local = choose_cardboard_box_mask_exact(
            cfg,
            masks,
            roi_rgb,
        )

        dedicated_box_candidate = map_dedicated_box_candidate_to_package_roi(
            cfg,
            dedicated_box_local,
        )

        if dedicated_box_candidate is None:
            print("No valid exact-box-rules candidate found in package ROI.")
        else:
            print(
                "Exact-box-rules candidate found in package ROI: "
                f"score={dedicated_box_candidate['score']:.3f}, "
                f"area={dedicated_box_candidate['area_ratio']:.3f}, "
                f"rect={dedicated_box_candidate['rectangularity']:.3f}, "
                f"color={dedicated_box_candidate['color_score']:.3f}"
            )

    best, accepted = choose_best_package_mask(
        cfg,
        masks,
        roi_rgb,
        dedicated_box_candidate,
    )

    if best is None:
        print("No valid package/product or cardboard box mask found.")
        return None

    # Measure the top face first so the box-type override can use physical
    # package thickness as a veto for strong, thin polymailers.
    top_face_depth_result = package_face_depth_mm(
        cfg,
        depth_measure_roi,
        best["mask"],
    )

    top_face_depth_mm = top_face_depth_result["distance_mm"]

    package_depth_mm = (
        max(0.0, cfg.base_depth_mm - top_face_depth_mm)
        if top_face_depth_mm is not None
        else None
    )

    classification = classify_package_type(
        cfg,
        depth_roi=depth_class_roi,
        package_mask=best["mask"],
        info=best,
        package_depth_mm=package_depth_mm,
    )

    cx_full = cfg.roi_x1 + best["center_roi"][0]
    cy_full = cfg.roi_y1 + best["center_roi"][1]
    center_full = (int(cx_full), int(cy_full))

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        top_face_depth_mm,
        intrinsics,
    )

    dimensions = estimate_package_dimensions_mm(
        cfg,
        info=best,
        depth_mm=top_face_depth_mm,
        intrinsics=intrinsics,
    )

    if classification["package_type"] == "polymailer":
        # Classification depth, not measurement depth. The measurement stream is
        # the right one for DIMENSIONS -- sparse but accurate in absolute depth.
        # Product-inside needs the opposite: it reads a few-mm RELATIVE bulge, so
        # density and resolution matter more than absolute accuracy. The
        # classification stream runs at full rgb_size against the measurement
        # stream's 640x400, which halves depth quantisation, and it carries
        # 97-98% valid pixels on the mailer against 91-94% -- the dropouts being
        # what tore holes in the footprint in the first place.
        if cfg.product_inside_enable:
            product_inside = detect_product_inside_polymailer(
                cfg,
                depth_roi=depth_class_roi,
                package_mask=best["mask"],
                center_full_depth_mm=top_face_depth_mm,
                intrinsics=intrinsics,
            )
        elif cfg.locator_enable:
            # NEW height-band locator (2026-09-03): absolute height above the
            # table, two nested single-blob segments, grasp at the grab-blob
            # centroid. Needs the FULL-frame streams (it crops internally).
            product_inside = locate_product_inside(
                cfg,
                depth_class_aligned,
                depth_measure_aligned,
                best["mask"],
                intrinsics,
            )
        else:
            product_inside = detect_product_inside_polymailer(
                cfg,
                depth_roi=depth_class_roi,
                package_mask=best["mask"],
                center_full_depth_mm=top_face_depth_mm,
                intrinsics=intrinsics,
            )
    else:
        product_inside = {
            "found": False,
            "center_roi": None,
            "center_full": None,
            "center_x_mm": None,
            "center_y_mm": None,
            "depth_mm": None,
            "area_px": 0,
            "area_ratio_of_package": 0.0,
            "mask": np.zeros_like(best["mask"], dtype=np.uint8),
            "core_mask": np.zeros_like(best["mask"], dtype=np.uint8),
            "core_area_px": 0,
            "core_found": False,
            "reason": "not_polymailer",
            "peak_mm": None,
            "noise_mm": None,
            "peak_noise_multiple": None,
            "threshold_mm": None,
        }

    # Fallback grasp points for the product inside: ranked alternatives for
    # when the cup fails to seal on the primary point. The primary is and
    # stays product_inside_center -- these are RETRIES ONLY, the best-scoring
    # other spots (no cup-width spacing rule, operator's call), kept only far
    # enough from the centre to be a different spot at all. Scored by the
    # object-mode scorer over the product mask (PackageConfig carries the
    # obj_grasp_* fields it reads).
    product_inside_grasp_candidates = []
    if product_inside["found"] and product_inside["center_roi"] is not None:
        from .object import score_object_grasp_candidates

        scored = score_object_grasp_candidates(
            cfg,
            {"mask": product_inside["mask"]},
            product_inside["center_roi"],
            depth_class_roi,
            product_inside["depth_mm"] or top_face_depth_mm,
            None,
            intrinsics,
        )
        for c in scored:
            if c.get("offset_mm", 0.0) < cfg.product_inside_grasp_min_offset_mm:
                continue
            product_inside_grasp_candidates.append(
                {
                    "x_mm": c.get("x_mm"),
                    "y_mm": c.get("y_mm"),
                    "z_mm": c.get("z_mm"),
                    "score": c.get("score"),
                    "point_full": c.get("point_full"),
                }
            )
            if len(product_inside_grasp_candidates) >= cfg.product_inside_grasp_retry_count:
                break

    # Same fallback treatment for BOXES, whose pick lands on the package
    # top-face centre: tape seams and dents break the seal there, so rank the
    # smoothest/flattest other cup spots on the top face. Shares the cup
    # geometry, offset floor and retry count with the product-inside path --
    # it is the same cup on the same station.
    grasp_candidates = []
    if classification["package_type"] == "box" and best["center_roi"] is not None:
        # Operator-specified (2026-09-04): exactly two retries, centre
        # +-box_grasp_min_offset_mm ALONG THE HORIZONTAL line through the
        # centre point -- one each side of the flap slit. Deterministic, no
        # scoring. Each point's depth is measured locally (measurement
        # stream, the one that sees cardboard) so a tilted box still gets
        # the right height per point.
        mm_per_px_box = (
            float(top_face_depth_mm) / float(intrinsics["fx"])
            if top_face_depth_mm is not None and intrinsics is not None
            else 0.68
        )
        off_px = cfg.box_grasp_min_offset_mm / mm_per_px_box
        cx0, cy0 = best["center_roi"]
        mh, mw = best["mask"].shape

        # Retries go PERPENDICULAR to the detected seam line, following the
        # box's own rotation (a square box's rectangle cannot disambiguate
        # the seam; the RGB edge test along the box axes can).
        from .box import box_seam_axes

        _, (dx, dy) = box_seam_axes(roi_rgb, best["mask"], (cx0, cy0))

        for sign in (1, -1):
            gx = int(round(cx0 + sign * off_px * dx))
            gy = int(round(cy0 + sign * off_px * dy))
            if not (0 <= gy < mh and 0 <= gx < mw) or best["mask"][gy, gx] == 0:
                continue
            from .object import center_depth_mm as _point_depth_mm

            gz, _ = _point_depth_mm(cfg, depth_measure_roi, best["mask"], (gx, gy))
            if gz is None:
                gz = top_face_depth_mm
            point_full = (cfg.roi_x1 + gx, cfg.roi_y1 + gy)
            gx_mm, gy_mm = pixel_to_camera_xy_mm(point_full, gz, intrinsics)
            grasp_candidates.append(
                {
                    "x_mm": gx_mm,
                    "y_mm": gy_mm,
                    "z_mm": gz,
                    "score": None,
                    "point_full": point_full,
                }
            )

    heatmap = make_package_depth_heatmap(
        cfg,
        depth_roi=depth_class_roi,
        package_mask=best["mask"],
        center_roi=best["center_roi"],
        product_inside=product_inside,
    )

    final_output = {
        "barcode_type": barcode["type"] if barcode is not None else None,
        "barcode_data": barcode["data"] if barcode is not None else None,
        "package_type": classification["package_type"],
        "package_type_confidence_percent": classification["package_type_confidence"] * 100.0,
        "mask_source": best["source"],
        "top_face_depth_mm": top_face_depth_mm,
        "package_depth_mm": package_depth_mm,
        "length_mm": dimensions["length_mm"],
        "width_mm": dimensions["short_side_mm"],
        "angle_deg": dimensions["angle_deg"],
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "product_inside_found": product_inside["found"],
        "product_inside_center_x_mm": product_inside["center_x_mm"],
        "product_inside_center_y_mm": product_inside["center_y_mm"],
        "product_inside_center_pixel_u": (
            product_inside["center_full"][0]
            if product_inside["center_full"] is not None
            else None
        ),
        "product_inside_center_pixel_v": (
            product_inside["center_full"][1]
            if product_inside["center_full"] is not None
            else None
        ),
        "product_inside_depth_mm": product_inside["depth_mm"],
        "product_inside_grasp_candidates": product_inside_grasp_candidates,
        "grasp_candidates": grasp_candidates,
    }

    print()
    print("PACKAGE FINAL OUTPUT:")
    print(f"  barcode_type:                    {final_output['barcode_type']}")
    print(f"  barcode_data:                    {final_output['barcode_data']}")
    print(f"  package_type:                    {final_output['package_type']}")
    print(f"  confidence_percent:              {final_output['package_type_confidence_percent']:.1f}")
    print(f"  mask_source:                     {final_output['mask_source']}")
    print(f"  top_face_depth_mm:               {fmt3(final_output['top_face_depth_mm'])}")
    print(f"  package_depth_mm:                {fmt3(final_output['package_depth_mm'])}")
    print(f"  length_mm:                       {fmt3(final_output['length_mm'])}")
    print(f"  width_mm:                        {fmt3(final_output['width_mm'])}")
    print(f"  angle_deg:                       {fmt3(final_output['angle_deg'])}")
    print(f"  center_x_mm:                     {fmt3(final_output['center_x_mm'])}")
    print(f"  center_y_mm:                     {fmt3(final_output['center_y_mm'])}")
    print(f"  product_inside_found:            {final_output['product_inside_found']}")
    print(f"  product_inside_center_x_mm:      {fmt3(final_output['product_inside_center_x_mm'])}")
    print(f"  product_inside_center_y_mm:      {fmt3(final_output['product_inside_center_y_mm'])}")

    return {
        "full_rgb": full_rgb,
        "roi_rgb": roi_rgb,
        "depth_class_roi": depth_class_roi,
        "depth_measure_roi": depth_measure_roi,
        "mask": best["mask"],
        "center_roi": best["center_roi"],
        "center_full": center_full,
        "info": best,
        "accepted": accepted,
        "all_mask_count": len(masks),
        "dedicated_box_mask_count": dedicated_box_mask_count,
        "dedicated_box_candidate_found": dedicated_box_candidate is not None,
        "top_face_depth_mm": top_face_depth_mm,
        "package_depth_mm": package_depth_mm,
        "top_face_depth_result": top_face_depth_result,
        "classification": classification,
        "dimensions": dimensions,
        "product_inside": product_inside,
        "final_output": final_output,
        "depth_heatmap": heatmap,
        "barcode": barcode,
    }


# =========================================================================
# NEW product-inside locator (2026-09-03): height-band segmentation on the
# ABSOLUTE height-above-table map. Lives here (not a separate module) so it
# is covered by the same tracked file as the rest of package detection.
# Operator insight behind it: a product under one end PROPS THE MAILER UP --
# the "tilt" largely IS the product, so plane-relative elevation subtracts
# the signal; height above the level table keeps it.
# =========================================================================

def build_depth_bundle(cfg, depth_class_aligned, depth_measure_aligned, package_mask, intrinsics):
    """Assemble the single-shot depth bundle the new locator will work from.

    Args:
        cfg: PackageConfig (ROI, valid-depth bounds, plane-fit settings).
        depth_class_aligned: full-frame RGB-aligned uint16 classification
            depth -- the frame the detection ran on.
        depth_measure_aligned: full-frame measurement-stereo depth (uint16),
            or None.
        package_mask: mailer mask in ROI coordinates (uint8).
        intrinsics: {"fx", ...} or None.

    Returns dict or None (not enough valid depth):
        depth        ROI float32 mm, NaN where invalid
        measure      ROI float32 mm or None, NaN where invalid
        mailer       ROI bool, eroded mailer interior with valid depth
        plane        ROI float32 mm, robust mailer reference plane
        elevation    ROI float32 mm, plane - depth (positive = raised),
                     raw, unsmoothed
        elevation_smooth  ROI float32 mm, noise-suppressed elevation (the
                     shipped detector's smoothing scale, for comparability)
        noise_mm     float, this frame's own depth speckle (robust sigma of
                     the high-pass residual) -- scale thresholds to this
        mm_per_px    float or None
    """
    depth = depth_class_aligned[
        cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2
    ].astype(np.float32)
    valid = (depth > cfg.min_valid_depth_mm) & (depth < cfg.max_valid_depth_mm)
    depth = np.where(valid, depth, np.nan).astype(np.float32)

    measure = None
    if depth_measure_aligned is not None:
        m = depth_measure_aligned[
            cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2
        ].astype(np.float32)
        measure = np.where(
            (m > cfg.min_valid_depth_mm) & (m < cfg.max_valid_depth_mm), m, np.nan
        ).astype(np.float32)

    package_mask = (package_mask > 0).astype(np.uint8)
    span = float(np.sqrt(max(int(package_mask.sum()), 1)))
    inner = erode_mask(
        package_mask, max(1, int(round(span * cfg.product_inside_edge_erode_frac)))
    )
    mailer = (inner == 1) & valid

    if int(mailer.sum()) < cfg.product_inside_min_valid_pixels:
        return None

    depth_for_fit = np.where(valid, depth, 0.0).astype(np.float32)
    plane = fit_reference_plane(
        depth_for_fit,
        mailer,
        max_points=cfg.product_inside_max_plane_points,
        iterations=cfg.product_inside_plane_iterations,
        trim_sigma=cfg.product_inside_plane_trim_sigma,
    )

    if plane is None:
        return None

    elevation = np.where(valid, plane - depth, np.nan).astype(np.float32)

    # Noise-suppressed elevation at the shipped detector's scale, plus the
    # frame's own speckle level measured from what the smoothing removed.
    raw_for_blur = np.where(mailer, plane - depth_for_fit, 0.0).astype(np.float32)
    smooth_px = max(2.0, span * cfg.product_inside_smooth_frac)
    elevation_smooth, _ = masked_gaussian_blur(
        raw_for_blur, mailer, smooth_px, return_support=True
    )
    high_pass = (raw_for_blur - elevation_smooth)[mailer]
    noise_mm = float(1.4826 * np.median(np.abs(high_pass - np.median(high_pass))))
    elevation_smooth = np.where(mailer, elevation_smooth, np.nan).astype(np.float32)

    mm_per_px = None
    if intrinsics is not None and intrinsics.get("fx"):
        mm_per_px = float(np.nanmedian(depth[mailer])) / float(intrinsics["fx"])

    # ABSOLUTE height above the table. Operator insight 2026-09-03: a product
    # under one end PROPS THE MAILER UP -- the "tilt" largely IS the product,
    # so the fitted plane's elevation subtracts the signal and leaves only
    # the central air bubble. Height above the level table keeps it. (The
    # fitted-plane elevation stays available for comparison; camera-axis
    # tilt, if any, shows in height as a fixed gradient and can be
    # calibrated per station later.)
    height_above_base = np.where(
        valid, float(cfg.base_depth_mm) - depth, np.nan
    ).astype(np.float32)

    return {
        "depth": depth,
        "measure": measure,
        "mailer": mailer,
        "plane": plane.astype(np.float32),
        "elevation": elevation,
        "elevation_smooth": elevation_smooth,
        "height_above_base": height_above_base,
        "noise_mm": max(noise_mm, 1e-3),
        "mm_per_px": mm_per_px,
    }


def _solid_blob(cfg, band_mask, mm_per_px):
    """Collapse a speckled band mask into ONE solid blob, or None.

    Operator spec: no scattered blobs -- merge the densest speckle cluster
    into a single region and ignore tiny outlying specks that would drag it.
    Small opening kills lone specks, closing fuses the dense cluster, the
    largest connected piece wins, holes are filled.
    """
    if mm_per_px is None or mm_per_px <= 0:
        mm_per_px = 0.68

    def _kernel(mm):
        px = max(1, int(round(mm / mm_per_px)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))

    blob = band_mask.astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, _kernel(cfg.locator_open_mm))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, _kernel(cfg.locator_close_mm))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, 4]))
    if stats[best, 4] < cfg.locator_min_region_px:
        return None

    blob = (labels == best).astype(np.uint8)

    # Appendage trim: thin legs that survived the close still drag the
    # centroid. A stronger opening removes anything narrower than
    # 2 x trim_open_mm; the cut only stands when the remaining body keeps
    # most of the area (a genuinely thin blob stays whole).
    if cfg.locator_trim_open_mm > 0:
        trimmed = cv2.morphologyEx(blob, cv2.MORPH_OPEN, _kernel(cfg.locator_trim_open_mm))
        tn, tlabels, tstats, _ = cv2.connectedComponentsWithStats(trimmed, 8)
        if tn > 1:
            tbest = 1 + int(np.argmax(tstats[1:, 4]))
            if tstats[tbest, 4] >= cfg.locator_trim_min_area_frac * int(blob.sum()):
                blob = (tlabels == tbest).astype(np.uint8)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled


def locate_product(cfg, bundle):
    """Stage-2 locator: nested height-band segments from the height map.

    Segment 1 (PRODUCT): the upper part of the mailer's own height range --
    "the product is in this region". Segment 2 (GRAB, inside segment 1): the
    very top band -- "the area we can most likely grab from". Bands are
    fractions of the frame's robust height range, so they self-scale to any
    product. Both come back as single solid blobs (see _solid_blob).

    Returns dict or None:
        lo_mm, hi_mm          robust height range of the mailer this frame
        product_thr_mm, grab_thr_mm   the two band thresholds
        product_mask, grab_mask       ROI uint8, one solid blob each
                                      (grab_mask may be None)
        product_center, grab_center   ROI (x, y) blob centroids
    """
    h = bundle["height_above_base"]
    m = bundle["mailer"]

    vals = h[m & np.isfinite(h)]
    if vals.size < cfg.locator_min_region_px:
        return None

    lo, hi = np.nanpercentile(vals, [2.0, 99.0])
    if hi - lo < 1.0:
        return None

    product_thr = lo + cfg.locator_product_band_frac * (hi - lo)
    grab_thr = lo + cfg.locator_grab_band_frac * (hi - lo)

    finite = np.isfinite(h)
    product_band = m & finite & (h >= product_thr)
    product_mask = _solid_blob(cfg, product_band, bundle.get("mm_per_px"))
    if product_mask is None:
        return None

    grab_band = (product_mask == 1) & finite & (h >= grab_thr)
    grab_mask = _solid_blob(cfg, grab_band, bundle.get("mm_per_px"))

    def _centroid(mask):
        ys, xs = np.where(mask == 1)
        return (int(round(xs.mean())), int(round(ys.mean())))

    # GRASP POINT (operator spec 2026-09-04): not the blob centroid but the
    # point ON the product closest to the MAILER's centre -- picking near
    # the mailer's middle leaves the least empty film hanging off the cup
    # during the carry. The blob is first shrunk by a cup radius so the
    # chosen point keeps the whole cup on the product region; if the blob is
    # too small to shrink, the centroid stands.
    mm_per_px = bundle.get("mm_per_px") or 0.68
    anchor = grab_mask if grab_mask is not None else product_mask
    r_px = max(1, int(round((cfg.obj_grasp_cup_diameter_mm / 2.0) / mm_per_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_px * 2 + 1, r_px * 2 + 1))
    core = cv2.erode(anchor, kernel)
    if int(core.sum()) == 0:
        core = anchor
    mys, mxs = np.where(m)
    mailer_c = (float(mxs.mean()), float(mys.mean()))
    cys, cxs = np.where(core == 1)
    d2 = (cxs - mailer_c[0]) ** 2 + (cys - mailer_c[1]) ** 2
    k = int(np.argmin(d2))
    grasp_point = (int(cxs[k]), int(cys[k]))

    return {
        "lo_mm": float(lo),
        "hi_mm": float(hi),
        "product_thr_mm": float(product_thr),
        "grab_thr_mm": float(grab_thr),
        "product_mask": product_mask,
        "grab_mask": grab_mask,
        "product_center": _centroid(product_mask),
        "grab_center": _centroid(grab_mask) if grab_mask is not None else None,
        "grasp_point": grasp_point,
    }


def locate_product_inside(cfg, depth_class_aligned, depth_measure_aligned, package_mask, intrinsics):
    """Full product_inside result from the height-band locator.

    Drop-in producer for the pipeline's product_inside dict (same keys the
    legacy detector returned, so the node, heatmap, JSON and grasp-fallback
    scoring all keep working unchanged). The grasp point is the GRAB blob's
    centroid; its depth is the median camera distance over that blob.
    """
    shape = package_mask.shape
    result = {
        "found": False,
        "center_roi": None,
        "center_full": None,
        "center_x_mm": None,
        "center_y_mm": None,
        "depth_mm": None,
        "area_px": 0,
        "area_ratio_of_package": 0.0,
        "mask": np.zeros(shape, dtype=np.uint8),
        "core_mask": np.zeros(shape, dtype=np.uint8),
        "core_area_px": 0,
        "core_found": False,
        "reason": "not_run",
        "peak_mm": None,
        "noise_mm": None,
        "peak_noise_multiple": None,
        "threshold_mm": None,
        "rect_w_mm": None,
        "rect_h_mm": None,
        "rect_angle_deg": None,
        "edge_drop_frac": None,
        "rect_fill": None,
        "solidity": None,
        "center_source": None,
        "candidate_score": None,
        "edge_sharpness": None,
    }

    bundle = build_depth_bundle(
        cfg, depth_class_aligned, depth_measure_aligned, package_mask, intrinsics
    )
    if bundle is None:
        result["reason"] = "not_enough_depth"
        return result

    result["noise_mm"] = bundle["noise_mm"]

    located = locate_product(cfg, bundle)
    if located is None:
        result["reason"] = "no_height_region"
        return result

    product_mask = located["product_mask"]
    grab_mask = located["grab_mask"]
    # Operator's pick (settled 2026-09-04 after trying all three variants):
    # the GRAB blob's centroid -- the centre of the dark-red top band. The
    # other two variants (product-region centroid, closest-to-mailer-centre
    # point) remain in `located` for comparison.
    center_roi = located["grab_center"] or located["product_center"]

    # Grasp depth: median camera distance over the grab blob (fall back to
    # the product blob when the grab band produced nothing solid).
    depth_src = grab_mask if grab_mask is not None else product_mask
    dvals = bundle["depth"][(depth_src == 1) & np.isfinite(bundle["depth"])]
    if dvals.size < cfg.min_surface_depth_count:
        dvals = bundle["depth"][(product_mask == 1) & np.isfinite(bundle["depth"])]
    if dvals.size == 0:
        result["reason"] = "no_height_region"
        return result
    depth_mm = float(np.median(dvals)) + cfg.measurement_depth_offset_mm

    center_full = (
        int(cfg.roi_x1 + center_roi[0]),
        int(cfg.roi_y1 + center_roi[1]),
    )
    cx_mm, cy_mm = pixel_to_camera_xy_mm(center_full, depth_mm, intrinsics)

    area = int(product_mask.sum())
    ys, xs = np.where(product_mask == 1)
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    (_, _), (rect_w, rect_h), rect_angle = rect
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
    mm_per_px = bundle["mm_per_px"] or 0.68

    result.update(
        {
            "found": True,
            "center_roi": (int(center_roi[0]), int(center_roi[1])),
            "center_full": center_full,
            "center_x_mm": cx_mm,
            "center_y_mm": cy_mm,
            "depth_mm": depth_mm,
            "area_px": area,
            "area_ratio_of_package": area / max(int((package_mask > 0).sum()), 1),
            "mask": product_mask,
            "core_mask": grab_mask if grab_mask is not None else np.zeros(shape, np.uint8),
            "core_area_px": int(grab_mask.sum()) if grab_mask is not None else 0,
            "core_found": grab_mask is not None,
            "reason": "ok",
            "peak_mm": float(located["hi_mm"] - located["lo_mm"]),
            "peak_noise_multiple": float(
                (located["hi_mm"] - located["lo_mm"]) / bundle["noise_mm"]
            ),
            "threshold_mm": float(located["product_thr_mm"]),
            "rect_w_mm": float(rect_w * mm_per_px),
            "rect_h_mm": float(rect_h * mm_per_px),
            "rect_angle_deg": float(rect_angle),
            "rect_fill": area / max(float(rect_w * rect_h), 1.0),
            "solidity": area / max(float(cv2.contourArea(hull)), 1.0),
            "center_source": "height_bands",
        }
    )
    return result
