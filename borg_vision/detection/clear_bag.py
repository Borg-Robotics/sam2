"""Clear-bag-mode detection logic: visible product mask scoring + depth-based
bag detection around the product.

Ported verbatim from run_clear_bag()/clear_bag_final.py with the module-level
constants replaced by fields of a ClearBagConfig passed as the first argument.
Generic, config-free helpers are reused from detection.package; the
clear-bag-specific helpers (bag seed/expansion, product scoring with its own
safe-center) live here.
"""

import cv2
import numpy as np

import torch

from ..utils import score_from_bad_good
from ..visualization.clear_bag import make_clear_bag_depth_heatmap
from .package import (
    close_mask,
    dilate_mask,
    erode_mask,
    largest_component,
    pixel_to_camera_xy_mm,
)


# ----------------------------------------------------------------- components


def component_containing_point(mask, point, search_radius_px):
    px, py = point

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    h, w = mask.shape[:2]

    x1 = max(0, px - search_radius_px)
    x2 = min(w, px + search_radius_px + 1)
    y1 = max(0, py - search_radius_px)
    y2 = min(h, py + search_radius_px + 1)

    label_patch = labels[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]

    valid_labels = label_patch[mask_patch == 1]

    if valid_labels.size == 0:
        return largest_component(mask)

    labels_unique, counts = np.unique(valid_labels, return_counts=True)

    best_label = None
    best_count = -1

    for label, count in zip(labels_unique, counts):
        if label == 0:
            continue

        if count > best_count:
            best_count = count
            best_label = label

    if best_label is None:
        return largest_component(mask)

    return (labels == best_label).astype(np.uint8)


# ---------------------------------------------------------- bag corner expand


def smooth_1d(values, kernel_px):
    values = np.asarray(values).astype(np.float32)

    if kernel_px <= 1:
        return values

    if kernel_px % 2 == 0:
        kernel_px += 1

    return cv2.GaussianBlur(values.reshape(1, -1), (kernel_px, 1), 0).reshape(-1)


def active_span_from_activity(cfg, activity, min_pixels, fraction):
    activity = np.asarray(activity).astype(np.float32)

    if activity.size == 0:
        return None

    smoothed = smooth_1d(activity, cfg.bag_edge_activity_smooth_px)
    max_activity = float(np.max(smoothed))

    if max_activity <= 0:
        return None

    threshold = max(float(min_pixels), max_activity * float(fraction))
    active = smoothed >= threshold

    indices = np.where(active)[0]

    if indices.size == 0:
        return None

    return int(indices[0]), int(indices[-1])


def corner_has_depth_support(cfg, mask, x1, y1, x2, y2):
    h, w = mask.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h))

    if x2 <= x1 or y2 <= y1:
        return False

    patch = mask[y1:y2, x1:x2]
    return int(patch.sum()) >= cfg.bag_corner_min_pixels


def expand_bag_rect_using_corner_depth(cfg, component, closer_than_base, product_mask):
    if not cfg.bag_corner_expand_enable:
        return component

    component = component.astype(np.uint8)
    closer_than_base = closer_than_base.astype(np.uint8)
    product_mask = product_mask.astype(np.uint8)

    support = np.maximum(component, closer_than_base)
    support = np.maximum(support, product_mask)
    support = dilate_mask(support, cfg.bag_corner_expand_dilate_px)
    support = close_mask(support, cfg.bag_close_kernel_px, iterations=1)

    row_activity = np.sum(support, axis=1)
    col_activity = np.sum(support, axis=0)

    x_span = active_span_from_activity(
        cfg,
        col_activity,
        cfg.bag_edge_min_activity_px,
        cfg.bag_edge_activity_fraction,
    )

    y_span = active_span_from_activity(
        cfg,
        row_activity,
        cfg.bag_edge_min_activity_px,
        cfg.bag_edge_activity_fraction,
    )

    if x_span is None or y_span is None:
        return component

    h, w = component.shape[:2]
    x1, x2 = x_span
    y1, y2 = y_span

    win = cfg.bag_corner_window_px

    top_left = corner_has_depth_support(cfg, closer_than_base, x1, y1, x1 + win, y1 + win)
    top_right = corner_has_depth_support(cfg, closer_than_base, x2 - win, y1, x2, y1 + win)
    bottom_left = corner_has_depth_support(cfg, closer_than_base, x1, y2 - win, x1 + win, y2)
    bottom_right = corner_has_depth_support(cfg, closer_than_base, x2 - win, y2 - win, x2, y2)

    cx, cy, cw, ch = cv2.boundingRect(component)

    if not (top_left or top_right):
        y1 = cy

    if not (bottom_left or bottom_right):
        y2 = cy + ch

    x1 = max(0, x1 - cfg.bag_expand_padding_x_px)
    x2 = min(w - 1, x2 + cfg.bag_expand_padding_x_px)
    y1 = max(0, y1 - cfg.bag_expand_padding_y_px)
    y2 = min(h - 1, y2 + cfg.bag_expand_padding_y_px)

    rect_mask = np.zeros_like(component, dtype=np.uint8)
    cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 1, thickness=-1)

    return rect_mask


# --------------------------------------------------------- product mask scoring


def clean_mask(cfg, mask):
    mask = mask.astype(np.uint8)

    if not cfg.use_mask_cleanup:
        return mask

    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(mask, cfg.mask_close_kernel_px, cfg.mask_close_iterations)
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(cfg, mask):
    safe_mask = erode_mask(mask, cfg.safe_erode_radius_px)
    safe_component = largest_component(safe_mask)

    if safe_component is None:
        safe_component = largest_component(mask)

    if safe_component is None:
        return None, safe_mask

    moments = cv2.moments(safe_component.astype(np.uint8))

    if moments["m00"] == 0:
        return None, safe_component

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy), safe_component.astype(np.uint8)


def is_roller_like_horizontal_strip(cfg, mask):
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))

    width_ratio = bw / max(w, 1)
    height_ratio = bh / max(h, 1)

    return (
        width_ratio >= cfg.roller_strip_min_width_ratio
        and height_ratio <= cfg.roller_strip_max_height_ratio
    )


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


def score_product_mask(cfg, mask, roi_rgb, sam_iou, sam_stability):
    mask = mask.astype(np.uint8)

    if cfg.hard_reject_roller_like and is_roller_like_horizontal_strip(cfg, mask):
        return None

    roi_h, roi_w = mask.shape[:2]
    roi_area = roi_h * roi_w

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < cfg.min_product_area_ratio:
        return None

    if area_ratio > cfg.max_product_area_ratio:
        return None

    x, y, w, h = cv2.boundingRect(mask)

    if w <= 0 or h <= 0:
        return None

    bbox_area = w * h
    rectangularity = area / max(bbox_area, 1)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    if rectangularity < cfg.min_product_rectangularity:
        return None

    if aspect_ratio > cfg.max_product_aspect_ratio:
        return None

    center_roi, safe_mask = get_mask_center(cfg, mask)

    if center_roi is None:
        return None

    safe_area = int(safe_mask.sum())
    safe_area_ratio = safe_area / max(roi_area, 1)

    roi_cx = roi_w / 2
    roi_cy = roi_h / 2

    dist = np.sqrt(
        (center_roi[0] - roi_cx) ** 2
        + (center_roi[1] - roi_cy) ** 2
    )

    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    area_score = 1.0 - min(
        abs(area_ratio - cfg.target_product_area_ratio) / cfg.target_product_area_ratio,
        1.0,
    )

    contrast_score, contrast_value = compute_rgb_contrast_score(cfg, roi_rgb, mask)
    texture_score, texture_value = compute_texture_score(cfg, roi_rgb, mask)

    quality_score = (
        0.25 * area_score
        + 0.20 * center_score
        + 0.30 * contrast_score
        + 0.15 * texture_score
        + 0.10 * min(safe_area_ratio / 0.03, 1.0)
    )

    score = (
        cfg.area_score_weight * area_score
        + cfg.center_score_weight * center_score
        + cfg.contrast_score_weight * contrast_score
        + cfg.texture_score_weight * texture_score
        + cfg.rect_score_weight * rectangularity
        + cfg.safe_area_score_weight * safe_area_ratio
        + cfg.sam_iou_score_weight * sam_iou
        + cfg.sam_stability_score_weight * sam_stability
    )

    return {
        "score": float(score),
        "quality_score": float(np.clip(quality_score, 0.0, 1.0)),
        "mask": mask,
        "safe_mask": safe_mask,
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "bbox": (int(x), int(y), int(w), int(h)),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "contrast_score": float(contrast_score),
        "contrast_value": float(contrast_value),
        "texture_score": float(texture_score),
        "texture_value": float(texture_value),
        "safe_area_ratio": float(safe_area_ratio),
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
    }


def choose_best_product_mask(cfg, masks, roi_rgb):
    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        cleaned_mask = clean_mask(cfg, raw_mask)

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        result = score_product_mask(cfg, cleaned_mask, roi_rgb, sam_iou, sam_stability)

        if result is not None:
            result["index"] = i

            if best is None or result["score"] > best["score"]:
                best = result

    return best


# ------------------------------------------------------------- depth measures


def robust_depth_from_values(cfg, values):
    values = values.astype(np.float32)

    if values.size < cfg.min_center_top_face_depth_count:
        return None, 0, None, None

    raw_median = float(np.median(values))
    abs_dev = np.abs(values - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = values[np.abs(values - raw_median) <= 12.0]
    else:
        filtered = values[abs_dev <= 3.5 * mad]

    if filtered.size < cfg.min_center_top_face_depth_count:
        filtered = values

    return (
        float(np.median(filtered)),
        int(filtered.size),
        float(np.mean(filtered)),
        float(np.min(filtered)),
    )


def center_top_face_depth_mm(cfg, depth_roi, product_mask, center_roi):
    cx, cy = center_roi
    h, w = depth_roi.shape[:2]

    radius = cfg.center_top_face_radius_px

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    patch_depth = depth_roi[y1:y2, x1:x2]
    patch_mask = product_mask[y1:y2, x1:x2]

    values = patch_depth[
        (patch_mask == 1)
        & (patch_depth > cfg.min_valid_depth_mm)
        & (patch_depth < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if values.size < cfg.min_center_top_face_depth_count:
        values = patch_depth[
            (patch_depth > cfg.min_valid_depth_mm)
            & (patch_depth < cfg.max_valid_depth_mm)
        ].astype(np.float32)

    depth_mm, count, mean_mm, min_mm = robust_depth_from_values(cfg, values)

    return {
        "depth_mm": depth_mm,
        "count": count,
        "mean_mm": mean_mm,
        "min_mm": min_mm,
        "radius_px": radius,
        "center_roi": (int(cx), int(cy)),
        "center_full": (int(cfg.roi_x1 + cx), int(cfg.roi_y1 + cy)),
    }


def estimate_base_depth_mm(cfg, depth_roi, product_mask):
    return float(cfg.base_depth_mm)


def get_rotated_rect_from_mask(mask):
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
        "angle_deg": float(angle),
        "points_roi": box_points,
    }


# -------------------------------------------------------------- bag detection


def detect_clear_bag_from_depth(cfg, depth_roi, product_mask, product_center_roi):
    roi_h, roi_w = depth_roi.shape[:2]
    roi_area = roi_h * roi_w

    if cfg.bag_use_median_blur_depth:
        depth_for_bag = cv2.medianBlur(
            depth_roi.astype(np.uint16),
            cfg.bag_depth_median_blur_ksize,
        )
    else:
        depth_for_bag = depth_roi.copy()

    base_depth_mm = estimate_base_depth_mm(cfg, depth_for_bag, product_mask)

    valid_depth = (
        (depth_for_bag > cfg.min_valid_depth_mm)
        & (depth_for_bag < cfg.max_valid_depth_mm)
    )

    closer_than_base = (
        valid_depth
        & (depth_for_bag <= base_depth_mm - cfg.bag_closer_than_base_min_mm)
        & (depth_for_bag >= base_depth_mm - cfg.bag_closer_than_base_max_mm)
    ).astype(np.uint8)

    px, py = product_center_roi

    seed = np.zeros_like(closer_than_base, dtype=np.uint8)
    cv2.circle(seed, (px, py), cfg.bag_seed_dilate_px, 1, -1)

    product_seed = dilate_mask(product_mask, cfg.bag_seed_dilate_px)
    seed = np.maximum(seed, product_seed)

    candidate = np.zeros_like(closer_than_base, dtype=np.uint8)
    candidate[(closer_than_base == 1) & (seed == 1)] = 1

    candidate = close_mask(candidate, cfg.bag_close_kernel_px, iterations=2)
    candidate = dilate_mask(candidate, cfg.bag_dilate_kernel_px)
    candidate = close_mask(candidate, cfg.bag_close_kernel_px, iterations=1)
    candidate = erode_mask(candidate, cfg.bag_erode_kernel_px)

    component = component_containing_point(
        candidate,
        product_center_roi,
        cfg.bag_component_keep_near_product_px,
    )

    if component is None:
        return {
            "bag_found": False,
            "bag_reason": "no_component_near_product",
            "bag_mask": candidate,
            "bag_bbox": None,
            "bag_center_roi": None,
            "base_depth_mm": base_depth_mm,
            "bag_area_ratio": 0.0,
            "rotated": None,
        }

    component = np.maximum(component.astype(np.uint8), product_mask.astype(np.uint8))
    component = close_mask(component, cfg.bag_close_kernel_px, iterations=1)

    component = expand_bag_rect_using_corner_depth(
        cfg,
        component=component,
        closer_than_base=closer_than_base,
        product_mask=product_mask,
    )

    area = int(component.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < cfg.bag_min_area_ratio:
        found = False
        reason = "too_small"
    elif area_ratio > cfg.bag_max_area_ratio:
        found = False
        reason = "too_large"
    else:
        found = True
        reason = "ok"

    x, y, w, h = cv2.boundingRect(component)
    bag_center_roi = (int(x + w / 2), int(y + h / 2))
    rotated = get_rotated_rect_from_mask(component)

    return {
        "bag_found": bool(found),
        "bag_reason": reason,
        "bag_mask": component,
        "bag_bbox": (int(x), int(y), int(w), int(h)),
        "bag_center_roi": bag_center_roi,
        "base_depth_mm": float(base_depth_mm),
        "bag_area_ratio": float(area_ratio),
        "rotated": rotated,
    }


def estimate_clear_bag_measurements(cfg, bag, intrinsics):
    empty = {
        "length_mm": None,
        "width_mm": None,
        "angle_deg": None,
        "center_x_mm": None,
        "center_y_mm": None,
        "points_full": None,
    }

    if bag is None or bag.get("bag_bbox") is None:
        return empty

    if intrinsics is None or bag.get("base_depth_mm") is None:
        return empty

    depth_mm = float(bag["base_depth_mm"])
    rotated = bag.get("rotated")

    if rotated is None:
        bx, by, bw, bh = bag["bag_bbox"]

        center_full = (
            cfg.roi_x1 + int(bx + bw / 2),
            cfg.roi_y1 + int(by + bh / 2),
        )

        raw_width_mm = (bw * depth_mm) / intrinsics["fx"]
        raw_height_mm = (bh * depth_mm) / intrinsics["fy"]

        angle_deg = 0.0
        points_full = None

    else:
        cx_roi, cy_roi = rotated["center_roi"]

        center_full = (
            cfg.roi_x1 + int(cx_roi),
            cfg.roi_y1 + int(cy_roi),
        )

        raw_width_mm = (rotated["width_px"] * depth_mm) / intrinsics["fx"]
        raw_height_mm = (rotated["height_px"] * depth_mm) / intrinsics["fy"]

        angle_deg = rotated["angle_deg"]

        points_roi = rotated["points_roi"]
        points_full_np = points_roi.copy()
        points_full_np[:, 0] += cfg.roi_x1
        points_full_np[:, 1] += cfg.roi_y1
        points_full = points_full_np.tolist()

    scaled_width_mm = raw_width_mm * cfg.poly_size_scale
    scaled_height_mm = raw_height_mm * cfg.poly_size_scale

    length_mm = max(scaled_width_mm, scaled_height_mm)
    width_mm = min(scaled_width_mm, scaled_height_mm)

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        depth_mm,
        intrinsics,
    )

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "angle_deg": float(angle_deg),
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "points_full": points_full,
    }


def run_sam2_product_and_bag(cfg, frame_bgr, depth_aligned, mask_generator, intrinsics):
    """Run SAM2 on the ROI, pick the best visible product mask, then detect the
    clear bag around it from depth. Returns a result dict or None when no valid
    product mask is found."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi = depth_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    best = choose_best_product_mask(cfg, masks, roi_rgb)

    if best is None:
        return None

    product_center_full = (
        int(cfg.roi_x1 + best["center_roi"][0]),
        int(cfg.roi_y1 + best["center_roi"][1]),
    )

    product_top_face_depth = center_top_face_depth_mm(
        cfg,
        depth_roi=depth_roi,
        product_mask=best["safe_mask"],
        center_roi=best["center_roi"],
    )

    product_inside_center_x_mm, product_inside_center_y_mm = pixel_to_camera_xy_mm(
        product_center_full,
        product_top_face_depth["depth_mm"],
        intrinsics,
    )

    bag = detect_clear_bag_from_depth(
        cfg,
        depth_roi=depth_roi,
        product_mask=best["mask"],
        product_center_roi=best["center_roi"],
    )

    bag_measurements = estimate_clear_bag_measurements(
        cfg,
        bag=bag,
        intrinsics=intrinsics,
    )

    depth_heatmap = make_clear_bag_depth_heatmap(
        cfg,
        depth_roi=depth_roi,
        product_mask=best["mask"],
        bag=bag,
    )

    if product_top_face_depth["depth_mm"] is None:
        clearbag_depth_mm = None
    else:
        clearbag_depth_mm = cfg.base_depth_mm - product_top_face_depth["depth_mm"]

    final_output = {
        "product_face_depth_mm": product_top_face_depth["depth_mm"],
        "clearbag_depth_mm": clearbag_depth_mm,
        "length_mm": bag_measurements["length_mm"],
        "width_mm": bag_measurements["width_mm"],
        "angle_deg": bag_measurements["angle_deg"],
        "center_x_mm": bag_measurements["center_x_mm"],
        "center_y_mm": bag_measurements["center_y_mm"],
        "product_inside_center_x_mm": product_inside_center_x_mm,
        "product_inside_center_y_mm": product_inside_center_y_mm,
    }

    return {
        "full_rgb": full_rgb,
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "mask": best["mask"],
        "safe_mask": best["safe_mask"],
        "center_roi": best["center_roi"],
        "center_full": product_center_full,
        "product_top_face_depth": product_top_face_depth,
        "info": best,
        "bag": bag,
        "bag_measurements": bag_measurements,
        "depth_heatmap": depth_heatmap,
        "final_output": final_output,
    }
