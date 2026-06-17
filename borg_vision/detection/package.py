"""Package-mode detection logic: mask scoring, classification, dimensions,
depth stats.

All functions are ports of the original product_detection_final.py with the
module-level constants replaced by fields of a PackageConfig passed as the
first argument. Function bodies are otherwise unchanged.
"""

import cv2
import numpy as np
import torch

from ..utils import fmt3, score_from_bad_good, score_from_range
from ..visualization import make_package_depth_heatmap


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
        "angle_deg": float(angle),
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


def choose_best_package_mask(cfg, masks, roi_rgb):
    best = None
    accepted = []

    print("\nChecking SAM 2 masks with package/product rules + exact box rules...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

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
            accepted.append(package_result)

            if best is None or package_result["score"] > best["score"]:
                best = package_result

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
            accepted.append(box_result)

            if best is None or box_result["score"] > best["score"]:
                best = box_result

    return best, accepted


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


def classify_package_type(cfg, depth_roi, package_mask, info):
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

    return {
        "package_type": package_type,
        "package_type_confidence": float(confidence),
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


def detect_product_inside_polymailer(cfg, depth_roi, package_mask, center_full_depth_mm, intrinsics):
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
        "reason": "not_run",
    }

    if not cfg.product_inside_enable:
        empty["reason"] = "disabled"
        return empty

    package_mask = package_mask.astype(np.uint8)
    package_area = int(package_mask.sum())

    if package_area <= 0:
        empty["reason"] = "empty_package_mask"
        return empty

    inner_mask = erode_mask(package_mask, cfg.product_inside_edge_erode_px)

    if int(inner_mask.sum()) < cfg.product_inside_min_valid_pixels:
        inner_mask = package_mask.copy()

    valid_inner = (
        (inner_mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    values = depth_roi[valid_inner].astype(np.float32)

    if values.size < cfg.product_inside_min_valid_pixels:
        empty["reason"] = "not_enough_depth"
        return empty

    poly_surface_depth = float(np.median(values))

    closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
    closer_mm[valid_inner] = poly_surface_depth - depth_roi[valid_inner].astype(np.float32)

    product_candidate = (
        valid_inner
        & (closer_mm >= cfg.product_inside_min_closer_than_poly_mm)
        & (closer_mm <= cfg.product_inside_max_closer_than_poly_mm)
    ).astype(np.uint8)

    product_candidate = close_mask(
        product_candidate,
        cfg.product_inside_close_kernel_px,
        iterations=1,
    )

    product_candidate = dilate_mask(product_candidate, cfg.product_inside_dilate_px)
    product_candidate = largest_component(product_candidate)

    if product_candidate is None:
        empty["reason"] = "no_component"
        return empty

    product_area = int(product_candidate.sum())
    area_ratio = product_area / max(package_area, 1)

    if area_ratio < cfg.product_inside_min_area_ratio_of_package:
        empty["reason"] = "too_small"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    if area_ratio > cfg.product_inside_max_area_ratio_of_package:
        empty["reason"] = "too_large"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    center_roi = get_mask_center(product_candidate)

    if center_roi is None:
        empty["reason"] = "no_center"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    cx_roi, cy_roi = center_roi
    center_full = (int(cfg.roi_x1 + cx_roi), int(cfg.roi_y1 + cy_roi))

    product_values = depth_roi[
        (product_candidate == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if product_values.size >= cfg.min_surface_depth_count:
        raw_product_depth = float(np.median(product_values))
        product_depth_mm = raw_product_depth + cfg.measurement_depth_offset_mm
    else:
        product_depth_mm = center_full_depth_mm

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        product_depth_mm,
        intrinsics,
    )

    return {
        "found": True,
        "center_roi": (int(cx_roi), int(cy_roi)),
        "center_full": center_full,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "depth_mm": product_depth_mm,
        "area_px": product_area,
        "area_ratio_of_package": float(area_ratio),
        "mask": product_candidate,
        "reason": "ok",
    }


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

    print("Running SAM 2 package segmentation on ROI...")

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    print(f"Generated {len(masks)} masks inside ROI")

    best, accepted = choose_best_package_mask(cfg, masks, roi_rgb)

    if best is None:
        print("No valid package/product or cardboard box mask found.")
        return None

    classification = classify_package_type(
        cfg,
        depth_roi=depth_class_roi,
        package_mask=best["mask"],
        info=best,
    )

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
        product_inside = detect_product_inside_polymailer(
            cfg,
            depth_roi=depth_measure_roi,
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
            "reason": "not_polymailer",
        }

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
