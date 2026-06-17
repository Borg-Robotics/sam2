"""Polymailer-mode detection logic: padded-envelope HSV mask scoring, depth
measurement and product-bulge detection.

Ported verbatim from run_polymailer()/polymailer_final.py with the module-level
constants replaced by fields of a PolymailerConfig passed as the first argument.
Generic, config-free helpers are reused from detection.package.
"""

import cv2
import numpy as np
import torch

from ..visualization.polymailer import make_polymailer_depth_heatmap
from .package import (
    close_mask,
    dilate_mask,
    erode_mask,
    get_mask_center,
    get_rotated_box_from_mask,
    largest_component,
    pixel_to_camera_xy_mm,
)


def open_mask(mask, kernel_px=7, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )


def clean_mask(cfg, mask):
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask.astype(np.uint8)

    cleaned = close_mask(
        mask,
        cfg.poly_mask_close_kernel_px,
        cfg.poly_mask_close_iterations,
    )
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.poly_max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask.astype(np.uint8)


def get_component_near_center(mask, center):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    cx, cy = center
    best_label = None
    best_score = None

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        bx = x + w / 2
        by = y + h / 2

        dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
        score = area - 2.0 * dist

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    return (labels == best_label).astype(np.uint8)


def center_depth_mm(cfg, depth_roi, mask, center_roi):
    cx, cy = center_roi
    h, w = depth_roi.shape[:2]

    radius = cfg.center_depth_radius_px

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    patch_depth = depth_roi[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    values = patch_depth[
        (patch_mask == 1)
        & (patch_depth > cfg.min_valid_depth_mm)
        & (patch_depth < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if values.size < cfg.min_center_depth_count:
        values = patch_depth[
            (patch_depth > cfg.min_valid_depth_mm)
            & (patch_depth < cfg.max_valid_depth_mm)
        ].astype(np.float32)

    if values.size < cfg.min_center_depth_count:
        return None, 0

    return float(np.median(values)), int(values.size)


def estimate_polymailer_depth_mm(cfg, polymailer_face_depth_mm):
    if polymailer_face_depth_mm is None:
        return None

    depth_mm = cfg.base_depth_mm - polymailer_face_depth_mm

    if depth_mm < 0:
        depth_mm = 0.0

    return float(depth_mm)


def estimate_polymailer_dimensions_mm(cfg, poly, intrinsics):
    empty = {
        "length_mm": None,
        "width_mm": None,
        "angle_deg": None,
        "points_full": None,
    }

    if intrinsics is None:
        return empty

    measure_mask = erode_mask(poly["mask"], cfg.poly_measure_erode_px)

    if int(measure_mask.sum()) < 100:
        measure_mask = poly["mask"].copy()

    rotated = get_rotated_box_from_mask(measure_mask)

    if rotated is None:
        return empty

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    # POLY_MEASURE_DEPTH_MM == BASE_DEPTH_MM in the original.
    raw_width_mm = (w_px * cfg.base_depth_mm) / fx
    raw_height_mm = (h_px * cfg.base_depth_mm) / fy

    length_mm = max(raw_width_mm, raw_height_mm) * cfg.poly_size_scale
    width_mm = min(raw_width_mm, raw_height_mm) * cfg.poly_size_scale

    points_roi = rotated["points_roi"]
    points_full = points_roi.copy()
    points_full[:, 0] += cfg.roi_x1
    points_full[:, 1] += cfg.roi_y1

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "angle_deg": float(rotated["angle_deg"]),
        "points_full": points_full.tolist(),
    }


def estimate_product_inside_polymailer(cfg, depth_roi, poly_mask, poly_center_roi):
    inner_mask = erode_mask(poly_mask, cfg.poly_inner_erode_px)

    if int(inner_mask.sum()) < 100:
        inner_mask = poly_mask.copy()

    valid_inside = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
        & (inner_mask == 1)
    )

    inside_values = depth_roi[valid_inside].astype(np.float32)

    if inside_values.size < 100:
        return {
            "found": False,
            "reason": "not_enough_depth",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": None,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    surface_depth_mm = float(np.percentile(inside_values, cfg.poly_surface_depth_percentile))

    product_candidate = (
        valid_inside
        & (depth_roi <= surface_depth_mm - cfg.poly_bulge_min_mm)
        & (depth_roi >= surface_depth_mm - cfg.poly_bulge_max_mm)
    ).astype(np.uint8)

    product_candidate = open_mask(product_candidate, cfg.poly_bulge_open_kernel_px, iterations=1)
    product_candidate = close_mask(product_candidate, cfg.poly_bulge_close_kernel_px, iterations=2)
    product_candidate = dilate_mask(product_candidate, cfg.poly_bulge_dilate_px)

    product_candidate[poly_mask == 0] = 0

    product_mask = get_component_near_center(product_candidate, poly_center_roi)

    if product_mask is None or int(product_mask.sum()) == 0:
        return {
            "found": False,
            "reason": "no_product_bulge",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": surface_depth_mm,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    product_area = int(product_mask.sum())
    poly_area = int(poly_mask.sum())
    area_ratio_of_poly = product_area / max(poly_area, 1)

    if area_ratio_of_poly < cfg.min_product_area_ratio_of_poly:
        found = False
        reason = "product_area_too_small"
    elif area_ratio_of_poly > cfg.max_product_area_ratio_of_poly:
        found = False
        reason = "product_area_too_large"
    else:
        found = True
        reason = "ok"

    product_values = depth_roi[
        (product_mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if product_values.size > 0:
        product_depth_mm = float(np.percentile(product_values, 20))
        bulge_height_mm = float(surface_depth_mm - product_depth_mm)
    else:
        product_depth_mm = None
        bulge_height_mm = None

    center_roi = get_mask_center(product_mask)
    bbox = cv2.boundingRect(product_mask.astype(np.uint8))

    return {
        "found": bool(found),
        "reason": reason,
        "mask": product_mask.astype(np.uint8),
        "surface_depth_mm": surface_depth_mm,
        "product_depth_mm": product_depth_mm,
        "bulge_height_mm": bulge_height_mm,
        "area_ratio_of_poly": float(area_ratio_of_poly),
        "bbox": bbox,
        "center_roi": center_roi,
    }


def choose_polymailer_mask(cfg, masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(cfg, raw_mask)

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.min_poly_area_ratio:
            continue

        if area_ratio > cfg.max_poly_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.min_poly_rectangularity:
            continue

        if aspect_ratio > cfg.max_poly_aspect_ratio:
            continue

        masked_hsv = hsv[mask == 1]

        if masked_hsv.size == 0:
            continue

        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))

        if mean_v < cfg.min_poly_value:
            continue

        hue_score = 1.0 - min(abs(mean_h - 18.0) / 25.0, 1.0)
        sat_score = 1.0 - min(abs(mean_s - 70.0) / 90.0, 1.0)
        val_score = 1.0 - min(abs(mean_v - 170.0) / 100.0, 1.0)

        color_score = 0.50 * hue_score + 0.25 * sat_score + 0.25 * val_score

        if color_score < cfg.min_poly_color_score:
            continue

        cx = x + bw / 2
        cy = y + bh / 2

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)

        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_poly_area_ratio) / cfg.target_poly_area_ratio,
            1.0,
        )

        score = (
            cfg.poly_color_score_weight * color_score
            + cfg.poly_rect_score_weight * rectangularity
            + cfg.poly_area_score_weight * area_score
            + cfg.poly_center_score_weight * center_score
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "color_score": float(color_score),
            "mean_hsv": (mean_h, mean_s, mean_v),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_polymailer(cfg, frame_bgr, depth_aligned, mask_generator, intrinsics):
    """Run SAM2 on the ROI, pick the best polymailer mask, measure its face
    depth + dimensions, and find any product bulge inside. Returns a result
    dict or None when no valid polymailer mask is found."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi = depth_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    poly = choose_polymailer_mask(cfg, masks, roi_rgb)

    if poly is None:
        return None

    poly_center_roi = get_mask_center(poly["mask"])

    if poly_center_roi is None:
        return None

    polymailer_face_depth_mm, depth_count = center_depth_mm(
        cfg,
        depth_roi,
        poly["mask"],
        poly_center_roi,
    )

    polymailer_depth_mm = estimate_polymailer_depth_mm(cfg, polymailer_face_depth_mm)

    dimensions = estimate_polymailer_dimensions_mm(cfg, poly, intrinsics)

    product = estimate_product_inside_polymailer(
        cfg,
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        poly_center_roi=poly_center_roi,
    )

    depth_heatmap = make_polymailer_depth_heatmap(
        cfg,
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        product_mask=product["mask"],
        surface_depth_mm=product["surface_depth_mm"],
    )

    poly_center_full = (
        cfg.roi_x1 + poly_center_roi[0],
        cfg.roi_y1 + poly_center_roi[1],
    )

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        poly_center_full,
        polymailer_face_depth_mm,
        intrinsics,
    )

    if product["center_roi"] is not None:
        product_center_full = (
            cfg.roi_x1 + product["center_roi"][0],
            cfg.roi_y1 + product["center_roi"][1],
        )
    else:
        product_center_full = None

    product_inside_center_face_depth = product["product_depth_mm"]

    product_inside_center_x_mm, product_inside_center_y_mm = pixel_to_camera_xy_mm(
        product_center_full,
        product_inside_center_face_depth,
        intrinsics,
    )

    if product["bbox"] is not None:
        px, py, pw, ph = product["bbox"]
        product_bbox_full = (
            cfg.roi_x1 + px,
            cfg.roi_y1 + py,
            pw,
            ph,
        )
    else:
        product_bbox_full = None

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "poly": poly,
        "poly_center_roi": poly_center_roi,
        "poly_center_full": poly_center_full,
        "polymailer_face_depth_mm": polymailer_face_depth_mm,
        "polymailer_depth_mm": polymailer_depth_mm,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "depth_count": depth_count,
        "dimensions": dimensions,
        "product": product,
        "product_center_full": product_center_full,
        "product_bbox_full": product_bbox_full,
        "product_inside_center_x_mm": product_inside_center_x_mm,
        "product_inside_center_y_mm": product_inside_center_y_mm,
        "product_inside_center_face_depth": product_inside_center_face_depth,
        "depth_heatmap": depth_heatmap,
        "final_output": {
            "polymailer_face_depth_mm": polymailer_face_depth_mm,
            "polymailer_depth_mm": polymailer_depth_mm,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["width_mm"],
            "angle_deg": dimensions["angle_deg"],
            "center_x_mm": center_x_mm,
            "center_y_mm": center_y_mm,
            "product_inside_found": bool(product["found"]),
            "product_inside_center_x_mm": product_inside_center_x_mm,
            "product_inside_center_y_mm": product_inside_center_y_mm,
            "product_inside_center_face_depth": product_inside_center_face_depth,
        },
    }
