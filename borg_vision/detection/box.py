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
    get_mask_center,
    get_rotated_box_from_mask,
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


def choose_cardboard_box_mask(cfg, masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

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
            cfg.box_color_score_weight * color_score
            + cfg.box_rect_score_weight * rectangularity
            + cfg.box_area_score_weight * area_score
            + cfg.box_center_score_weight * center_score
            + cfg.box_sam_iou_score_weight * sam_iou
            + cfg.box_sam_stability_score_weight * sam_stability
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "center_roi": center_roi,
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
