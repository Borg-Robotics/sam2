"""Object-mode detection logic: generic single-object segmentation + center
depth.

Ported verbatim from run_object()/object_detection_final.py with the
module-level constants replaced by fields of an ObjectConfig passed as the
first argument. Function bodies are otherwise unchanged.
"""

import cv2
import numpy as np
import torch

from .package import get_rotated_box_from_mask, pixel_to_camera_xy_mm


def mask_touches_roi_border(mask, margin_px):
    h, w = mask.shape[:2]

    top = mask[:margin_px, :]
    bottom = mask[h - margin_px:h, :]
    left = mask[:, :margin_px]
    right = mask[:, w - margin_px:w]

    return (
        np.any(top == 1)
        or np.any(bottom == 1)
        or np.any(left == 1)
        or np.any(right == 1)
    )


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


def clean_mask(cfg, mask):
    mask = mask.astype(np.uint8)

    if not cfg.use_mask_cleanup:
        return mask

    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.mask_close_kernel_px, cfg.mask_close_kernel_px),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=cfg.mask_close_iterations,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    if cfg.use_convex_hull:
        contours, _ = cv2.findContours(
            cleaned.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)

            hull_mask = np.zeros_like(cleaned, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)

            hull_area = int(hull_mask.sum())

            if hull_area <= original_area * cfg.max_cleaned_area_growth:
                return hull_mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def estimate_object_dimensions_mm(cfg, obj, depth_mm, intrinsics):
    """Object footprint and height in millimetres.

    Mirrors the box mode: length/width are the sides of the minimum-area
    rectangle scaled from pixels using the measured face depth, and height is
    how far the face stands above the base surface (cfg.base_depth_mm).
    angle_deg comes from the same rectangle fit, so it always describes the
    same edge as `length`.
    """
    empty = {
        "length_mm": None,
        "width_mm": None,
        "height_mm": None,
        "angle_deg": None,
    }

    rotated = get_rotated_box_from_mask(obj["mask"])

    if rotated is None:
        return empty

    angle_deg = float(rotated["angle_deg"])

    if depth_mm is None or intrinsics is None:
        return {**empty, "angle_deg": angle_deg}

    width_mm = (rotated["width_px"] * depth_mm) / intrinsics["fx"]
    height_px_mm = (rotated["height_px"] * depth_mm) / intrinsics["fy"]

    height_mm = cfg.base_depth_mm - depth_mm

    if height_mm < 0:
        height_mm = 0.0

    return {
        "length_mm": float(max(width_mm, height_px_mm)),
        "width_mm": float(min(width_mm, height_px_mm)),
        "height_mm": float(height_mm),
        "angle_deg": angle_deg,
    }


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


def choose_best_object_mask(cfg, masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    best = None

    if cfg.debug_print_masks:
        print()
        print("Checking SAM 2 object masks...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(cfg, raw_mask)

        if cfg.reject_masks_touching_roi_border:
            if mask_touches_roi_border(mask, cfg.roi_border_margin_px):
                continue

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.min_area_ratio:
            continue

        if area_ratio > cfg.max_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        bbox_area = bw * bh
        rectangularity = area / max(bbox_area, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.min_rectangularity:
            continue

        if aspect_ratio > cfg.max_aspect_ratio:
            continue

        center = get_mask_center(mask)

        if center is None:
            continue

        cx, cy = center

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_area_ratio) / cfg.target_area_ratio,
            1.0,
        )

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        score = (
            cfg.center_score_weight * center_score
            + cfg.area_score_weight * area_score
            + cfg.rect_score_weight * rectangularity
            + cfg.sam_iou_score_weight * sam_iou
            + cfg.sam_stability_score_weight * sam_stability
        )

        if cfg.debug_print_masks:
            print(
                f"mask={i:03d} "
                f"score={score:.3f} "
                f"area={area_ratio:.3f} "
                f"rect={rectangularity:.3f} "
                f"aspect={aspect_ratio:.2f} "
                f"center={center_score:.2f} "
                f"sam_iou={sam_iou:.2f} "
                f"stable={sam_stability:.2f} "
                f"bbox=({x},{y},{bw},{bh})"
            )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "center_roi": center,
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "sam_iou": sam_iou,
            "sam_stability": sam_stability,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_object_segmentation(
    cfg,
    frame_bgr,
    depth_aligned,
    mask_generator,
    intrinsics=None,
):
    """Run SAM2 on the ROI, pick the best object mask, and measure its center
    depth. Returns a result dict or None when no valid mask is found.

    With intrinsics the center is also projected into camera-frame millimetres
    (x right, y down, z = the measured distance); without them center_x_mm and
    center_y_mm come back None."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi = depth_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    obj = choose_best_object_mask(cfg, masks, roi_rgb)

    if obj is None:
        return None

    center_full = (
        cfg.roi_x1 + obj["center_roi"][0],
        cfg.roi_y1 + obj["center_roi"][1],
    )

    distance_mm, depth_count = center_depth_mm(
        cfg,
        depth_roi,
        obj["mask"],
        obj["center_roi"],
    )

    dimensions = estimate_object_dimensions_mm(
        cfg,
        obj,
        distance_mm,
        intrinsics,
    )
    angle_deg = dimensions["angle_deg"]

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        distance_mm,
        intrinsics,
    )

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "object": obj,
        "center_full": center_full,
        "distance_mm": distance_mm,
        "depth_count": depth_count,
        "angle_deg": angle_deg,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "dimensions": dimensions,
        # Convenience scalars exposed in final_output for the result dataclass.
        "final_output": {
            "center_pixel_u": int(center_full[0]),
            "center_pixel_v": int(center_full[1]),
            "distance_mm": (
                float(distance_mm) if distance_mm is not None else None
            ),
            "depth_count": int(depth_count),
            "angle_deg": angle_deg,
            "center_x_mm": center_x_mm,
            "center_y_mm": center_y_mm,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["width_mm"],
            "height_mm": dimensions["height_mm"],
        },
    }
