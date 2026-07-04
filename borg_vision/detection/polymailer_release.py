"""Pure detection functions for the polymailer product-release monitor.

Ported verbatim from polymailer_product_release.py (the "moving-slit
product-exit test" prototype), with every module-level tuning constant
replaced by the matching PolymailerReleaseConfig field (threaded through as
the `cfg` first argument). ROI-sized masks derive their dimensions from
cfg.roi_* — all masks and points live in ROI coordinates.

Notes vs the prototype:
- close_mask / largest_component / mask_iou / mask_overlap_fraction are
  reused from detection.package (verified behaviorally identical).
- clean_mask / mask_center are the prototype's own variants (kernel 15,
  growth cap 2.5) and intentionally NOT merged with the polymailer mode's.
- The prototype defined crop_edge_touch_fraction twice; only the second
  (shadowing) definition is kept, matching runtime behavior.
"""

import time

import cv2
import numpy as np

from .package import (
    close_mask,
    largest_component,
    mask_iou,
    mask_overlap_fraction,
)


def clean_mask(mask):
    raw = mask.astype(np.uint8)
    original_area = int(raw.sum())

    if original_area <= 0:
        return raw

    cleaned = close_mask(raw, kernel_px=15, iterations=1)
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return raw

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * 2.5:
        return cleaned.astype(np.uint8)

    return raw

def mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] <= 0:
        return None

    return np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=np.float32,
    )

def erode_release_core(mask, radius_px, minimum_area_px):
    """Remove uncertain boundary pixels for full-release verification."""
    source = mask.astype(np.uint8)

    if radius_px <= 0:
        return source

    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    core = cv2.erode(source, kernel, iterations=1)

    if int(core.sum()) < minimum_area_px:
        return source

    return core.astype(np.uint8)

def full_release_core_metrics(cfg, candidate_mask, poly_mask, slit_geometry):
    """Measure separation while ignoring thin SAM/flow outline overlap."""
    candidate_core = erode_release_core(
        candidate_mask,
        cfg.full_release_product_core_erode_px,
        cfg.full_release_min_core_area_px,
    )
    poly_core = erode_release_core(
        poly_mask,
        cfg.full_release_poly_core_erode_px,
        cfg.full_release_min_core_area_px,
    )

    slit_metrics = signed_slit_metrics(cfg, candidate_core, slit_geometry)

    # Distance from every product-core pixel to the nearest polymailer-core
    # pixel. A positive gap prevents a half-emerged product whose visible mask
    # simply stops at the slit from being called fully released.
    if int(poly_core.sum()) > 0 and int(candidate_core.sum()) > 0:
        distance_from_poly = cv2.distanceTransform(
            (poly_core == 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        core_poly_gap_px = float(np.min(distance_from_poly[candidate_core == 1]))
    else:
        core_poly_gap_px = 0.0

    return {
        "core_poly_overlap": float(
            mask_overlap_fraction(candidate_core, poly_core)
        ),
        "core_poly_gap_px": core_poly_gap_px,
        "core_outward_fraction": float(
            slit_metrics["outward_fraction"]
        ),
        "core_inward_fraction": float(
            slit_metrics["inward_fraction"]
        ),
        "core_line_band_fraction": float(
            slit_metrics["line_band_fraction"]
        ),
        "core_min_line_distance_px": float(
            slit_metrics["min_line_distance_px"]
        ),
        "core_min_signed_distance_px": float(
            slit_metrics["min_signed_distance_px"]
        ),
        "core_trailing_edge_clearance_px": float(
            slit_metrics["trailing_edge_clearance_px"]
        ),
    }

def masked_mean_hsv(roi_rgb, mask):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    values = hsv[mask == 1]

    if values.size == 0:
        return None

    return (
        float(np.mean(values[:, 0])),
        float(np.mean(values[:, 1])),
        float(np.mean(values[:, 2])),
    )

def hsv_similarity(hsv_a, hsv_b):
    if hsv_a is None or hsv_b is None:
        return 0.0

    h_a, s_a, v_a = hsv_a
    h_b, s_b, v_b = hsv_b

    hue_distance = abs(h_a - h_b)
    hue_distance = min(hue_distance, 180.0 - hue_distance) / 90.0
    saturation_distance = abs(s_a - s_b) / 255.0
    value_distance = abs(v_a - v_b) / 255.0

    distance = (
        0.50 * hue_distance
        + 0.25 * saturation_distance
        + 0.25 * value_distance
    )

    return float(max(0.0, 1.0 - min(distance, 1.0)))

def choose_polymailer_mask(cfg, masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2.0
    roi_cy = h / 2.0
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    best = None

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if not (cfg.min_poly_area_ratio <= area_ratio <= cfg.max_poly_area_ratio):
            continue

        x, y, width, height = cv2.boundingRect(mask)

        if width <= 0 or height <= 0:
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

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
        color_score = (
            0.50 * hue_score
            + 0.25 * sat_score
            + 0.25 * val_score
        )

        if color_score < cfg.min_poly_color_score:
            continue

        center_x = x + width / 2.0
        center_y = y + height / 2.0
        distance = np.hypot(center_x - roi_cx, center_y - roi_cy)
        max_distance = np.hypot(roi_cx, roi_cy)
        center_score = 1.0 - min(distance / max(max_distance, 1.0), 1.0)
        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_poly_area_ratio)
            / cfg.target_poly_area_ratio,
            1.0,
        )

        score = (
            3.0 * color_score
            + 1.5 * rectangularity
            + 1.2 * area_score
            + 1.0 * center_score
            + 0.25 * float(sam_mask.get("predicted_iou", 0.0))
            + 0.25 * float(sam_mask.get("stability_score", 0.0))
        )

        candidate = {
            "index": index,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, width, height),
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "color_score": float(color_score),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best

def make_polymailer_refresh_crop(cfg, poly_mask):
    """Build a padded crop around the currently tracked complete bag mask."""
    if poly_mask is None or int(poly_mask.sum()) <= 0:
        return (0, 0, (cfg.roi_x2 - cfg.roi_x1), (cfg.roi_y2 - cfg.roi_y1))

    x, y, width, height = cv2.boundingRect(poly_mask.astype(np.uint8))
    x1 = max(0, x - cfg.poly_refresh_crop_padding_px)
    y1 = max(0, y - cfg.poly_refresh_crop_padding_px)
    x2 = min((cfg.roi_x2 - cfg.roi_x1), x + width + cfg.poly_refresh_crop_padding_px)
    y2 = min((cfg.roi_y2 - cfg.roi_y1), y + height + cfg.poly_refresh_crop_padding_px)

    if x2 - x1 < cfg.poly_refresh_crop_min_size_px:
        missing = cfg.poly_refresh_crop_min_size_px - (x2 - x1)
        x1 = max(0, x1 - missing // 2)
        x2 = min((cfg.roi_x2 - cfg.roi_x1), x2 + missing - missing // 2)
        if x2 - x1 < cfg.poly_refresh_crop_min_size_px:
            if x1 == 0:
                x2 = min((cfg.roi_x2 - cfg.roi_x1), cfg.poly_refresh_crop_min_size_px)
            else:
                x1 = max(0, (cfg.roi_x2 - cfg.roi_x1) - cfg.poly_refresh_crop_min_size_px)

    if y2 - y1 < cfg.poly_refresh_crop_min_size_px:
        missing = cfg.poly_refresh_crop_min_size_px - (y2 - y1)
        y1 = max(0, y1 - missing // 2)
        y2 = min((cfg.roi_y2 - cfg.roi_y1), y2 + missing - missing // 2)
        if y2 - y1 < cfg.poly_refresh_crop_min_size_px:
            if y1 == 0:
                y2 = min((cfg.roi_y2 - cfg.roi_y1), cfg.poly_refresh_crop_min_size_px)
            else:
                y1 = max(0, (cfg.roi_y2 - cfg.roi_y1) - cfg.poly_refresh_crop_min_size_px)

    return (int(x1), int(y1), int(x2), int(y2))

def make_slit_band_mask(cfg, slit_points):
    band = np.zeros(((cfg.roi_y2 - cfg.roi_y1), (cfg.roi_x2 - cfg.roi_x1)), dtype=np.uint8)

    if slit_points is None:
        return band

    point_a = tuple(np.round(slit_points[0]).astype(int))
    point_b = tuple(np.round(slit_points[1]).astype(int))
    cv2.line(
        band,
        point_a,
        point_b,
        1,
        thickness=cfg.poly_refresh_slit_band_width_px,
        lineType=cv2.LINE_AA,
    )
    return (band > 0).astype(np.uint8)

def choose_live_polymailer_refresh_mask(
    cfg,
    masks,
    roi_rgb,
    predicted_poly_mask,
    baseline_poly_hsv,
    slit_points,
    crop_box,
):
    """Select the current bag mask using color plus optical-flow prediction."""
    if predicted_poly_mask is None:
        return None

    predicted = predicted_poly_mask.astype(np.uint8)
    predicted_area = int(predicted.sum())
    predicted_center = mask_center(predicted)

    if predicted_area <= 0 or predicted_center is None:
        return None

    slit_band = make_slit_band_mask(cfg, slit_points)
    best = None

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area <= 0:
            continue

        area_ratio = area / max(predicted_area, 1)

        if not (
            cfg.poly_refresh_min_area_ratio_to_predicted
            <= area_ratio
            <= cfg.poly_refresh_max_area_ratio_to_predicted
        ):
            continue

        candidate_center = mask_center(mask)

        if candidate_center is None:
            continue

        center_distance = float(
            np.linalg.norm(candidate_center - predicted_center)
        )

        if center_distance > cfg.poly_refresh_max_center_distance_px:
            continue

        intersection = int(
            np.count_nonzero((mask == 1) & (predicted == 1))
        )
        candidate_inside_predicted = intersection / max(area, 1)
        predicted_inside_candidate = intersection / max(predicted_area, 1)
        iou = mask_iou(mask, predicted)

        if (
            candidate_inside_predicted
            < cfg.poly_refresh_min_candidate_inside_predicted
            and iou < cfg.poly_refresh_min_iou
        ):
            continue

        candidate_hsv = masked_mean_hsv(roi_rgb, mask)
        color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )

        if color_similarity < cfg.poly_refresh_min_color_similarity:
            continue

        slit_band_pixels = int(
            np.count_nonzero((mask == 1) & (slit_band == 1))
        )

        if slit_band_pixels < cfg.poly_refresh_min_slit_band_pixels:
            continue

        edge_touch = crop_edge_touch_fraction(cfg, mask, crop_box)

        if edge_touch > cfg.poly_refresh_max_crop_edge_touch_fraction:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        center_score = 1.0 - min(
            center_distance / cfg.poly_refresh_max_center_distance_px,
            1.0,
        )
        area_score = 1.0 - min(abs(area_ratio - 1.0), 1.0)
        slit_score = min(
            slit_band_pixels
            / max(cfg.poly_refresh_min_slit_band_pixels * 5, 1),
            1.0,
        )

        score = (
            3.6 * color_similarity
            + 2.2 * candidate_inside_predicted
            + 1.3 * iou
            + 0.8 * predicted_inside_candidate
            + 0.8 * center_score
            + 0.6 * area_score
            + 0.6 * slit_score
            + 0.25 * sam_iou
            + 0.25 * sam_stability
            - 0.7 * edge_touch
        )

        candidate = {
            "index": index,
            "score": float(score),
            "mask": mask.astype(np.uint8),
            "area_ratio": float(area_ratio),
            "candidate_inside_predicted": float(
                candidate_inside_predicted
            ),
            "predicted_inside_candidate": float(
                predicted_inside_candidate
            ),
            "iou": float(iou),
            "color_similarity": float(color_similarity),
            "center_distance_px": float(center_distance),
            "slit_band_pixels": int(slit_band_pixels),
            "edge_touch_fraction": float(edge_touch),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best

def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))

    if length < 1e-6:
        return None

    return vector / length

def make_slit_geometry(
    cfg,
    poly_mask,
    slit_points,
    baseline_slit_midpoint=None,
    baseline_slit_length=None,
    expansion_state=None,
):
    if poly_mask is None or slit_points is None:
        return None

    point_a = slit_points[0].astype(np.float32)
    point_b = slit_points[1].astype(np.float32)
    tangent = normalize_vector(point_b - point_a)
    center = mask_center(poly_mask)

    if tangent is None or center is None:
        return None

    midpoint = (point_a + point_b) * 0.5
    outward = normalize_vector(midpoint - center)

    if outward is None:
        outward = np.array(
            [-tangent[1], tangent[0]],
            dtype=np.float32,
        )

    current_slit_length = float(
        np.linalg.norm(point_b - point_a)
    )

    pose_scale = 1.0
    pull_away_px = 0.0
    lateral_motion_px = 0.0
    total_motion_px = 0.0

    outward_extra_px = 0.0
    side_extra_px = 0.0
    inward_extra_px = 0.0

    dynamic_ready = (
        cfg.dynamic_corridor_enable
        and baseline_slit_midpoint is not None
        and baseline_slit_length is not None
        and baseline_slit_length > 1e-6
        and expansion_state is not None
    )

    if dynamic_ready:
        pose_scale = float(
            np.clip(
                current_slit_length / float(baseline_slit_length),
                cfg.dynamic_corridor_min_pose_scale,
                cfg.dynamic_corridor_max_pose_scale,
            )
        )

        movement = (
            midpoint.astype(np.float32)
            - np.asarray(
                baseline_slit_midpoint,
                dtype=np.float32,
            )
        )

        total_motion_px = float(np.linalg.norm(movement))

        # Pulling the bag away from the opening normally moves the slit in the
        # direction opposite its outward vector. Grow most strongly in that
        # direction because the product tends to remain near the table.
        pull_away_px = max(
            0.0,
            -float(np.dot(movement, outward)),
        )

        # Sideways movement grows both ends of the corridor.
        lateral_motion_px = abs(
            float(np.dot(movement, tangent))
        )

        requested_outward_extra = min(
            cfg.dynamic_corridor_max_extra_outward_px,
            (
                pull_away_px * cfg.dynamic_corridor_pull_factor
                + total_motion_px
                * cfg.dynamic_corridor_total_outward_factor
            ),
        )

        requested_side_extra = min(
            cfg.dynamic_corridor_max_extra_side_px,
            (
                lateral_motion_px * cfg.dynamic_corridor_lateral_factor
                + total_motion_px
                * cfg.dynamic_corridor_total_side_factor
            ),
        )

        requested_inward_extra = min(
            cfg.dynamic_corridor_max_extra_inward_px,
            (
                total_motion_px
                * cfg.dynamic_corridor_total_inward_factor
            ),
        )

        now_monotonic = time.monotonic()
        last_update_time = expansion_state.get(
            "_last_update_time",
            None,
        )

        if last_update_time is None:
            update_dt = 1.0 / max(float(cfg.fps), 1.0)
        else:
            update_dt = float(
                np.clip(
                    now_monotonic - float(last_update_time),
                    0.0,
                    0.12,
                )
            )

        expansion_state["_last_update_time"] = now_monotonic

        current_outward = float(
            expansion_state.get("outward_extra_px", 0.0)
        )
        current_side = float(
            expansion_state.get("side_extra_px", 0.0)
        )
        current_inward = float(
            expansion_state.get("inward_extra_px", 0.0)
        )
        current_pose_scale = float(
            expansion_state.get("pose_scale", 1.0)
        )

        if cfg.dynamic_corridor_grow_only:
            target_outward = max(
                current_outward,
                float(requested_outward_extra),
            )
            target_side = max(
                current_side,
                float(requested_side_extra),
            )
            target_inward = max(
                current_inward,
                float(requested_inward_extra),
            )
            target_pose_scale = max(
                current_pose_scale,
                float(pose_scale),
            )
        else:
            target_outward = float(requested_outward_extra)
            target_side = float(requested_side_extra)
            target_inward = float(requested_inward_extra)
            target_pose_scale = float(pose_scale)

        max_outward_step = (
            cfg.dynamic_corridor_outward_growth_px_per_second
            * update_dt
        )
        max_side_step = (
            cfg.dynamic_corridor_side_growth_px_per_second
            * update_dt
        )
        max_inward_step = (
            cfg.dynamic_corridor_inward_growth_px_per_second
            * update_dt
        )
        max_pose_step = (
            cfg.dynamic_corridor_pose_growth_per_second
            * update_dt
        )

        expansion_state["outward_extra_px"] = float(
            current_outward
            + np.clip(
                target_outward - current_outward,
                -max_outward_step,
                max_outward_step,
            )
        )
        expansion_state["side_extra_px"] = float(
            current_side
            + np.clip(
                target_side - current_side,
                -max_side_step,
                max_side_step,
            )
        )
        expansion_state["inward_extra_px"] = float(
            current_inward
            + np.clip(
                target_inward - current_inward,
                -max_inward_step,
                max_inward_step,
            )
        )
        expansion_state["pose_scale"] = float(
            current_pose_scale
            + np.clip(
                target_pose_scale - current_pose_scale,
                -max_pose_step,
                max_pose_step,
            )
        )

        expansion_state["pull_away_px"] = float(
            pull_away_px
        )
        expansion_state["lateral_motion_px"] = float(
            lateral_motion_px
        )
        expansion_state["total_motion_px"] = float(
            total_motion_px
        )

        outward_extra_px = float(
            expansion_state.get("outward_extra_px", 0.0)
        )
        side_extra_px = float(
            expansion_state.get("side_extra_px", 0.0)
        )
        inward_extra_px = float(
            expansion_state.get("inward_extra_px", 0.0)
        )
        pose_scale = float(
            expansion_state.get("pose_scale", pose_scale)
        )

    side_margin_px = (
        cfg.slit_side_margin_px * pose_scale
        + side_extra_px
    )

    outward_distance_px = (
        cfg.slit_outward_distance_px * pose_scale
        + outward_extra_px
    )

    inward_distance_px = (
        cfg.slit_inward_distance_px * pose_scale
        + inward_extra_px
    )

    # Keep the blue contact corridor local to the actual slit. Only the yellow
    # SAM-search corridor receives the large pull-away expansion.
    contact_side_margin_px = (
        cfg.slit_side_margin_px * pose_scale
    )
    contact_outward_distance_px = (
        cfg.slit_contact_outward_px * pose_scale
    )
    contact_inward_distance_px = (
        cfg.slit_contact_inward_px * pose_scale
    )

    side_a = point_a - tangent * side_margin_px
    side_b = point_b + tangent * side_margin_px

    contact_side_a = (
        point_a - tangent * contact_side_margin_px
    )
    contact_side_b = (
        point_b + tangent * contact_side_margin_px
    )

    raw_corridor_polygon = np.stack(
        [
            side_a + outward * outward_distance_px,
            side_b + outward * outward_distance_px,
            side_b - outward * inward_distance_px,
            side_a - outward * inward_distance_px,
        ]
    )

    corridor_polygon = raw_corridor_polygon

    if dynamic_ready:
        previous_polygon = expansion_state.get(
            "_smoothed_corridor_polygon",
            None,
        )

        if (
            isinstance(previous_polygon, np.ndarray)
            and previous_polygon.shape == raw_corridor_polygon.shape
        ):
            point_motion = float(
                np.max(
                    np.linalg.norm(
                        raw_corridor_polygon - previous_polygon,
                        axis=1,
                    )
                )
            )

            alpha = (
                cfg.dynamic_corridor_fast_motion_alpha
                if point_motion
                >= cfg.dynamic_corridor_fast_motion_threshold_px
                else cfg.dynamic_corridor_polygon_smooth_alpha
            )

            corridor_polygon = (
                (1.0 - alpha) * previous_polygon
                + alpha * raw_corridor_polygon
            ).astype(np.float32)

        expansion_state["_smoothed_corridor_polygon"] = (
            corridor_polygon.copy()
        )

    contact_polygon = np.stack(
        [
            (
                contact_side_a
                + outward * contact_outward_distance_px
            ),
            (
                contact_side_b
                + outward * contact_outward_distance_px
            ),
            (
                contact_side_b
                - outward * contact_inward_distance_px
            ),
            (
                contact_side_a
                - outward * contact_inward_distance_px
            ),
        ]
    )

    corridor_mask = polygon_to_mask(cfg, corridor_polygon)
    contact_mask = polygon_to_mask(cfg, contact_polygon)

    # Always keep the true slit/contact region inside the smoothed SAM crop.
    crop_source_mask = cv2.bitwise_or(
        corridor_mask,
        contact_mask,
    )
    crop_box = mask_bounding_crop(
    cfg,
        crop_source_mask,
        cfg.slit_crop_padding_px,
    )
    crop_box = quantize_crop_box(cfg, crop_box)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "midpoint": midpoint,
        "tangent": tangent,
        "outward": outward,
        "corridor_polygon": corridor_polygon,
        "contact_polygon": contact_polygon,
        "corridor_mask": corridor_mask,
        "contact_mask": contact_mask,
        "crop_box": crop_box,
        "dynamic_corridor_enabled": bool(dynamic_ready),
        "corridor_pose_scale": float(pose_scale),
        "corridor_pull_away_px": float(pull_away_px),
        "corridor_lateral_motion_px": float(
            lateral_motion_px
        ),
        "corridor_total_motion_px": float(
            total_motion_px
        ),
        "corridor_outward_extra_px": float(
            outward_extra_px
        ),
        "corridor_side_extra_px": float(side_extra_px),
        "corridor_inward_extra_px": float(
            inward_extra_px
        ),
        "corridor_outward_distance_px": float(
            outward_distance_px
        ),
        "corridor_side_margin_px": float(side_margin_px),
        "corridor_inward_distance_px": float(
            inward_distance_px
        ),
    }

def polygon_to_mask(cfg, polygon):
    mask = np.zeros(((cfg.roi_y2 - cfg.roi_y1), (cfg.roi_x2 - cfg.roi_x1)), dtype=np.uint8)
    polygon_int = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon_int, 1)
    return mask

def mask_bounding_crop(cfg, mask, padding):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return (0, 0, (cfg.roi_x2 - cfg.roi_x1), (cfg.roi_y2 - cfg.roi_y1))

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min((cfg.roi_x2 - cfg.roi_x1), int(xs.max()) + padding + 1)
    y2 = min((cfg.roi_y2 - cfg.roi_y1), int(ys.max()) + padding + 1)

    width = x2 - x1
    height = y2 - y1

    if width < cfg.slit_crop_min_size_px:
        center = (x1 + x2) // 2
        half = cfg.slit_crop_min_size_px // 2
        x1 = max(0, center - half)
        x2 = min((cfg.roi_x2 - cfg.roi_x1), x1 + cfg.slit_crop_min_size_px)
        x1 = max(0, x2 - cfg.slit_crop_min_size_px)

    if height < cfg.slit_crop_min_size_px:
        center = (y1 + y2) // 2
        half = cfg.slit_crop_min_size_px // 2
        y1 = max(0, center - half)
        y2 = min((cfg.roi_y2 - cfg.roi_y1), y1 + cfg.slit_crop_min_size_px)
        y1 = max(0, y2 - cfg.slit_crop_min_size_px)

    return (x1, y1, x2, y2)

def quantize_crop_box(cfg, crop_box, step_px=None):
    """Expand a crop outward to stable step-aligned boundaries."""
    if step_px is None:
        step_px = cfg.dynamic_corridor_crop_quantize_px
    x1, y1, x2, y2 = [int(value) for value in crop_box]
    step = max(1, int(step_px))

    x1 = max(0, (x1 // step) * step)
    y1 = max(0, (y1 // step) * step)

    x2 = min(
        (cfg.roi_x2 - cfg.roi_x1),
        ((x2 + step - 1) // step) * step,
    )
    y2 = min(
        (cfg.roi_y2 - cfg.roi_y1),
        ((y2 + step - 1) // step) * step,
    )

    if x2 <= x1:
        x2 = min((cfg.roi_x2 - cfg.roi_x1), x1 + step)

    if y2 <= y1:
        y2 = min((cfg.roi_y2 - cfg.roi_y1), y1 + step)

    return (x1, y1, x2, y2)

def make_baseline_change_mask(cfg, current_roi_rgb, baseline_roi_rgb):
    if current_roi_rgb is None or baseline_roi_rgb is None:
        return None

    current_gray = cv2.cvtColor(current_roi_rgb, cv2.COLOR_RGB2GRAY)
    baseline_gray = cv2.cvtColor(baseline_roi_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
    baseline_gray = cv2.GaussianBlur(baseline_gray, (5, 5), 0)
    difference = cv2.absdiff(current_gray, baseline_gray)
    changed = (difference >= cfg.baseline_diff_threshold).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.baseline_diff_open_kernel_px, cfg.baseline_diff_open_kernel_px),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.baseline_diff_close_kernel_px, cfg.baseline_diff_close_kernel_px),
    )

    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )
    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return changed.astype(np.uint8)

def roi_edge_touch_fraction(mask, border_px=4):
    border = np.zeros_like(mask, dtype=np.uint8)
    border[:border_px, :] = 1
    border[-border_px:, :] = 1
    border[:, :border_px] = 1
    border[:, -border_px:] = 1
    return mask_overlap_fraction(mask, border)

def crop_edge_touch_fraction(
    cfg,
    mask,
    crop_box,
    border_px=None,
):
    """Return how much of a full-ROI mask touches the active SAM crop edge."""
    if border_px is None:
        border_px = cfg.product_crop_edge_border_px
    if crop_box is None:
        return 0.0

    x1, y1, x2, y2 = [int(value) for value in crop_box]
    x1 = max(0, min((cfg.roi_x2 - cfg.roi_x1), x1))
    x2 = max(0, min((cfg.roi_x2 - cfg.roi_x1), x2))
    y1 = max(0, min((cfg.roi_y2 - cfg.roi_y1), y1))
    y2 = max(0, min((cfg.roi_y2 - cfg.roi_y1), y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    thickness = max(1, int(border_px))
    border = np.zeros_like(mask, dtype=np.uint8)

    border[y1:min(y2, y1 + thickness), x1:x2] = 1
    border[max(y1, y2 - thickness):y2, x1:x2] = 1
    border[y1:y2, x1:min(x2, x1 + thickness)] = 1
    border[y1:y2, max(x1, x2 - thickness):x2] = 1

    return mask_overlap_fraction(mask, border)

def slit_span_fraction(
    cfg,
    mask,
    slit_geometry,
    margin_px=None,
):
    """Return the fraction positioned across the physical slit segment."""
    if margin_px is None:
        margin_px = cfg.emerging_slit_span_margin_px
    if slit_geometry is None:
        return 0.0

    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return 0.0

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]

    tangent = slit_geometry.get("tangent")

    if tangent is None:
        tangent = normalize_vector(
            slit_geometry["point_b"] - slit_geometry["point_a"]
        )

    if tangent is None:
        return 0.0

    tangent_position = relative @ tangent

    slit_length = float(
        np.linalg.norm(
            slit_geometry["point_b"] - slit_geometry["point_a"]
        )
    )
    half_span = slit_length * 0.5 + float(margin_px)
    inside = np.abs(tangent_position) <= half_span

    return float(np.count_nonzero(inside) / max(len(points), 1))

def copy_candidate(candidate):
    """Copy and latch the final product segmentation."""
    if candidate is None:
        return None

    copied = dict(candidate)

    if candidate.get("mask") is not None:
        copied["mask"] = candidate["mask"].copy()

    if candidate.get("center") is not None:
        copied["center"] = candidate["center"].copy()

    if candidate.get("mean_hsv") is not None:
        copied["mean_hsv"] = tuple(candidate["mean_hsv"])

    copied["bbox"] = tuple(candidate["bbox"])
    copied["mode"] = "latched_released_product"
    return copied

def outward_fraction_for_mask(mask, slit_geometry, margin_px=5.0):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return 0.0, 0

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]
    signed = relative @ slit_geometry["outward"]
    outward_pixels = int(np.count_nonzero(signed >= margin_px))
    return outward_pixels / max(len(points), 1), outward_pixels

def signed_slit_metrics(cfg, mask, slit_geometry):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return {
            "outward_fraction": 0.0,
            "inward_fraction": 0.0,
            "line_band_fraction": 0.0,
            "outward_pixels": 0,
            "min_line_distance_px": float("inf"),
            "min_signed_distance_px": float("-inf"),
            "max_signed_distance_px": float("-inf"),
            "trailing_edge_clearance_px": 0.0,
        }

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]
    signed = relative @ slit_geometry["outward"]

    outward = signed >= 5.0
    inward = signed <= -5.0
    line_band = np.abs(signed) <= cfg.emerging_line_band_px

    min_signed_distance = float(np.min(signed))
    max_signed_distance = float(np.max(signed))

    return {
        "outward_fraction": float(np.mean(outward)),
        "inward_fraction": float(np.mean(inward)),
        "line_band_fraction": float(np.mean(line_band)),
        "outward_pixels": int(np.count_nonzero(outward)),
        "min_line_distance_px": float(np.min(np.abs(signed))),
        # Positive only when every pixel is completely on the outward side.
        # This is the clearance of the product's trailing edge from the slit.
        "min_signed_distance_px": min_signed_distance,
        "max_signed_distance_px": max_signed_distance,
        "trailing_edge_clearance_px": max(0.0, min_signed_distance),
    }

def estimate_baseline_base_hsv(roi_rgb, poly_mask):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    expanded_poly = cv2.dilate(
        poly_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        iterations=1,
    )

    sample_mask = expanded_poly == 0
    sample_mask[:12, :] = False
    sample_mask[-12:, :] = False
    sample_mask[:, :12] = False
    sample_mask[:, -12:] = False

    # The fixture/base is dark. Restricting the sample prevents bright rails,
    # labels, and cables from dominating the base appearance model.
    sample_mask &= hsv[:, :, 2] <= 145
    values = hsv[sample_mask]

    if values.shape[0] < 200:
        values = hsv[expanded_poly == 0]

    if values.shape[0] < 50:
        return None

    return (
        float(np.median(values[:, 0])),
        float(np.median(values[:, 1])),
        float(np.median(values[:, 2])),
    )

def build_baseline_static_signatures(cfg, masks, poly_mask, roi_rgb):
    signatures = []
    roi_area = (cfg.roi_x2 - cfg.roi_x1) * (cfg.roi_y2 - cfg.roi_y1)

    for sam_mask in masks:
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < cfg.baseline_static_min_mask_area_px:
            continue

        if area / max(roi_area, 1) > cfg.baseline_static_max_mask_area_ratio:
            continue

        if mask_iou(mask, poly_mask) > 0.20:
            continue

        center = mask_center(mask)

        if center is None:
            continue

        signatures.append(
            {
                "mask": mask,
                "area": area,
                "center": center,
                "mean_hsv": masked_mean_hsv(roi_rgb, mask),
            }
        )

    return signatures

def baseline_static_match(cfg, candidate_mask, candidate_center, static_signatures):
    candidate_area = max(int(candidate_mask.sum()), 1)
    best_iou = 0.0

    for signature in static_signatures:
        signature_mask = signature["mask"]
        overlap_iou = mask_iou(candidate_mask, signature_mask)
        best_iou = max(best_iou, overlap_iou)

        intersection = int(
            np.count_nonzero(
                (candidate_mask == 1)
                & (signature_mask == 1)
            )
        )
        candidate_inside_static = (
            intersection / max(candidate_area, 1)
        )

        if (
            overlap_iou >= cfg.baseline_static_reject_iou
            or candidate_inside_static
            >= cfg.baseline_static_reject_candidate_containment
        ):
            return True, best_iou

        if candidate_center is None:
            continue

        center_distance = float(
            np.linalg.norm(candidate_center - signature["center"])
        )
        area_ratio = candidate_area / max(signature["area"], 1)

        if (
            center_distance <= cfg.baseline_static_reject_center_distance_px
            and cfg.baseline_static_reject_area_ratio_low
            <= area_ratio
            <= cfg.baseline_static_reject_area_ratio_high
            and overlap_iou >= 0.08
        ):
            return True, best_iou

    return False, best_iou

def baseline_unchanged_fraction(
    cfg,
    mask,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_mask,
):
    if baseline_roi_rgb is None or baseline_poly_mask is None:
        return 0.0, 0.0

    visible_mask = (mask == 1) & (baseline_poly_mask == 0)
    candidate_area = max(int(mask.sum()), 1)
    visible_count = int(np.count_nonzero(visible_mask))
    visible_fraction = visible_count / candidate_area

    if visible_count == 0:
        return 0.0, visible_fraction

    difference = cv2.absdiff(current_roi_rgb, baseline_roi_rgb)
    difference = np.mean(difference.astype(np.float32), axis=2)
    unchanged = difference[visible_mask] <= cfg.baseline_unchanged_pixel_threshold

    return float(np.mean(unchanged)), float(visible_fraction)

def candidate_height_above_local_base(cfg, depth_roi, mask, poly_mask):
    if depth_roi is None:
        return None, 0, 0, None, None

    valid_depth = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    candidate_values = depth_roi[(mask == 1) & valid_depth].astype(np.float32)

    if candidate_values.size < cfg.emerging_min_depth_sample_count:
        return None, int(candidate_values.size), 0, None, None

    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            cfg.emerging_depth_ring_inner_px * 2 + 1,
            cfg.emerging_depth_ring_inner_px * 2 + 1,
        ),
    )
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            cfg.emerging_depth_ring_outer_px * 2 + 1,
            cfg.emerging_depth_ring_outer_px * 2 + 1,
        ),
    )

    inner = cv2.dilate(mask.astype(np.uint8), inner_kernel, iterations=1)
    outer = cv2.dilate(mask.astype(np.uint8), outer_kernel, iterations=1)
    ring = (outer == 1) & (inner == 0) & (poly_mask == 0) & valid_depth
    base_values = depth_roi[ring].astype(np.float32)

    if base_values.size < cfg.emerging_min_depth_sample_count:
        return (
            None,
            int(candidate_values.size),
            int(base_values.size),
            float(np.median(candidate_values)),
            None,
        )

    candidate_depth = float(np.median(candidate_values))
    base_depth = float(np.median(base_values))
    height_mm = float(base_depth - candidate_depth)

    return (
        height_mm,
        int(candidate_values.size),
        int(base_values.size),
        candidate_depth,
        base_depth,
    )

def find_emerging_product_candidates(
    cfg,
    masks,
    poly_mask,
    slit_geometry,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_hsv,
    baseline_base_hsv,
    baseline_poly_mask,
    baseline_static_signatures,
    depth_roi,
):
    if poly_mask is None or slit_geometry is None:
        return []

    poly_area = max(int(poly_mask.sum()), 1)
    baseline_change_mask = make_baseline_change_mask(
    cfg,
        current_roi_rgb,
        baseline_roi_rgb,
    )
    corridor_mask = slit_geometry["corridor_mask"]
    contact_mask = slit_geometry["contact_mask"]
    candidates = []

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < cfg.emerging_min_area_px:
            continue

        area_ratio_of_poly = area / poly_area

        if area_ratio_of_poly < cfg.emerging_min_area_ratio_of_poly:
            continue

        if area_ratio_of_poly > cfg.emerging_max_area_ratio_of_poly:
            continue

        x, y, width, height = cv2.boundingRect(mask)

        if (
            width < cfg.emerging_min_bbox_width_px
            or height < cfg.emerging_min_bbox_height_px
        ):
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

        if rectangularity < cfg.emerging_min_rectangularity:
            continue

        crop_edge_fraction = crop_edge_touch_fraction(
    cfg,
            mask,
            slit_geometry.get("crop_box"),
        )

        if (
            crop_edge_fraction
            > cfg.emerging_max_crop_edge_touch_fraction
        ):
            continue

        physical_slit_span_fraction = slit_span_fraction(
    cfg,
            mask,
            slit_geometry,
        )

        if (
            physical_slit_span_fraction
            < cfg.emerging_min_slit_span_fraction
        ):
            continue

        center = mask_center(mask)

        if center is None:
            continue

        static_match, static_match_iou = baseline_static_match(
    cfg,
            mask,
            center,
            baseline_static_signatures,
        )

        if static_match:
            continue

        unchanged_fraction, baseline_visible_fraction = (
            baseline_unchanged_fraction(
    cfg,
                mask,
                current_roi_rgb,
                baseline_roi_rgb,
                baseline_poly_mask,
            )
        )

        if (
            baseline_visible_fraction >= cfg.baseline_visible_min_fraction
            and unchanged_fraction
            >= cfg.baseline_unchanged_reject_fraction
        ):
            continue

        poly_overlap = mask_overlap_fraction(mask, poly_mask)

        if poly_overlap > cfg.emerging_max_poly_overlap:
            continue

        outside_poly_fraction = 1.0 - poly_overlap

        if outside_poly_fraction < cfg.emerging_min_outside_poly_fraction:
            continue

        corridor_overlap = mask_overlap_fraction(mask, corridor_mask)

        if corridor_overlap < cfg.emerging_min_corridor_overlap:
            continue

        contact_overlap = mask_overlap_fraction(mask, contact_mask)
        slit_metrics = signed_slit_metrics(cfg, mask, slit_geometry)
        outward_fraction = slit_metrics["outward_fraction"]
        inward_fraction = slit_metrics["inward_fraction"]
        line_band_fraction = slit_metrics["line_band_fraction"]
        outward_pixels = slit_metrics["outward_pixels"]
        min_line_distance_px = slit_metrics["min_line_distance_px"]

        if outward_fraction < cfg.emerging_min_outward_fraction:
            continue

        if outward_pixels < cfg.emerging_min_outward_pixels:
            continue

        crosses_slit = (
            inward_fraction >= cfg.emerging_min_inward_fraction_for_crossing
            and line_band_fraction >= cfg.emerging_min_line_band_fraction
        )

        large_outside_near_slit = (
            area_ratio_of_poly
            >= cfg.emerging_outside_only_min_area_ratio_of_poly
            and min_line_distance_px
            <= cfg.emerging_outside_only_max_line_distance_px
            and line_band_fraction
            >= cfg.emerging_min_line_band_fraction * 0.55
        )

        if not crosses_slit and not large_outside_near_slit:
            continue

        if (
            contact_overlap < cfg.emerging_min_contact_overlap
            and poly_overlap < 0.01
            and not large_outside_near_slit
        ):
            continue

        if baseline_change_mask is not None:
            changed_fraction = mask_overlap_fraction(
                mask,
                baseline_change_mask,
            )
        else:
            changed_fraction = 1.0

        if changed_fraction < cfg.emerging_min_baseline_change_fraction:
            continue

        edge_touch = roi_edge_touch_fraction(mask)

        if edge_touch > cfg.emerging_max_roi_edge_touch_fraction:
            continue

        candidate_hsv = masked_mean_hsv(current_roi_rgb, mask)
        bag_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )
        base_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_base_hsv,
        )

        if (
            bag_color_similarity >= cfg.emerging_bag_color_reject_similarity
            and poly_overlap >= 0.06
        ):
            continue

        (
            height_above_base_mm,
            candidate_depth_count,
            base_depth_count,
            candidate_depth_mm,
            local_base_depth_mm,
        ) = candidate_height_above_local_base(
    cfg,
            depth_roi,
            mask,
            poly_mask,
        )

        depth_is_strong = (
            height_above_base_mm is not None
            and height_above_base_mm
            >= cfg.emerging_min_height_above_base_mm
        )
        visual_bypass = (
            area_ratio_of_poly
            >= cfg.emerging_strong_visual_bypass_area_ratio
            and changed_fraction
            >= cfg.emerging_strong_visual_bypass_change_fraction
            and base_color_similarity
            <= cfg.baseline_base_color_bypass_max_similarity
            and (crosses_slit or min_line_distance_px <= 20.0)
        )

        strong_real_slit_crossing = (
            crosses_slit
            and poly_overlap
            >= cfg.base_side_strong_crossing_min_poly_overlap
            and changed_fraction
            >= cfg.base_side_strong_crossing_min_change
            and depth_is_strong
        )

        base_side_like = (
            base_color_similarity
            >= cfg.base_side_color_edge_reject_similarity
            and (
                crop_edge_fraction >= 0.02
                or (
                    aspect_ratio >= cfg.base_side_long_aspect_ratio
                    and rectangularity
                    <= cfg.base_side_low_rectangularity
                )
            )
        )

        if base_side_like and not strong_real_slit_crossing:
            continue

        if (
            not depth_is_strong
            and base_color_similarity
            >= cfg.baseline_base_color_reject_similarity
        ):
            continue

        if not depth_is_strong and not visual_bypass:
            continue

        duplicate = False

        for accepted in candidates:
            if mask_iou(mask, accepted["mask"]) >= cfg.emerging_duplicate_iou:
                duplicate = True
                break

        if duplicate:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        depth_score = 0.0

        if height_above_base_mm is not None:
            depth_score = min(max(height_above_base_mm, 0.0) / 35.0, 1.0)

        score = (
            2.6 * outward_fraction
            + 2.2 * corridor_overlap
            + 1.8 * contact_overlap
            + 1.5 * outside_poly_fraction
            + 1.4 * changed_fraction
            + 1.2 * line_band_fraction
            + 1.2 * depth_score
            + 0.7 * min(area_ratio_of_poly / 0.08, 1.0)
            + 0.5 * sam_stability
            + 0.3 * sam_iou
            - 0.8 * bag_color_similarity
            - 1.0 * base_color_similarity
            - 0.9 * unchanged_fraction
            - 0.7 * static_match_iou
        )

        if crosses_slit or poly_overlap > 0.03:
            mode = "sliding_through_slit"
        else:
            mode = "outside_slit"

        candidates.append(
            {
                "index": index,
                "mask": mask,
                "bbox": (x, y, width, height),
                "center": center,
                "area": area,
                "score": float(score),
                "mode": mode,
                "area_ratio_of_poly": float(area_ratio_of_poly),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect_ratio),
                "crop_edge_touch_fraction": float(
                    crop_edge_fraction
                ),
                "slit_span_fraction": float(
                    physical_slit_span_fraction
                ),
                "corridor_overlap": float(corridor_overlap),
                "contact_overlap": float(contact_overlap),
                "poly_overlap": float(poly_overlap),
                "outside_poly_fraction": float(outside_poly_fraction),
                "outward_fraction": float(outward_fraction),
                "inward_fraction": float(inward_fraction),
                "line_band_fraction": float(line_band_fraction),
                "min_line_distance_px": float(min_line_distance_px),
                "outward_pixels": outward_pixels,
                "baseline_change_fraction": float(changed_fraction),
                "baseline_visible_fraction": float(baseline_visible_fraction),
                "baseline_unchanged_fraction": float(unchanged_fraction),
                "static_match_iou": float(static_match_iou),
                "bag_color_similarity": float(bag_color_similarity),
                "base_color_similarity": float(base_color_similarity),
                "mean_hsv": candidate_hsv,
                "height_above_base_mm": (
                    None
                    if height_above_base_mm is None
                    else float(height_above_base_mm)
                ),
                "candidate_depth_count": candidate_depth_count,
                "base_depth_count": base_depth_count,
                "candidate_depth_mm": candidate_depth_mm,
                "local_base_depth_mm": local_base_depth_mm,
                "depth_bypassed": bool(not depth_is_strong),
                "sam_iou": sam_iou,
                "sam_stability": sam_stability,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates

def candidate_matches_previous(cfg, candidate, previous_candidate):
    if candidate is None or previous_candidate is None:
        return False

    area_a = max(float(candidate.get("area", 0)), 1.0)
    area_b = max(float(previous_candidate.get("area", 0)), 1.0)
    area_ratio = area_a / area_b

    if not (
        cfg.product_match_min_area_ratio
        <= area_ratio
        <= cfg.product_match_max_area_ratio
    ):
        return False

    overlap_iou = mask_iou(
        candidate["mask"],
        previous_candidate["mask"],
    )

    if overlap_iou >= cfg.product_match_min_iou:
        return True

    center_a = candidate.get("center")
    center_b = previous_candidate.get("center")

    if center_a is None or center_b is None:
        return False

    center_distance = float(np.linalg.norm(center_a - center_b))
    return center_distance <= cfg.product_match_max_center_distance_px

def make_product_verification_crop(cfg, candidate, slit_geometry):
    """Build a moving local crop around the last visible product mask."""
    if candidate is None:
        if slit_geometry is not None:
            return slit_geometry["crop_box"]
        return (0, 0, (cfg.roi_x2 - cfg.roi_x1), (cfg.roi_y2 - cfg.roi_y1))

    x, y, width, height = candidate["bbox"]
    padding = cfg.full_release_verify_crop_padding_px

    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min((cfg.roi_x2 - cfg.roi_x1), int(x + width) + padding)
    y2 = min((cfg.roi_y2 - cfg.roi_y1), int(y + height) + padding)

    # Keep part of the current slit in the crop so a still-partly-attached
    # product cannot disappear merely because the bag moved slightly.
    if slit_geometry is not None:
        slit_x1, slit_y1, slit_x2, slit_y2 = slit_geometry["crop_box"]
        midpoint = slit_geometry["midpoint"]
        near_x1 = max(0, int(midpoint[0]) - 90)
        near_y1 = max(0, int(midpoint[1]) - 90)
        near_x2 = min((cfg.roi_x2 - cfg.roi_x1), int(midpoint[0]) + 91)
        near_y2 = min((cfg.roi_y2 - cfg.roi_y1), int(midpoint[1]) + 91)
        x1 = min(x1, max(slit_x1, near_x1))
        y1 = min(y1, max(slit_y1, near_y1))
        x2 = max(x2, min(slit_x2, near_x2))
        y2 = max(y2, min(slit_y2, near_y2))

    width_now = x2 - x1
    height_now = y2 - y1

    if width_now < cfg.full_release_verify_crop_min_size_px:
        center_x = (x1 + x2) // 2
        half = cfg.full_release_verify_crop_min_size_px // 2
        x1 = max(0, center_x - half)
        x2 = min((cfg.roi_x2 - cfg.roi_x1), x1 + cfg.full_release_verify_crop_min_size_px)
        x1 = max(0, x2 - cfg.full_release_verify_crop_min_size_px)

    if height_now < cfg.full_release_verify_crop_min_size_px:
        center_y = (y1 + y2) // 2
        half = cfg.full_release_verify_crop_min_size_px // 2
        y1 = max(0, center_y - half)
        y2 = min((cfg.roi_y2 - cfg.roi_y1), y1 + cfg.full_release_verify_crop_min_size_px)
        y1 = max(0, y2 - cfg.full_release_verify_crop_min_size_px)

    return (x1, y1, x2, y2)

def candidate_reference_match(cfg, candidate_mask, candidate_center, reference_candidate):
    if reference_candidate is None or candidate_center is None:
        return False, 0.0, float("inf"), 0.0

    candidate_area = max(float(candidate_mask.sum()), 1.0)
    reference_area = max(float(reference_candidate.get("area", 0)), 1.0)
    area_ratio = candidate_area / reference_area

    if not (
        cfg.full_release_reference_min_area_ratio
        <= area_ratio
        <= cfg.full_release_reference_max_area_ratio
    ):
        return False, 0.0, float("inf"), area_ratio

    reference_mask = reference_candidate.get("mask")
    overlap_iou = 0.0

    if reference_mask is not None:
        overlap_iou = mask_iou(candidate_mask, reference_mask)

    reference_center = reference_candidate.get("center")
    center_distance = float("inf")

    if reference_center is not None:
        center_distance = float(
            np.linalg.norm(candidate_center - reference_center)
        )

    matched = (
        overlap_iou >= cfg.full_release_reference_min_iou
        or center_distance
        <= cfg.full_release_reference_max_center_distance_px
    )

    return matched, overlap_iou, center_distance, area_ratio

def find_released_product_candidates(
    cfg,
    masks,
    reference_candidate,
    poly_mask,
    slit_geometry,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_hsv,
    baseline_base_hsv,
    baseline_poly_mask,
    baseline_static_signatures,
    depth_roi,
):
    """Track the already-discovered product after it crosses the slit."""
    if reference_candidate is None or poly_mask is None or slit_geometry is None:
        return []

    poly_area = max(int(poly_mask.sum()), 1)
    baseline_change_mask = make_baseline_change_mask(
    cfg,
        current_roi_rgb,
        baseline_roi_rgb,
    )
    candidates = []

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < cfg.emerging_min_area_px:
            continue

        area_ratio_of_poly = area / poly_area

        if area_ratio_of_poly < cfg.emerging_min_area_ratio_of_poly * 0.70:
            continue

        if area_ratio_of_poly > cfg.emerging_max_area_ratio_of_poly:
            continue

        x, y, width, height = cv2.boundingRect(mask)

        if (
            width < cfg.emerging_min_bbox_width_px
            or height < cfg.emerging_min_bbox_height_px
        ):
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

        if rectangularity < cfg.emerging_min_rectangularity:
            continue

        crop_edge_fraction = crop_edge_touch_fraction(
    cfg,
            mask,
            slit_geometry.get("crop_box"),
        )

        if (
            crop_edge_fraction
            > cfg.released_max_crop_edge_touch_fraction
        ):
            continue

        center = mask_center(mask)

        if center is None:
            continue

        matched, match_iou, center_distance, reference_area_ratio = (
            candidate_reference_match(
    cfg,
                mask,
                center,
                reference_candidate,
            )
        )

        if not matched:
            continue

        static_match, static_match_iou = baseline_static_match(
    cfg,
            mask,
            center,
            baseline_static_signatures,
        )

        if static_match:
            continue

        unchanged_fraction, baseline_visible_fraction = (
            baseline_unchanged_fraction(
    cfg,
                mask,
                current_roi_rgb,
                baseline_roi_rgb,
                baseline_poly_mask,
            )
        )

        if (
            baseline_visible_fraction >= cfg.baseline_visible_min_fraction
            and unchanged_fraction
            >= cfg.baseline_unchanged_reject_fraction
        ):
            continue

        if baseline_change_mask is not None:
            changed_fraction = mask_overlap_fraction(
                mask,
                baseline_change_mask,
            )
        else:
            changed_fraction = 1.0

        if changed_fraction < cfg.emerging_min_baseline_change_fraction * 0.75:
            continue

        edge_touch = roi_edge_touch_fraction(mask)

        if edge_touch > cfg.emerging_max_roi_edge_touch_fraction:
            continue

        candidate_hsv = masked_mean_hsv(current_roi_rgb, mask)
        bag_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )
        base_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_base_hsv,
        )
        reference_color_similarity = hsv_similarity(
            candidate_hsv,
            reference_candidate.get("mean_hsv"),
        )

        poly_overlap = mask_overlap_fraction(mask, poly_mask)

        if (
            bag_color_similarity >= cfg.emerging_bag_color_reject_similarity
            and poly_overlap >= 0.06
        ):
            continue

        slit_metrics = signed_slit_metrics(cfg, mask, slit_geometry)
        contact_overlap = mask_overlap_fraction(
            mask,
            slit_geometry["contact_mask"],
        )
        core_metrics = full_release_core_metrics(
    cfg,
            mask,
            poly_mask,
            slit_geometry,
        )

        (
            height_above_base_mm,
            candidate_depth_count,
            base_depth_count,
            candidate_depth_mm,
            local_base_depth_mm,
        ) = candidate_height_above_local_base(
    cfg,
            depth_roi,
            mask,
            poly_mask,
        )

        depth_is_strong = (
            height_above_base_mm is not None
            and height_above_base_mm
            >= cfg.full_release_min_height_above_base_mm
        )
        visual_table_support = (
            changed_fraction >= cfg.full_release_visual_table_min_change
            and base_color_similarity
            <= cfg.full_release_visual_table_max_base_color_similarity
        )

        base_side_like = (
            base_color_similarity
            >= cfg.base_side_color_edge_reject_similarity
            and (
                crop_edge_fraction >= 0.025
                or (
                    aspect_ratio >= cfg.base_side_long_aspect_ratio
                    and rectangularity
                    <= cfg.base_side_low_rectangularity
                )
            )
        )

        strong_reference_support = (
            match_iou >= 0.12
            and reference_color_similarity >= 0.55
            and depth_is_strong
        )

        if base_side_like and not strong_reference_support:
            continue

        if (
            not depth_is_strong
            and base_color_similarity
            >= cfg.baseline_base_color_reject_similarity
        ):
            continue

        if not depth_is_strong and not visual_table_support:
            continue

        duplicate = False

        for accepted in candidates:
            if mask_iou(mask, accepted["mask"]) >= cfg.emerging_duplicate_iou:
                duplicate = True
                break

        if duplicate:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        center_score = 1.0 - min(
            center_distance / cfg.full_release_reference_max_center_distance_px,
            1.0,
        )
        area_score = 1.0 - min(abs(np.log(max(reference_area_ratio, 1e-6))) / 1.4, 1.0)
        depth_score = 0.0

        if height_above_base_mm is not None:
            depth_score = min(max(height_above_base_mm, 0.0) / 35.0, 1.0)

        score = (
            2.8 * match_iou
            + 2.2 * center_score
            + 1.4 * area_score
            + 1.4 * changed_fraction
            + 1.2 * depth_score
            + 0.8 * reference_color_similarity
            + 0.5 * sam_stability
            + 0.3 * sam_iou
            + 0.8 * (1.0 - poly_overlap)
            - 0.9 * base_color_similarity
            - 0.7 * bag_color_similarity
            - 0.7 * unchanged_fraction
            - 0.5 * static_match_iou
        )

        candidates.append(
            {
                "index": index,
                "mask": mask,
                "bbox": (x, y, width, height),
                "center": center,
                "area": area,
                "score": float(score),
                "mode": "released_product",
                "area_ratio_of_poly": float(area_ratio_of_poly),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect_ratio),
                "crop_edge_touch_fraction": float(
                    crop_edge_fraction
                ),
                "corridor_overlap": float(
                    mask_overlap_fraction(
                        mask,
                        slit_geometry["corridor_mask"],
                    )
                ),
                "contact_overlap": float(contact_overlap),
                "poly_overlap": float(poly_overlap),
                "outside_poly_fraction": float(1.0 - poly_overlap),
                "core_poly_overlap": core_metrics["core_poly_overlap"],
                "core_outward_fraction": core_metrics[
                    "core_outward_fraction"
                ],
                "core_inward_fraction": core_metrics[
                    "core_inward_fraction"
                ],
                "core_line_band_fraction": core_metrics[
                    "core_line_band_fraction"
                ],
                "core_min_line_distance_px": core_metrics[
                    "core_min_line_distance_px"
                ],
                "core_min_signed_distance_px": core_metrics[
                    "core_min_signed_distance_px"
                ],
                "core_trailing_edge_clearance_px": core_metrics[
                    "core_trailing_edge_clearance_px"
                ],
                "core_poly_gap_px": core_metrics["core_poly_gap_px"],
                "outward_fraction": float(slit_metrics["outward_fraction"]),
                "inward_fraction": float(slit_metrics["inward_fraction"]),
                "line_band_fraction": float(slit_metrics["line_band_fraction"]),
                "min_line_distance_px": float(slit_metrics["min_line_distance_px"]),
                "min_signed_distance_px": float(
                    slit_metrics["min_signed_distance_px"]
                ),
                "trailing_edge_clearance_px": float(
                    slit_metrics["trailing_edge_clearance_px"]
                ),
                "outward_pixels": int(slit_metrics["outward_pixels"]),
                "baseline_change_fraction": float(changed_fraction),
                "baseline_visible_fraction": float(baseline_visible_fraction),
                "baseline_unchanged_fraction": float(unchanged_fraction),
                "static_match_iou": float(static_match_iou),
                "bag_color_similarity": float(bag_color_similarity),
                "base_color_similarity": float(base_color_similarity),
                "reference_color_similarity": float(reference_color_similarity),
                "reference_match_iou": float(match_iou),
                "reference_center_distance_px": float(center_distance),
                "reference_area_ratio": float(reference_area_ratio),
                "mean_hsv": candidate_hsv,
                "height_above_base_mm": (
                    None
                    if height_above_base_mm is None
                    else float(height_above_base_mm)
                ),
                "candidate_depth_count": candidate_depth_count,
                "base_depth_count": base_depth_count,
                "candidate_depth_mm": candidate_depth_mm,
                "local_base_depth_mm": local_base_depth_mm,
                "depth_bypassed": bool(not depth_is_strong),
                "sam_iou": sam_iou,
                "sam_stability": sam_stability,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates

def candidate_center_step(candidate, previous_candidate):
    if candidate is None or previous_candidate is None:
        return None

    center_a = candidate.get("center")
    center_b = previous_candidate.get("center")

    if center_a is None or center_b is None:
        return None

    return float(np.linalg.norm(center_a - center_b))

def evaluate_full_release(cfg, candidate, previous_candidate):
    if candidate is None:
        return False, False, False, None, None

    strict_slit_clearance = (
        candidate.get("trailing_edge_clearance_px", 0.0)
        >= cfg.full_release_min_raw_trailing_clearance_px
        and candidate.get("core_poly_gap_px", 0.0)
        >= cfg.full_release_min_core_poly_gap_px
    )

    strict_separation = (
        candidate["poly_overlap"] <= cfg.full_release_max_poly_overlap
        and candidate["inward_fraction"]
        <= cfg.full_release_max_inward_fraction
        and candidate["outward_fraction"]
        >= cfg.full_release_min_outward_fraction
        and strict_slit_clearance
        and (
            candidate["contact_overlap"]
            <= cfg.full_release_max_contact_overlap
            or candidate["min_line_distance_px"]
            >= cfg.full_release_min_line_distance_px
        )
    )

    # A thin overlap between the colored outlines is not proof that the
    # physical product remains in the bag. The tolerant branch requires the
    # eroded product core to be clearly outside the eroded polymailer core.
    boundary_tolerant_separation = (
        candidate["poly_overlap"]
        <= cfg.full_release_max_raw_poly_overlap_with_clear_core
        and candidate.get("core_poly_overlap", 1.0)
        <= cfg.full_release_max_core_poly_overlap
        and candidate.get("core_inward_fraction", 1.0)
        <= cfg.full_release_max_core_inward_fraction
        and candidate.get("core_outward_fraction", 0.0)
        >= cfg.full_release_min_core_outward_fraction
        and candidate.get("core_trailing_edge_clearance_px", 0.0)
        >= cfg.full_release_min_core_trailing_clearance_px
        and candidate.get("core_poly_gap_px", 0.0)
        >= cfg.full_release_min_core_poly_gap_px
        and candidate.get("core_line_band_fraction", 1.0)
        <= cfg.full_release_max_core_line_band_fraction
    )

    separated = strict_separation or boundary_tolerant_separation
    candidate["strict_slit_clearance"] = bool(strict_slit_clearance)
    candidate["strict_separation"] = bool(strict_separation)
    candidate["boundary_tolerant_separation"] = bool(
        boundary_tolerant_separation
    )

    depth_on_table = (
        candidate["height_above_base_mm"] is not None
        and candidate["height_above_base_mm"]
        >= cfg.full_release_min_height_above_base_mm
    )
    visual_on_table = (
        candidate["baseline_change_fraction"]
        >= cfg.full_release_visual_table_min_change
        and candidate["base_color_similarity"]
        <= cfg.full_release_visual_table_max_base_color_similarity
    )
    on_table = depth_on_table or visual_on_table

    center_step = candidate_center_step(candidate, previous_candidate)
    area_ratio = None
    stationary = False

    if previous_candidate is not None:
        previous_area = max(float(previous_candidate.get("area", 0)), 1.0)
        area_ratio = float(candidate["area"] / previous_area)
        stationary = (
            center_step is not None
            and center_step <= cfg.full_release_max_center_step_px
            and cfg.full_release_min_stable_area_ratio
            <= area_ratio
            <= cfg.full_release_max_stable_area_ratio
        )

    return separated, on_table, stationary, center_step, area_ratio
