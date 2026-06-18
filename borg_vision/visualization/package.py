"""Package-mode overlays, heatmap, save helpers and JSON result building.

Ported verbatim from the original visualization.py; shared helpers now come
from visualization/common.py.
"""

import cv2
import numpy as np

from ..barcode import draw_barcode_overlay
from ..utils import fmt3, json_number
from .common import (
    draw_roi_axes,
    make_depth_vis,
    put_text_lines,
    save_binary_mask,
)


def make_package_depth_heatmap(cfg, depth_roi, package_mask, center_roi, product_inside=None):
    h, w = depth_roi.shape[:2]

    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    if np.any(valid):
        closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
        closer_mm[valid] = cfg.base_depth_mm - depth_roi[valid].astype(np.float32)
        closer_mm = np.clip(closer_mm, 0.0, cfg.heatmap_max_closer_than_base_mm)

        gray = cv2.normalize(
            closer_mm,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        gray[~valid] = 0
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    green = np.zeros_like(heatmap)
    green[:, :] = (0, 255, 0)

    heatmap[package_mask == 1] = (
        0.45 * heatmap[package_mask == 1]
        + 0.55 * green[package_mask == 1]
    ).astype(np.uint8)

    if product_inside is not None and product_inside.get("found"):
        product_mask = product_inside["mask"].astype(np.uint8)

        yellow = np.zeros_like(heatmap)
        yellow[:, :] = (0, 255, 255)

        heatmap[product_mask == 1] = (
            0.35 * heatmap[product_mask == 1]
            + 0.65 * yellow[product_mask == 1]
        ).astype(np.uint8)

        pcx, pcy = product_inside["center_roi"]
        cv2.circle(heatmap, (pcx, pcy), 8, (0, 255, 255), -1)
        cv2.circle(heatmap, (pcx, pcy), 16, (0, 255, 255), 2)

    if center_roi is not None:
        cx, cy = center_roi
        cv2.circle(heatmap, (cx, cy), 8, (255, 255, 255), -1)
        cv2.circle(heatmap, (cx, cy), 14, (255, 255, 255), 2)

    return heatmap


def draw_rotated_or_axis_box(cfg, result_rgb, info, dimensions=None):
    points_full = None

    if dimensions is not None:
        points_full = dimensions.get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)
        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 0), 3)

        for p in points_full_np:
            cv2.circle(result_rgb, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

        return

    rotated = info.get("rotated")

    if rotated is not None:
        points_roi = rotated["points_roi"]
        points_full_np = points_roi.copy()
        points_full_np[:, 0] += cfg.roi_x1
        points_full_np[:, 1] += cfg.roi_y1

        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 0), 3)

        for p in points_full_np:
            cv2.circle(result_rgb, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

        return

    x, y, w, h = info["bbox"]

    cv2.rectangle(
        result_rgb,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (255, 0, 0),
        3,
    )


def draw_result(cfg, frame_bgr, depth_class_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(cfg, depth_class_aligned)

    roi_rgb = result["roi_rgb"].copy()
    mask = result["mask"]
    center_full = result["center_full"]
    info = result["info"]
    classification = result["classification"]
    final_output = result["final_output"]
    dimensions = result["dimensions"]
    product_inside = result["product_inside"]
    barcode = result["barcode"]

    roi_rgb[mask == 1] = (
        0.50 * roi_rgb[mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    if product_inside.get("found"):
        product_mask = product_inside["mask"].astype(np.uint8)
        roi_rgb[product_mask == 1] = (
            0.35 * roi_rgb[product_mask == 1]
            + 0.65 * np.array([255, 255, 0])
        ).astype(np.uint8)

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

    draw_roi_axes(cfg, result_rgb)
    draw_roi_axes(cfg, depth_vis)
    draw_rotated_or_axis_box(cfg, result_rgb, info, dimensions)
    draw_rotated_or_axis_box(cfg, depth_vis, info, dimensions)

    cv2.circle(result_rgb, center_full, 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, center_full, 14, (255, 255, 0), 2)

    cv2.circle(depth_vis, center_full, 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, center_full, 14, (0, 255, 255), 2)

    if product_inside.get("found") and product_inside.get("center_full") is not None:
        pcx, pcy = product_inside["center_full"]

        cv2.circle(result_rgb, (pcx, pcy), 8, (255, 255, 0), -1)
        cv2.circle(result_rgb, (pcx, pcy), 18, (255, 255, 0), 3)

        cv2.circle(depth_vis, (pcx, pcy), 8, (0, 255, 255), -1)
        cv2.circle(depth_vis, (pcx, pcy), 18, (0, 255, 255), 3)

    result_bgr_for_barcode = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    if cfg.barcode_draw_enable and barcode is not None:
        result_bgr_for_barcode = draw_barcode_overlay(result_bgr_for_barcode, barcode)

    result_rgb = cv2.cvtColor(result_bgr_for_barcode, cv2.COLOR_BGR2RGB)

    poly_sig = classification["polymailer_depth_signature"]

    lines = [
        "PACKAGE FINAL OUTPUT",
        f"barcode={final_output['barcode_data']}",
        f"type={final_output['package_type']}",
        f"confidence={final_output['package_type_confidence_percent']:.1f}%",
        f"mask_source={final_output['mask_source']}",
        f"top_face_depth_mm={fmt3(final_output['top_face_depth_mm'])}",
        f"package_depth_mm={fmt3(final_output['package_depth_mm'])}",
        f"length_mm={fmt3(final_output['length_mm'])}",
        f"width_mm={fmt3(final_output['width_mm'])}",
        f"angle_deg={fmt3(final_output['angle_deg'])}",
        f"center_x_mm={fmt3(final_output['center_x_mm'])}",
        f"center_y_mm={fmt3(final_output['center_y_mm'])}",
        f"product_inside={final_output['product_inside_found']}",
        f"prod_x_mm={fmt3(final_output['product_inside_center_x_mm'])}",
        f"prod_y_mm={fmt3(final_output['product_inside_center_y_mm'])}",
        f"poly_score={fmt3(poly_sig['score'])}",
    ]

    put_text_lines(result_rgb, lines)

    depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    heatmap_full = np.zeros_like(result_rgb)
    heatmap_roi_rgb = cv2.cvtColor(result["depth_heatmap"], cv2.COLOR_BGR2RGB)
    heatmap_full[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    draw_roi_axes(cfg, heatmap_full)

    cv2.putText(
        heatmap_full,
        "PACKAGE DEPTH HEATMAP",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined_rgb = np.hstack([result_rgb, depth_vis_rgb, heatmap_full])

    return cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)


def save_accepted_masks(save_dir, timestamp, result):
    accepted_dir = save_dir / f"accepted_masks_{timestamp}"
    accepted_dir.mkdir(exist_ok=True)

    for item_number, item in enumerate(result["accepted"]):
        idx = item.get("index")
        source = item.get("source", "unknown")

        if idx is None:
            idx_text = f"merged_{item_number:03d}"
        else:
            idx_text = f"{int(idx):03d}"

        path = accepted_dir / f"accepted_mask_{idx_text}_{source}_score_{item['score']:.3f}.png"
        save_binary_mask(path, item["mask"])


def make_json_result(cfg, result, timestamp):
    final_output = result["final_output"]
    classification = result["classification"]
    poly_sig = classification["polymailer_depth_signature"]
    depth_features = classification["depth_features"]
    info = result["info"]
    product_inside = result["product_inside"]
    barcode = result["barcode"]

    return {
        "timestamp": timestamp,
        "success": True,
        "mode": "barcode_gate_package_detection",
        "barcode": {
            "type": barcode["type"] if barcode is not None else None,
            "data": barcode["data"] if barcode is not None else None,
            "rect": list(barcode["rect"]) if barcode is not None else None,
            "decoded_count": barcode["decoded_count"] if barcode is not None else 0,
        },
        "package_type": final_output["package_type"],
        "package_type_confidence_percent": json_number(final_output["package_type_confidence_percent"]),
        "mask_source": final_output["mask_source"],
        "selected_mask_index": (
            int(info["index"])
            if info.get("index") is not None
            else None
        ),
        "selected_mask_merged_from_indices": info.get("merged_from_indices"),
        "selected_mask_score": json_number(info["score"]),
        "selected_mask_area_ratio": json_number(info["area_ratio"]),
        "selected_mask_rectangularity": json_number(info["rectangularity"]),
        "selected_mask_aspect_ratio": json_number(info["aspect_ratio"]),
        "selected_mask_color_score": json_number(info["color_score"]),
        "selected_mask_selection_override": info.get("selection_override"),
        "dedicated_box_sam_enabled": bool(cfg.dedicated_box_sam_enable),
        "dedicated_box_mask_count": int(result.get("dedicated_box_mask_count", 0)),
        "dedicated_box_candidate_found": bool(
            result.get("dedicated_box_candidate_found", False)
        ),
        "top_face_depth_mm": json_number(final_output["top_face_depth_mm"]),
        "package_depth_mm": json_number(final_output["package_depth_mm"]),
        "length_mm": json_number(final_output["length_mm"]),
        "width_mm": json_number(final_output["width_mm"]),
        "angle_deg": json_number(final_output["angle_deg"]),
        "center_x_mm": json_number(final_output["center_x_mm"]),
        "center_y_mm": json_number(final_output["center_y_mm"]),
        "product_inside_found": bool(final_output["product_inside_found"]),
        "product_inside_center_x_mm": json_number(final_output["product_inside_center_x_mm"]),
        "product_inside_center_y_mm": json_number(final_output["product_inside_center_y_mm"]),
        "product_inside_center_pixel_u": final_output["product_inside_center_pixel_u"],
        "product_inside_center_pixel_v": final_output["product_inside_center_pixel_v"],
        "product_inside_depth_mm": json_number(final_output["product_inside_depth_mm"]),
        "product_inside_debug": {
            "reason": product_inside["reason"],
            "area_px": int(product_inside["area_px"]),
            "area_ratio_of_package": json_number(product_inside["area_ratio_of_package"]),
        },
        "size_scale": json_number(cfg.package_size_scale),
        "base_depth_mm": json_number(cfg.base_depth_mm),
        "measurement_depth_offset_mm": json_number(cfg.measurement_depth_offset_mm),
        "top_face_depth_debug": {
            "raw_distance_mm": json_number(result["top_face_depth_result"]["raw_distance_mm"]),
            "offset_distance_mm": json_number(result["top_face_depth_result"]["distance_mm"]),
            "depth_count": int(result["top_face_depth_result"]["depth_count"]),
            "sample_count_total": int(result["top_face_depth_result"]["sample_count_total"]),
            "mean_depth_mm": json_number(result["top_face_depth_result"]["mean_depth_mm"]),
            "min_depth_mm": json_number(result["top_face_depth_result"]["min_depth_mm"]),
        },
        "depth_classification_debug": {
            "segmentation_box_override": bool(
                classification.get("segmentation_box_override", False)
            ),
            "segmentation_box_override_reason": classification.get(
                "segmentation_box_override_reason"
            ),
            "segmentation_box_override_vetoed": bool(
                classification.get("segmentation_box_override_vetoed", False)
            ),
            "segmentation_box_override_veto_reason": classification.get(
                "segmentation_box_override_veto_reason"
            ),
            "depth_classifier_type_before_override": classification.get(
                "depth_classifier_type_before_override"
            ),
            "depth_classifier_confidence_before_override": json_number(
                classification.get("depth_classifier_confidence_before_override")
            ),
            "package_depth_mm_for_type": json_number(
                classification.get("package_depth_mm_for_type")
            ),
            "box_score": json_number(classification["box_score"]),
            "raw_box_score": json_number(classification["raw_box_score"]),
            "poly_signal_count": int(classification["poly_signal_count"]),
            "poly_signature_score": json_number(poly_sig["score"]),
            "plane_residual_std_mm": json_number(depth_features["plane_residual_std_mm"]),
            "plane_residual_range_mm": json_number(depth_features["plane_residual_range_mm"]),
            "center_edge_depth_delta_mm": json_number(depth_features["center_edge_depth_delta_mm"]),
            "valid_points": int(depth_features["valid_points"]),
            "valid_fraction_on_mask": json_number(depth_features["valid_fraction_on_mask"]),
        },
        "camera": {
            "rgb_size": list(cfg.rgb_size),
            "classification_depth_size": list(cfg.rgb_size),
            "measurement_stereo_size": list(cfg.stereo_size),
            "fps": cfg.fps,
        },
    }
