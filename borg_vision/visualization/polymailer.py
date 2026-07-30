"""Polymailer-mode heatmap, overlay and JSON result building.

Ported verbatim from run_polymailer()/polymailer_final.py; shared helpers
(depth colormap, ROI axes) come from visualization/common.py.
"""

import cv2
import numpy as np

from ..utils import fmt3, json_number
from .common import draw_roi_axes, make_depth_vis


def make_polymailer_depth_heatmap(cfg, depth_roi, poly_mask, product_mask, surface_depth_mm):
    h, w = depth_roi.shape[:2]
    debug = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
        & (poly_mask == 1)
    )

    if surface_depth_mm is not None and np.any(valid):
        diff = np.zeros_like(depth_roi, dtype=np.float32)
        diff[valid] = surface_depth_mm - depth_roi[valid].astype(np.float32)

        diff_clip = np.clip(diff, 0, cfg.poly_bulge_max_mm)

        gray = cv2.normalize(
            diff_clip,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        debug[valid] = heat[valid]

    purple = np.zeros_like(debug)
    purple[:, :] = (255, 0, 255)

    debug[poly_mask == 1] = (
        0.65 * debug[poly_mask == 1]
        + 0.35 * purple[poly_mask == 1]
    ).astype(np.uint8)

    orange = np.zeros_like(debug)
    orange[:, :] = (0, 140, 255)

    debug[product_mask == 1] = (
        0.35 * debug[product_mask == 1]
        + 0.65 * orange[product_mask == 1]
    ).astype(np.uint8)

    return debug


def draw_rotated_poly_outline(cfg, result_rgb, depth_vis, poly, dimensions):
    points_full = dimensions.get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)

        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 255), 3)
        cv2.polylines(depth_vis, [points_full_np], True, (255, 0, 255), 3)

        return

    x, y, w, h = poly["bbox"]

    cv2.rectangle(
        result_rgb,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (255, 0, 255),
        3,
    )

    cv2.rectangle(
        depth_vis,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (255, 0, 255),
        3,
    )


def draw_result(cfg, frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(cfg, depth_aligned)

    roi_rgb = result["roi_rgb"].copy()

    poly = result["poly"]
    poly_mask = poly["mask"]
    dimensions = result["dimensions"]

    product = result["product"]
    product_mask = product["mask"]

    roi_rgb[poly_mask == 1] = (
        0.72 * roi_rgb[poly_mask == 1]
        + 0.28 * np.array([255, 0, 255])
    ).astype(np.uint8)

    if int(product_mask.sum()) > 0:
        roi_rgb[product_mask == 1] = (
            0.35 * roi_rgb[product_mask == 1]
            + 0.65 * np.array([255, 100, 0])
        ).astype(np.uint8)

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

    draw_roi_axes(cfg, result_rgb)
    draw_roi_axes(cfg, depth_vis)

    draw_rotated_poly_outline(cfg, result_rgb, depth_vis, poly, dimensions)

    pcx, pcy = result["poly_center_full"]

    cv2.circle(result_rgb, (pcx, pcy), 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, (pcx, pcy), 16, (255, 255, 0), 2)

    cv2.circle(depth_vis, (pcx, pcy), 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, (pcx, pcy), 16, (0, 255, 255), 2)

    if result["product_center_full"] is not None:
        cx, cy = result["product_center_full"]

        cv2.circle(result_rgb, (cx, cy), 8, (255, 180, 0), -1)
        cv2.circle(result_rgb, (cx, cy), 16, (255, 180, 0), 2)

        cv2.circle(depth_vis, (cx, cy), 8, (0, 140, 255), -1)
        cv2.circle(depth_vis, (cx, cy), 16, (0, 140, 255), 2)

    edges = result["edges"]

    for point in (edges["top_point_full"], edges["bottom_point_full"]):
        ex, ey = point
        cv2.circle(result_rgb, (ex, ey), 7, (0, 255, 128), -1)
        cv2.circle(result_rgb, (ex, ey), 14, (0, 255, 128), 2)
        cv2.circle(depth_vis, (ex, ey), 7, (0, 255, 128), -1)
        cv2.circle(depth_vis, (ex, ey), 14, (0, 255, 128), 2)

    cut = result["cut"]

    if cut["cut_side"] is not None:
        bx, _, bw, _ = poly["bbox"]
        cut_x1 = cfg.roi_x1 + bx
        cut_x2 = cfg.roi_x1 + bx + bw
        cut_y = cut["cut_row_full"]

        cv2.line(result_rgb, (cut_x1, cut_y), (cut_x2, cut_y), (0, 255, 0), 3)
        cv2.line(depth_vis, (cut_x1, cut_y), (cut_x2, cut_y), (0, 255, 0), 3)

        label = f"CUT {cut['cut_side']}"
        label_y = cut_y - 12 if cut["cut_side"] == "TOP" else cut_y + 30

        cv2.putText(
            result_rgb,
            label,
            (cut_x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        "POLYMAILER OUTPUT",
        f"polymailer_face_depth_mm={fmt3(result['polymailer_face_depth_mm'])}",
        f"polymailer_depth_mm={fmt3(result['polymailer_depth_mm'])}",
        f"length_mm={fmt3(dimensions['length_mm'])}",
        f"width_mm={fmt3(dimensions['width_mm'])}",
        f"angle_deg={fmt3(dimensions['angle_deg'])}",
        f"center_x_mm={fmt3(result['center_x_mm'])}",
        f"center_y_mm={fmt3(result['center_y_mm'])}",
        f"product_inside_center_x_mm={fmt3(result['product_inside_center_x_mm'])}",
        f"product_inside_center_y_mm={fmt3(result['product_inside_center_y_mm'])}",
        f"product_inside_center_face_depth={fmt3(result['product_inside_center_face_depth'])}",
        f"cut_side={cut['cut_side']}",
        f"top_gap_mm={fmt3(cut['top_gap_mm'])}  bottom_gap_mm={fmt3(cut['bottom_gap_mm'])}",
        f"top_edge xyz=({fmt3(edges['top_x_mm'])}, {fmt3(edges['top_y_mm'])}, {fmt3(edges['top_z_mm'])})",
        f"bottom_edge xyz=({fmt3(edges['bottom_x_mm'])}, {fmt3(edges['bottom_y_mm'])}, {fmt3(edges['bottom_z_mm'])})",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            result_rgb,
            line,
            (30, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    heatmap_full = np.zeros_like(result_rgb)
    heatmap_roi_rgb = cv2.cvtColor(result["depth_heatmap"], cv2.COLOR_BGR2RGB)

    heatmap_full[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    draw_roi_axes(cfg, heatmap_full)

    cv2.putText(
        heatmap_full,
        "DEPTH HEATMAP",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined_rgb = np.hstack([result_rgb, depth_vis_rgb, heatmap_full])

    return cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)


def make_json_result(cfg, result, timestamp):
    dimensions = result["dimensions"]
    cut = result["cut"]
    edges = result["edges"]

    return {
        "polymailer_face_depth_mm": json_number(result["polymailer_face_depth_mm"]),
        "polymailer_depth_mm": json_number(result["polymailer_depth_mm"]),
        "length_mm": json_number(dimensions["length_mm"]),
        "width_mm": json_number(dimensions["width_mm"]),
        "angle_deg": json_number(dimensions["angle_deg"]),
        "center_x_mm": json_number(result["center_x_mm"]),
        "center_y_mm": json_number(result["center_y_mm"]),
        "product_inside_center_x_mm": json_number(result["product_inside_center_x_mm"]),
        "product_inside_center_y_mm": json_number(result["product_inside_center_y_mm"]),
        "product_inside_center_face_depth": json_number(result["product_inside_center_face_depth"]),
        "cut_side": cut["cut_side"],
        "top_gap_mm": json_number(cut["top_gap_mm"]),
        "bottom_gap_mm": json_number(cut["bottom_gap_mm"]),
        "top_edge_x_mm": json_number(edges["top_x_mm"]),
        "top_edge_y_mm": json_number(edges["top_y_mm"]),
        "top_edge_z_mm": json_number(edges["top_z_mm"]),
        "bottom_edge_x_mm": json_number(edges["bottom_x_mm"]),
        "bottom_edge_y_mm": json_number(edges["bottom_y_mm"]),
        "bottom_edge_z_mm": json_number(edges["bottom_z_mm"]),
    }
