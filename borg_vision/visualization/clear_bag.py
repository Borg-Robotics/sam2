"""Clear-bag-mode heatmap, overlay and JSON result building.

Ported verbatim from run_clear_bag()/clear_bag_final.py; shared helpers (depth
colormap, ROI axes, text lines) come from visualization/common.py.
"""

import cv2
import numpy as np

from ..utils import fmt3, json_number
from .common import draw_roi_axes, make_depth_vis, put_text_lines


def make_clear_bag_depth_heatmap(cfg, depth_roi, product_mask, bag):
    h, w = depth_roi.shape[:2]
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    base_depth_mm = bag.get("base_depth_mm")

    valid = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    )

    if base_depth_mm is not None and np.any(valid):
        closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
        closer_mm[valid] = float(base_depth_mm) - depth_roi[valid].astype(np.float32)
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

    if bag.get("bag_mask") is not None:
        bag_mask = bag["bag_mask"].astype(np.uint8)
        purple = np.zeros_like(heatmap)
        purple[:, :] = (255, 0, 255)

        heatmap[bag_mask == 1] = (
            0.60 * heatmap[bag_mask == 1]
            + 0.40 * purple[bag_mask == 1]
        ).astype(np.uint8)

    if product_mask is not None:
        green = np.zeros_like(heatmap)
        green[:, :] = (0, 255, 0)

        heatmap[product_mask == 1] = (
            0.45 * heatmap[product_mask == 1]
            + 0.55 * green[product_mask == 1]
        ).astype(np.uint8)

    if bag.get("rotated") is not None:
        points = bag["rotated"]["points_roi"]
        cv2.polylines(heatmap, [points], True, (255, 0, 255), 3)

    elif bag.get("bag_bbox") is not None:
        bx, by, bw, bh = bag["bag_bbox"]
        cv2.rectangle(
            heatmap,
            (bx, by),
            (bx + bw, by + bh),
            (255, 0, 255),
            3,
        )

    if bag.get("bag_center_roi") is not None:
        cx, cy = bag["bag_center_roi"]
        cv2.circle(heatmap, (cx, cy), 8, (0, 255, 255), -1)
        cv2.circle(heatmap, (cx, cy), 14, (0, 255, 255), 2)

    return heatmap


def draw_result(cfg, frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(cfg, depth_aligned)

    roi_rgb = result["roi_rgb"].copy()

    product_mask = result["mask"]
    safe_mask = result["safe_mask"]
    center_full = result["center_full"]
    bag = result["bag"]
    final_output = result["final_output"]

    if bag["bag_mask"] is not None:
        bag_mask = bag["bag_mask"]
        roi_rgb[bag_mask == 1] = (
            0.85 * roi_rgb[bag_mask == 1]
            + 0.15 * np.array([255, 0, 255])
        ).astype(np.uint8)

    roi_rgb[product_mask == 1] = (
        0.50 * roi_rgb[product_mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    roi_rgb[safe_mask == 1] = (
        0.45 * roi_rgb[safe_mask == 1]
        + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

    draw_roi_axes(cfg, result_rgb)
    draw_roi_axes(cfg, depth_vis)

    points_full = result["bag_measurements"].get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)
        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 255), 3)
        cv2.polylines(depth_vis, [points_full_np], True, (255, 0, 255), 3)

    elif bag["bag_bbox"] is not None:
        bx, by, bw, bh = bag["bag_bbox"]

        cv2.rectangle(
            result_rgb,
            (cfg.roi_x1 + bx, cfg.roi_y1 + by),
            (cfg.roi_x1 + bx + bw, cfg.roi_y1 + by + bh),
            (255, 0, 255),
            3,
        )

        cv2.rectangle(
            depth_vis,
            (cfg.roi_x1 + bx, cfg.roi_y1 + by),
            (cfg.roi_x1 + bx + bw, cfg.roi_y1 + by + bh),
            (255, 0, 255),
            3,
        )

    x, y, w, h = result["info"]["bbox"]

    cv2.rectangle(
        result_rgb,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (255, 0, 0),
        2,
    )

    cv2.circle(result_rgb, center_full, 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, center_full, 14, (255, 255, 0), 2)

    cv2.circle(depth_vis, center_full, 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, center_full, 14, (0, 255, 255), 2)

    text_lines = [
        "CLEAR BAG FINAL OUTPUT",
        f"product_face_depth_mm={fmt3(final_output['product_face_depth_mm'])}",
        f"clearbag_depth_mm={fmt3(final_output['clearbag_depth_mm'])}",
        f"length_mm={fmt3(final_output['length_mm'])}",
        f"width_mm={fmt3(final_output['width_mm'])}",
        f"angle_deg={fmt3(final_output['angle_deg'])}",
        f"center_x_mm={fmt3(final_output['center_x_mm'])}",
        f"center_y_mm={fmt3(final_output['center_y_mm'])}",
        f"product_inside_center_x_mm={fmt3(final_output['product_inside_center_x_mm'])}",
        f"product_inside_center_y_mm={fmt3(final_output['product_inside_center_y_mm'])}",
    ]

    put_text_lines(result_rgb, text_lines)

    depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    heatmap_full = np.zeros_like(result_rgb)
    heatmap_roi_rgb = cv2.cvtColor(result["depth_heatmap"], cv2.COLOR_BGR2RGB)
    heatmap_full[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    draw_roi_axes(cfg, heatmap_full)

    cv2.putText(
        heatmap_full,
        "BAG DEPTH HEATMAP",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined_rgb = np.hstack([result_rgb, depth_vis_rgb, heatmap_full])

    return cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)


def make_json_result(cfg, result, timestamp=None):
    final_output = result["final_output"]

    return {
        "product_face_depth_mm": json_number(final_output["product_face_depth_mm"]),
        "clearbag_depth_mm": json_number(final_output["clearbag_depth_mm"]),
        "length_mm": json_number(final_output["length_mm"]),
        "width_mm": json_number(final_output["width_mm"]),
        "angle_deg": json_number(final_output["angle_deg"]),
        "center_x_mm": json_number(final_output["center_x_mm"]),
        "center_y_mm": json_number(final_output["center_y_mm"]),
        "product_inside_center_x_mm": json_number(final_output["product_inside_center_x_mm"]),
        "product_inside_center_y_mm": json_number(final_output["product_inside_center_y_mm"]),
    }
