"""Box-mode overlay and JSON result building.

Ported verbatim from run_box()/box_detection_final.py; shared helpers (depth
colormap, ROI axes) come from visualization/common.py.
"""

import cv2
import numpy as np

from ..utils import fmt3, json_number
from .common import draw_roi_axes, make_depth_vis


def draw_rotated_or_axis_box(cfg, result_rgb, depth_vis, box, dimensions):
    points_full = dimensions.get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)

        cv2.polylines(result_rgb, [points_full_np], True, (0, 255, 0), 3)
        cv2.polylines(depth_vis, [points_full_np], True, (0, 255, 0), 3)

        for p in points_full_np:
            cv2.circle(result_rgb, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)

        return

    x, y, w, h = box["bbox"]

    cv2.rectangle(
        result_rgb,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (0, 255, 0),
        3,
    )

    cv2.rectangle(
        depth_vis,
        (cfg.roi_x1 + x, cfg.roi_y1 + y),
        (cfg.roi_x1 + x + w, cfg.roi_y1 + y + h),
        (0, 255, 0),
        3,
    )


def draw_result(cfg, frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(cfg, depth_aligned)

    roi_rgb = result["roi_rgb"].copy()

    box = result["box"]
    mask = box["mask"]
    dimensions = result["dimensions"]

    roi_rgb[mask == 1] = (
        0.50 * roi_rgb[mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

    draw_roi_axes(cfg, result_rgb)
    draw_roi_axes(cfg, depth_vis)

    draw_rotated_or_axis_box(cfg, result_rgb, depth_vis, box, dimensions)

    cx, cy = result["center_full"]

    cv2.circle(result_rgb, (cx, cy), 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, (cx, cy), 16, (255, 255, 0), 2)

    cv2.circle(depth_vis, (cx, cy), 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, (cx, cy), 16, (0, 255, 255), 2)

    lines = [
        "BOX MEASUREMENTS",
        f"box_face_depth_mm={fmt3(result['box_face_depth_mm'])}",
        f"box_depth_mm={fmt3(result['box_depth_mm'])}",
        f"length_mm={fmt3(dimensions['length_mm'])}",
        f"width_mm={fmt3(dimensions['short_side_mm'])}",
        f"angle_deg={fmt3(dimensions['angle_deg'])}",
        f"center_x_mm={fmt3(result['center_x_mm'])}",
        f"center_y_mm={fmt3(result['center_y_mm'])}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            result_rgb,
            line,
            (30, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
    combined_rgb = np.hstack([result_rgb, depth_vis_rgb])

    return cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)


def make_json_result(cfg, result, timestamp):
    dimensions = result["dimensions"]

    return {
        "box_face_depth_mm": json_number(result["box_face_depth_mm"]),
        "box_depth_mm": json_number(result["box_depth_mm"]),
        "length_mm": json_number(dimensions["length_mm"]),
        "width_mm": json_number(dimensions["short_side_mm"]),
        "angle_deg": json_number(dimensions["angle_deg"]),
        "center_x_mm": json_number(result["center_x_mm"]),
        "center_y_mm": json_number(result["center_y_mm"]),
    }
