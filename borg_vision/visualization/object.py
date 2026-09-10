"""Object-mode overlay and JSON result building.

Ported verbatim from run_object()/object_detection_final.py; shared helpers
(depth colormap) come from visualization/common.py.
"""

import cv2
import numpy as np

from ..utils import fmt3, json_number
from .common import make_depth_vis


def draw_object_outline(cfg, result_rgb, depth_vis, obj):
    """Trace the object's actual silhouette.

    Follows the mask contour rather than a bounding box, so the outline sits on
    the object's real edges. Falls back to the axis-aligned bbox if the mask
    yields no contour.
    """
    contours, _ = cv2.findContours(
        obj["mask"].astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        # Mask coordinates are ROI-relative; shift them into the full frame.
        outline = max(contours, key=cv2.contourArea) + [cfg.roi_x1, cfg.roi_y1]

        cv2.polylines(result_rgb, [outline], True, (0, 255, 0), 3)
        cv2.polylines(depth_vis, [outline], True, (0, 255, 0), 3)

        return

    x, y, w, h = obj["bbox"]

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

    obj = result["object"]
    mask = obj["mask"]

    roi_rgb[mask == 1] = (
        0.50 * roi_rgb[mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # inside_box overlay: the RED RECTANGLE is the detected box (its inner
    # walls); grasp points keep object_wall_clearance_mm inside it. The
    # tinted fallback appears only when no box rectangle could be found.
    box_quad = result.get("box_quad")
    if box_quad is not None:
        cv2.polylines(roi_rgb, [box_quad], True, (255, 0, 0), 3)
    # YELLOW outline: where the cup may land so that the gripper body fits
    # inside the walls at some quarter turn (box_mover's own test). Every
    # grasp point and retry must be inside it.
    feasible = result.get("wall_feasible")
    if feasible is not None and int(feasible.sum()) > 0:
        cnts, _ = cv2.findContours(
            np.ascontiguousarray(feasible.astype(np.uint8)),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_rgb, cnts, -1, (255, 255, 0), 1)
    wall_mask = result.get("wall_mask")
    if wall_mask is not None and int(wall_mask.sum()) > 0:
        roi_rgb[wall_mask == 1] = (
            0.5 * roi_rgb[wall_mask == 1] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

    draw_object_outline(cfg, result_rgb, depth_vis, obj)

    cx, cy = result["center_full"]

    cv2.circle(result_rgb, (cx, cy), 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, (cx, cy), 16, (255, 255, 0), 2)

    cv2.circle(depth_vis, (cx, cy), 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, (cx, cy), 16, (0, 255, 255), 2)

    # The pick (rank 0, green) and its three retries (orange), so the debug
    # image alone shows where the cup is going and what it would try next.
    # The cup footprint is drawn at its real radius, which is what makes an
    # overhanging or edge-hugging pick obvious at a glance.
    for rank, cand in enumerate(result.get("grasp_candidates") or []):
        point = cand.get("point_full")

        if point is None:
            continue

        cx_c, cy_c = point
        colour = (0, 255, 0) if rank == 0 else (0, 200, 255)
        cv2.circle(result_rgb, (cx_c, cy_c), 4, colour, -1)
        cv2.circle(result_rgb, (cx_c, cy_c), 22, colour, 2)
        cv2.putText(
            result_rgb,
            "PICK" if rank == 0 else str(rank),
            (cx_c + 26, cy_c + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(depth_vis, (cx_c, cy_c), 22, colour, 2)

    distance_mm = result["distance_mm"]

    distance_text = (
        "center_distance=None"
        if distance_mm is None
        else f"center_distance={distance_mm:.1f}mm"
    )

    dimensions = result.get("dimensions") or {}

    lines = [
        "OBJECT SEGMENT + DEPTH",
        distance_text,
        f"length_mm={fmt3(dimensions.get('length_mm'))}",
        f"width_mm={fmt3(dimensions.get('width_mm'))}",
        f"height_mm={fmt3(dimensions.get('height_mm'))}",
        f"angle_deg={fmt3(result.get('angle_deg'))}",
        f"grasp xyz=({fmt3(result.get('grasp_x_mm'))}, "
        f"{fmt3(result.get('grasp_y_mm'))}, {fmt3(result.get('grasp_z_mm'))})",
        f"center_xy_mm=({fmt3(result.get('center_x_mm'))}, "
        f"{fmt3(result.get('center_y_mm'))})",
        f"center=({cx},{cy})",
        f"depth_count={result['depth_count']}",
        f"score={obj['score']:.2f}",
        f"area={obj['area_ratio']:.3f}",
        f"rect={obj['rectangularity']:.2f}",
        f"aspect={obj['aspect_ratio']:.2f}",
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
    dimensions = result.get("dimensions") or {}

    return {
        "timestamp": timestamp,
        "success": True,
        "center_pixel": {
            "u": int(result["center_full"][0]),
            "v": int(result["center_full"][1]),
        },
        "distance_from_object_center_to_camera_mm": (
            float(result["distance_mm"])
            if result["distance_mm"] is not None
            else None
        ),
        "depth_count": int(result["depth_count"]),
        "angle_deg": json_number(result.get("angle_deg")),
        "center_x_mm": json_number(result.get("center_x_mm")),
        "center_y_mm": json_number(result.get("center_y_mm")),
        "length_mm": json_number(dimensions.get("length_mm")),
        "width_mm": json_number(dimensions.get("width_mm")),
        "height_mm": json_number(dimensions.get("height_mm")),
        "grasp_x_mm": json_number(result.get("grasp_x_mm")),
        "grasp_y_mm": json_number(result.get("grasp_y_mm")),
        "grasp_z_mm": json_number(result.get("grasp_z_mm")),
        "grasp_candidates": [
            {
                "x_mm": json_number(c.get("x_mm")),
                "y_mm": json_number(c.get("y_mm")),
                "z_mm": json_number(c.get("z_mm")),
            }
            # Retries only -- the list's head IS grasp_point, so it is dropped.
            for c in (result.get("grasp_candidates") or [])[1:]
        ],
        # roi / depth_alignment / selected_mask deliberately omitted: nothing read
        # them back and they are recoverable from cfg and the annotated PNG.
        # See "Diagnostics deliberately NOT in the JSON" in the README.
    }
