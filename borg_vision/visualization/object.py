"""Object-mode overlay and JSON result building.

Ported verbatim from run_object()/object_detection_final.py; shared helpers
(depth colormap) come from visualization/common.py.
"""

import cv2
import numpy as np

from .common import make_depth_vis


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

    result_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_rgb

    cv2.rectangle(result_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

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

    cx, cy = result["center_full"]

    cv2.circle(result_rgb, (cx, cy), 8, (255, 255, 0), -1)
    cv2.circle(result_rgb, (cx, cy), 16, (255, 255, 0), 2)

    cv2.circle(depth_vis, (cx, cy), 8, (0, 255, 255), -1)
    cv2.circle(depth_vis, (cx, cy), 16, (0, 255, 255), 2)

    distance_mm = result["distance_mm"]

    distance_text = (
        "center_distance=None"
        if distance_mm is None
        else f"center_distance={distance_mm:.1f}mm"
    )

    lines = [
        "OBJECT SEGMENT + DEPTH",
        distance_text,
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
    obj = result["object"]

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
        "roi": {
            "x1": cfg.roi_x1,
            "y1": cfg.roi_y1,
            "x2": cfg.roi_x2,
            "y2": cfg.roi_y2,
        },
        "depth_alignment": {
            "scale_x": cfg.depth_align_scale_x,
            "scale_y": cfg.depth_align_scale_y,
            "shift_x_px": cfg.depth_align_x_shift_px,
            "shift_y_px": cfg.depth_align_y_shift_px,
        },
        "selected_mask": {
            "index": int(obj["index"]),
            "score": float(obj["score"]),
            "area_ratio": float(obj["area_ratio"]),
            "bbox_roi": [int(v) for v in obj["bbox"]],
            "rectangularity": float(obj["rectangularity"]),
            "aspect_ratio": float(obj["aspect_ratio"]),
            "center_score": float(obj["center_score"]),
        },
    }
