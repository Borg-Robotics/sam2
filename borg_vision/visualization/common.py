"""Shared visualization helpers used by every mode.

Depth colormaps, the live ROI heatmap, ROI axis/label drawing and mask saving
are mode-independent. Per-mode overlays/JSON live in visualization/<mode>.py.
"""

import cv2
import numpy as np


def make_depth_vis(cfg, depth):
    valid = (depth > cfg.min_valid_depth_mm) & (depth < cfg.max_valid_depth_mm)

    if not np.any(valid):
        gray = np.zeros_like(depth, dtype=np.uint8)
    else:
        vals = depth[valid]
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)
        clipped = np.clip(depth, lo, hi)

        gray = cv2.normalize(
            clipped,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        gray[~valid] = 0

    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def make_live_depth_heatmap(cfg, depth):
    heatmap = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    depth_roi = depth[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2]
    valid = (depth_roi > cfg.min_valid_depth_mm) & (depth_roi < cfg.max_valid_depth_mm)

    if np.any(valid):
        vals = depth_roi[valid].astype(np.float32)
        near = np.percentile(vals, 5)
        far = np.percentile(vals, 95)

        clipped = np.clip(depth_roi.astype(np.float32), near, far)
        norm = ((far - clipped) / max(far - near, 1.0) * 255.0).astype(np.uint8)
        norm[~valid] = 0

        roi_heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        heatmap[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_heat

    cv2.rectangle(heatmap, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
    draw_roi_axes(cfg, heatmap)

    cv2.putText(
        heatmap,
        "CLASSIFICATION DEPTH HEATMAP",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return heatmap


def put_text_lines(img, lines, x=30, y=40, line_h=30):
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_roi_axes(cfg, image):
    origin_x = cfg.roi_x1 + 35
    origin_y = cfg.roi_y1 + 35
    arrow_len = 90

    cv2.arrowedLine(
        image,
        (origin_x, origin_y),
        (origin_x + arrow_len, origin_y),
        (0, 255, 255),
        4,
        tipLength=0.25,
    )

    cv2.putText(
        image,
        "X+",
        (origin_x + arrow_len + 10, origin_y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        image,
        (origin_x, origin_y),
        (origin_x, origin_y + arrow_len),
        (0, 255, 255),
        4,
        tipLength=0.25,
    )

    cv2.putText(
        image,
        "Y+",
        (origin_x - 15, origin_y + arrow_len + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
