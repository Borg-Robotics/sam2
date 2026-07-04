"""Visualization for the polymailer product-release monitor.

Ported verbatim from polymailer_product_release.py (overlay_mask,
draw_mask_outline, draw_polygon, make_live_view, save_live_result). The live
view is no longer shown in a GUI window in production; the detector renders
it on demand and the action server saves it as event snapshots.

One deviation from the prototype: save_live_result takes an optional out_dir
(and creates it) so the ROS server can group artifacts per goal; it defaults
to cfg.save_dir like the prototype's module-level SAVE_DIR.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def overlay_mask(image, mask, color, alpha=0.35):
    """Blend a solid BGR color over pixels where mask is nonzero."""
    output = image.copy()
    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        return output

    color_array = np.array(color, dtype=np.float32)
    source_pixels = output[mask_bool].astype(np.float32)
    blended_pixels = (
        (1.0 - float(alpha)) * source_pixels
        + float(alpha) * color_array
    )

    output[mask_bool] = np.clip(blended_pixels, 0, 255).astype(np.uint8)
    return output

def draw_mask_outline(image, mask, color, thickness=2):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)

def draw_polygon(image, polygon, color, thickness=2):
    if polygon is None:
        return

    points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, color, thickness)

def make_live_view(
    cfg,
    frame_bgr,
    current_poly_mask,
    baseline_mask,
    slit_geometry,
    emerging_candidates,
    latest_sam,
    state,
    frame_count,
    flow_point_count,
    tracker_ok,
    candidate_streak,
    exit_confirmed,
    full_out_streak,
    full_out_confirmed,
    selected_slit_side,
):
    display = frame_bgr.copy()
    roi_view = display[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    if baseline_mask is not None:
        draw_mask_outline(roi_view, baseline_mask, (255, 255, 0), 1)

    if current_poly_mask is not None:
        roi_view = overlay_mask(
            roi_view,
            current_poly_mask,
            (255, 0, 255),
            alpha=0.22,
        )
        draw_mask_outline(roi_view, current_poly_mask, (255, 0, 255), 3)

    if slit_geometry is not None:
        draw_polygon(
            roi_view,
            slit_geometry["corridor_polygon"],
            (0, 255, 255),
            2,
        )
        draw_polygon(
            roi_view,
            slit_geometry["contact_polygon"],
            (255, 100, 0),
            1,
        )

        point_a = tuple(np.round(slit_geometry["point_a"]).astype(int))
        point_b = tuple(np.round(slit_geometry["point_b"]).astype(int))
        cv2.line(roi_view, point_a, point_b, (255, 0, 0), 5)

        midpoint = slit_geometry["midpoint"]
        outward_end = midpoint + slit_geometry["outward"] * 90.0
        cv2.arrowedLine(
            roi_view,
            tuple(np.round(midpoint).astype(int)),
            tuple(np.round(outward_end).astype(int)),
            (255, 0, 0),
            3,
            tipLength=0.25,
        )

        x1, y1, x2, y2 = slit_geometry["crop_box"]
        cv2.rectangle(roi_view, (x1, y1), (x2, y2), (0, 180, 255), 1)

    for candidate_index, candidate in enumerate(emerging_candidates[:3]):
        if full_out_confirmed:
            color = (0, 255, 0)
        elif exit_confirmed:
            color = (255, 220, 0)
        else:
            color = (0, 140, 255)

        alpha = 0.48 if candidate_index == 0 else 0.22
        thickness = 4 if candidate_index == 0 else 2
        roi_view = overlay_mask(
            roi_view,
            candidate["mask"],
            color,
            alpha=alpha,
        )
        draw_mask_outline(roi_view, candidate["mask"], color, thickness)
        x, y, width, height = candidate["bbox"]
        cv2.rectangle(
            roi_view,
            (x, y),
            (x + width, y + height),
            color,
            thickness,
        )
        if full_out_confirmed:
            label = "FULL PRODUCT ON TABLE"
        elif candidate["mode"] == "released_product":
            label = "VERIFYING FULL RELEASE"
        elif candidate["mode"] == "sliding_through_slit":
            label = "PRODUCT SLIDING OUT"
        else:
            label = "PRODUCT OUTSIDE SLIT"
        cv2.putText(
            roi_view,
            label,
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    display[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2] = roi_view
    cv2.rectangle(
        display,
        (cfg.roi_x1, cfg.roi_y1),
        (cfg.roi_x2, cfg.roi_y2),
        (0, 255, 255),
        2,
    )

    sam_mode = "waiting"
    sam_ms = None
    sam_rate = None
    sam_frame = None
    sam_masks = 0
    crop_size = None
    error = None

    if latest_sam is not None:
        sam_mode = latest_sam["mode"]
        sam_ms = latest_sam["sam_seconds"] * 1000.0
        sam_frame = latest_sam["frame_id"]
        sam_masks = len(latest_sam["masks"])
        error = latest_sam["error"]
        x1, y1, x2, y2 = latest_sam["crop_box"]
        crop_size = f"{x2 - x1}x{y2 - y1}"

        if latest_sam["sam_seconds"] > 0:
            sam_rate = 1.0 / latest_sam["sam_seconds"]

    lines = [
        f"MOVING SLIT: {state}",
        f"camera_frame={frame_count} sam_frame={sam_frame}",
        f"sam_mode={sam_mode} crop={crop_size} masks={sam_masks}",
        f"sam_time_ms={sam_ms:.1f}" if sam_ms is not None else "sam_time_ms=waiting",
        f"sam_rate_hz={sam_rate:.2f}" if sam_rate is not None else "sam_rate_hz=waiting",
        f"flow_points={flow_point_count} tracker_ok={tracker_ok}",
        f"candidates={len(emerging_candidates)} exit_streak={candidate_streak}",
        f"full_out_streak={full_out_streak}/{cfg.full_release_required_stable_updates}",
        f"selected_slit_side={selected_slit_side.upper()}",
        "T=top B=bottom L=left R=right SPACE=start X=reset S=save Q=quit",
    ]

    if emerging_candidates:
        best = emerging_candidates[0]
        lines.append(
            "best: "
            f"score={best['score']:.2f} "
            f"outward={best['outward_fraction']:.2f} "
            f"contact={best['contact_overlap']:.2f} "
            f"bag_overlap={best['poly_overlap']:.2f} "
            f"core_overlap={best.get('core_poly_overlap', 0.0):.2f} "
            f"trail_clear={best.get('core_trailing_edge_clearance_px', 0.0):.1f}px "
            f"bag_gap={best.get('core_poly_gap_px', 0.0):.1f}px"
        )

    if error:
        lines.append(f"SAM ERROR: {error[:90]}")

    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (20, 32 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if full_out_confirmed:
        banner = "FULL PRODUCT OUT - DISCARD MAILER"
        banner_color = (0, 160, 0)
    elif exit_confirmed:
        banner = "PRODUCT EXIT SEEN - VERIFYING FULL RELEASE"
        banner_color = (180, 110, 0)
    else:
        banner = None
        banner_color = None

    if banner is not None:
        text_size, _ = cv2.getTextSize(
            banner,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            3,
        )
        x = max(20, (display.shape[1] - text_size[0]) // 2)
        cv2.rectangle(
            display,
            (x - 18, display.shape[0] - 78),
            (x + text_size[0] + 18, display.shape[0] - 20),
            banner_color,
            -1,
        )
        cv2.putText(
            display,
            banner,
            (x, display.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    return display

def save_live_result(
    cfg,
    frame_bgr,
    display_bgr,
    baseline_mask,
    current_poly_mask,
    slit_geometry,
    best_candidate,
    out_dir=None,
):
    save_dir = Path(out_dir) if out_dir is not None else Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = save_dir / f"moving_slit_raw_{timestamp}.jpg"
    view_path = save_dir / f"moving_slit_view_{timestamp}.png"
    cv2.imwrite(str(raw_path), frame_bgr)
    cv2.imwrite(str(view_path), display_bgr)

    print()
    print(f"Saved raw image: {raw_path}")
    print(f"Saved live view: {view_path}")

    if baseline_mask is not None:
        path = save_dir / f"baseline_polymailer_mask_{timestamp}.png"
        cv2.imwrite(str(path), baseline_mask.astype(np.uint8) * 255)
        print(f"Saved baseline mask: {path}")

    if current_poly_mask is not None:
        path = save_dir / f"tracked_polymailer_mask_{timestamp}.png"
        cv2.imwrite(str(path), current_poly_mask.astype(np.uint8) * 255)
        print(f"Saved tracked mask: {path}")

    if slit_geometry is not None:
        slit_image = np.zeros(((cfg.roi_y2 - cfg.roi_y1), (cfg.roi_x2 - cfg.roi_x1), 3), dtype=np.uint8)
        draw_polygon(
            slit_image,
            slit_geometry["corridor_polygon"],
            (0, 255, 255),
            2,
        )
        point_a = tuple(np.round(slit_geometry["point_a"]).astype(int))
        point_b = tuple(np.round(slit_geometry["point_b"]).astype(int))
        cv2.line(slit_image, point_a, point_b, (255, 0, 0), 5)
        path = save_dir / f"tracked_slit_{timestamp}.png"
        cv2.imwrite(str(path), slit_image)
        print(f"Saved tracked slit: {path}")

    if best_candidate is not None:
        path = save_dir / f"emerging_product_mask_{timestamp}.png"
        cv2.imwrite(str(path), best_candidate["mask"].astype(np.uint8) * 255)
        print(f"Saved product mask: {path}")
