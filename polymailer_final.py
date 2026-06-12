# polymailer_final.py

import cv2
import torch
import depthai as dai
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


SAVE_DIR = Path("polymailer_segment_depth_heatmap_results")
SAVE_DIR.mkdir(exist_ok=True)

FPS = 20
WARMUP_SECONDS = 5.0

CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

ROI_X1 = 330
ROI_Y1 = 60
ROI_X2 = 940
ROI_Y2 = 700

BASE_DEPTH_MM = 705.0
POLY_MEASURE_DEPTH_MM = BASE_DEPTH_MM
POLY_MEASURE_ERODE_PX = 8
POLY_SIZE_SCALE = 1.04

SAM_POINTS_PER_SIDE = 24
SAM_PRED_IOU_THRESH = 0.78
SAM_STABILITY_SCORE_THRESH = 0.82
SAM_MIN_MASK_REGION_AREA = 500

DEPTH_ALIGN_X_SHIFT_PX = -40
DEPTH_ALIGN_Y_SHIFT_PX = 0
DEPTH_ALIGN_SCALE_X = 1.25
DEPTH_ALIGN_SCALE_Y = 1.25

MIN_VALID_DEPTH_MM = 450
MAX_VALID_DEPTH_MM = 1200

IR_LASER_INTENSITY = 1.0
IR_FLOOD_INTENSITY = 0.0
CONFIDENCE_THRESHOLD = 120
USE_SUBPIXEL = True
USE_LEFT_RIGHT_CHECK = True
USE_MANUAL_STEREO_EXPOSURE = True
STEREO_EXPOSURE_US = 1000
STEREO_ISO = 400

CENTER_DEPTH_RADIUS_PX = 35

MIN_POLY_AREA_RATIO = 0.04
MAX_POLY_AREA_RATIO = 0.75
TARGET_POLY_AREA_RATIO = 0.55

MIN_POLY_RECTANGULARITY = 0.35
MAX_POLY_ASPECT_RATIO = 3.5

MIN_POLY_VALUE = 80
MIN_POLY_COLOR_SCORE = 0.30

POLY_INNER_ERODE_PX = 22
POLY_SURFACE_DEPTH_PERCENTILE = 75

POLY_BULGE_MIN_MM = 7
POLY_BULGE_MAX_MM = 140

POLY_BULGE_OPEN_KERNEL_PX = 7
POLY_BULGE_CLOSE_KERNEL_PX = 28
POLY_BULGE_DILATE_PX = 5

MIN_PRODUCT_AREA_RATIO_OF_POLY = 0.010
MAX_PRODUCT_AREA_RATIO_OF_POLY = 0.65


def fmt3(value):
    if value is None:
        return "None"
    value = round(float(value), 3)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def json_number(value):
    if value is None:
        return None
    value = round(float(value), 3)
    if value.is_integer():
        return int(value)
    return value


def estimate_polymailer_depth_mm(polymailer_face_depth_mm):
    if polymailer_face_depth_mm is None:
        return None

    depth_mm = BASE_DEPTH_MM - polymailer_face_depth_mm

    if depth_mm < 0:
        depth_mm = 0.0

    return float(depth_mm)


def draw_roi_axes(image):
    origin_x = ROI_X1 + 35
    origin_y = ROI_Y1 + 35
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


def align_depth_to_rgb(depth):
    h, w = depth.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    matrix = np.float32(
        [
            [
                DEPTH_ALIGN_SCALE_X,
                0,
                (1.0 - DEPTH_ALIGN_SCALE_X) * cx + DEPTH_ALIGN_X_SHIFT_PX,
            ],
            [
                0,
                DEPTH_ALIGN_SCALE_Y,
                (1.0 - DEPTH_ALIGN_SCALE_Y) * cy + DEPTH_ALIGN_Y_SHIFT_PX,
            ],
        ]
    )

    return cv2.warpAffine(
        depth,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_depth_vis(depth):
    valid = (depth > MIN_VALID_DEPTH_MM) & (depth < MAX_VALID_DEPTH_MM)

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


def make_live_depth_heatmap(depth):
    heatmap = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    depth_roi = depth[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    valid = (depth_roi > MIN_VALID_DEPTH_MM) & (depth_roi < MAX_VALID_DEPTH_MM)

    if np.any(valid):
        vals = depth_roi[valid].astype(np.float32)
        near = np.percentile(vals, 5)
        far = np.percentile(vals, 95)

        clipped = np.clip(depth_roi.astype(np.float32), near, far)
        norm = ((far - clipped) / max(far - near, 1.0) * 255.0).astype(np.uint8)
        norm[~valid] = 0

        roi_heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        heatmap[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_heat

    cv2.rectangle(heatmap, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    draw_roi_axes(heatmap)

    cv2.putText(
        heatmap,
        "LIVE DEPTH HEATMAP",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return heatmap


def apply_manual_exposure(camera_node, label):
    if not USE_MANUAL_STEREO_EXPOSURE:
        return

    try:
        camera_node.initialControl.setManualExposure(
            STEREO_EXPOSURE_US,
            STEREO_ISO,
        )
        print(f"{label}: manual exposure {STEREO_EXPOSURE_US} us ISO {STEREO_ISO}")
    except Exception as e:
        print(f"{label}: could not set manual exposure: {e}")


def set_ir(device):
    try:
        device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
        print(f"IR laser set to {IR_LASER_INTENSITY}")
    except Exception as e:
        print(f"Could not set IR laser: {e}")

    try:
        device.setIrFloodLightIntensity(IR_FLOOD_INTENSITY)
        print(f"IR flood set to {IR_FLOOD_INTENSITY}")
    except Exception as e:
        print(f"Could not set IR flood: {e}")


def get_rgb_intrinsics(device):
    try:
        calibration = device.readCalibration()
        intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            1280,
            720,
        )

        return {
            "fx": float(intrinsics[0][0]),
            "fy": float(intrinsics[1][1]),
            "cx": float(intrinsics[0][2]),
            "cy": float(intrinsics[1][2]),
        }

    except Exception as e:
        print(f"Could not read RGB intrinsics: {e}")
        return None


def pixel_to_camera_xy_mm(center_full, depth_mm, intrinsics):
    if center_full is None or depth_mm is None or intrinsics is None:
        return None, None

    u, v = center_full

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x_mm = (u - cx) * depth_mm / fx
    y_mm = (v - cy) * depth_mm / fy

    return float(x_mm), float(y_mm)


def build_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build()

    rgb_output = cam.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS,
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    apply_manual_exposure(left, "left stereo")
    apply_manual_exposure(right, "right stereo")

    left_out = left.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    right_out = right.requestOutput(
        size=(1280, 720),
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo.setSubpixel(USE_SUBPIXEL)

    try:
        stereo.setOutputSize(1280, 720)
        print("Stereo depth output size set to 1280x720.")
    except Exception as e:
        print(f"Could not set stereo output size: {e}")

    try:
        stereo.initialConfig.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
        print(f"Confidence threshold set to {CONFIDENCE_THRESHOLD}.")
    except Exception as e:
        print(f"Could not set confidence threshold: {e}")

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    rgb_queue = rgb_output.createOutputQueue()
    depth_queue = stereo.depth.createOutputQueue()

    return pipeline, rgb_queue, depth_queue


def close_mask(mask, kernel_px=21, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )


def open_mask(mask, kernel_px=7, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )


def erode_mask(mask, radius_px):
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilate_mask(mask, radius_px):
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


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


def clean_mask(mask):
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask.astype(np.uint8)

    cleaned = close_mask(mask, kernel_px=21, iterations=1)
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * 2.8:
        return cleaned.astype(np.uint8)

    return mask.astype(np.uint8)


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def get_rotated_box_from_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    (cx, cy), (w_px, h_px), angle = rect

    return {
        "center_roi": (float(cx), float(cy)),
        "width_px": float(w_px),
        "height_px": float(h_px),
        "angle_deg": float(angle),
        "points_roi": box_points,
    }


def estimate_polymailer_dimensions_mm(poly, intrinsics):
    empty = {
        "length_mm": None,
        "width_mm": None,
        "angle_deg": None,
        "points_full": None,
    }

    if intrinsics is None:
        return empty

    measure_mask = erode_mask(poly["mask"], POLY_MEASURE_ERODE_PX)

    if int(measure_mask.sum()) < 100:
        measure_mask = poly["mask"].copy()

    rotated = get_rotated_box_from_mask(measure_mask)

    if rotated is None:
        return empty

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    raw_width_mm = (w_px * POLY_MEASURE_DEPTH_MM) / fx
    raw_height_mm = (h_px * POLY_MEASURE_DEPTH_MM) / fy

    length_mm = max(raw_width_mm, raw_height_mm) * POLY_SIZE_SCALE
    width_mm = min(raw_width_mm, raw_height_mm) * POLY_SIZE_SCALE

    points_roi = rotated["points_roi"]
    points_full = points_roi.copy()
    points_full[:, 0] += ROI_X1
    points_full[:, 1] += ROI_Y1

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "angle_deg": float(rotated["angle_deg"]),
        "points_full": points_full.tolist(),
    }


def get_component_near_center(mask, center):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    cx, cy = center
    best_label = None
    best_score = None

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        bx = x + w / 2
        by = y + h / 2

        dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
        score = area - 2.0 * dist

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    return (labels == best_label).astype(np.uint8)


def center_depth_mm(depth_roi, mask, center_roi):
    cx, cy = center_roi
    h, w = depth_roi.shape[:2]

    x1 = max(0, cx - CENTER_DEPTH_RADIUS_PX)
    x2 = min(w, cx + CENTER_DEPTH_RADIUS_PX + 1)
    y1 = max(0, cy - CENTER_DEPTH_RADIUS_PX)
    y2 = min(h, cy + CENTER_DEPTH_RADIUS_PX + 1)

    patch_depth = depth_roi[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    values = patch_depth[
        (patch_mask == 1)
        & (patch_depth > MIN_VALID_DEPTH_MM)
        & (patch_depth < MAX_VALID_DEPTH_MM)
    ].astype(np.float32)

    if values.size < 30:
        values = patch_depth[
            (patch_depth > MIN_VALID_DEPTH_MM)
            & (patch_depth < MAX_VALID_DEPTH_MM)
        ].astype(np.float32)

    if values.size < 30:
        return None, 0

    return float(np.median(values)), int(values.size)


def estimate_product_inside_polymailer(depth_roi, poly_mask, poly_center_roi):
    inner_mask = erode_mask(poly_mask, POLY_INNER_ERODE_PX)

    if int(inner_mask.sum()) < 100:
        inner_mask = poly_mask.copy()

    valid_inside = (
        (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
        & (inner_mask == 1)
    )

    inside_values = depth_roi[valid_inside].astype(np.float32)

    if inside_values.size < 100:
        return {
            "found": False,
            "reason": "not_enough_depth",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": None,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    surface_depth_mm = float(np.percentile(inside_values, POLY_SURFACE_DEPTH_PERCENTILE))

    product_candidate = (
        valid_inside
        & (depth_roi <= surface_depth_mm - POLY_BULGE_MIN_MM)
        & (depth_roi >= surface_depth_mm - POLY_BULGE_MAX_MM)
    ).astype(np.uint8)

    product_candidate = open_mask(product_candidate, POLY_BULGE_OPEN_KERNEL_PX, iterations=1)
    product_candidate = close_mask(product_candidate, POLY_BULGE_CLOSE_KERNEL_PX, iterations=2)
    product_candidate = dilate_mask(product_candidate, POLY_BULGE_DILATE_PX)

    product_candidate[poly_mask == 0] = 0

    product_mask = get_component_near_center(product_candidate, poly_center_roi)

    if product_mask is None or int(product_mask.sum()) == 0:
        return {
            "found": False,
            "reason": "no_product_bulge",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": surface_depth_mm,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    product_area = int(product_mask.sum())
    poly_area = int(poly_mask.sum())
    area_ratio_of_poly = product_area / max(poly_area, 1)

    if area_ratio_of_poly < MIN_PRODUCT_AREA_RATIO_OF_POLY:
        found = False
        reason = "product_area_too_small"
    elif area_ratio_of_poly > MAX_PRODUCT_AREA_RATIO_OF_POLY:
        found = False
        reason = "product_area_too_large"
    else:
        found = True
        reason = "ok"

    product_values = depth_roi[
        (product_mask == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    ].astype(np.float32)

    if product_values.size > 0:
        product_depth_mm = float(np.percentile(product_values, 20))
        bulge_height_mm = float(surface_depth_mm - product_depth_mm)
    else:
        product_depth_mm = None
        bulge_height_mm = None

    center_roi = get_mask_center(product_mask)
    bbox = cv2.boundingRect(product_mask.astype(np.uint8))

    return {
        "found": bool(found),
        "reason": reason,
        "mask": product_mask.astype(np.uint8),
        "surface_depth_mm": surface_depth_mm,
        "product_depth_mm": product_depth_mm,
        "bulge_height_mm": bulge_height_mm,
        "area_ratio_of_poly": float(area_ratio_of_poly),
        "bbox": bbox,
        "center_roi": center_roi,
    }


def make_polymailer_depth_heatmap(depth_roi, poly_mask, product_mask, surface_depth_mm):
    h, w = depth_roi.shape[:2]
    debug = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (
        (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
        & (poly_mask == 1)
    )

    if surface_depth_mm is not None and np.any(valid):
        diff = np.zeros_like(depth_roi, dtype=np.float32)
        diff[valid] = surface_depth_mm - depth_roi[valid].astype(np.float32)

        diff_clip = np.clip(diff, 0, POLY_BULGE_MAX_MM)

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


def choose_polymailer_mask(masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(raw_mask)

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < MIN_POLY_AREA_RATIO:
            continue

        if area_ratio > MAX_POLY_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < MIN_POLY_RECTANGULARITY:
            continue

        if aspect_ratio > MAX_POLY_ASPECT_RATIO:
            continue

        masked_hsv = hsv[mask == 1]

        if masked_hsv.size == 0:
            continue

        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))

        if mean_v < MIN_POLY_VALUE:
            continue

        hue_score = 1.0 - min(abs(mean_h - 18.0) / 25.0, 1.0)
        sat_score = 1.0 - min(abs(mean_s - 70.0) / 90.0, 1.0)
        val_score = 1.0 - min(abs(mean_v - 170.0) / 100.0, 1.0)

        color_score = 0.50 * hue_score + 0.25 * sat_score + 0.25 * val_score

        if color_score < MIN_POLY_COLOR_SCORE:
            continue

        cx = x + bw / 2
        cy = y + bh / 2

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)

        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - TARGET_POLY_AREA_RATIO) / TARGET_POLY_AREA_RATIO,
            1.0,
        )

        score = (
            3.0 * color_score
            + 1.5 * rectangularity
            + 1.2 * area_score
            + 1.0 * center_score
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "color_score": float(color_score),
            "mean_hsv": (mean_h, mean_s, mean_v),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_polymailer(frame_bgr, depth_aligned, mask_generator, intrinsics):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    depth_roi = depth_aligned[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    poly = choose_polymailer_mask(masks, roi_rgb)

    if poly is None:
        print("No valid polymailer mask found.")
        return None

    poly_center_roi = get_mask_center(poly["mask"])

    if poly_center_roi is None:
        print("Could not get polymailer center.")
        return None

    polymailer_face_depth_mm, depth_count = center_depth_mm(
        depth_roi,
        poly["mask"],
        poly_center_roi,
    )

    polymailer_depth_mm = estimate_polymailer_depth_mm(polymailer_face_depth_mm)

    dimensions = estimate_polymailer_dimensions_mm(poly, intrinsics)

    product = estimate_product_inside_polymailer(
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        poly_center_roi=poly_center_roi,
    )

    depth_heatmap = make_polymailer_depth_heatmap(
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        product_mask=product["mask"],
        surface_depth_mm=product["surface_depth_mm"],
    )

    poly_center_full = (
        ROI_X1 + poly_center_roi[0],
        ROI_Y1 + poly_center_roi[1],
    )

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        poly_center_full,
        polymailer_face_depth_mm,
        intrinsics,
    )

    if product["center_roi"] is not None:
        product_center_full = (
            ROI_X1 + product["center_roi"][0],
            ROI_Y1 + product["center_roi"][1],
        )
    else:
        product_center_full = None

    product_inside_center_face_depth = product["product_depth_mm"]

    product_inside_center_x_mm, product_inside_center_y_mm = pixel_to_camera_xy_mm(
        product_center_full,
        product_inside_center_face_depth,
        intrinsics,
    )

    if product["bbox"] is not None:
        px, py, pw, ph = product["bbox"]
        product_bbox_full = (
            ROI_X1 + px,
            ROI_Y1 + py,
            pw,
            ph,
        )
    else:
        product_bbox_full = None

    print()
    print("POLYMAILER OUTPUT:")
    print(f"  polymailer_face_depth_mm:          {fmt3(polymailer_face_depth_mm)}")
    print(f"  polymailer_depth_mm:               {fmt3(polymailer_depth_mm)}")
    print(f"  length_mm:                         {fmt3(dimensions['length_mm'])}")
    print(f"  width_mm:                          {fmt3(dimensions['width_mm'])}")
    print(f"  angle_deg:                         {fmt3(dimensions['angle_deg'])}")
    print(f"  center_x_mm:                       {fmt3(center_x_mm)}")
    print(f"  center_y_mm:                       {fmt3(center_y_mm)}")
    print(f"  product_inside_center_x_mm:        {fmt3(product_inside_center_x_mm)}")
    print(f"  product_inside_center_y_mm:        {fmt3(product_inside_center_y_mm)}")
    print(f"  product_inside_center_face_depth:  {fmt3(product_inside_center_face_depth)}")

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "poly": poly,
        "poly_center_roi": poly_center_roi,
        "poly_center_full": poly_center_full,
        "polymailer_face_depth_mm": polymailer_face_depth_mm,
        "polymailer_depth_mm": polymailer_depth_mm,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "depth_count": depth_count,
        "dimensions": dimensions,
        "product": product,
        "product_center_full": product_center_full,
        "product_bbox_full": product_bbox_full,
        "product_inside_center_x_mm": product_inside_center_x_mm,
        "product_inside_center_y_mm": product_inside_center_y_mm,
        "product_inside_center_face_depth": product_inside_center_face_depth,
        "depth_heatmap": depth_heatmap,
    }


def draw_rotated_poly_outline(result_rgb, depth_vis, poly, dimensions):
    points_full = dimensions.get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)

        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 255), 3)
        cv2.polylines(depth_vis, [points_full_np], True, (255, 0, 255), 3)

        return

    x, y, w, h = poly["bbox"]

    cv2.rectangle(
        result_rgb,
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
        (255, 0, 255),
        3,
    )

    cv2.rectangle(
        depth_vis,
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
        (255, 0, 255),
        3,
    )


def draw_result(frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(depth_aligned)

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

    result_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_rgb

    cv2.rectangle(result_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

    draw_roi_axes(result_rgb)
    draw_roi_axes(depth_vis)

    draw_rotated_poly_outline(result_rgb, depth_vis, poly, dimensions)

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

    heatmap_full[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    draw_roi_axes(heatmap_full)

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


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main():
    print("Starting OAK + SAM 2 polymailer final output")
    print("Keys:")
    print("  SPACE = capture and output polymailer measurements")
    print("  q     = quit")
    print()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 2 on device: {device_name}")

    sam2_model = build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device=device_name,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )

    pipeline, rgb_queue, depth_queue = build_pipeline()
    pipeline.start()

    start_time = time.time()

    with pipeline:
        device = pipeline.getDefaultDevice()
        set_ir(device)
        intrinsics = get_rgb_intrinsics(device)

        while pipeline.isRunning():
            rgb_msg = rgb_queue.get()
            depth_msg = depth_queue.get()

            rgb = rgb_msg.getCvFrame()
            depth_raw = depth_msg.getFrame()
            depth_aligned = align_depth_to_rgb(depth_raw)

            elapsed = time.time() - start_time
            warmed = elapsed >= WARMUP_SECONDS

            preview_rgb = rgb.copy()
            preview_depth = make_depth_vis(depth_aligned)
            preview_heatmap = make_live_depth_heatmap(depth_aligned)

            cv2.rectangle(preview_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

            draw_roi_axes(preview_rgb)
            draw_roi_axes(preview_depth)

            status = "READY - SPACE to measure" if warmed else f"WARMING {WARMUP_SECONDS - elapsed:.1f}s"

            cv2.putText(
                preview_rgb,
                "POLYMAILER OUTPUT",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview_rgb,
                status,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            combined_preview = np.hstack([preview_rgb, preview_depth, preview_heatmap])
            cv2.imshow("OAK Polymailer Output", combined_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before running.")
                    continue

                result = run_sam2_polymailer(
                    rgb,
                    depth_aligned,
                    mask_generator,
                    intrinsics,
                )

                if result is None:
                    continue

                output_bgr = draw_result(
                    rgb,
                    depth_aligned,
                    result,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                raw_path = SAVE_DIR / f"raw_rgb_{timestamp}.jpg"
                result_path = SAVE_DIR / f"polymailer_output_{timestamp}.png"
                poly_mask_path = SAVE_DIR / f"polymailer_mask_{timestamp}.png"
                product_mask_path = SAVE_DIR / f"product_bulge_mask_{timestamp}.png"
                json_path = SAVE_DIR / f"polymailer_output_{timestamp}.json"

                cv2.imwrite(str(raw_path), rgb)
                cv2.imwrite(str(result_path), output_bgr)
                save_binary_mask(poly_mask_path, result["poly"]["mask"])
                save_binary_mask(product_mask_path, result["product"]["mask"])

                json_result = {
                    "polymailer_face_depth_mm": json_number(result["polymailer_face_depth_mm"]),
                    "polymailer_depth_mm": json_number(result["polymailer_depth_mm"]),
                    "length_mm": json_number(result["dimensions"]["length_mm"]),
                    "width_mm": json_number(result["dimensions"]["width_mm"]),
                    "angle_deg": json_number(result["dimensions"]["angle_deg"]),
                    "center_x_mm": json_number(result["center_x_mm"]),
                    "center_y_mm": json_number(result["center_y_mm"]),
                    "product_inside_center_x_mm": json_number(result["product_inside_center_x_mm"]),
                    "product_inside_center_y_mm": json_number(result["product_inside_center_y_mm"]),
                    "product_inside_center_face_depth": json_number(result["product_inside_center_face_depth"]),
                }

                with open(json_path, "w") as f:
                    json.dump(json_result, f, indent=2)

                print()
                print(f"Saved result image: {result_path}")
                print(f"Saved polymailer mask: {poly_mask_path}")
                print(f"Saved product mask: {product_mask_path}")
                print(f"Saved JSON: {json_path}")

                cv2.imshow("Polymailer Output Result", output_bgr)
                cv2.waitKey(0)
                cv2.destroyWindow("Polymailer Output Result")

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()