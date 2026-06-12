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


SAVE_DIR = Path("clear_bag_final_results")
SAVE_DIR.mkdir(exist_ok=True)

FPS = 20
WARMUP_SECONDS = 5.0

CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

ROI_X1 = 330
ROI_Y1 = 60
ROI_X2 = 940
ROI_Y2 = 700

RGB_SIZE = (1280, 720)

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

BASE_DEPTH_MM = 705.0
POLY_SIZE_SCALE = 1.04

IR_LASER_INTENSITY = 1.0
IR_FLOOD_INTENSITY = 0.0
CONFIDENCE_THRESHOLD = 120
USE_SUBPIXEL = True
USE_LEFT_RIGHT_CHECK = True
USE_MANUAL_STEREO_EXPOSURE = True
STEREO_EXPOSURE_US = 1000
STEREO_ISO = 400

MIN_PRODUCT_AREA_RATIO = 0.003
MAX_PRODUCT_AREA_RATIO = 0.40
TARGET_PRODUCT_AREA_RATIO = 0.08

MIN_PRODUCT_RECTANGULARITY = 0.08
MAX_PRODUCT_ASPECT_RATIO = 10.0

HARD_REJECT_ROLLER_LIKE = True
ROLLER_STRIP_MIN_WIDTH_RATIO = 0.70
ROLLER_STRIP_MAX_HEIGHT_RATIO = 0.25

USE_MASK_CLEANUP = True
MASK_CLOSE_KERNEL_PX = 17
MASK_CLOSE_ITERATIONS = 1
MAX_CLEANED_AREA_GROWTH = 2.5
SAFE_ERODE_RADIUS_PX = 12

RING_DILATE_PX = 25
GOOD_CONTRAST = 35.0
BAD_CONTRAST = 5.0
GOOD_TEXTURE = 28.0
BAD_TEXTURE = 4.0

AREA_SCORE_WEIGHT = 1.2
CENTER_SCORE_WEIGHT = 1.0
CONTRAST_SCORE_WEIGHT = 2.4
TEXTURE_SCORE_WEIGHT = 1.0
RECT_SCORE_WEIGHT = 0.5
SAFE_AREA_SCORE_WEIGHT = 0.4
SAM_IOU_SCORE_WEIGHT = 0.4
SAM_STABILITY_SCORE_WEIGHT = 0.4

CENTER_TOP_FACE_RADIUS_PX = 35
MIN_CENTER_TOP_FACE_DEPTH_COUNT = 30

BAG_CLOSER_THAN_BASE_MIN_MM = 25
BAG_CLOSER_THAN_BASE_MAX_MM = 170

BAG_SEED_DILATE_PX = 135
BAG_COMPONENT_KEEP_NEAR_PRODUCT_PX = 190

BAG_CLOSE_KERNEL_PX = 19
BAG_DILATE_KERNEL_PX = 7
BAG_ERODE_KERNEL_PX = 5

BAG_MIN_AREA_RATIO = 0.025
BAG_MAX_AREA_RATIO = 0.70

BAG_USE_MEDIAN_BLUR_DEPTH = True
BAG_DEPTH_MEDIAN_BLUR_KSIZE = 5

BAG_CORNER_EXPAND_ENABLE = True
BAG_CORNER_EXPAND_DILATE_PX = 9
BAG_EDGE_ACTIVITY_SMOOTH_PX = 21
BAG_EDGE_MIN_ACTIVITY_PX = 12
BAG_EDGE_ACTIVITY_FRACTION = 0.12
BAG_CORNER_WINDOW_PX = 45
BAG_CORNER_MIN_PIXELS = 25
BAG_EXPAND_PADDING_X_PX = 0
BAG_EXPAND_PADDING_Y_PX = 0

HEATMAP_MAX_CLOSER_THAN_BASE_MM = 180.0

DEBUG_PRINT_MASKS = False


def json_number(value):
    if value is None:
        return None

    value = round(float(value), 3)

    if value.is_integer():
        return int(value)

    return value


def fmt3(value):
    if value is None:
        return "None"

    value = round(float(value), 3)

    if value.is_integer():
        return str(int(value))

    return f"{value:.3f}".rstrip("0").rstrip(".")


def score_from_range(value, bad_value, good_value):
    if value <= bad_value:
        return 0.0

    if value >= good_value:
        return 1.0

    return float((value - bad_value) / (good_value - bad_value))


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
            RGB_SIZE[0],
            RGB_SIZE[1],
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
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS,
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    apply_manual_exposure(left, "left stereo")
    apply_manual_exposure(right, "right stereo")

    left_out = left.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    right_out = right.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo.setSubpixel(USE_SUBPIXEL)

    try:
        stereo.setOutputSize(RGB_SIZE[0], RGB_SIZE[1])
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


def close_mask(mask, kernel_px, iterations=1):
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


def component_containing_point(mask, point, search_radius_px):
    px, py = point

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    h, w = mask.shape[:2]

    x1 = max(0, px - search_radius_px)
    x2 = min(w, px + search_radius_px + 1)
    y1 = max(0, py - search_radius_px)
    y2 = min(h, py + search_radius_px + 1)

    label_patch = labels[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]

    valid_labels = label_patch[mask_patch == 1]

    if valid_labels.size == 0:
        return largest_component(mask)

    labels_unique, counts = np.unique(valid_labels, return_counts=True)

    best_label = None
    best_count = -1

    for label, count in zip(labels_unique, counts):
        if label == 0:
            continue

        if count > best_count:
            best_count = count
            best_label = label

    if best_label is None:
        return largest_component(mask)

    return (labels == best_label).astype(np.uint8)


def smooth_1d(values, kernel_px):
    values = np.asarray(values).astype(np.float32)

    if kernel_px <= 1:
        return values

    if kernel_px % 2 == 0:
        kernel_px += 1

    return cv2.GaussianBlur(values.reshape(1, -1), (kernel_px, 1), 0).reshape(-1)


def active_span_from_activity(activity, min_pixels, fraction):
    activity = np.asarray(activity).astype(np.float32)

    if activity.size == 0:
        return None

    smoothed = smooth_1d(activity, BAG_EDGE_ACTIVITY_SMOOTH_PX)
    max_activity = float(np.max(smoothed))

    if max_activity <= 0:
        return None

    threshold = max(float(min_pixels), max_activity * float(fraction))
    active = smoothed >= threshold

    indices = np.where(active)[0]

    if indices.size == 0:
        return None

    return int(indices[0]), int(indices[-1])


def corner_has_depth_support(mask, x1, y1, x2, y2):
    h, w = mask.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h))

    if x2 <= x1 or y2 <= y1:
        return False

    patch = mask[y1:y2, x1:x2]
    return int(patch.sum()) >= BAG_CORNER_MIN_PIXELS


def expand_bag_rect_using_corner_depth(component, closer_than_base, product_mask):
    if not BAG_CORNER_EXPAND_ENABLE:
        return component

    component = component.astype(np.uint8)
    closer_than_base = closer_than_base.astype(np.uint8)
    product_mask = product_mask.astype(np.uint8)

    support = np.maximum(component, closer_than_base)
    support = np.maximum(support, product_mask)
    support = dilate_mask(support, BAG_CORNER_EXPAND_DILATE_PX)
    support = close_mask(support, BAG_CLOSE_KERNEL_PX, iterations=1)

    row_activity = np.sum(support, axis=1)
    col_activity = np.sum(support, axis=0)

    x_span = active_span_from_activity(
        col_activity,
        BAG_EDGE_MIN_ACTIVITY_PX,
        BAG_EDGE_ACTIVITY_FRACTION,
    )

    y_span = active_span_from_activity(
        row_activity,
        BAG_EDGE_MIN_ACTIVITY_PX,
        BAG_EDGE_ACTIVITY_FRACTION,
    )

    if x_span is None or y_span is None:
        return component

    h, w = component.shape[:2]
    x1, x2 = x_span
    y1, y2 = y_span

    top_left = corner_has_depth_support(
        closer_than_base,
        x1,
        y1,
        x1 + BAG_CORNER_WINDOW_PX,
        y1 + BAG_CORNER_WINDOW_PX,
    )

    top_right = corner_has_depth_support(
        closer_than_base,
        x2 - BAG_CORNER_WINDOW_PX,
        y1,
        x2,
        y1 + BAG_CORNER_WINDOW_PX,
    )

    bottom_left = corner_has_depth_support(
        closer_than_base,
        x1,
        y2 - BAG_CORNER_WINDOW_PX,
        x1 + BAG_CORNER_WINDOW_PX,
        y2,
    )

    bottom_right = corner_has_depth_support(
        closer_than_base,
        x2 - BAG_CORNER_WINDOW_PX,
        y2 - BAG_CORNER_WINDOW_PX,
        x2,
        y2,
    )

    cx, cy, cw, ch = cv2.boundingRect(component)

    if not (top_left or top_right):
        y1 = cy

    if not (bottom_left or bottom_right):
        y2 = cy + ch

    x1 = max(0, x1 - BAG_EXPAND_PADDING_X_PX)
    x2 = min(w - 1, x2 + BAG_EXPAND_PADDING_X_PX)
    y1 = max(0, y1 - BAG_EXPAND_PADDING_Y_PX)
    y2 = min(h - 1, y2 + BAG_EXPAND_PADDING_Y_PX)

    rect_mask = np.zeros_like(component, dtype=np.uint8)
    cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 1, thickness=-1)

    return rect_mask


def clean_mask(mask):
    mask = mask.astype(np.uint8)

    if not USE_MASK_CLEANUP:
        return mask

    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(mask, MASK_CLOSE_KERNEL_PX, MASK_CLOSE_ITERATIONS)
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * MAX_CLEANED_AREA_GROWTH:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    safe_mask = erode_mask(mask, SAFE_ERODE_RADIUS_PX)
    safe_component = largest_component(safe_mask)

    if safe_component is None:
        safe_component = largest_component(mask)

    if safe_component is None:
        return None, safe_mask

    moments = cv2.moments(safe_component.astype(np.uint8))

    if moments["m00"] == 0:
        return None, safe_component

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy), safe_component.astype(np.uint8)


def is_roller_like_horizontal_strip(mask):
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))

    width_ratio = bw / max(w, 1)
    height_ratio = bh / max(h, 1)

    return (
        width_ratio >= ROLLER_STRIP_MIN_WIDTH_RATIO
        and height_ratio <= ROLLER_STRIP_MAX_HEIGHT_RATIO
    )


def compute_rgb_contrast_score(roi_rgb, mask):
    if int(mask.sum()) < 20:
        return 0.0, 0.0

    dilated = dilate_mask(mask, RING_DILATE_PX)
    ring = dilated.copy()
    ring[mask == 1] = 0

    if int(ring.sum()) < 20:
        return 0.0, 0.0

    mask_pixels = roi_rgb[mask == 1].astype(np.float32)
    ring_pixels = roi_rgb[ring == 1].astype(np.float32)

    if mask_pixels.size == 0 or ring_pixels.size == 0:
        return 0.0, 0.0

    mask_mean = np.mean(mask_pixels, axis=0)
    ring_mean = np.mean(ring_pixels, axis=0)

    contrast = float(np.mean(np.abs(mask_mean - ring_mean)))
    contrast_score = score_from_range(contrast, BAD_CONTRAST, GOOD_CONTRAST)

    return contrast_score, contrast


def compute_texture_score(roi_rgb, mask):
    if int(mask.sum()) < 20:
        return 0.0, 0.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    values = gray[mask == 1].astype(np.float32)

    if values.size < 20:
        return 0.0, 0.0

    texture = float(np.std(values))
    texture_score = score_from_range(texture, BAD_TEXTURE, GOOD_TEXTURE)

    return texture_score, texture


def score_product_mask(mask, roi_rgb, sam_iou, sam_stability):
    mask = mask.astype(np.uint8)

    if HARD_REJECT_ROLLER_LIKE and is_roller_like_horizontal_strip(mask):
        return None

    roi_h, roi_w = mask.shape[:2]
    roi_area = roi_h * roi_w

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < MIN_PRODUCT_AREA_RATIO:
        return None

    if area_ratio > MAX_PRODUCT_AREA_RATIO:
        return None

    x, y, w, h = cv2.boundingRect(mask)

    if w <= 0 or h <= 0:
        return None

    bbox_area = w * h
    rectangularity = area / max(bbox_area, 1)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    if rectangularity < MIN_PRODUCT_RECTANGULARITY:
        return None

    if aspect_ratio > MAX_PRODUCT_ASPECT_RATIO:
        return None

    center_roi, safe_mask = get_mask_center(mask)

    if center_roi is None:
        return None

    safe_area = int(safe_mask.sum())
    safe_area_ratio = safe_area / max(roi_area, 1)

    roi_cx = roi_w / 2
    roi_cy = roi_h / 2

    dist = np.sqrt(
        (center_roi[0] - roi_cx) ** 2
        + (center_roi[1] - roi_cy) ** 2
    )

    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    area_score = 1.0 - min(
        abs(area_ratio - TARGET_PRODUCT_AREA_RATIO) / TARGET_PRODUCT_AREA_RATIO,
        1.0,
    )

    contrast_score, contrast_value = compute_rgb_contrast_score(roi_rgb, mask)
    texture_score, texture_value = compute_texture_score(roi_rgb, mask)

    quality_score = (
        0.25 * area_score
        + 0.20 * center_score
        + 0.30 * contrast_score
        + 0.15 * texture_score
        + 0.10 * min(safe_area_ratio / 0.03, 1.0)
    )

    score = (
        AREA_SCORE_WEIGHT * area_score
        + CENTER_SCORE_WEIGHT * center_score
        + CONTRAST_SCORE_WEIGHT * contrast_score
        + TEXTURE_SCORE_WEIGHT * texture_score
        + RECT_SCORE_WEIGHT * rectangularity
        + SAFE_AREA_SCORE_WEIGHT * safe_area_ratio
        + SAM_IOU_SCORE_WEIGHT * sam_iou
        + SAM_STABILITY_SCORE_WEIGHT * sam_stability
    )

    return {
        "score": float(score),
        "quality_score": float(np.clip(quality_score, 0.0, 1.0)),
        "mask": mask,
        "safe_mask": safe_mask,
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "bbox": (int(x), int(y), int(w), int(h)),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "contrast_score": float(contrast_score),
        "contrast_value": float(contrast_value),
        "texture_score": float(texture_score),
        "texture_value": float(texture_value),
        "safe_area_ratio": float(safe_area_ratio),
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
    }


def choose_best_product_mask(masks, roi_rgb):
    best = None

    print("\nChecking SAM 2 clear-bag product masks...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        cleaned_mask = clean_mask(raw_mask)

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        result = score_product_mask(
            cleaned_mask,
            roi_rgb,
            sam_iou,
            sam_stability,
        )

        if result is not None:
            result["index"] = i

            if best is None or result["score"] > best["score"]:
                best = result

    return best


def robust_depth_from_values(values):
    values = values.astype(np.float32)

    if values.size < MIN_CENTER_TOP_FACE_DEPTH_COUNT:
        return None, 0, None, None

    raw_median = float(np.median(values))
    abs_dev = np.abs(values - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = values[np.abs(values - raw_median) <= 12.0]
    else:
        filtered = values[abs_dev <= 3.5 * mad]

    if filtered.size < MIN_CENTER_TOP_FACE_DEPTH_COUNT:
        filtered = values

    return (
        float(np.median(filtered)),
        int(filtered.size),
        float(np.mean(filtered)),
        float(np.min(filtered)),
    )


def center_top_face_depth_mm(depth_roi, product_mask, center_roi):
    cx, cy = center_roi
    h, w = depth_roi.shape[:2]

    x1 = max(0, cx - CENTER_TOP_FACE_RADIUS_PX)
    x2 = min(w, cx + CENTER_TOP_FACE_RADIUS_PX + 1)
    y1 = max(0, cy - CENTER_TOP_FACE_RADIUS_PX)
    y2 = min(h, cy + CENTER_TOP_FACE_RADIUS_PX + 1)

    patch_depth = depth_roi[y1:y2, x1:x2]
    patch_mask = product_mask[y1:y2, x1:x2]

    values = patch_depth[
        (patch_mask == 1)
        & (patch_depth > MIN_VALID_DEPTH_MM)
        & (patch_depth < MAX_VALID_DEPTH_MM)
    ].astype(np.float32)

    if values.size < MIN_CENTER_TOP_FACE_DEPTH_COUNT:
        values = patch_depth[
            (patch_depth > MIN_VALID_DEPTH_MM)
            & (patch_depth < MAX_VALID_DEPTH_MM)
        ].astype(np.float32)

    depth_mm, count, mean_mm, min_mm = robust_depth_from_values(values)

    return {
        "depth_mm": depth_mm,
        "count": count,
        "mean_mm": mean_mm,
        "min_mm": min_mm,
        "radius_px": CENTER_TOP_FACE_RADIUS_PX,
        "center_roi": (int(cx), int(cy)),
        "center_full": (int(ROI_X1 + cx), int(ROI_Y1 + cy)),
    }


def estimate_base_depth_mm(depth_roi, product_mask):
    return float(BASE_DEPTH_MM)


def get_rotated_rect_from_mask(mask):
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


def detect_clear_bag_from_depth(depth_roi, product_mask, product_center_roi):
    roi_h, roi_w = depth_roi.shape[:2]
    roi_area = roi_h * roi_w

    if BAG_USE_MEDIAN_BLUR_DEPTH:
        depth_for_bag = cv2.medianBlur(
            depth_roi.astype(np.uint16),
            BAG_DEPTH_MEDIAN_BLUR_KSIZE,
        )
    else:
        depth_for_bag = depth_roi.copy()

    base_depth_mm = estimate_base_depth_mm(depth_for_bag, product_mask)

    valid_depth = (
        (depth_for_bag > MIN_VALID_DEPTH_MM)
        & (depth_for_bag < MAX_VALID_DEPTH_MM)
    )

    closer_than_base = (
        valid_depth
        & (depth_for_bag <= base_depth_mm - BAG_CLOSER_THAN_BASE_MIN_MM)
        & (depth_for_bag >= base_depth_mm - BAG_CLOSER_THAN_BASE_MAX_MM)
    ).astype(np.uint8)

    px, py = product_center_roi

    seed = np.zeros_like(closer_than_base, dtype=np.uint8)
    cv2.circle(seed, (px, py), BAG_SEED_DILATE_PX, 1, -1)

    product_seed = dilate_mask(product_mask, BAG_SEED_DILATE_PX)
    seed = np.maximum(seed, product_seed)

    candidate = np.zeros_like(closer_than_base, dtype=np.uint8)
    candidate[(closer_than_base == 1) & (seed == 1)] = 1

    candidate = close_mask(candidate, BAG_CLOSE_KERNEL_PX, iterations=2)
    candidate = dilate_mask(candidate, BAG_DILATE_KERNEL_PX)
    candidate = close_mask(candidate, BAG_CLOSE_KERNEL_PX, iterations=1)
    candidate = erode_mask(candidate, BAG_ERODE_KERNEL_PX)

    component = component_containing_point(
        candidate,
        product_center_roi,
        BAG_COMPONENT_KEEP_NEAR_PRODUCT_PX,
    )

    if component is None:
        return {
            "bag_found": False,
            "bag_reason": "no_component_near_product",
            "bag_mask": candidate,
            "bag_bbox": None,
            "bag_center_roi": None,
            "base_depth_mm": base_depth_mm,
            "bag_area_ratio": 0.0,
            "rotated": None,
        }

    component = np.maximum(component.astype(np.uint8), product_mask.astype(np.uint8))
    component = close_mask(component, BAG_CLOSE_KERNEL_PX, iterations=1)

    component = expand_bag_rect_using_corner_depth(
        component=component,
        closer_than_base=closer_than_base,
        product_mask=product_mask,
    )

    area = int(component.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < BAG_MIN_AREA_RATIO:
        found = False
        reason = "too_small"
    elif area_ratio > BAG_MAX_AREA_RATIO:
        found = False
        reason = "too_large"
    else:
        found = True
        reason = "ok"

    x, y, w, h = cv2.boundingRect(component)
    bag_center_roi = (int(x + w / 2), int(y + h / 2))
    rotated = get_rotated_rect_from_mask(component)

    return {
        "bag_found": bool(found),
        "bag_reason": reason,
        "bag_mask": component,
        "bag_bbox": (int(x), int(y), int(w), int(h)),
        "bag_center_roi": bag_center_roi,
        "base_depth_mm": float(base_depth_mm),
        "bag_area_ratio": float(area_ratio),
        "rotated": rotated,
    }


def estimate_clear_bag_measurements(bag, intrinsics):
    empty = {
        "length_mm": None,
        "width_mm": None,
        "angle_deg": None,
        "center_x_mm": None,
        "center_y_mm": None,
        "points_full": None,
    }

    if bag is None or bag.get("bag_bbox") is None:
        return empty

    if intrinsics is None or bag.get("base_depth_mm") is None:
        return empty

    depth_mm = float(bag["base_depth_mm"])
    rotated = bag.get("rotated")

    if rotated is None:
        bx, by, bw, bh = bag["bag_bbox"]

        center_full = (
            ROI_X1 + int(bx + bw / 2),
            ROI_Y1 + int(by + bh / 2),
        )

        raw_width_mm = (bw * depth_mm) / intrinsics["fx"]
        raw_height_mm = (bh * depth_mm) / intrinsics["fy"]

        angle_deg = 0.0
        points_full = None

    else:
        cx_roi, cy_roi = rotated["center_roi"]

        center_full = (
            ROI_X1 + int(cx_roi),
            ROI_Y1 + int(cy_roi),
        )

        raw_width_mm = (rotated["width_px"] * depth_mm) / intrinsics["fx"]
        raw_height_mm = (rotated["height_px"] * depth_mm) / intrinsics["fy"]

        angle_deg = rotated["angle_deg"]

        points_roi = rotated["points_roi"]
        points_full_np = points_roi.copy()
        points_full_np[:, 0] += ROI_X1
        points_full_np[:, 1] += ROI_Y1
        points_full = points_full_np.tolist()

    scaled_width_mm = raw_width_mm * POLY_SIZE_SCALE
    scaled_height_mm = raw_height_mm * POLY_SIZE_SCALE

    length_mm = max(scaled_width_mm, scaled_height_mm)
    width_mm = min(scaled_width_mm, scaled_height_mm)

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        depth_mm,
        intrinsics,
    )

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "angle_deg": float(angle_deg),
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "points_full": points_full,
    }


def make_clear_bag_depth_heatmap(depth_roi, product_mask, bag):
    h, w = depth_roi.shape[:2]
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    base_depth_mm = bag.get("base_depth_mm")

    valid = (
        (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    if base_depth_mm is not None and np.any(valid):
        closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
        closer_mm[valid] = float(base_depth_mm) - depth_roi[valid].astype(np.float32)
        closer_mm = np.clip(closer_mm, 0.0, HEATMAP_MAX_CLOSER_THAN_BASE_MM)

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


def run_sam2_product_and_bag(frame_bgr, depth_aligned, mask_generator, intrinsics):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = full_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    depth_roi = depth_aligned[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    print("Running SAM 2 on clear-bag product ROI...")

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    print(f"Generated {len(masks)} masks inside ROI")

    best = choose_best_product_mask(masks, roi_rgb)

    if best is None:
        print("No valid visible product mask found inside clear bag.")
        return None

    product_center_full = (
        int(ROI_X1 + best["center_roi"][0]),
        int(ROI_Y1 + best["center_roi"][1]),
    )

    product_top_face_depth = center_top_face_depth_mm(
        depth_roi=depth_roi,
        product_mask=best["safe_mask"],
        center_roi=best["center_roi"],
    )

    product_inside_center_x_mm, product_inside_center_y_mm = pixel_to_camera_xy_mm(
        product_center_full,
        product_top_face_depth["depth_mm"],
        intrinsics,
    )

    bag = detect_clear_bag_from_depth(
        depth_roi=depth_roi,
        product_mask=best["mask"],
        product_center_roi=best["center_roi"],
    )

    bag_measurements = estimate_clear_bag_measurements(
        bag=bag,
        intrinsics=intrinsics,
    )

    depth_heatmap = make_clear_bag_depth_heatmap(
        depth_roi=depth_roi,
        product_mask=best["mask"],
        bag=bag,
    )

    if product_top_face_depth["depth_mm"] is None:
        clearbag_depth_mm = None
    else:
        clearbag_depth_mm = BASE_DEPTH_MM - product_top_face_depth["depth_mm"]

    final_output = {
        "product_face_depth_mm": product_top_face_depth["depth_mm"],
        "clearbag_depth_mm": clearbag_depth_mm,
        "length_mm": bag_measurements["length_mm"],
        "width_mm": bag_measurements["width_mm"],
        "angle_deg": bag_measurements["angle_deg"],
        "center_x_mm": bag_measurements["center_x_mm"],
        "center_y_mm": bag_measurements["center_y_mm"],
        "product_inside_center_x_mm": product_inside_center_x_mm,
        "product_inside_center_y_mm": product_inside_center_y_mm,
    }

    print()
    print("CLEAR BAG FINAL OUTPUT:")
    print(f"  product_face_depth_mm:           {fmt3(final_output['product_face_depth_mm'])}")
    print(f"  clearbag_depth_mm:               {fmt3(final_output['clearbag_depth_mm'])}")
    print(f"  length_mm:                       {fmt3(final_output['length_mm'])}")
    print(f"  width_mm:                        {fmt3(final_output['width_mm'])}")
    print(f"  angle_deg:                       {fmt3(final_output['angle_deg'])}")
    print(f"  center_x_mm:                     {fmt3(final_output['center_x_mm'])}")
    print(f"  center_y_mm:                     {fmt3(final_output['center_y_mm'])}")
    print(f"  product_inside_center_x_mm:      {fmt3(final_output['product_inside_center_x_mm'])}")
    print(f"  product_inside_center_y_mm:      {fmt3(final_output['product_inside_center_y_mm'])}")

    return {
        "full_rgb": full_rgb,
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "mask": best["mask"],
        "safe_mask": best["safe_mask"],
        "center_roi": best["center_roi"],
        "center_full": product_center_full,
        "product_top_face_depth": product_top_face_depth,
        "info": best,
        "bag": bag,
        "bag_measurements": bag_measurements,
        "depth_heatmap": depth_heatmap,
        "final_output": final_output,
    }


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


def draw_result(frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(depth_aligned)

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

    result_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_rgb

    cv2.rectangle(result_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

    draw_roi_axes(result_rgb)
    draw_roi_axes(depth_vis)

    points_full = result["bag_measurements"].get("points_full")

    if points_full is not None:
        points_full_np = np.array(points_full, dtype=np.int32)
        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 255), 3)
        cv2.polylines(depth_vis, [points_full_np], True, (255, 0, 255), 3)

    elif bag["bag_bbox"] is not None:
        bx, by, bw, bh = bag["bag_bbox"]

        cv2.rectangle(
            result_rgb,
            (ROI_X1 + bx, ROI_Y1 + by),
            (ROI_X1 + bx + bw, ROI_Y1 + by + bh),
            (255, 0, 255),
            3,
        )

        cv2.rectangle(
            depth_vis,
            (ROI_X1 + bx, ROI_Y1 + by),
            (ROI_X1 + bx + bw, ROI_Y1 + by + bh),
            (255, 0, 255),
            3,
        )

    x, y, w, h = result["info"]["bbox"]

    cv2.rectangle(
        result_rgb,
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
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
    heatmap_full[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    draw_roi_axes(heatmap_full)

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


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def make_json_result(result):
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


def main():
    print("Starting OAK + SAM 2 clear bag final detector")
    print("Keys:")
    print("  SPACE = capture and output final measurements")
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
            live_heatmap = make_live_depth_heatmap(depth_aligned)

            cv2.rectangle(preview_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

            draw_roi_axes(preview_rgb)
            draw_roi_axes(preview_depth)

            status = "READY - SPACE to measure" if warmed else f"WARMING {WARMUP_SECONDS - elapsed:.1f}s"

            cv2.putText(
                preview_rgb,
                "CLEAR BAG FINAL OUTPUT",
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

            preview_depth_rgb = cv2.cvtColor(preview_depth, cv2.COLOR_BGR2RGB)
            preview_rgb_rgb = cv2.cvtColor(preview_rgb, cv2.COLOR_BGR2RGB)
            live_heatmap_rgb = cv2.cvtColor(live_heatmap, cv2.COLOR_BGR2RGB)

            combined_preview_rgb = np.hstack(
                [
                    preview_rgb_rgb,
                    preview_depth_rgb,
                    live_heatmap_rgb,
                ]
            )

            combined_preview_bgr = cv2.cvtColor(combined_preview_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("OAK Clear Bag Final Output", combined_preview_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before running.")
                    continue

                result = run_sam2_product_and_bag(
                    rgb,
                    depth_aligned,
                    mask_generator,
                    intrinsics,
                )

                if result is None:
                    continue

                result_bgr = draw_result(rgb, depth_aligned, result)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                raw_path = SAVE_DIR / f"raw_rgb_{timestamp}.jpg"
                result_path = SAVE_DIR / f"clear_bag_final_output_{timestamp}.png"
                product_mask_path = SAVE_DIR / f"clear_bag_product_mask_{timestamp}.png"
                bag_mask_path = SAVE_DIR / f"clear_bag_mask_{timestamp}.png"
                heatmap_path = SAVE_DIR / f"clear_bag_depth_heatmap_{timestamp}.png"
                json_path = SAVE_DIR / f"clear_bag_final_output_{timestamp}.json"

                cv2.imwrite(str(raw_path), rgb)
                cv2.imwrite(str(result_path), result_bgr)
                save_binary_mask(product_mask_path, result["mask"])
                save_binary_mask(bag_mask_path, result["bag"]["bag_mask"])
                cv2.imwrite(str(heatmap_path), result["depth_heatmap"])

                with open(json_path, "w") as f:
                    json.dump(make_json_result(result), f, indent=2)

                print()
                print(f"Saved raw RGB:       {raw_path}")
                print(f"Saved result image:  {result_path}")
                print(f"Saved product mask:  {product_mask_path}")
                print(f"Saved bag mask:      {bag_mask_path}")
                print(f"Saved heatmap:       {heatmap_path}")
                print(f"Saved JSON:          {json_path}")

                cv2.imshow("Clear Bag Final Output Result", result_bgr)
                cv2.waitKey(0)
                cv2.destroyWindow("Clear Bag Final Output Result")

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()