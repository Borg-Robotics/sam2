import cv2
import torch
import depthai as dai
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json

from pyzbar.pyzbar import decode

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


SAVE_DIR = Path("package_barcode_detection_results")
SAVE_DIR.mkdir(exist_ok=True)

FPS = 20
WARMUP_SECONDS = 5.0

CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

RGB_SIZE = (1280, 720)
STEREO_SIZE = (640, 400)

ROI_X1 = 250
ROI_Y1 = 60
ROI_X2 = 1020
ROI_Y2 = 700

BARCODE_DECODE_UPSCALE = 2.0
BARCODE_DRAW_ENABLE = True

SAM_POINTS_PER_SIDE = 24
SAM_PRED_IOU_THRESH = 0.78
SAM_STABILITY_SCORE_THRESH = 0.82
SAM_MIN_MASK_REGION_AREA = 500

DEPTH_ALIGN_X_SHIFT_PX = 35
DEPTH_ALIGN_Y_SHIFT_PX = 0
DEPTH_ALIGN_SCALE_X = 1.25
DEPTH_ALIGN_SCALE_Y = 1.25

MIN_VALID_DEPTH_MM = 450
MAX_VALID_DEPTH_MM = 1200

BASE_DEPTH_MM = 700.0
MEASUREMENT_DEPTH_OFFSET_MM = 10.0
PACKAGE_SIZE_SCALE = 1.04

IR_LASER_INTENSITY = 1.0
IR_FLOOD_INTENSITY = 0.0
CONFIDENCE_THRESHOLD = 120

USE_SUBPIXEL = True
USE_LEFT_RIGHT_CHECK = True

USE_MANUAL_STEREO_EXPOSURE = True
STEREO_EXPOSURE_US = 1000
STEREO_ISO = 400

MIN_PACKAGE_AREA_RATIO = 0.003
MAX_PACKAGE_AREA_RATIO = 0.70
TARGET_PACKAGE_AREA_RATIO = 0.18

MIN_PACKAGE_RECTANGULARITY = 0.08
MAX_PACKAGE_ASPECT_RATIO = 10.0
MIN_PACKAGE_CENTER_SCORE = 0.18

PACKAGE_MASK_CLOSE_KERNEL_PX = 19
PACKAGE_MASK_CLOSE_ITERATIONS = 1
PACKAGE_MAX_CLEANED_AREA_GROWTH = 2.8

SAFE_ERODE_RADIUS_PX = 12

HARD_REJECT_ROLLER_LIKE = True
ROLLER_STRIP_MIN_WIDTH_RATIO = 0.72
ROLLER_STRIP_MAX_HEIGHT_RATIO = 0.22

REJECT_EDGE_STRIPS = True
EDGE_STRIP_MARGIN_PX = 24
EDGE_STRIP_MAX_WIDTH_RATIO = 0.16
EDGE_STRIP_MAX_HEIGHT_RATIO = 0.16

RING_DILATE_PX = 25

GOOD_CONTRAST = 35.0
BAD_CONTRAST = 5.0
GOOD_TEXTURE = 28.0
BAD_TEXTURE = 4.0

AREA_SCORE_WEIGHT = 1.25
CENTER_SCORE_WEIGHT = 1.20
CONTRAST_SCORE_WEIGHT = 1.30
TEXTURE_SCORE_WEIGHT = 0.75
RECT_SCORE_WEIGHT = 0.85
BBOX_SIZE_SCORE_WEIGHT = 0.65
SAFE_AREA_SCORE_WEIGHT = 0.45
SAM_IOU_SCORE_WEIGHT = 0.45
SAM_STABILITY_SCORE_WEIGHT = 0.45

MIN_BOX_AREA_RATIO = 0.04
MAX_BOX_AREA_RATIO = 0.75
TARGET_BOX_AREA_RATIO = 0.45

MIN_BOX_RECTANGULARITY = 0.35
MAX_BOX_ASPECT_RATIO = 4.0

MIN_BOX_VALUE = 70
MIN_BOX_COLOR_SCORE = 0.25

BOX_MASK_CLOSE_KERNEL_PX = 21
BOX_MASK_CLOSE_ITERATIONS = 1
BOX_MAX_CLEANED_AREA_GROWTH = 2.8

BOX_FLAT_STD_GOOD_MM = 6.0
BOX_FLAT_STD_BAD_MM = 18.0

BOX_RESIDUAL_RANGE_GOOD_MM = 20.0
BOX_RESIDUAL_RANGE_BAD_MM = 55.0

BOX_RECTANGULARITY_GOOD = 0.90
BOX_RECTANGULARITY_BAD = 0.60

BOX_SCORE_THRESHOLD = 0.60
BOX_SCORE_POLY_SIGNATURE_PENALTY = 0.65

POLY_CENTER_EDGE_SOFT_MM = 14.0
POLY_CENTER_EDGE_HARD_MM = 24.0

CENTER_EDGE_OVERRIDE_MIN_VALID_FRACTION = 0.20
CENTER_EDGE_OVERRIDE_MIN_VALID_POINTS = 20000

SPARSE_DEPTH_BOX_MAX_VALID_FRACTION = 0.08
SPARSE_DEPTH_BOX_MIN_RECTANGULARITY = 0.88
SPARSE_DEPTH_BOX_MIN_AREA_RATIO = 0.12

CENTER_EDGE_KERNEL_PX = 70
MAX_PLANE_POINTS = 8000

POLY_SIGNATURE_MIN_VALID_FRACTION = 0.18
POLY_SIGNATURE_MIN_VALID_POINTS = 15000
POLY_SIGNATURE_MIN_AREA_RATIO = 0.16

POLY_SIGNATURE_OVERRIDE_THRESHOLD = 0.30

POLY_CENTER_CLOSER_SOFT_MM = 4.0
POLY_CENTER_CLOSER_HARD_MM = 16.0

POLY_FULL_DEPTH_RANGE_SOFT_MM = 7.0
POLY_FULL_DEPTH_RANGE_HARD_MM = 24.0

POLY_EDGE_DEPTH_RANGE_SOFT_MM = 7.0
POLY_EDGE_DEPTH_RANGE_HARD_MM = 26.0

POLY_SIDE_SPREAD_SOFT_MM = 4.0
POLY_SIDE_SPREAD_HARD_MM = 20.0

POLY_MULTI_SIGNAL_MIN_COUNT = 3
POLY_MULTI_SIGNAL_CONFIDENCE = 0.82

MIN_SURFACE_DEPTH_COUNT = 30

PRODUCT_INSIDE_ENABLE = True
PRODUCT_INSIDE_EDGE_ERODE_PX = 45
PRODUCT_INSIDE_MIN_CLOSER_THAN_POLY_MM = 8.0
PRODUCT_INSIDE_MAX_CLOSER_THAN_POLY_MM = 90.0
PRODUCT_INSIDE_CLOSE_KERNEL_PX = 17
PRODUCT_INSIDE_DILATE_PX = 5
PRODUCT_INSIDE_MIN_AREA_RATIO_OF_PACKAGE = 0.015
PRODUCT_INSIDE_MAX_AREA_RATIO_OF_PACKAGE = 0.65
PRODUCT_INSIDE_MIN_VALID_PIXELS = 80

HEATMAP_MAX_CLOSER_THAN_BASE_MM = 180.0

DEBUG_PRINT_MASKS = False
DEBUG_SAVE_ALL_ACCEPTED_MASKS = False


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


def score_from_range(value, good_value, bad_value, reverse=False):
    if value is None:
        return 0.5

    value = float(value)

    if not reverse:
        if value <= good_value:
            return 1.0
        if value >= bad_value:
            return 0.0
        return 1.0 - ((value - good_value) / (bad_value - good_value))

    if value >= good_value:
        return 1.0
    if value <= bad_value:
        return 0.0
    return (value - bad_value) / (good_value - bad_value)


def score_from_bad_good(value, bad_value, good_value):
    if value <= bad_value:
        return 0.0
    if value >= good_value:
        return 1.0
    return float((value - bad_value) / (good_value - bad_value))


def detect_barcode(frame_bgr):
    results = decode(frame_bgr)
    scale_used = 1.0

    if len(results) == 0 and BARCODE_DECODE_UPSCALE is not None and BARCODE_DECODE_UPSCALE > 1.0:
        upscaled = cv2.resize(
            frame_bgr,
            None,
            fx=BARCODE_DECODE_UPSCALE,
            fy=BARCODE_DECODE_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        results = decode(upscaled)
        scale_used = BARCODE_DECODE_UPSCALE

    if len(results) == 0:
        return None

    r = results[0]

    try:
        barcode_data = r.data.decode("utf-8")
    except Exception:
        barcode_data = str(r.data)

    x, y, w, h = r.rect

    if scale_used != 1.0:
        x = int(x / scale_used)
        y = int(y / scale_used)
        w = int(w / scale_used)
        h = int(h / scale_used)

    return {
        "type": str(r.type),
        "data": barcode_data,
        "rect": (int(x), int(y), int(w), int(h)),
        "decoded_count": int(len(results)),
    }


def draw_barcode_overlay(frame_bgr, barcode):
    if barcode is None:
        return frame_bgr

    x, y, w, h = barcode["rect"]

    cv2.rectangle(
        frame_bgr,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        frame_bgr,
        f"{barcode['type']}: {barcode['data']}",
        (x, max(y - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame_bgr


def resize_depth_to_rgb_size(depth):
    return cv2.resize(depth, RGB_SIZE, interpolation=cv2.INTER_NEAREST)


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

    x_mm = (u - intrinsics["cx"]) * depth_mm / intrinsics["fx"]
    y_mm = (v - intrinsics["cy"]) * depth_mm / intrinsics["fy"]

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

    left_class_out = left.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    right_class_out = right.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    left_measure_out = left.requestOutput(
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    right_measure_out = right.requestOutput(
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo_class = pipeline.create(dai.node.StereoDepth)
    stereo_class.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo_class.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo_class.setSubpixel(USE_SUBPIXEL)

    try:
        stereo_class.setOutputSize(RGB_SIZE[0], RGB_SIZE[1])
        print("Classification stereo depth output size set to 1280x720.")
    except Exception as e:
        print(f"Could not set classification stereo output size: {e}")

    try:
        stereo_class.initialConfig.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
        print(f"Classification confidence threshold set to {CONFIDENCE_THRESHOLD}.")
    except Exception as e:
        print(f"Could not set classification confidence threshold: {e}")

    stereo_measure = pipeline.create(dai.node.StereoDepth)
    stereo_measure.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo_measure.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo_measure.setSubpixel(USE_SUBPIXEL)

    try:
        stereo_measure.initialConfig.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
        print(f"Measurement confidence threshold set to {CONFIDENCE_THRESHOLD}.")
    except Exception as e:
        print(f"Could not set measurement confidence threshold: {e}")

    left_class_out.link(stereo_class.left)
    right_class_out.link(stereo_class.right)

    left_measure_out.link(stereo_measure.left)
    right_measure_out.link(stereo_measure.right)

    rgb_queue = rgb_output.createOutputQueue()
    depth_class_queue = stereo_class.depth.createOutputQueue()
    depth_measure_queue = stereo_measure.depth.createOutputQueue()

    return pipeline, rgb_queue, depth_class_queue, depth_measure_queue


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
        "CLASSIFICATION DEPTH HEATMAP",
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


def clean_package_mask(mask):
    mask = mask.astype(np.uint8)
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(
        mask,
        PACKAGE_MASK_CLOSE_KERNEL_PX,
        PACKAGE_MASK_CLOSE_ITERATIONS,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * PACKAGE_MAX_CLEANED_AREA_GROWTH:
        return cleaned.astype(np.uint8)

    return mask


def clean_box_mask(mask):
    mask = mask.astype(np.uint8)
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    cleaned = close_mask(
        mask,
        BOX_MASK_CLOSE_KERNEL_PX,
        BOX_MASK_CLOSE_ITERATIONS,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * BOX_MAX_CLEANED_AREA_GROWTH:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def get_safe_center(mask):
    safe_mask = erode_mask(mask, SAFE_ERODE_RADIUS_PX)
    safe_component = largest_component(safe_mask)

    if safe_component is None:
        safe_component = largest_component(mask)

    if safe_component is None:
        return None, safe_mask

    center = get_mask_center(safe_component)

    if center is None:
        return None, safe_component

    return center, safe_component.astype(np.uint8)


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


def is_roller_like_horizontal_strip(mask):
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))

    width_ratio = bw / max(w, 1)
    height_ratio = bh / max(h, 1)

    return (
        width_ratio >= ROLLER_STRIP_MIN_WIDTH_RATIO
        and height_ratio <= ROLLER_STRIP_MAX_HEIGHT_RATIO
    )


def is_edge_strip(mask):
    roi_h, roi_w = mask.shape[:2]
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    if w <= 0 or h <= 0:
        return True

    bbox_width_ratio = w / max(roi_w, 1)
    bbox_height_ratio = h / max(roi_h, 1)

    touches_left = x <= EDGE_STRIP_MARGIN_PX
    touches_right = (x + w) >= (roi_w - EDGE_STRIP_MARGIN_PX)
    touches_top = y <= EDGE_STRIP_MARGIN_PX
    touches_bottom = (y + h) >= (roi_h - EDGE_STRIP_MARGIN_PX)

    vertical_edge_strip = (
        (touches_left or touches_right)
        and bbox_width_ratio <= EDGE_STRIP_MAX_WIDTH_RATIO
    )

    horizontal_edge_strip = (
        (touches_top or touches_bottom)
        and bbox_height_ratio <= EDGE_STRIP_MAX_HEIGHT_RATIO
    )

    return bool(vertical_edge_strip or horizontal_edge_strip)


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
    contrast_score = score_from_bad_good(contrast, BAD_CONTRAST, GOOD_CONTRAST)

    return contrast_score, contrast


def compute_texture_score(roi_rgb, mask):
    if int(mask.sum()) < 20:
        return 0.0, 0.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    values = gray[mask == 1].astype(np.float32)

    if values.size < 20:
        return 0.0, 0.0

    texture = float(np.std(values))
    texture_score = score_from_bad_good(texture, BAD_TEXTURE, GOOD_TEXTURE)

    return texture_score, texture


def score_package_product_mask(mask, roi_rgb, sam_iou, sam_stability):
    mask = mask.astype(np.uint8)

    if HARD_REJECT_ROLLER_LIKE and is_roller_like_horizontal_strip(mask):
        return None, "roller_like"

    if REJECT_EDGE_STRIPS and is_edge_strip(mask):
        return None, "edge_strip"

    roi_h, roi_w = mask.shape[:2]
    roi_area = roi_h * roi_w

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < MIN_PACKAGE_AREA_RATIO:
        return None, "too_small"

    if area_ratio > MAX_PACKAGE_AREA_RATIO:
        return None, "too_large"

    x, y, w, h = cv2.boundingRect(mask)

    if w <= 0 or h <= 0:
        return None, "bad_bbox"

    bbox_area = w * h
    rectangularity = area / max(bbox_area, 1)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    if rectangularity < MIN_PACKAGE_RECTANGULARITY:
        return None, "low_rectangularity"

    if aspect_ratio > MAX_PACKAGE_ASPECT_RATIO:
        return None, "bad_aspect"

    center_roi, safe_mask = get_safe_center(mask)

    if center_roi is None:
        return None, "no_center"

    roi_cx = roi_w / 2
    roi_cy = roi_h / 2

    dist = np.sqrt(
        (center_roi[0] - roi_cx) ** 2
        + (center_roi[1] - roi_cy) ** 2
    )

    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    if center_score < MIN_PACKAGE_CENTER_SCORE:
        return None, "center_too_far"

    bbox_width_ratio = w / max(roi_w, 1)
    bbox_height_ratio = h / max(roi_h, 1)

    area_score = 1.0 - min(
        abs(area_ratio - TARGET_PACKAGE_AREA_RATIO) / TARGET_PACKAGE_AREA_RATIO,
        1.0,
    )

    bbox_size_score = min(
        min(bbox_width_ratio / 0.45, bbox_height_ratio / 0.45),
        1.0,
    )

    safe_area_ratio = int(safe_mask.sum()) / max(roi_area, 1)
    safe_area_score = min(safe_area_ratio / 0.03, 1.0)

    contrast_score, contrast_value = compute_rgb_contrast_score(roi_rgb, mask)
    texture_score, texture_value = compute_texture_score(roi_rgb, mask)

    score = (
        AREA_SCORE_WEIGHT * area_score
        + CENTER_SCORE_WEIGHT * center_score
        + CONTRAST_SCORE_WEIGHT * contrast_score
        + TEXTURE_SCORE_WEIGHT * texture_score
        + RECT_SCORE_WEIGHT * rectangularity
        + BBOX_SIZE_SCORE_WEIGHT * bbox_size_score
        + SAFE_AREA_SCORE_WEIGHT * safe_area_score
        + SAM_IOU_SCORE_WEIGHT * sam_iou
        + SAM_STABILITY_SCORE_WEIGHT * sam_stability
    )

    rotated = get_rotated_box_from_mask(mask)

    return {
        "source": "package_product_rules",
        "index": None,
        "score": float(score),
        "mask": mask,
        "safe_mask": safe_mask,
        "bbox": (int(x), int(y), int(w), int(h)),
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "bbox_width_ratio": float(bbox_width_ratio),
        "bbox_height_ratio": float(bbox_height_ratio),
        "bbox_size_score": float(bbox_size_score),
        "safe_area_ratio": float(safe_area_ratio),
        "safe_area_score": float(safe_area_score),
        "contrast_score": float(contrast_score),
        "contrast_value": float(contrast_value),
        "texture_score": float(texture_score),
        "texture_value": float(texture_value),
        "color_score": None,
        "mean_hsv": None,
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
        "rotated": rotated,
    }, "ok"


def score_cardboard_box_mask(mask, roi_rgb, sam_iou, sam_stability):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    mask = mask.astype(np.uint8)

    area = int(mask.sum())
    area_ratio = area / max(roi_area, 1)

    if area_ratio < MIN_BOX_AREA_RATIO:
        return None, "box_too_small"

    if area_ratio > MAX_BOX_AREA_RATIO:
        return None, "box_too_large"

    x, y, bw, bh = cv2.boundingRect(mask)

    if bw <= 0 or bh <= 0:
        return None, "box_bad_bbox"

    rectangularity = area / max(bw * bh, 1)
    aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

    if rectangularity < MIN_BOX_RECTANGULARITY:
        return None, "box_low_rectangularity"

    if aspect_ratio > MAX_BOX_ASPECT_RATIO:
        return None, "box_bad_aspect"

    masked_hsv = hsv[mask == 1]

    if masked_hsv.size == 0:
        return None, "box_no_hsv"

    mean_h = float(np.mean(masked_hsv[:, 0]))
    mean_s = float(np.mean(masked_hsv[:, 1]))
    mean_v = float(np.mean(masked_hsv[:, 2]))

    if mean_v < MIN_BOX_VALUE:
        return None, "box_too_dark"

    hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
    sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
    val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)

    color_score = (
        0.50 * hue_score
        + 0.25 * sat_score
        + 0.25 * val_score
    )

    if color_score < MIN_BOX_COLOR_SCORE:
        return None, "box_low_color_score"

    center_roi = get_mask_center(mask)

    if center_roi is None:
        return None, "box_no_center"

    cx, cy = center_roi

    dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
    max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
    center_score = 1.0 - min(dist / max_dist, 1.0)

    area_score = 1.0 - min(
        abs(area_ratio - TARGET_BOX_AREA_RATIO) / TARGET_BOX_AREA_RATIO,
        1.0,
    )

    score = (
        2.7 * color_score
        + 1.6 * rectangularity
        + 1.2 * area_score
        + 1.0 * center_score
        + 0.4 * sam_iou
        + 0.4 * sam_stability
    )

    rotated = get_rotated_box_from_mask(mask)

    return {
        "source": "cardboard_box_rules",
        "index": None,
        "score": float(score),
        "mask": mask,
        "safe_mask": mask,
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "center_roi": center_roi,
        "area": int(area),
        "area_ratio": float(area_ratio),
        "rectangularity": float(rectangularity),
        "aspect_ratio": float(aspect_ratio),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "bbox_width_ratio": float(bw / max(w, 1)),
        "bbox_height_ratio": float(bh / max(h, 1)),
        "bbox_size_score": None,
        "safe_area_ratio": None,
        "safe_area_score": None,
        "contrast_score": None,
        "contrast_value": None,
        "texture_score": None,
        "texture_value": None,
        "color_score": float(color_score),
        "mean_hsv": (mean_h, mean_s, mean_v),
        "sam_iou": float(sam_iou),
        "sam_stability": float(sam_stability),
        "rotated": rotated,
    }, "ok"


def choose_best_package_mask(masks, roi_rgb):
    best = None
    accepted = []

    print("\nChecking SAM 2 masks with package/product rules + exact box rules...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        package_mask = clean_package_mask(raw_mask)
        package_result, package_reason = score_package_product_mask(
            package_mask,
            roi_rgb,
            sam_iou,
            sam_stability,
        )

        if package_result is not None:
            package_result["index"] = i
            package_result["reason"] = package_reason
            accepted.append(package_result)

            if best is None or package_result["score"] > best["score"]:
                best = package_result

        box_mask = clean_box_mask(raw_mask)
        box_result, box_reason = score_cardboard_box_mask(
            box_mask,
            roi_rgb,
            sam_iou,
            sam_stability,
        )

        if box_result is not None:
            box_result["index"] = i
            box_result["reason"] = box_reason
            accepted.append(box_result)

            if best is None or box_result["score"] > best["score"]:
                best = box_result

    return best, accepted


def robust_depth_values(values):
    return values[
        (values > MIN_VALID_DEPTH_MM)
        & (values < MAX_VALID_DEPTH_MM)
    ].astype(np.float32)


def robust_median_depth(values):
    values = values.astype(np.float32)

    if values.size < MIN_SURFACE_DEPTH_COUNT:
        return None, 0, None, None

    raw_median = float(np.median(values))
    abs_dev = np.abs(values - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = values[np.abs(values - raw_median) <= 12.0]
    else:
        filtered = values[abs_dev <= 3.5 * mad]

    if filtered.size < MIN_SURFACE_DEPTH_COUNT:
        filtered = values

    return (
        float(np.median(filtered)),
        int(filtered.size),
        float(np.mean(filtered)),
        float(np.min(filtered)),
    )


def package_face_depth_mm(depth_roi, mask):
    valid_sample = (
        (mask == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    values = depth_roi[valid_sample].astype(np.float32)

    distance_mm, depth_count, mean_depth, min_depth = robust_median_depth(values)

    return {
        "raw_distance_mm": distance_mm,
        "distance_mm": (
            distance_mm + MEASUREMENT_DEPTH_OFFSET_MM
            if distance_mm is not None
            else None
        ),
        "depth_count": depth_count,
        "sample_count_total": int(values.size),
        "mean_depth_mm": (
            mean_depth + MEASUREMENT_DEPTH_OFFSET_MM
            if mean_depth is not None
            else None
        ),
        "min_depth_mm": (
            min_depth + MEASUREMENT_DEPTH_OFFSET_MM
            if min_depth is not None
            else None
        ),
        "measurement_depth_offset_mm": MEASUREMENT_DEPTH_OFFSET_MM,
    }


def robust_stats_from_values(values, total_count):
    valid = robust_depth_values(values)
    valid_fraction = valid.size / max(total_count, 1)

    if valid.size == 0:
        return {
            "valid_fraction": 0.0,
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
            "range_p90_p10": None,
            "p10": None,
            "p90": None,
        }

    raw_median = float(np.median(valid))
    abs_dev = np.abs(valid - raw_median)
    mad = float(np.median(abs_dev))

    if mad < 1.0:
        filtered = valid[np.abs(valid - raw_median) <= 12.0]
    else:
        filtered = valid[abs_dev <= 3.5 * mad]

    if filtered.size == 0:
        filtered = valid

    p10 = float(np.percentile(valid, 10))
    p90 = float(np.percentile(valid, 90))

    return {
        "valid_fraction": float(valid_fraction),
        "median": float(np.median(filtered)),
        "mean": float(np.mean(filtered)),
        "std": float(np.std(filtered)),
        "min": float(np.min(filtered)),
        "max": float(np.max(filtered)),
        "count": int(filtered.size),
        "range_p90_p10": float(p90 - p10),
        "p10": p10,
        "p90": p90,
    }


def depth_stats_inside_mask(depth_roi, mask):
    selected_depth = depth_roi[mask == 1]
    return robust_stats_from_values(selected_depth, selected_depth.size)


def get_mask_rectangularity(mask):
    area = int(mask.sum())

    if area <= 0:
        return 0.0

    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    if w <= 0 or h <= 0:
        return 0.0

    return float(area / max(w * h, 1))


def fit_plane_depth_features(depth_roi, package_mask):
    mask = package_mask.astype(np.uint8)

    valid = (
        (mask == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    valid_points = int(valid.sum())
    mask_area = int(mask.sum())
    valid_fraction_on_mask = valid_points / max(mask_area, 1)

    empty = {
        "success": False,
        "valid_points": valid_points,
        "valid_fraction_on_mask": float(valid_fraction_on_mask),
        "plane_residual_std_mm": None,
        "plane_residual_range_mm": None,
        "center_edge_depth_delta_mm": None,
    }

    if valid_points < 100:
        return empty

    yy, xx = np.where(valid)
    zz = depth_roi[valid].astype(np.float32)

    if zz.size > MAX_PLANE_POINTS:
        rng = np.random.default_rng(12345)
        idx = rng.choice(zz.size, size=MAX_PLANE_POINTS, replace=False)
        xx_fit = xx[idx].astype(np.float32)
        yy_fit = yy[idx].astype(np.float32)
        zz_fit = zz[idx]
    else:
        xx_fit = xx.astype(np.float32)
        yy_fit = yy.astype(np.float32)
        zz_fit = zz

    a = np.column_stack(
        [
            xx_fit,
            yy_fit,
            np.ones_like(xx_fit),
        ]
    )

    try:
        coeffs, _, _, _ = np.linalg.lstsq(a, zz_fit, rcond=None)
        pred = a @ coeffs
        residuals = zz_fit - pred

        residual_std = float(np.std(residuals))
        residual_range = float(np.percentile(residuals, 90) - np.percentile(residuals, 10))

    except Exception:
        residual_std = None
        residual_range = None

    center = get_mask_center(mask)
    center_edge_delta = None

    if center is not None:
        cx, cy = center
        yy_all, xx_all = np.indices(mask.shape)

        center_circle = (
            ((xx_all - cx) ** 2 + (yy_all - cy) ** 2)
            <= CENTER_EDGE_KERNEL_PX ** 2
        )

        eroded = erode_mask(mask, CENTER_EDGE_KERNEL_PX // 2)
        edge = mask.copy()
        edge[eroded == 1] = 0

        center_values = depth_roi[
            (mask == 1)
            & center_circle
            & (depth_roi > MIN_VALID_DEPTH_MM)
            & (depth_roi < MAX_VALID_DEPTH_MM)
        ].astype(np.float32)

        edge_values = depth_roi[
            (edge == 1)
            & (depth_roi > MIN_VALID_DEPTH_MM)
            & (depth_roi < MAX_VALID_DEPTH_MM)
        ].astype(np.float32)

        if center_values.size >= 50 and edge_values.size >= 50:
            center_median = float(np.median(center_values))
            edge_median = float(np.median(edge_values))
            center_edge_delta = center_median - edge_median

    return {
        "success": True,
        "valid_points": valid_points,
        "valid_fraction_on_mask": float(valid_fraction_on_mask),
        "plane_residual_std_mm": residual_std,
        "plane_residual_range_mm": residual_range,
        "center_edge_depth_delta_mm": (
            float(center_edge_delta)
            if center_edge_delta is not None
            else None
        ),
    }


def compute_polymailer_depth_signature(depth_roi, package_mask):
    mask = package_mask.astype(np.uint8)
    mask_area = int(mask.sum())

    empty = {
        "success": False,
        "valid_points": 0,
        "valid_fraction_on_mask": 0.0,
        "full_range_mm": None,
        "edge_range_mm": None,
        "center_range_mm": None,
        "center_closer_than_edge_mm": None,
        "center_edge_signed_mm": None,
        "side_spread_mm": None,
        "score": 0.0,
        "center_closer_score": 0.0,
        "full_range_score": 0.0,
        "edge_range_score": 0.0,
        "side_spread_score": 0.0,
    }

    if mask_area <= 0:
        return empty

    full_stats = depth_stats_inside_mask(depth_roi, mask)
    valid_points = full_stats["count"]
    valid_fraction = full_stats["valid_fraction"]

    if valid_points < 50:
        return empty

    center = get_mask_center(mask)

    if center is None:
        return empty

    cx, cy = center
    yy, xx = np.indices(mask.shape)

    center_region = (
        (mask == 1)
        & (((xx - cx) ** 2 + (yy - cy) ** 2) <= CENTER_EDGE_KERNEL_PX ** 2)
    ).astype(np.uint8)

    eroded = erode_mask(mask, CENTER_EDGE_KERNEL_PX // 2)
    edge_region = mask.copy()
    edge_region[eroded == 1] = 0

    center_stats = depth_stats_inside_mask(depth_roi, center_region)
    edge_stats = depth_stats_inside_mask(depth_roi, edge_region)

    center_closer_than_edge_mm = None
    center_edge_signed_mm = None

    if center_stats["median"] is not None and edge_stats["median"] is not None:
        center_edge_signed_mm = center_stats["median"] - edge_stats["median"]
        center_closer_than_edge_mm = edge_stats["median"] - center_stats["median"]

    x, y, w, h = cv2.boundingRect(mask)

    thirds = [
        (x, y, max(1, w // 3), h),
        (x + max(1, w // 3), y, max(1, w // 3), h),
        (x + max(1, 2 * w // 3), y, max(1, w - 2 * (w // 3)), h),
        (x, y, w, max(1, h // 3)),
        (x, y + max(1, 2 * h // 3), w, max(1, h - 2 * (h // 3))),
    ]

    side_medians = []

    for tx, ty, tw, th in thirds:
        patch_mask = mask[ty:ty + th, tx:tx + tw]
        patch_depth = depth_roi[ty:ty + th, tx:tx + tw]

        patch_valid = patch_depth[
            (patch_mask == 1)
            & (patch_depth > MIN_VALID_DEPTH_MM)
            & (patch_depth < MAX_VALID_DEPTH_MM)
        ].astype(np.float32)

        if patch_valid.size >= 50:
            side_medians.append(float(np.median(patch_valid)))

    if len(side_medians) >= 2:
        side_spread_mm = float(max(side_medians) - min(side_medians))
    else:
        side_spread_mm = None

    center_closer_score = score_from_range(
        center_closer_than_edge_mm,
        POLY_CENTER_CLOSER_HARD_MM,
        POLY_CENTER_CLOSER_SOFT_MM,
        reverse=True,
    )

    full_range_score = score_from_range(
        full_stats["range_p90_p10"],
        POLY_FULL_DEPTH_RANGE_HARD_MM,
        POLY_FULL_DEPTH_RANGE_SOFT_MM,
        reverse=True,
    )

    edge_range_score = score_from_range(
        edge_stats["range_p90_p10"],
        POLY_EDGE_DEPTH_RANGE_HARD_MM,
        POLY_EDGE_DEPTH_RANGE_SOFT_MM,
        reverse=True,
    )

    side_spread_score = score_from_range(
        side_spread_mm,
        POLY_SIDE_SPREAD_HARD_MM,
        POLY_SIDE_SPREAD_SOFT_MM,
        reverse=True,
    )

    score = (
        0.35 * center_closer_score
        + 0.25 * full_range_score
        + 0.25 * edge_range_score
        + 0.15 * side_spread_score
    )

    score = float(np.clip(score, 0.0, 1.0))

    return {
        "success": True,
        "valid_points": int(valid_points),
        "valid_fraction_on_mask": float(valid_fraction),
        "full_range_mm": full_stats["range_p90_p10"],
        "edge_range_mm": edge_stats["range_p90_p10"],
        "center_range_mm": center_stats["range_p90_p10"],
        "center_closer_than_edge_mm": (
            float(center_closer_than_edge_mm)
            if center_closer_than_edge_mm is not None
            else None
        ),
        "center_edge_signed_mm": (
            float(center_edge_signed_mm)
            if center_edge_signed_mm is not None
            else None
        ),
        "side_spread_mm": (
            float(side_spread_mm)
            if side_spread_mm is not None
            else None
        ),
        "score": score,
        "center_closer_score": float(center_closer_score),
        "full_range_score": float(full_range_score),
        "edge_range_score": float(edge_range_score),
        "side_spread_score": float(side_spread_score),
    }


def count_polymailer_depth_signals(poly_signature):
    signals = 0

    center_closer = poly_signature.get("center_closer_than_edge_mm")
    full_range = poly_signature.get("full_range_mm")
    edge_range = poly_signature.get("edge_range_mm")
    side_spread = poly_signature.get("side_spread_mm")

    if center_closer is not None and center_closer >= 6.0:
        signals += 1

    if full_range is not None and full_range >= 14.0:
        signals += 1

    if edge_range is not None and edge_range >= 14.0:
        signals += 1

    if side_spread is not None and side_spread >= 10.0:
        signals += 1

    return int(signals)


def classify_package_type(depth_roi, package_mask, info):
    depth_features = fit_plane_depth_features(depth_roi, package_mask)
    poly_signature = compute_polymailer_depth_signature(depth_roi, package_mask)

    rectangularity = get_mask_rectangularity(package_mask)

    residual_std = depth_features["plane_residual_std_mm"]
    residual_range = depth_features["plane_residual_range_mm"]
    center_edge_delta = depth_features["center_edge_depth_delta_mm"]
    valid_points = depth_features["valid_points"]
    valid_fraction_on_mask = depth_features["valid_fraction_on_mask"]

    mask_area_ratio = info.get("area_ratio", 0.0)

    flatness_score = score_from_range(
        residual_std,
        BOX_FLAT_STD_GOOD_MM,
        BOX_FLAT_STD_BAD_MM,
        reverse=False,
    )

    residual_range_score = score_from_range(
        residual_range,
        BOX_RESIDUAL_RANGE_GOOD_MM,
        BOX_RESIDUAL_RANGE_BAD_MM,
        reverse=False,
    )

    rectangularity_score = score_from_range(
        rectangularity,
        BOX_RECTANGULARITY_GOOD,
        BOX_RECTANGULARITY_BAD,
        reverse=True,
    )

    if center_edge_delta is None:
        center_edge_abs = None
        center_edge_box_score = 0.6
    else:
        center_edge_abs = abs(float(center_edge_delta))

        center_edge_box_score = score_from_range(
            center_edge_abs,
            POLY_CENTER_EDGE_SOFT_MM,
            POLY_CENTER_EDGE_HARD_MM,
            reverse=False,
        )

    center_edge_override_allowed = (
        center_edge_abs is not None
        and valid_fraction_on_mask >= CENTER_EDGE_OVERRIDE_MIN_VALID_FRACTION
        and valid_points >= CENTER_EDGE_OVERRIDE_MIN_VALID_POINTS
    )

    sparse_rectangular_box = (
        valid_fraction_on_mask <= SPARSE_DEPTH_BOX_MAX_VALID_FRACTION
        and rectangularity >= SPARSE_DEPTH_BOX_MIN_RECTANGULARITY
        and mask_area_ratio >= SPARSE_DEPTH_BOX_MIN_AREA_RATIO
    )

    poly_signal_count = count_polymailer_depth_signals(poly_signature)

    polymailer_signature_override_allowed = (
        poly_signature["success"]
        and poly_signature["valid_fraction_on_mask"] >= POLY_SIGNATURE_MIN_VALID_FRACTION
        and poly_signature["valid_points"] >= POLY_SIGNATURE_MIN_VALID_POINTS
        and mask_area_ratio >= POLY_SIGNATURE_MIN_AREA_RATIO
        and (
            poly_signature["score"] >= POLY_SIGNATURE_OVERRIDE_THRESHOLD
            or poly_signal_count >= POLY_MULTI_SIGNAL_MIN_COUNT
        )
    )

    raw_box_score = (
        0.35 * flatness_score
        + 0.25 * residual_range_score
        + 0.15 * rectangularity_score
        + 0.25 * center_edge_box_score
    )

    raw_box_score = float(np.clip(raw_box_score, 0.0, 1.0))

    box_score = raw_box_score * (
        1.0 - BOX_SCORE_POLY_SIGNATURE_PENALTY * poly_signature["score"]
    )

    box_score = float(np.clip(box_score, 0.0, 1.0))

    if polymailer_signature_override_allowed:
        package_type = "polymailer"

        if poly_signal_count >= POLY_MULTI_SIGNAL_MIN_COUNT:
            confidence = POLY_MULTI_SIGNAL_CONFIDENCE
        else:
            confidence = 0.72 + 0.25 * poly_signature["score"]

        confidence = float(np.clip(confidence, 0.72, 0.95))

    elif sparse_rectangular_box:
        package_type = "box"
        confidence = 0.78

    elif (
        center_edge_override_allowed
        and center_edge_abs is not None
        and center_edge_abs >= POLY_CENTER_EDGE_HARD_MM
    ):
        package_type = "polymailer"
        confidence = 0.75 + min((center_edge_abs - POLY_CENTER_EDGE_HARD_MM) / 40.0, 0.20)
        confidence = float(np.clip(confidence, 0.75, 0.95))

    else:
        polymailer_score = 1.0 - box_score

        if box_score >= BOX_SCORE_THRESHOLD:
            package_type = "box"
            confidence = box_score
        else:
            package_type = "polymailer"
            confidence = polymailer_score

        confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        "package_type": package_type,
        "package_type_confidence": float(confidence),
        "box_score": float(box_score),
        "raw_box_score": float(raw_box_score),
        "flatness_score": float(flatness_score),
        "residual_range_score": float(residual_range_score),
        "rectangularity_score": float(rectangularity_score),
        "center_edge_box_score": float(center_edge_box_score),
        "poly_signal_count": int(poly_signal_count),
        "depth_features": depth_features,
        "polymailer_depth_signature": poly_signature,
    }


def estimate_package_dimensions_mm(info, depth_mm, intrinsics):
    empty = {
        "rotated_width_px": None,
        "rotated_height_px": None,
        "angle_deg": None,
        "width_mm": None,
        "height_mm": None,
        "length_mm": None,
        "short_side_mm": None,
        "points_roi": None,
        "points_full": None,
    }

    if depth_mm is None or intrinsics is None:
        return empty

    rotated = get_rotated_box_from_mask(info["mask"])

    if rotated is None:
        return empty

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    width_mm = (w_px * depth_mm) / intrinsics["fx"]
    height_mm = (h_px * depth_mm) / intrinsics["fy"]

    width_mm *= PACKAGE_SIZE_SCALE
    height_mm *= PACKAGE_SIZE_SCALE

    length_mm = max(width_mm, height_mm)
    short_side_mm = min(width_mm, height_mm)

    points_roi = rotated["points_roi"]
    points_full = points_roi.copy()
    points_full[:, 0] += ROI_X1
    points_full[:, 1] += ROI_Y1

    return {
        "rotated_width_px": float(w_px),
        "rotated_height_px": float(h_px),
        "angle_deg": float(rotated["angle_deg"]),
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "length_mm": float(length_mm),
        "short_side_mm": float(short_side_mm),
        "points_roi": points_roi.tolist(),
        "points_full": points_full.tolist(),
    }


def detect_product_inside_polymailer(depth_roi, package_mask, center_full_depth_mm, intrinsics):
    empty = {
        "found": False,
        "center_roi": None,
        "center_full": None,
        "center_x_mm": None,
        "center_y_mm": None,
        "depth_mm": None,
        "area_px": 0,
        "area_ratio_of_package": 0.0,
        "mask": np.zeros_like(package_mask, dtype=np.uint8),
        "reason": "not_run",
    }

    if not PRODUCT_INSIDE_ENABLE:
        empty["reason"] = "disabled"
        return empty

    package_mask = package_mask.astype(np.uint8)
    package_area = int(package_mask.sum())

    if package_area <= 0:
        empty["reason"] = "empty_package_mask"
        return empty

    inner_mask = erode_mask(package_mask, PRODUCT_INSIDE_EDGE_ERODE_PX)

    if int(inner_mask.sum()) < PRODUCT_INSIDE_MIN_VALID_PIXELS:
        inner_mask = package_mask.copy()

    valid_inner = (
        (inner_mask == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    values = depth_roi[valid_inner].astype(np.float32)

    if values.size < PRODUCT_INSIDE_MIN_VALID_PIXELS:
        empty["reason"] = "not_enough_depth"
        return empty

    poly_surface_depth = float(np.median(values))

    closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
    closer_mm[valid_inner] = poly_surface_depth - depth_roi[valid_inner].astype(np.float32)

    product_candidate = (
        valid_inner
        & (closer_mm >= PRODUCT_INSIDE_MIN_CLOSER_THAN_POLY_MM)
        & (closer_mm <= PRODUCT_INSIDE_MAX_CLOSER_THAN_POLY_MM)
    ).astype(np.uint8)

    product_candidate = close_mask(
        product_candidate,
        PRODUCT_INSIDE_CLOSE_KERNEL_PX,
        iterations=1,
    )

    product_candidate = dilate_mask(product_candidate, PRODUCT_INSIDE_DILATE_PX)
    product_candidate = largest_component(product_candidate)

    if product_candidate is None:
        empty["reason"] = "no_component"
        return empty

    product_area = int(product_candidate.sum())
    area_ratio = product_area / max(package_area, 1)

    if area_ratio < PRODUCT_INSIDE_MIN_AREA_RATIO_OF_PACKAGE:
        empty["reason"] = "too_small"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    if area_ratio > PRODUCT_INSIDE_MAX_AREA_RATIO_OF_PACKAGE:
        empty["reason"] = "too_large"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    center_roi = get_mask_center(product_candidate)

    if center_roi is None:
        empty["reason"] = "no_center"
        empty["mask"] = product_candidate
        empty["area_px"] = product_area
        empty["area_ratio_of_package"] = float(area_ratio)
        return empty

    cx_roi, cy_roi = center_roi
    center_full = (int(ROI_X1 + cx_roi), int(ROI_Y1 + cy_roi))

    product_values = depth_roi[
        (product_candidate == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    ].astype(np.float32)

    if product_values.size >= MIN_SURFACE_DEPTH_COUNT:
        raw_product_depth = float(np.median(product_values))
        product_depth_mm = raw_product_depth + MEASUREMENT_DEPTH_OFFSET_MM
    else:
        product_depth_mm = center_full_depth_mm

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        product_depth_mm,
        intrinsics,
    )

    return {
        "found": True,
        "center_roi": (int(cx_roi), int(cy_roi)),
        "center_full": center_full,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "depth_mm": product_depth_mm,
        "area_px": product_area,
        "area_ratio_of_package": float(area_ratio),
        "mask": product_candidate,
        "reason": "ok",
    }


def make_package_depth_heatmap(depth_roi, package_mask, center_roi, product_inside=None):
    h, w = depth_roi.shape[:2]

    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (
        (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    if np.any(valid):
        closer_mm = np.zeros_like(depth_roi, dtype=np.float32)
        closer_mm[valid] = BASE_DEPTH_MM - depth_roi[valid].astype(np.float32)
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


def run_sam_package_depth_type(
    frame_bgr,
    depth_class_aligned,
    depth_measure_aligned,
    mask_generator,
    intrinsics,
    barcode,
):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    roi_rgb = full_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    depth_class_roi = depth_class_aligned[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    depth_measure_roi = depth_measure_aligned[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    print("Running SAM 2 package segmentation on ROI...")

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    print(f"Generated {len(masks)} masks inside ROI")

    best, accepted = choose_best_package_mask(masks, roi_rgb)

    if best is None:
        print("No valid package/product or cardboard box mask found.")
        return None

    classification = classify_package_type(
        depth_roi=depth_class_roi,
        package_mask=best["mask"],
        info=best,
    )

    top_face_depth_result = package_face_depth_mm(
        depth_measure_roi,
        best["mask"],
    )

    top_face_depth_mm = top_face_depth_result["distance_mm"]

    package_depth_mm = (
        max(0.0, BASE_DEPTH_MM - top_face_depth_mm)
        if top_face_depth_mm is not None
        else None
    )

    cx_full = ROI_X1 + best["center_roi"][0]
    cy_full = ROI_Y1 + best["center_roi"][1]
    center_full = (int(cx_full), int(cy_full))

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        top_face_depth_mm,
        intrinsics,
    )

    dimensions = estimate_package_dimensions_mm(
        info=best,
        depth_mm=top_face_depth_mm,
        intrinsics=intrinsics,
    )

    if classification["package_type"] == "polymailer":
        product_inside = detect_product_inside_polymailer(
            depth_roi=depth_measure_roi,
            package_mask=best["mask"],
            center_full_depth_mm=top_face_depth_mm,
            intrinsics=intrinsics,
        )
    else:
        product_inside = {
            "found": False,
            "center_roi": None,
            "center_full": None,
            "center_x_mm": None,
            "center_y_mm": None,
            "depth_mm": None,
            "area_px": 0,
            "area_ratio_of_package": 0.0,
            "mask": np.zeros_like(best["mask"], dtype=np.uint8),
            "reason": "not_polymailer",
        }

    heatmap = make_package_depth_heatmap(
        depth_roi=depth_class_roi,
        package_mask=best["mask"],
        center_roi=best["center_roi"],
        product_inside=product_inside,
    )

    final_output = {
        "barcode_type": barcode["type"] if barcode is not None else None,
        "barcode_data": barcode["data"] if barcode is not None else None,
        "package_type": classification["package_type"],
        "package_type_confidence_percent": classification["package_type_confidence"] * 100.0,
        "mask_source": best["source"],
        "top_face_depth_mm": top_face_depth_mm,
        "package_depth_mm": package_depth_mm,
        "length_mm": dimensions["length_mm"],
        "width_mm": dimensions["short_side_mm"],
        "angle_deg": dimensions["angle_deg"],
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "product_inside_found": product_inside["found"],
        "product_inside_center_x_mm": product_inside["center_x_mm"],
        "product_inside_center_y_mm": product_inside["center_y_mm"],
        "product_inside_center_pixel_u": (
            product_inside["center_full"][0]
            if product_inside["center_full"] is not None
            else None
        ),
        "product_inside_center_pixel_v": (
            product_inside["center_full"][1]
            if product_inside["center_full"] is not None
            else None
        ),
        "product_inside_depth_mm": product_inside["depth_mm"],
    }

    print()
    print("PACKAGE FINAL OUTPUT:")
    print(f"  barcode_type:                    {final_output['barcode_type']}")
    print(f"  barcode_data:                    {final_output['barcode_data']}")
    print(f"  package_type:                    {final_output['package_type']}")
    print(f"  confidence_percent:              {final_output['package_type_confidence_percent']:.1f}")
    print(f"  mask_source:                     {final_output['mask_source']}")
    print(f"  top_face_depth_mm:               {fmt3(final_output['top_face_depth_mm'])}")
    print(f"  package_depth_mm:                {fmt3(final_output['package_depth_mm'])}")
    print(f"  length_mm:                       {fmt3(final_output['length_mm'])}")
    print(f"  width_mm:                        {fmt3(final_output['width_mm'])}")
    print(f"  angle_deg:                       {fmt3(final_output['angle_deg'])}")
    print(f"  center_x_mm:                     {fmt3(final_output['center_x_mm'])}")
    print(f"  center_y_mm:                     {fmt3(final_output['center_y_mm'])}")
    print(f"  product_inside_found:            {final_output['product_inside_found']}")
    print(f"  product_inside_center_x_mm:      {fmt3(final_output['product_inside_center_x_mm'])}")
    print(f"  product_inside_center_y_mm:      {fmt3(final_output['product_inside_center_y_mm'])}")

    return {
        "full_rgb": full_rgb,
        "roi_rgb": roi_rgb,
        "depth_class_roi": depth_class_roi,
        "depth_measure_roi": depth_measure_roi,
        "mask": best["mask"],
        "center_roi": best["center_roi"],
        "center_full": center_full,
        "info": best,
        "accepted": accepted,
        "all_mask_count": len(masks),
        "top_face_depth_mm": top_face_depth_mm,
        "package_depth_mm": package_depth_mm,
        "top_face_depth_result": top_face_depth_result,
        "classification": classification,
        "dimensions": dimensions,
        "product_inside": product_inside,
        "final_output": final_output,
        "depth_heatmap": heatmap,
        "barcode": barcode,
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


def draw_rotated_or_axis_box(result_rgb, info, dimensions=None):
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
        points_full_np[:, 0] += ROI_X1
        points_full_np[:, 1] += ROI_Y1

        cv2.polylines(result_rgb, [points_full_np], True, (255, 0, 0), 3)

        for p in points_full_np:
            cv2.circle(result_rgb, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

        return

    x, y, w, h = info["bbox"]

    cv2.rectangle(
        result_rgb,
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
        (255, 0, 0),
        3,
    )


def draw_result(frame_bgr, depth_class_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(depth_class_aligned)

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

    result_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_rgb

    cv2.rectangle(result_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

    draw_roi_axes(result_rgb)
    draw_roi_axes(depth_vis)
    draw_rotated_or_axis_box(result_rgb, info, dimensions)
    draw_rotated_or_axis_box(depth_vis, info, dimensions)

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

    if BARCODE_DRAW_ENABLE and barcode is not None:
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
    heatmap_full[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = heatmap_roi_rgb

    cv2.rectangle(heatmap_full, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    draw_roi_axes(heatmap_full)

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


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def save_accepted_masks(timestamp, result):
    accepted_dir = SAVE_DIR / f"accepted_masks_{timestamp}"
    accepted_dir.mkdir(exist_ok=True)

    for item in result["accepted"]:
        idx = item["index"]
        source = item["source"]
        path = accepted_dir / f"accepted_mask_{idx:03d}_{source}_score_{item['score']:.3f}.png"
        save_binary_mask(path, item["mask"])


def make_json_result(result, timestamp):
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
        "selected_mask_index": int(info["index"]),
        "selected_mask_score": json_number(info["score"]),
        "selected_mask_area_ratio": json_number(info["area_ratio"]),
        "selected_mask_rectangularity": json_number(info["rectangularity"]),
        "selected_mask_aspect_ratio": json_number(info["aspect_ratio"]),
        "selected_mask_color_score": json_number(info["color_score"]),
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
        "size_scale": json_number(PACKAGE_SIZE_SCALE),
        "base_depth_mm": json_number(BASE_DEPTH_MM),
        "measurement_depth_offset_mm": json_number(MEASUREMENT_DEPTH_OFFSET_MM),
        "top_face_depth_debug": {
            "raw_distance_mm": json_number(result["top_face_depth_result"]["raw_distance_mm"]),
            "offset_distance_mm": json_number(result["top_face_depth_result"]["distance_mm"]),
            "depth_count": int(result["top_face_depth_result"]["depth_count"]),
            "sample_count_total": int(result["top_face_depth_result"]["sample_count_total"]),
            "mean_depth_mm": json_number(result["top_face_depth_result"]["mean_depth_mm"]),
            "min_depth_mm": json_number(result["top_face_depth_result"]["min_depth_mm"]),
        },
        "depth_classification_debug": {
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
            "rgb_size": list(RGB_SIZE),
            "classification_depth_size": list(RGB_SIZE),
            "measurement_stereo_size": list(STEREO_SIZE),
            "fps": FPS,
        },
    }


def main():
    print("Starting OAK + Barcode Gate + SAM 2 package detector")
    print("Keys:")
    print("  SPACE = start barcode search, then run package detection once")
    print("  q     = quit")
    print()
    print("FLOW:")
    print("  1. Press SPACE")
    print("  2. System searches for barcode")
    print("  3. When barcode is found, package detection runs once")
    print("  4. System waits for SPACE again")
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

    pipeline, rgb_queue, depth_class_queue, depth_measure_queue = build_pipeline()
    pipeline.start()

    start_time = time.time()
    barcode_scan_active = False
    last_barcode = None

    with pipeline:
        device = pipeline.getDefaultDevice()
        set_ir(device)
        intrinsics = get_rgb_intrinsics(device)

        while pipeline.isRunning():
            rgb_msg = rgb_queue.get()
            depth_class_msg = depth_class_queue.get()
            depth_measure_msg = depth_measure_queue.get()

            rgb = rgb_msg.getCvFrame()

            depth_class_raw = depth_class_msg.getFrame()
            depth_class_aligned = align_depth_to_rgb(depth_class_raw)

            depth_measure_640_raw = depth_measure_msg.getFrame()
            depth_measure_scaled = resize_depth_to_rgb_size(depth_measure_640_raw)
            depth_measure_aligned = align_depth_to_rgb(depth_measure_scaled)

            elapsed = time.time() - start_time
            warmed = elapsed >= WARMUP_SECONDS

            preview_rgb = rgb.copy()
            preview_depth = make_depth_vis(depth_class_aligned)
            preview_heatmap = make_live_depth_heatmap(depth_class_aligned)

            cv2.rectangle(preview_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

            draw_roi_axes(preview_rgb)
            draw_roi_axes(preview_depth)

            if not warmed:
                status = f"WARMING {WARMUP_SECONDS - elapsed:.1f}s"
            elif barcode_scan_active:
                status = "SCANNING BARCODE..."
            else:
                status = "READY - PRESS SPACE TO START"

            cv2.putText(
                preview_rgb,
                "BARCODE GATE PACKAGE DETECTOR",
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

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before starting barcode scan.")
                else:
                    barcode_scan_active = True
                    last_barcode = None
                    print()
                    print("SPACE pressed. Searching for barcode...")

            if barcode_scan_active and warmed:
                barcode = detect_barcode(rgb)

                if barcode is not None:
                    last_barcode = barcode
                    preview_rgb = draw_barcode_overlay(preview_rgb, barcode)

                    print()
                    print(f"BARCODE FOUND: {barcode['type']} {barcode['data']}")
                    print("Running package detection...")

                    result = run_sam_package_depth_type(
                        rgb,
                        depth_class_aligned,
                        depth_measure_aligned,
                        mask_generator,
                        intrinsics,
                        last_barcode,
                    )

                    barcode_scan_active = False

                    if result is not None:
                        result_bgr = draw_result(
                            rgb,
                            depth_class_aligned,
                            result,
                        )

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        raw_path = SAVE_DIR / f"raw_rgb_{timestamp}.jpg"
                        result_path = SAVE_DIR / f"package_final_{timestamp}.png"
                        mask_path = SAVE_DIR / f"package_mask_{timestamp}.png"
                        product_mask_path = SAVE_DIR / f"product_inside_mask_{timestamp}.png"
                        heatmap_path = SAVE_DIR / f"package_depth_heatmap_{timestamp}.png"
                        json_path = SAVE_DIR / f"package_final_{timestamp}.json"

                        cv2.imwrite(str(raw_path), rgb)
                        cv2.imwrite(str(result_path), result_bgr)
                        save_binary_mask(mask_path, result["mask"])
                        save_binary_mask(product_mask_path, result["product_inside"]["mask"])
                        cv2.imwrite(str(heatmap_path), result["depth_heatmap"])

                        if DEBUG_SAVE_ALL_ACCEPTED_MASKS:
                            save_accepted_masks(timestamp, result)

                        with open(json_path, "w") as f:
                            json.dump(make_json_result(result, timestamp), f, indent=2)

                        print()
                        print(f"Saved raw RGB:             {raw_path}")
                        print(f"Saved result image:        {result_path}")
                        print(f"Saved package mask:        {mask_path}")
                        print(f"Saved product inside mask: {product_mask_path}")
                        print(f"Saved heatmap:             {heatmap_path}")
                        print(f"Saved JSON:                {json_path}")
                        print()
                        print("Done. Press SPACE to scan the next barcode/package.")

                        cv2.imshow("Package Final Result", result_bgr)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Package Final Result")

                    else:
                        print("Package detection failed. Press SPACE to try again.")

                    last_barcode = None

            combined_preview = np.hstack([preview_rgb, preview_depth, preview_heatmap])
            cv2.imshow("OAK Barcode Gate Package Detector", combined_preview)

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()