# box_detection_final.py

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


SAVE_DIR = Path("cardboard_box_depth_results")
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
STEREO_SIZE = (640, 400)

BASE_DEPTH_MM = 705.0

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

MIN_SURFACE_DEPTH_COUNT = 30

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


def resize_depth_to_rgb_size(depth):
    return cv2.resize(depth, RGB_SIZE, interpolation=cv2.INTER_NEAREST)


def align_depth_to_rgb(depth):
    h, w = depth.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    matrix = np.float32(
        [
            [DEPTH_ALIGN_SCALE_X, 0, (1.0 - DEPTH_ALIGN_SCALE_X) * cx + DEPTH_ALIGN_X_SHIFT_PX],
            [0, DEPTH_ALIGN_SCALE_Y, (1.0 - DEPTH_ALIGN_SCALE_Y) * cy + DEPTH_ALIGN_Y_SHIFT_PX],
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


def estimate_box_dimensions_mm(box, depth_mm, intrinsics):
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

    rotated = get_rotated_box_from_mask(box["mask"])

    if rotated is None:
        return empty

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    width_mm = (w_px * depth_mm) / fx
    height_mm = (h_px * depth_mm) / fy

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


def estimate_box_depth_mm(box_face_depth_mm):
    if box_face_depth_mm is None:
        return None

    box_depth = BASE_DEPTH_MM - box_face_depth_mm

    if box_depth < 0:
        box_depth = 0.0

    return float(box_depth)


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
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    right_out = right.requestOutput(
        size=STEREO_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo.setSubpixel(USE_SUBPIXEL)

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

    median_depth = float(np.median(filtered))
    mean_depth = float(np.mean(filtered))
    min_depth = float(np.min(filtered))

    return median_depth, int(filtered.size), mean_depth, min_depth


def box_face_depth_mm(depth_roi, mask):
    valid_sample = (
        (mask == 1)
        & (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    values = depth_roi[valid_sample].astype(np.float32)

    distance_mm, depth_count, mean_depth, min_depth = robust_median_depth(values)

    return {
        "distance_mm": distance_mm,
        "depth_count": depth_count,
        "sample_count_total": int(values.size),
        "mean_depth_mm": mean_depth,
        "min_depth_mm": min_depth,
    }


def choose_cardboard_box_mask(masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_box_mask(raw_mask)

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < MIN_BOX_AREA_RATIO:
            continue

        if area_ratio > MAX_BOX_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < MIN_BOX_RECTANGULARITY:
            continue

        if aspect_ratio > MAX_BOX_ASPECT_RATIO:
            continue

        masked_hsv = hsv[mask == 1]

        if masked_hsv.size == 0:
            continue

        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))

        if mean_v < MIN_BOX_VALUE:
            continue

        hue_score = 1.0 - min(abs(mean_h - 18.0) / 30.0, 1.0)
        sat_score = 1.0 - min(abs(mean_s - 65.0) / 100.0, 1.0)
        val_score = 1.0 - min(abs(mean_v - 170.0) / 120.0, 1.0)

        color_score = (
            0.50 * hue_score
            + 0.25 * sat_score
            + 0.25 * val_score
        )

        if color_score < MIN_BOX_COLOR_SCORE:
            continue

        center_roi = get_mask_center(mask)

        if center_roi is None:
            continue

        cx, cy = center_roi

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - TARGET_BOX_AREA_RATIO) / TARGET_BOX_AREA_RATIO,
            1.0,
        )

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        score = (
            2.7 * color_score
            + 1.6 * rectangularity
            + 1.2 * area_score
            + 1.0 * center_score
            + 0.4 * sam_iou
            + 0.4 * sam_stability
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "center_roi": center_roi,
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "color_score": float(color_score),
            "mean_hsv": (mean_h, mean_s, mean_v),
            "sam_iou": sam_iou,
            "sam_stability": sam_stability,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_cardboard_box(frame_bgr, depth_scaled, depth_aligned, mask_generator, intrinsics):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    depth_roi_scaled = depth_scaled[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    box = choose_cardboard_box_mask(masks, roi_rgb)

    if box is None:
        print("No valid cardboard box mask found.")
        return None

    center_full = (
        ROI_X1 + box["center_roi"][0],
        ROI_Y1 + box["center_roi"][1],
    )

    scaled_depth_result = box_face_depth_mm(depth_roi_scaled, box["mask"])

    box_face_depth = scaled_depth_result["distance_mm"]
    box_depth = estimate_box_depth_mm(box_face_depth)

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        box_face_depth,
        intrinsics,
    )

    dimensions = estimate_box_dimensions_mm(
        box,
        box_face_depth,
        intrinsics,
    )

    print()
    print("BOX MEASUREMENTS:")
    print(f"  box_face_depth_mm: {fmt3(box_face_depth)}")
    print(f"  box_depth_mm:      {fmt3(box_depth)}")
    print(f"  length_mm:         {fmt3(dimensions['length_mm'])}")
    print(f"  width_mm:          {fmt3(dimensions['short_side_mm'])}")
    print(f"  angle_deg:         {fmt3(dimensions['angle_deg'])}")
    print(f"  center_x_mm:       {fmt3(center_x_mm)}")
    print(f"  center_y_mm:       {fmt3(center_y_mm)}")

    return {
        "roi_rgb": roi_rgb,
        "box": box,
        "center_full": center_full,
        "box_face_depth_mm": box_face_depth,
        "box_depth_mm": box_depth,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "dimensions": dimensions,
        "scaled_depth_result": scaled_depth_result,
    }


def draw_rotated_or_axis_box(result_rgb, depth_vis, box, dimensions):
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
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
        (0, 255, 0),
        3,
    )

    cv2.rectangle(
        depth_vis,
        (ROI_X1 + x, ROI_Y1 + y),
        (ROI_X1 + x + w, ROI_Y1 + y + h),
        (0, 255, 0),
        3,
    )


def draw_result(frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(depth_aligned)

    roi_rgb = result["roi_rgb"].copy()

    box = result["box"]
    mask = box["mask"]
    dimensions = result["dimensions"]

    roi_rgb[mask == 1] = (
        0.50 * roi_rgb[mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    result_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_rgb

    cv2.rectangle(result_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

    draw_roi_axes(result_rgb)
    draw_roi_axes(depth_vis)

    draw_rotated_or_axis_box(result_rgb, depth_vis, box, dimensions)

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


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main():
    print("Starting OAK + SAM 2 cardboard box measurements")
    print("Keys:")
    print("  SPACE = capture and measure box")
    print("  q     = quit")
    print()
    print("Output:")
    print("  box_face_depth_mm")
    print("  box_depth_mm")
    print("  length_mm")
    print("  width_mm")
    print("  angle_deg")
    print("  center_x_mm")
    print("  center_y_mm")
    print()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 2 on device: {device_name}")

    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device_name)

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

            depth_640_raw = depth_msg.getFrame()
            depth_scaled = resize_depth_to_rgb_size(depth_640_raw)
            depth_aligned = align_depth_to_rgb(depth_scaled)

            elapsed = time.time() - start_time
            warmed = elapsed >= WARMUP_SECONDS

            preview_rgb = rgb.copy()
            preview_depth = make_depth_vis(depth_aligned)

            cv2.rectangle(preview_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

            draw_roi_axes(preview_rgb)
            draw_roi_axes(preview_depth)

            status = "READY - SPACE to measure" if warmed else f"WARMING {WARMUP_SECONDS - elapsed:.1f}s"

            cv2.putText(
                preview_rgb,
                "BOX MEASUREMENTS",
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

            combined_preview = np.hstack([preview_rgb, preview_depth])
            cv2.imshow("OAK Cardboard Box Measurements", combined_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before running.")
                    continue

                result = run_sam2_cardboard_box(
                    rgb,
                    depth_scaled,
                    depth_aligned,
                    mask_generator,
                    intrinsics,
                )

                if result is None:
                    continue

                result_bgr = draw_result(rgb, depth_aligned, result)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                raw_path = SAVE_DIR / f"raw_rgb_{timestamp}.jpg"
                result_path = SAVE_DIR / f"cardboard_box_measurements_{timestamp}.png"
                mask_path = SAVE_DIR / f"cardboard_box_mask_{timestamp}.png"
                json_path = SAVE_DIR / f"cardboard_box_measurements_{timestamp}.json"

                cv2.imwrite(str(raw_path), rgb)
                cv2.imwrite(str(result_path), result_bgr)
                save_binary_mask(mask_path, result["box"]["mask"])

                json_result = {
                    "box_face_depth_mm": json_number(result["box_face_depth_mm"]),
                    "box_depth_mm": json_number(result["box_depth_mm"]),
                    "length_mm": json_number(result["dimensions"]["length_mm"]),
                    "width_mm": json_number(result["dimensions"]["short_side_mm"]),
                    "angle_deg": json_number(result["dimensions"]["angle_deg"]),
                    "center_x_mm": json_number(result["center_x_mm"]),
                    "center_y_mm": json_number(result["center_y_mm"]),
                }

                with open(json_path, "w") as f:
                    json.dump(json_result, f, indent=2)

                print()
                print(f"Saved result image: {result_path}")
                print(f"Saved mask:         {mask_path}")
                print(f"Saved JSON:         {json_path}")

                cv2.imshow("Cardboard Box Measurements Result", result_bgr)
                cv2.waitKey(0)
                cv2.destroyWindow("Cardboard Box Measurements Result")

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()