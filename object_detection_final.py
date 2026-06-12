# object_segmentation_depth.py

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


SAVE_DIR = Path("object_segmentation_depth_results")
SAVE_DIR.mkdir(exist_ok=True)

FPS = 20
WARMUP_SECONDS = 5.0

CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Locked ROI from polymailer script
ROI_X1 = 330
ROI_Y1 = 60
ROI_X2 = 940
ROI_Y2 = 700

# Fast SAM settings
SAM_POINTS_PER_SIDE = 24
SAM_PRED_IOU_THRESH = 0.78
SAM_STABILITY_SCORE_THRESH = 0.82
SAM_MIN_MASK_REGION_AREA = 500

# Depth settings
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

# Object segmentation rules
MIN_AREA_RATIO = 0.008
MAX_AREA_RATIO = 0.35
TARGET_AREA_RATIO = 0.08

MIN_RECTANGULARITY = 0.18
MAX_ASPECT_RATIO = 7.0

REJECT_MASKS_TOUCHING_ROI_BORDER = True
ROI_BORDER_MARGIN_PX = 12

USE_MASK_CLEANUP = True
USE_CONVEX_HULL = True
MASK_CLOSE_KERNEL_PX = 25
MASK_CLOSE_ITERATIONS = 2
MAX_CLEANED_AREA_GROWTH = 2.8


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


def mask_touches_roi_border(mask, margin_px):
    h, w = mask.shape[:2]

    top = mask[:margin_px, :]
    bottom = mask[h - margin_px:h, :]
    left = mask[:, :margin_px]
    right = mask[:, w - margin_px:w]

    return (
        np.any(top == 1)
        or np.any(bottom == 1)
        or np.any(left == 1)
        or np.any(right == 1)
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


def clean_mask(mask):
    mask = mask.astype(np.uint8)

    if not USE_MASK_CLEANUP:
        return mask

    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MASK_CLOSE_KERNEL_PX, MASK_CLOSE_KERNEL_PX),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MASK_CLOSE_ITERATIONS,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    if USE_CONVEX_HULL:
        contours, _ = cv2.findContours(
            cleaned.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)

            hull_mask = np.zeros_like(cleaned, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)

            hull_area = int(hull_mask.sum())

            if hull_area <= original_area * MAX_CLEANED_AREA_GROWTH:
                return hull_mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * MAX_CLEANED_AREA_GROWTH:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


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


def choose_best_object_mask(masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    best = None

    print()
    print("Checking SAM 2 object masks...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(raw_mask)

        if REJECT_MASKS_TOUCHING_ROI_BORDER:
            if mask_touches_roi_border(mask, ROI_BORDER_MARGIN_PX):
                continue

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < MIN_AREA_RATIO:
            continue

        if area_ratio > MAX_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        bbox_area = bw * bh
        rectangularity = area / max(bbox_area, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < MIN_RECTANGULARITY:
            continue

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        center = get_mask_center(mask)

        if center is None:
            continue

        cx, cy = center

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - TARGET_AREA_RATIO) / TARGET_AREA_RATIO,
            1.0,
        )

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        score = (
            1.4 * center_score
            + 1.2 * area_score
            + 1.0 * rectangularity
            + 0.5 * sam_iou
            + 0.5 * sam_stability
        )

        print(
            f"mask={i:03d} "
            f"score={score:.3f} "
            f"area={area_ratio:.3f} "
            f"rect={rectangularity:.3f} "
            f"aspect={aspect_ratio:.2f} "
            f"center={center_score:.2f} "
            f"sam_iou={sam_iou:.2f} "
            f"stable={sam_stability:.2f} "
            f"bbox=({x},{y},{bw},{bh})"
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "center_roi": center,
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "sam_iou": sam_iou,
            "sam_stability": sam_stability,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_object_segmentation(frame_bgr, depth_aligned, mask_generator):
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
    depth_roi = depth_aligned[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    print()
    print("Running SAM 2 on ROI...")

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    print(f"Generated {len(masks)} masks inside ROI")

    obj = choose_best_object_mask(masks, roi_rgb)

    if obj is None:
        print("No valid object mask found.")
        return None

    center_full = (
        ROI_X1 + obj["center_roi"][0],
        ROI_Y1 + obj["center_roi"][1],
    )

    distance_mm, depth_count = center_depth_mm(
        depth_roi,
        obj["mask"],
        obj["center_roi"],
    )

    print()
    print("OBJECT SEGMENT + DEPTH:")
    print(f"  selected_mask_index: {obj['index']}")
    print(f"  score:               {obj['score']:.3f}")
    print(f"  area_ratio:          {obj['area_ratio']:.3f}")
    print(f"  bbox_roi:            {obj['bbox']}")
    print(f"  center_full:         {center_full}")
    print(f"  distance_mm:         {distance_mm}")
    print(f"  depth_count:         {depth_count}")
    print(f"  rectangularity:      {obj['rectangularity']:.3f}")
    print(f"  aspect_ratio:        {obj['aspect_ratio']:.3f}")
    print(f"  center_score:        {obj['center_score']:.3f}")

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "object": obj,
        "center_full": center_full,
        "distance_mm": distance_mm,
        "depth_count": depth_count,
    }


def draw_result(frame_bgr, depth_aligned, result):
    result_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_vis = make_depth_vis(depth_aligned)

    roi_rgb = result["roi_rgb"].copy()

    obj = result["object"]
    mask = obj["mask"]

    roi_rgb[mask == 1] = (
        0.50 * roi_rgb[mask == 1]
        + 0.50 * np.array([0, 255, 0])
    ).astype(np.uint8)

    result_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_rgb

    cv2.rectangle(result_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
    cv2.rectangle(depth_vis, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

    x, y, w, h = obj["bbox"]

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


def save_binary_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main():
    print("Starting OAK + SAM 2 object segmentation with depth")
    print("Keys:")
    print("  SPACE = capture, segment object, measure center depth")
    print("  q     = quit")
    print()
    print(f"Locked ROI: {ROI_X1},{ROI_Y1} to {ROI_X2},{ROI_Y2}")
    print(
        f"Depth alignment: scale=({DEPTH_ALIGN_SCALE_X},{DEPTH_ALIGN_SCALE_Y}) "
        f"shift=({DEPTH_ALIGN_X_SHIFT_PX},{DEPTH_ALIGN_Y_SHIFT_PX})"
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print()
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

            cv2.rectangle(preview_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 3)

            status = (
                "READY - SPACE to segment + depth"
                if warmed
                else f"WARMING {WARMUP_SECONDS - elapsed:.1f}s"
            )

            cv2.putText(
                preview_rgb,
                "OBJECT SEGMENT + DEPTH",
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
            cv2.imshow("OAK Object RGB + Depth", combined_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before segmenting.")
                    continue

                print()
                print("Captured frame. Running SAM 2 object segmentation + depth...")

                result = run_sam2_object_segmentation(
                    rgb,
                    depth_aligned,
                    mask_generator,
                )

                if result is None:
                    continue

                result_bgr = draw_result(
                    rgb,
                    depth_aligned,
                    result,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                raw_path = SAVE_DIR / f"raw_rgb_{timestamp}.jpg"
                depth_raw_path = SAVE_DIR / f"depth_raw_{timestamp}.npy"
                depth_aligned_path = SAVE_DIR / f"depth_aligned_{timestamp}.npy"
                result_path = SAVE_DIR / f"object_segment_depth_{timestamp}.png"
                mask_path = SAVE_DIR / f"object_mask_{timestamp}.png"
                json_path = SAVE_DIR / f"object_segment_depth_{timestamp}.json"

                cv2.imwrite(str(raw_path), rgb)
                np.save(str(depth_raw_path), depth_raw)
                np.save(str(depth_aligned_path), depth_aligned)
                cv2.imwrite(str(result_path), result_bgr)
                save_binary_mask(mask_path, result["object"]["mask"])

                json_result = {
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
                        "x1": ROI_X1,
                        "y1": ROI_Y1,
                        "x2": ROI_X2,
                        "y2": ROI_Y2,
                    },
                    "depth_alignment": {
                        "scale_x": DEPTH_ALIGN_SCALE_X,
                        "scale_y": DEPTH_ALIGN_SCALE_Y,
                        "shift_x_px": DEPTH_ALIGN_X_SHIFT_PX,
                        "shift_y_px": DEPTH_ALIGN_Y_SHIFT_PX,
                    },
                    "selected_mask": {
                        "index": int(result["object"]["index"]),
                        "score": float(result["object"]["score"]),
                        "area_ratio": float(result["object"]["area_ratio"]),
                        "bbox_roi": [int(v) for v in result["object"]["bbox"]],
                        "rectangularity": float(result["object"]["rectangularity"]),
                        "aspect_ratio": float(result["object"]["aspect_ratio"]),
                        "center_score": float(result["object"]["center_score"]),
                    },
                }

                with open(json_path, "w") as f:
                    json.dump(json_result, f, indent=2)

                print()
                print(f"Saved raw RGB:        {raw_path}")
                print(f"Saved raw depth:      {depth_raw_path}")
                print(f"Saved aligned depth:  {depth_aligned_path}")
                print(f"Saved result image:   {result_path}")
                print(f"Saved object mask:    {mask_path}")
                print(f"Saved JSON:           {json_path}")

                cv2.imshow("Object Segment Depth Result", result_bgr)
                cv2.waitKey(0)
                cv2.destroyWindow("Object Segment Depth Result")

    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()