#!/usr/bin/env python3

import cv2
import torch
import depthai as dai
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import queue
import time

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


SAVE_DIR = Path("polymailer_live_moving_slit_results")
SAVE_DIR.mkdir(exist_ok=True)

FPS = 20
WARMUP_SECONDS = 3.0

# Capture the baseline polymailer and begin slit tracking automatically once
# warmup ends and SAM has found a valid polymailer, instead of waiting for
# SPACE. SPACE still works. "x" restarts the warmup timer, so after a reset
# there is another WARMUP_SECONDS to place the next bag before it re-baselines.
AUTO_START = True

CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Which OAK device to use when several are plugged in. Set this to a device
# id (e.g. "14442C10E171DDD600") to always use the same camera. Leave it as
# an empty string to get a numbered menu at startup instead.
DEVICE_ID = ""

# Keep the same polymailer ROI.
ROI_X1 = 330
ROI_Y1 = 30
ROI_X2 = 940
ROI_Y2 = 700
ROI_WIDTH = ROI_X2 - ROI_X1
ROI_HEIGHT = ROI_Y2 - ROI_Y1
RGB_SIZE = (1280, 720)

# Stereo depth is used only as a physical-object check. Static pieces of the
# black base should have almost the same depth as the surrounding base, while
# a released product should rise above it.
BASE_DEPTH_MM = 705.0
MIN_VALID_DEPTH_MM = 450
MAX_VALID_DEPTH_MM = 1200
DEPTH_ALIGN_X_SHIFT_PX = -40
DEPTH_ALIGN_Y_SHIFT_PX = 0
DEPTH_ALIGN_SCALE_X = 1.25
DEPTH_ALIGN_SCALE_Y = 1.25
IR_LASER_INTENSITY = 1.0
IR_FLOOD_INTENSITY = 0.0
CONFIDENCE_THRESHOLD = 120
USE_SUBPIXEL = True
USE_LEFT_RIGHT_CHECK = True
USE_MANUAL_STEREO_EXPOSURE = True
STEREO_EXPOSURE_US = 1000
STEREO_ISO = 400

# Do not start product decisions immediately after SPACE. The bag must first
# move enough for the lifting/pulling action to have started.
EXIT_ARM_DELAY_SECONDS = 0.60
EXIT_ARM_MIN_SLIT_MOTION_PX = 18.0

# Faster SAM settings.
# Before SPACE, SAM searches the complete polymailer ROI.
# After SPACE, SAM searches only a smaller moving crop around the slit.
FULL_SAM_POINTS_PER_SIDE = 16
SLIT_SAM_POINTS_PER_SIDE = 12
SAM_PRED_IOU_THRESH = 0.76
SAM_STABILITY_SCORE_THRESH = 0.80
SAM_MIN_MASK_REGION_AREA = 300
FULL_SAM_INTERVAL_FRAMES = 2
SLIT_SAM_INTERVAL_FRAMES = 1

# Choose which physical edge of the flat polymailer will be cut.
# The selection is image-relative at the instant SPACE is pressed.
SLIT_SIDE_TOP = "top"
SLIT_SIDE_RIGHT = "right"
SLIT_SIDE_BOTTOM = "bottom"
SLIT_SIDE_LEFT = "left"
VALID_SLIT_SIDES = (
    SLIT_SIDE_TOP,
    SLIT_SIDE_RIGHT,
    SLIT_SIDE_BOTTOM,
    SLIT_SIDE_LEFT,
)
DEFAULT_SLIT_SIDE = SLIT_SIDE_TOP
USE_CUDA_AUTOCAST = True

# Initial polymailer selection.
MIN_POLY_AREA_RATIO = 0.04
MAX_POLY_AREA_RATIO = 0.75
TARGET_POLY_AREA_RATIO = 0.55
MIN_POLY_RECTANGULARITY = 0.35
MAX_POLY_ASPECT_RATIO = 3.5
MIN_POLY_VALUE = 80
MIN_POLY_COLOR_SCORE = 0.30

# Optical-flow tracking of the moving polymailer and slit.
FLOW_MAX_CORNERS = 300
FLOW_QUALITY_LEVEL = 0.01
FLOW_MIN_DISTANCE = 7
FLOW_BLOCK_SIZE = 7
FLOW_WIN_SIZE = (31, 31)
FLOW_MAX_LEVEL = 4
FLOW_MIN_GOOD_POINTS = 12
FLOW_RESEED_POINT_COUNT = 55
FLOW_RESEED_INTERVAL_FRAMES = 12
FLOW_FORWARD_BACKWARD_MAX_ERROR = 1.8
FLOW_RANSAC_REPROJ_THRESHOLD = 3.0

# Robust moving-slit tracking. The previous version allowed a full projective
# homography, which could shear the saved bag mask and rotate the slit across
# the middle of the polymailer. This tracker uses similarity transforms only
# and gives extra weight to points close to the physical slit.
FLOW_LOCAL_SLIT_BAND_INWARD_PX = 150.0
FLOW_LOCAL_SLIT_BAND_OUTWARD_PX = 28.0
FLOW_LOCAL_SLIT_SIDE_MARGIN_PX = 65.0
FLOW_MIN_LOCAL_SLIT_POINTS = 8
FLOW_MIN_AFFINE_INLIER_RATIO = 0.45
FLOW_MIN_FRAME_SCALE = 0.82
FLOW_MAX_FRAME_SCALE = 1.22
FLOW_MAX_FRAME_ROTATION_DEG = 24.0
FLOW_MAX_FRAME_TRANSLATION_PX = 85.0
FLOW_MAX_SLIT_LENGTH_RATIO = 1.28
FLOW_MIN_SLIT_LENGTH_RATIO = 0.78
FLOW_SLIT_SNAP_MIN_PARALLEL = 0.68
FLOW_SLIT_SNAP_MAX_MIDPOINT_DISTANCE_PX = 105.0
FLOW_SLIT_SNAP_BLEND = 0.72

# Dynamic slit corridor. The normal points away from the polymailer center.
SLIT_SIDE_MARGIN_PX = 75
SLIT_OUTWARD_DISTANCE_PX = 220
SLIT_INWARD_DISTANCE_PX = 85
SLIT_CONTACT_OUTWARD_PX = 75
SLIT_CONTACT_INWARD_PX = 55
SLIT_CROP_PADDING_PX = 18
SLIT_CROP_MIN_SIZE_PX = 160

# Dynamic yellow-corridor growth.
#
# The yellow corridor still follows the selected moving slit, but it now grows
# as the polymailer is pulled away or sideways. This prevents the product from
# being left outside a fixed-size SAM search rectangle while the bag moves.
#
# Growth is monotonic during one cycle. Press X to reset it.
DYNAMIC_CORRIDOR_ENABLE = True
DYNAMIC_CORRIDOR_GROW_ONLY = True

DYNAMIC_CORRIDOR_PULL_FACTOR = 1.35
DYNAMIC_CORRIDOR_TOTAL_OUTWARD_FACTOR = 0.20
DYNAMIC_CORRIDOR_LATERAL_FACTOR = 1.05
DYNAMIC_CORRIDOR_TOTAL_SIDE_FACTOR = 0.12
DYNAMIC_CORRIDOR_TOTAL_INWARD_FACTOR = 0.08

DYNAMIC_CORRIDOR_MAX_EXTRA_OUTWARD_PX = 360.0
DYNAMIC_CORRIDOR_MAX_EXTRA_SIDE_PX = 190.0
DYNAMIC_CORRIDOR_MAX_EXTRA_INWARD_PX = 80.0

DYNAMIC_CORRIDOR_MIN_POSE_SCALE = 1.0
DYNAMIC_CORRIDOR_MAX_POSE_SCALE = 1.45

# Smooth dynamic-corridor updates to remove visible jitter.
#
# Expansion still grows enough to keep the product inside the yellow region,
# but it is rate-limited instead of jumping several dozen pixels in one frame.
DYNAMIC_CORRIDOR_OUTWARD_GROWTH_PX_PER_SECOND = 260.0
DYNAMIC_CORRIDOR_SIDE_GROWTH_PX_PER_SECOND = 190.0
DYNAMIC_CORRIDOR_INWARD_GROWTH_PX_PER_SECOND = 110.0
DYNAMIC_CORRIDOR_POSE_GROWTH_PER_SECOND = 0.55

# Low-pass smoothing for the yellow corridor polygon. Fast movement receives a
# larger alpha so the corridor can still follow the polymailer without lag.
DYNAMIC_CORRIDOR_POLYGON_SMOOTH_ALPHA = 0.34
DYNAMIC_CORRIDOR_FAST_MOTION_ALPHA = 0.70
DYNAMIC_CORRIDOR_FAST_MOTION_THRESHOLD_PX = 48.0

# Quantizing crop boundaries prevents the SAM input tensor size from changing
# by one or two pixels on every camera frame.
DYNAMIC_CORRIDOR_CROP_QUANTIZE_PX = 8

# Only one SAM request may be in flight. The old newest-frame queue copied a
# large crop and several metadata arrays at camera FPS even though SAM could
# only process roughly 5 FPS.
SAM_ONE_IN_FLIGHT_ONLY = True

# Emerging-product rules. These are intentionally stricter than the first
# moving-slit test because the black fixture contains many small SAM masks.
EMERGING_MIN_AREA_PX = 2200
EMERGING_MIN_AREA_RATIO_OF_POLY = 0.012
EMERGING_MAX_AREA_RATIO_OF_POLY = 0.62
EMERGING_MIN_BBOX_WIDTH_PX = 34
EMERGING_MIN_BBOX_HEIGHT_PX = 34
EMERGING_MIN_RECTANGULARITY = 0.12
EMERGING_MIN_CORRIDOR_OVERLAP = 0.08
EMERGING_MIN_CONTACT_OVERLAP = 0.008
EMERGING_MIN_OUTWARD_FRACTION = 0.12
EMERGING_MIN_OUTWARD_PIXELS = 500
EMERGING_MIN_OUTSIDE_POLY_FRACTION = 0.12
EMERGING_MAX_POLY_OVERLAP = 0.82
EMERGING_MIN_BASELINE_CHANGE_FRACTION = 0.12
EMERGING_MAX_ROI_EDGE_TOUCH_FRACTION = 0.35
EMERGING_DUPLICATE_IOU = 0.80
EMERGING_BAG_COLOR_REJECT_SIMILARITY = 0.92

# Stronger rejection for black fixture/base-side segments.
#
# The yellow SAM crop can become large after the bag is pulled away. These
# checks prevent clipped tray edges and baseline fixture fragments from being
# accepted as the released product.
PRODUCT_CROP_EDGE_BORDER_PX = 10
EMERGING_MAX_CROP_EDGE_TOUCH_FRACTION = 0.10
RELEASED_MAX_CROP_EDGE_TOUCH_FRACTION = 0.16
EMERGING_MIN_SLIT_SPAN_FRACTION = 0.28
EMERGING_SLIT_SPAN_MARGIN_PX = 55.0

BASELINE_STATIC_REJECT_CANDIDATE_CONTAINMENT = 0.55

BASE_SIDE_COLOR_EDGE_REJECT_SIMILARITY = 0.72
BASE_SIDE_LONG_ASPECT_RATIO = 3.4
BASE_SIDE_LOW_RECTANGULARITY = 0.52
BASE_SIDE_STRONG_CROSSING_MIN_POLY_OVERLAP = 0.035
BASE_SIDE_STRONG_CROSSING_MIN_CHANGE = 0.30

# A candidate must either visibly cross the slit line or be a sufficiently
# large object immediately outside it. This rejects tiny tray cells that happen
# to sit in the yellow corridor.
EMERGING_LINE_BAND_PX = 24.0
EMERGING_MIN_LINE_BAND_FRACTION = 0.018
EMERGING_MIN_INWARD_FRACTION_FOR_CROSSING = 0.025
EMERGING_OUTSIDE_ONLY_MIN_AREA_RATIO_OF_POLY = 0.020
EMERGING_OUTSIDE_ONLY_MAX_LINE_DISTANCE_PX = 38.0

# Reject static objects already present before SPACE.
BASELINE_STATIC_MIN_MASK_AREA_PX = 500
BASELINE_STATIC_MAX_MASK_AREA_RATIO = 0.70
BASELINE_STATIC_REJECT_IOU = 0.28
BASELINE_STATIC_REJECT_CENTER_DISTANCE_PX = 28.0
BASELINE_STATIC_REJECT_AREA_RATIO_LOW = 0.58
BASELINE_STATIC_REJECT_AREA_RATIO_HIGH = 1.72
BASELINE_VISIBLE_MIN_FRACTION = 0.22
BASELINE_UNCHANGED_PIXEL_THRESHOLD = 18.0
BASELINE_UNCHANGED_REJECT_FRACTION = 0.68
BASELINE_BASE_COLOR_REJECT_SIMILARITY = 0.86
BASELINE_BASE_COLOR_BYPASS_MAX_SIMILARITY = 0.72

# Depth rejection for black-base fragments. A real product is normally closer
# to the camera than the local base surrounding it. Large, clearly new masks
# can bypass this when stereo depth is sparse.
EMERGING_DEPTH_RING_INNER_PX = 8
EMERGING_DEPTH_RING_OUTER_PX = 34
EMERGING_MIN_DEPTH_SAMPLE_COUNT = 60
EMERGING_MIN_HEIGHT_ABOVE_BASE_MM = 8.0
EMERGING_STRONG_VISUAL_BYPASS_AREA_RATIO = 0.030
EMERGING_STRONG_VISUAL_BYPASS_CHANGE_FRACTION = 0.45

BASELINE_DIFF_THRESHOLD = 22
BASELINE_DIFF_OPEN_KERNEL_PX = 5
BASELINE_DIFF_CLOSE_KERNEL_PX = 11

PRODUCT_CONFIRM_REQUIRED_SAM_UPDATES = 2
PRODUCT_MATCH_MAX_CENTER_DISTANCE_PX = 85
PRODUCT_MATCH_MIN_IOU = 0.08
PRODUCT_MATCH_MIN_AREA_RATIO = 0.45
PRODUCT_MATCH_MAX_AREA_RATIO = 2.20

# After the product first crosses the selected moving slit, keep tracking the
# same product until the complete mask is separated from the polymailer and
# has stopped moving on the table. This does not change the existing exit
# detector. It adds a second verification stage after PRODUCT EXIT CONFIRMED.
FULL_RELEASE_VERIFY_CROP_PADDING_PX = 150
FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX = 240
FULL_RELEASE_REFERENCE_MIN_AREA_RATIO = 0.35
FULL_RELEASE_REFERENCE_MAX_AREA_RATIO = 4.00
FULL_RELEASE_REFERENCE_MAX_CENTER_DISTANCE_PX = 165.0
FULL_RELEASE_REFERENCE_MIN_IOU = 0.025
FULL_RELEASE_MAX_POLY_OVERLAP = 0.035
FULL_RELEASE_MAX_INWARD_FRACTION = 0.045
FULL_RELEASE_MIN_OUTWARD_FRACTION = 0.72
FULL_RELEASE_MAX_CONTACT_OVERLAP = 0.025
FULL_RELEASE_MIN_LINE_DISTANCE_PX = 14.0

# The product must not merely be mostly visible above the slit. Its trailing
# edge must clear the moving slit and the product core must have a real gap
# from the polymailer core before the mailer can be discarded.
FULL_RELEASE_MIN_RAW_TRAILING_CLEARANCE_PX = 5.0
FULL_RELEASE_MIN_CORE_TRAILING_CLEARANCE_PX = 10.0
FULL_RELEASE_MIN_CORE_POLY_GAP_PX = 7.0
FULL_RELEASE_MAX_CORE_LINE_BAND_FRACTION = 0.08
FULL_RELEASE_MIN_CONFIRM_DELAY_SECONDS = 0.75

FULL_RELEASE_MIN_HEIGHT_ABOVE_BASE_MM = 5.0
FULL_RELEASE_MAX_CENTER_STEP_PX = 18.0
FULL_RELEASE_MIN_STABLE_AREA_RATIO = 0.80
FULL_RELEASE_MAX_STABLE_AREA_RATIO = 1.25
FULL_RELEASE_REQUIRED_STABLE_UPDATES = 3
FULL_RELEASE_TIMEOUT_SECONDS = 6.0
FULL_RELEASE_VISUAL_TABLE_MIN_CHANGE = 0.58
FULL_RELEASE_VISUAL_TABLE_MAX_BASE_COLOR_SIMILARITY = 0.66

# SAM and optical-flow outlines can overlap by a few pixels even when the
# physical product is already clear of the polymailer. Full-release
# verification also evaluates eroded core masks so boundary-only overlap does
# not incorrectly request another shake.
FULL_RELEASE_PRODUCT_CORE_ERODE_PX = 6
FULL_RELEASE_POLY_CORE_ERODE_PX = 10
FULL_RELEASE_MIN_CORE_AREA_PX = 180
FULL_RELEASE_MAX_RAW_POLY_OVERLAP_WITH_CLEAR_CORE = 0.12
FULL_RELEASE_MAX_CORE_POLY_OVERLAP = 0.025
FULL_RELEASE_MAX_CORE_INWARD_FRACTION = 0.070
FULL_RELEASE_MIN_CORE_OUTWARD_FRACTION = 0.72

# Fast-exit fallback.
#
# The moving-slit detector needs the product to straddle the slit for a couple
# of SAM updates. SAM runs at roughly 5 Hz, so a product dumped out in a few
# tenths of a second crosses the slit entirely between two updates and is never
# seen as "emerging" at all.
#
# This second path does not try to watch the crossing. It watches for the
# arrival instead: a baseline difference runs on every camera frame (~1 ms) and
# looks for new, non-bag content on the outward side of the slit. Once that
# blob stops moving, one SAM scan is run on it and the result enters the normal
# exit + full-release verification. Set FAST_EXIT_ENABLE to False to get the
# original slit-only behaviour back.
FAST_EXIT_ENABLE = True

# Minimum size of the arrival blob before it is worth a SAM scan.
FAST_EXIT_MIN_AREA_PX = 2200

# The flow-tracked bag is dilated before being cut out of the difference, so
# the bag's own moving boundary does not leak in as "new" content.
FAST_EXIT_BAG_DILATE_PX = 25

# A released product lands and stops. Requiring it to hold still separates it
# from the table being revealed while the bag is still being pulled away.
FAST_EXIT_SETTLE_MAX_CENTER_STEP_PX = 6.0
FAST_EXIT_SETTLE_FRAMES = 8

# Most of the blob must be on the exit side of the slit.
FAST_EXIT_MIN_OUTWARD_FRACTION = 0.55

# SAM crop around the settled blob.
FAST_EXIT_CROP_PADDING_PX = 90
FAST_EXIT_CROP_MIN_SIZE_PX = 240

# The SAM mask must line up with the blob that triggered the scan. This
# replaces the slit-relative gating that the fast path has to skip.
FAST_EXIT_MIN_ARRIVAL_OVERLAP = 0.45

# Do not resubmit a fast-exit scan more often than this while one keeps failing
# the candidate gates.
FAST_EXIT_RETRY_INTERVAL_SECONDS = 0.80


# Optical flow is fast, but a flexible polymailer can bend and change shape in
# ways that a similarity transform cannot model forever. Periodically run a
# small SAM scan around the complete tracked bag and re-anchor the polymailer
# mask to the current image. This keeps the bag boundary from drifting over the
# black fixture or over a fully released product.
POLY_REFRESH_ENABLE = True
POLY_REFRESH_INTERVAL_SECONDS = 0.75
POLY_REFRESH_CROP_PADDING_PX = 55
POLY_REFRESH_CROP_MIN_SIZE_PX = 260
POLY_REFRESH_MIN_AREA_RATIO_TO_PREDICTED = 0.24
POLY_REFRESH_MAX_AREA_RATIO_TO_PREDICTED = 1.55
POLY_REFRESH_MIN_COLOR_SIMILARITY = 0.58
POLY_REFRESH_MIN_CANDIDATE_INSIDE_PREDICTED = 0.34
POLY_REFRESH_MIN_IOU = 0.12
POLY_REFRESH_MAX_CENTER_DISTANCE_PX = 145.0
POLY_REFRESH_SLIT_BAND_WIDTH_PX = 34
POLY_REFRESH_MIN_SLIT_BAND_PIXELS = 45
POLY_REFRESH_MAX_CROP_EDGE_TOUCH_FRACTION = 0.38
POLY_REFRESH_PRINT_EVERY = 5

OVERLAY_ALPHA = 0.30


class LiveSAMWorker:
    """Runs full-ROI or moving-slit SAM in one background thread."""

    def __init__(self, full_generator, slit_generator, device_name):
        self.full_generator = full_generator
        self.slit_generator = slit_generator
        self.device_name = device_name
        self.use_autocast = bool(
            USE_CUDA_AUTOCAST and device_name == "cuda"
        )
        self.input_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_result = None
        self.in_flight = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)

    def submit(
        self,
        frame_id,
        image_rgb,
        mode,
        crop_box,
        metadata=None,
    ):
        if SAM_ONE_IN_FLIGHT_ONLY:
            with self.lock:
                if self.in_flight:
                    return False
                self.in_flight = True

        try:
            # Crop views are often non-contiguous. This creates at most one
            # contiguous SAM input copy per completed SAM inference instead of
            # copying a replacement request on every camera frame.
            image_for_sam = np.ascontiguousarray(image_rgb)

            item = {
                "frame_id": frame_id,
                "image_rgb": image_for_sam,
                "mode": mode,
                "crop_box": tuple(crop_box),
                "metadata": copy_metadata(metadata),
                "submitted_at": time.perf_counter(),
            }

            self.input_queue.put_nowait(item)
            return True

        except queue.Full:
            if SAM_ONE_IN_FLIGHT_ONLY:
                with self.lock:
                    self.in_flight = False
            return False

        except Exception:
            if SAM_ONE_IN_FLIGHT_ONLY:
                with self.lock:
                    self.in_flight = False
            raise

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def _generate(self, generator, image_rgb):
        with torch.inference_mode():
            if self.use_autocast:
                try:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        return generator.generate(image_rgb)
                except Exception as exc:
                    print()
                    print(
                        "CUDA autocast failed once; retrying SAM in normal "
                        f"precision: {exc}"
                    )
                    self.use_autocast = False

            return generator.generate(image_rgb)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            frame_id = item["frame_id"]
            image_rgb = item["image_rgb"]
            mode = item["mode"]
            crop_box = item["crop_box"]
            metadata = item["metadata"]
            submitted_at = item["submitted_at"]
            started_at = time.perf_counter()

            try:
                generator = (
                    self.full_generator
                    if mode == "full"
                    else self.slit_generator
                )
                local_masks = self._generate(generator, image_rgb)
                masks = map_masks_to_full_roi(local_masks, crop_box)
                error = None
            except Exception as exc:
                masks = []
                error = str(exc)

            finished_at = time.perf_counter()

            result = {
                "frame_id": frame_id,
                "mode": mode,
                "crop_box": crop_box,
                "masks": masks,
                "sam_seconds": finished_at - started_at,
                "queue_delay_seconds": started_at - submitted_at,
                "finished_at": finished_at,
                "error": error,
                "metadata": metadata,
            }

            with self.lock:
                self.latest_result = result
                self.in_flight = False


def copy_metadata(metadata):
    if metadata is None:
        return None

    copied = {}

    # These arrays are freshly allocated for the current camera frame or
    # geometry update and are not modified afterward. Keeping references avoids
    # several multi-megabyte copies per SAM request.
    safe_reference_keys = {
        "roi_rgb",
        "roi_bgr",
        "depth_roi",
        "corridor_mask",
        "contact_mask",
        "corridor_polygon",
        "contact_polygon",
    }

    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            if key in safe_reference_keys:
                copied[key] = value
            else:
                copied[key] = value.copy()
        else:
            copied[key] = value

    return copied


def map_masks_to_full_roi(local_masks, crop_box):
    x1, y1, x2, y2 = crop_box
    crop_width = x2 - x1
    crop_height = y2 - y1
    mapped = []

    for local_mask_data in local_masks:
        local_mask = local_mask_data["segmentation"].astype(np.uint8)

        if local_mask.shape != (crop_height, crop_width):
            local_mask = cv2.resize(
                local_mask,
                (crop_width, crop_height),
                interpolation=cv2.INTER_NEAREST,
            )

        full_mask = np.zeros(
            (ROI_HEIGHT, ROI_WIDTH),
            dtype=np.uint8,
        )
        full_mask[y1:y2, x1:x2] = local_mask

        mapped_data = dict(local_mask_data)
        mapped_data["segmentation"] = full_mask.astype(bool)
        mapped.append(mapped_data)

    return mapped


def configure_torch(device_name):
    if device_name != "cuda":
        return

    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        properties = torch.cuda.get_device_properties(0)
        if properties.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("CUDA TF32 acceleration enabled.")
    except Exception as exc:
        print(f"Could not configure TF32: {exc}")


def apply_manual_exposure(camera_node, label):
    if not USE_MANUAL_STEREO_EXPOSURE:
        return

    try:
        camera_node.initialControl.setManualExposure(
            STEREO_EXPOSURE_US,
            STEREO_ISO,
        )
        print(
            f"{label}: manual exposure "
            f"{STEREO_EXPOSURE_US} us ISO {STEREO_ISO}"
        )
    except Exception as exc:
        print(f"{label}: could not set manual exposure: {exc}")


def align_depth_to_rgb(depth):
    height, width = depth.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    matrix = np.float32(
        [
            [
                DEPTH_ALIGN_SCALE_X,
                0,
                (1.0 - DEPTH_ALIGN_SCALE_X) * center_x
                + DEPTH_ALIGN_X_SHIFT_PX,
            ],
            [
                0,
                DEPTH_ALIGN_SCALE_Y,
                (1.0 - DEPTH_ALIGN_SCALE_Y) * center_y
                + DEPTH_ALIGN_Y_SHIFT_PX,
            ],
        ]
    )

    return cv2.warpAffine(
        depth,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def set_ir(device):
    try:
        device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
        print(f"IR laser set to {IR_LASER_INTENSITY}")
    except Exception as exc:
        print(f"Could not set IR laser: {exc}")

    try:
        device.setIrFloodLightIntensity(IR_FLOOD_INTENSITY)
        print(f"IR flood set to {IR_FLOOD_INTENSITY}")
    except Exception as exc:
        print(f"Could not set IR flood: {exc}")


def select_device():
    infos = dai.Device.getAllAvailableDevices()

    if not infos:
        raise RuntimeError("No OAK devices found. Check the USB connection.")

    if DEVICE_ID:
        for info in infos:
            if info.deviceId == DEVICE_ID:
                print(f"Using configured device: {info.deviceId} ({info.name})")
                return dai.Device(info)

        available = ", ".join(info.deviceId for info in infos)
        raise RuntimeError(
            f"Configured DEVICE_ID {DEVICE_ID} is not connected. "
            f"Available devices: {available}"
        )

    if len(infos) == 1:
        info = infos[0]
        print(f"Using only connected device: {info.deviceId} ({info.name})")
        return dai.Device(info)

    print("Multiple OAK devices connected:")
    for i, info in enumerate(infos):
        print(f"  [{i}] {info.deviceId}  (port {info.name})")

    while True:
        choice = input("Select device index: ").strip()

        if choice.isdigit() and 0 <= int(choice) < len(infos):
            info = infos[int(choice)]
            print(f"Using device: {info.deviceId} ({info.name})")
            return dai.Device(info)

        print("Invalid selection, try again.")


def build_pipeline(device):
    pipeline = dai.Pipeline(device)

    camera = pipeline.create(dai.node.Camera).build()

    rgb_output = camera.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS,
    )

    left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )

    apply_manual_exposure(left, "left stereo")
    apply_manual_exposure(right, "right stereo")

    left_output = left.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )
    right_output = right.requestOutput(
        size=RGB_SIZE,
        type=dai.ImgFrame.Type.GRAY8,
        fps=FPS,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.FAST_DENSITY
    )
    stereo.setLeftRightCheck(USE_LEFT_RIGHT_CHECK)
    stereo.setSubpixel(USE_SUBPIXEL)

    try:
        stereo.setOutputSize(RGB_SIZE[0], RGB_SIZE[1])
    except Exception as exc:
        print(f"Could not set stereo output size: {exc}")

    try:
        stereo.initialConfig.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    except Exception as exc:
        print(f"Could not set stereo confidence: {exc}")

    left_output.link(stereo.left)
    right_output.link(stereo.right)

    rgb_queue = rgb_output.createOutputQueue(
        maxSize=2,
        blocking=False,
    )
    depth_queue = stereo.depth.createOutputQueue(
        maxSize=2,
        blocking=False,
    )

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


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def clean_mask(mask):
    raw = mask.astype(np.uint8)
    original_area = int(raw.sum())

    if original_area <= 0:
        return raw

    cleaned = close_mask(raw, kernel_px=15, iterations=1)
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return raw

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * 2.5:
        return cleaned.astype(np.uint8)

    return raw


def mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] <= 0:
        return None

    return np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=np.float32,
    )


def mask_iou(mask_a, mask_b):
    intersection = int(np.count_nonzero((mask_a == 1) & (mask_b == 1)))
    union = int(np.count_nonzero((mask_a == 1) | (mask_b == 1)))

    if union <= 0:
        return 0.0

    return intersection / union


def mask_overlap_fraction(mask, reference_mask):
    area = int(mask.sum())

    if area <= 0:
        return 0.0

    intersection = int(np.count_nonzero((mask == 1) & (reference_mask == 1)))
    return intersection / area


def erode_release_core(mask, radius_px, minimum_area_px):
    """Remove uncertain boundary pixels for full-release verification."""
    source = mask.astype(np.uint8)

    if radius_px <= 0:
        return source

    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    core = cv2.erode(source, kernel, iterations=1)

    if int(core.sum()) < minimum_area_px:
        return source

    return core.astype(np.uint8)


def full_release_core_metrics(candidate_mask, poly_mask, slit_geometry):
    """Measure separation while ignoring thin SAM/flow outline overlap."""
    candidate_core = erode_release_core(
        candidate_mask,
        FULL_RELEASE_PRODUCT_CORE_ERODE_PX,
        FULL_RELEASE_MIN_CORE_AREA_PX,
    )
    poly_core = erode_release_core(
        poly_mask,
        FULL_RELEASE_POLY_CORE_ERODE_PX,
        FULL_RELEASE_MIN_CORE_AREA_PX,
    )

    slit_metrics = signed_slit_metrics(candidate_core, slit_geometry)

    # Distance from every product-core pixel to the nearest polymailer-core
    # pixel. A positive gap prevents a half-emerged product whose visible mask
    # simply stops at the slit from being called fully released.
    if int(poly_core.sum()) > 0 and int(candidate_core.sum()) > 0:
        distance_from_poly = cv2.distanceTransform(
            (poly_core == 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        core_poly_gap_px = float(np.min(distance_from_poly[candidate_core == 1]))
    else:
        core_poly_gap_px = 0.0

    return {
        "core_poly_overlap": float(
            mask_overlap_fraction(candidate_core, poly_core)
        ),
        "core_poly_gap_px": core_poly_gap_px,
        "core_outward_fraction": float(
            slit_metrics["outward_fraction"]
        ),
        "core_inward_fraction": float(
            slit_metrics["inward_fraction"]
        ),
        "core_line_band_fraction": float(
            slit_metrics["line_band_fraction"]
        ),
        "core_min_line_distance_px": float(
            slit_metrics["min_line_distance_px"]
        ),
        "core_min_signed_distance_px": float(
            slit_metrics["min_signed_distance_px"]
        ),
        "core_trailing_edge_clearance_px": float(
            slit_metrics["trailing_edge_clearance_px"]
        ),
    }


def masked_mean_hsv(roi_rgb, mask):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    values = hsv[mask == 1]

    if values.size == 0:
        return None

    return (
        float(np.mean(values[:, 0])),
        float(np.mean(values[:, 1])),
        float(np.mean(values[:, 2])),
    )


def hsv_similarity(hsv_a, hsv_b):
    if hsv_a is None or hsv_b is None:
        return 0.0

    h_a, s_a, v_a = hsv_a
    h_b, s_b, v_b = hsv_b

    hue_distance = abs(h_a - h_b)
    hue_distance = min(hue_distance, 180.0 - hue_distance) / 90.0
    saturation_distance = abs(s_a - s_b) / 255.0
    value_distance = abs(v_a - v_b) / 255.0

    distance = (
        0.50 * hue_distance
        + 0.25 * saturation_distance
        + 0.25 * value_distance
    )

    return float(max(0.0, 1.0 - min(distance, 1.0)))


def choose_polymailer_mask(masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2.0
    roi_cy = h / 2.0
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    best = None

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if not (MIN_POLY_AREA_RATIO <= area_ratio <= MAX_POLY_AREA_RATIO):
            continue

        x, y, width, height = cv2.boundingRect(mask)

        if width <= 0 or height <= 0:
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

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
        color_score = (
            0.50 * hue_score
            + 0.25 * sat_score
            + 0.25 * val_score
        )

        if color_score < MIN_POLY_COLOR_SCORE:
            continue

        center_x = x + width / 2.0
        center_y = y + height / 2.0
        distance = np.hypot(center_x - roi_cx, center_y - roi_cy)
        max_distance = np.hypot(roi_cx, roi_cy)
        center_score = 1.0 - min(distance / max(max_distance, 1.0), 1.0)
        area_score = 1.0 - min(
            abs(area_ratio - TARGET_POLY_AREA_RATIO)
            / TARGET_POLY_AREA_RATIO,
            1.0,
        )

        score = (
            3.0 * color_score
            + 1.5 * rectangularity
            + 1.2 * area_score
            + 1.0 * center_score
            + 0.25 * float(sam_mask.get("predicted_iou", 0.0))
            + 0.25 * float(sam_mask.get("stability_score", 0.0))
        )

        candidate = {
            "index": index,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, width, height),
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "color_score": float(color_score),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def make_polymailer_refresh_crop(poly_mask):
    """Build a padded crop around the currently tracked complete bag mask."""
    if poly_mask is None or int(poly_mask.sum()) <= 0:
        return (0, 0, ROI_WIDTH, ROI_HEIGHT)

    x, y, width, height = cv2.boundingRect(poly_mask.astype(np.uint8))
    x1 = max(0, x - POLY_REFRESH_CROP_PADDING_PX)
    y1 = max(0, y - POLY_REFRESH_CROP_PADDING_PX)
    x2 = min(ROI_WIDTH, x + width + POLY_REFRESH_CROP_PADDING_PX)
    y2 = min(ROI_HEIGHT, y + height + POLY_REFRESH_CROP_PADDING_PX)

    if x2 - x1 < POLY_REFRESH_CROP_MIN_SIZE_PX:
        missing = POLY_REFRESH_CROP_MIN_SIZE_PX - (x2 - x1)
        x1 = max(0, x1 - missing // 2)
        x2 = min(ROI_WIDTH, x2 + missing - missing // 2)
        if x2 - x1 < POLY_REFRESH_CROP_MIN_SIZE_PX:
            if x1 == 0:
                x2 = min(ROI_WIDTH, POLY_REFRESH_CROP_MIN_SIZE_PX)
            else:
                x1 = max(0, ROI_WIDTH - POLY_REFRESH_CROP_MIN_SIZE_PX)

    if y2 - y1 < POLY_REFRESH_CROP_MIN_SIZE_PX:
        missing = POLY_REFRESH_CROP_MIN_SIZE_PX - (y2 - y1)
        y1 = max(0, y1 - missing // 2)
        y2 = min(ROI_HEIGHT, y2 + missing - missing // 2)
        if y2 - y1 < POLY_REFRESH_CROP_MIN_SIZE_PX:
            if y1 == 0:
                y2 = min(ROI_HEIGHT, POLY_REFRESH_CROP_MIN_SIZE_PX)
            else:
                y1 = max(0, ROI_HEIGHT - POLY_REFRESH_CROP_MIN_SIZE_PX)

    return (int(x1), int(y1), int(x2), int(y2))


def crop_edge_touch_fraction(mask, crop_box):
    """Measure how much of a mask lies against the dynamic crop boundary."""
    x1, y1, x2, y2 = crop_box
    local = mask[y1:y2, x1:x2].astype(np.uint8)
    area = int(local.sum())

    if area <= 0 or local.size == 0:
        return 1.0

    border = np.zeros_like(local, dtype=np.uint8)
    border_width = min(5, max(1, min(local.shape[:2]) // 20))
    border[:border_width, :] = 1
    border[-border_width:, :] = 1
    border[:, :border_width] = 1
    border[:, -border_width:] = 1
    touching = int(np.count_nonzero((local == 1) & (border == 1)))
    return touching / max(area, 1)


def make_slit_band_mask(slit_points):
    band = np.zeros((ROI_HEIGHT, ROI_WIDTH), dtype=np.uint8)

    if slit_points is None:
        return band

    point_a = tuple(np.round(slit_points[0]).astype(int))
    point_b = tuple(np.round(slit_points[1]).astype(int))
    cv2.line(
        band,
        point_a,
        point_b,
        1,
        thickness=POLY_REFRESH_SLIT_BAND_WIDTH_PX,
        lineType=cv2.LINE_AA,
    )
    return (band > 0).astype(np.uint8)


def choose_live_polymailer_refresh_mask(
    masks,
    roi_rgb,
    predicted_poly_mask,
    baseline_poly_hsv,
    slit_points,
    crop_box,
):
    """Select the current bag mask using color plus optical-flow prediction."""
    if predicted_poly_mask is None:
        return None

    predicted = predicted_poly_mask.astype(np.uint8)
    predicted_area = int(predicted.sum())
    predicted_center = mask_center(predicted)

    if predicted_area <= 0 or predicted_center is None:
        return None

    slit_band = make_slit_band_mask(slit_points)
    best = None

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area <= 0:
            continue

        area_ratio = area / max(predicted_area, 1)

        if not (
            POLY_REFRESH_MIN_AREA_RATIO_TO_PREDICTED
            <= area_ratio
            <= POLY_REFRESH_MAX_AREA_RATIO_TO_PREDICTED
        ):
            continue

        candidate_center = mask_center(mask)

        if candidate_center is None:
            continue

        center_distance = float(
            np.linalg.norm(candidate_center - predicted_center)
        )

        if center_distance > POLY_REFRESH_MAX_CENTER_DISTANCE_PX:
            continue

        intersection = int(
            np.count_nonzero((mask == 1) & (predicted == 1))
        )
        candidate_inside_predicted = intersection / max(area, 1)
        predicted_inside_candidate = intersection / max(predicted_area, 1)
        iou = mask_iou(mask, predicted)

        if (
            candidate_inside_predicted
            < POLY_REFRESH_MIN_CANDIDATE_INSIDE_PREDICTED
            and iou < POLY_REFRESH_MIN_IOU
        ):
            continue

        candidate_hsv = masked_mean_hsv(roi_rgb, mask)
        color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )

        if color_similarity < POLY_REFRESH_MIN_COLOR_SIMILARITY:
            continue

        slit_band_pixels = int(
            np.count_nonzero((mask == 1) & (slit_band == 1))
        )

        if slit_band_pixels < POLY_REFRESH_MIN_SLIT_BAND_PIXELS:
            continue

        edge_touch = crop_edge_touch_fraction(mask, crop_box)

        if edge_touch > POLY_REFRESH_MAX_CROP_EDGE_TOUCH_FRACTION:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        center_score = 1.0 - min(
            center_distance / POLY_REFRESH_MAX_CENTER_DISTANCE_PX,
            1.0,
        )
        area_score = 1.0 - min(abs(area_ratio - 1.0), 1.0)
        slit_score = min(
            slit_band_pixels
            / max(POLY_REFRESH_MIN_SLIT_BAND_PIXELS * 5, 1),
            1.0,
        )

        score = (
            3.6 * color_similarity
            + 2.2 * candidate_inside_predicted
            + 1.3 * iou
            + 0.8 * predicted_inside_candidate
            + 0.8 * center_score
            + 0.6 * area_score
            + 0.6 * slit_score
            + 0.25 * sam_iou
            + 0.25 * sam_stability
            - 0.7 * edge_touch
        )

        candidate = {
            "index": index,
            "score": float(score),
            "mask": mask.astype(np.uint8),
            "area_ratio": float(area_ratio),
            "candidate_inside_predicted": float(
                candidate_inside_predicted
            ),
            "predicted_inside_candidate": float(
                predicted_inside_candidate
            ),
            "iou": float(iou),
            "color_similarity": float(color_similarity),
            "center_distance_px": float(center_distance),
            "slit_band_pixels": int(slit_band_pixels),
            "edge_touch_fraction": float(edge_touch),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def get_initial_slit_edge(poly_mask, slit_side=DEFAULT_SLIT_SIDE):
    """Return the selected image-relative edge of the polymailer rectangle."""
    if slit_side not in VALID_SLIT_SIDES:
        raise ValueError(f"Unsupported slit side: {slit_side}")

    contours, _ = cv2.findContours(
        poly_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)

    edges = []

    for index in range(4):
        point_a = corners[index]
        point_b = corners[(index + 1) % 4]
        midpoint = (point_a + point_b) * 0.5
        edges.append(
            {
                "points": np.stack([point_a, point_b]).astype(np.float32),
                "midpoint": midpoint.astype(np.float32),
            }
        )

    if slit_side == SLIT_SIDE_TOP:
        selected = min(edges, key=lambda edge: float(edge["midpoint"][1]))
    elif slit_side == SLIT_SIDE_BOTTOM:
        selected = max(edges, key=lambda edge: float(edge["midpoint"][1]))
    elif slit_side == SLIT_SIDE_LEFT:
        selected = min(edges, key=lambda edge: float(edge["midpoint"][0]))
    else:
        selected = max(edges, key=lambda edge: float(edge["midpoint"][0]))

    slit_edge = selected["points"].copy()

    # Give each selected edge a consistent point order. This makes the
    # displayed slit and tangent direction stable between runs.
    if slit_side in (SLIT_SIDE_TOP, SLIT_SIDE_BOTTOM):
        if slit_edge[0, 0] > slit_edge[1, 0]:
            slit_edge = slit_edge[::-1].copy()
    else:
        if slit_edge[0, 1] > slit_edge[1, 1]:
            slit_edge = slit_edge[::-1].copy()

    return slit_edge.astype(np.float32)


def transform_points_homography(points, homography):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)


class MovingSlitTracker:
    """Tracks the polymailer and slit without projective homography drift."""

    def __init__(self):
        self.active = False
        self.prev_gray = None
        self.points = None
        self.poly_mask = None
        self.slit_points = None
        self.slit_side = DEFAULT_SLIT_SIDE
        self.frame_index = 0
        self.last_good_point_count = 0
        self.last_local_point_count = 0
        self.last_transform_ok = False
        self.last_transform_source = "none"
        self.last_scale = 1.0
        self.last_rotation_deg = 0.0
        self.last_translation_px = 0.0
        self.last_inlier_ratio = 0.0

    def clear(self):
        self.__init__()

    def initialize(self, roi_bgr, poly_mask, slit_side=DEFAULT_SLIT_SIDE):
        slit_points = get_initial_slit_edge(poly_mask, slit_side)

        if slit_points is None:
            return False

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self.poly_mask = poly_mask.astype(np.uint8).copy()
        self.slit_points = slit_points.astype(np.float32)
        self.slit_side = slit_side
        self.prev_gray = gray
        self.points = self._detect_points(gray, self.poly_mask)
        self.frame_index = 0
        self.last_good_point_count = 0 if self.points is None else len(self.points)
        self.last_local_point_count = 0
        self.last_transform_ok = True
        self.last_transform_source = "initial"
        self.active = self.points is not None and len(self.points) >= FLOW_MIN_GOOD_POINTS
        return self.active

    def _detect_points(self, gray, mask):
        feature_mask = cv2.erode(
            mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=FLOW_QUALITY_LEVEL,
            minDistance=FLOW_MIN_DISTANCE,
            mask=feature_mask,
            blockSize=FLOW_BLOCK_SIZE,
        )

    @staticmethod
    def _apply_affine(points, affine):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(points, affine).reshape(-1, 2)

    @staticmethod
    def _affine_properties(affine):
        a = float(affine[0, 0])
        c = float(affine[1, 0])
        scale = float(np.sqrt(a * a + c * c))
        rotation_deg = float(np.degrees(np.arctan2(c, a)))
        translation_px = float(
            np.linalg.norm(
                np.array([affine[0, 2], affine[1, 2]], dtype=np.float32)
            )
        )
        return scale, rotation_deg, translation_px

    def _estimate_similarity(self, old_points, new_points):
        if len(old_points) < 3 or len(new_points) < 3:
            return None, 0.0

        affine, inliers = cv2.estimateAffinePartial2D(
            old_points.astype(np.float32),
            new_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=FLOW_RANSAC_REPROJ_THRESHOLD,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )

        if affine is None:
            return None, 0.0

        if inliers is None:
            inlier_ratio = 1.0
        else:
            inlier_ratio = float(np.mean(inliers.reshape(-1) != 0))

        scale, rotation_deg, translation_px = self._affine_properties(affine)

        sane = (
            FLOW_MIN_FRAME_SCALE <= scale <= FLOW_MAX_FRAME_SCALE
            and abs(rotation_deg) <= FLOW_MAX_FRAME_ROTATION_DEG
            and translation_px <= FLOW_MAX_FRAME_TRANSLATION_PX
            and inlier_ratio >= FLOW_MIN_AFFINE_INLIER_RATIO
        )

        if not sane:
            return None, inlier_ratio

        return affine.astype(np.float64), inlier_ratio

    def _local_slit_point_mask(self, points):
        if self.slit_points is None or self.poly_mask is None:
            return np.zeros(len(points), dtype=bool)

        point_a = self.slit_points[0].astype(np.float32)
        point_b = self.slit_points[1].astype(np.float32)
        midpoint = (point_a + point_b) * 0.5
        tangent = point_b - point_a
        slit_length = float(np.linalg.norm(tangent))

        if slit_length < 1.0:
            return np.zeros(len(points), dtype=bool)

        tangent /= slit_length
        center = mask_center(self.poly_mask)

        if center is None:
            return np.zeros(len(points), dtype=bool)

        outward = midpoint - np.asarray(center, dtype=np.float32)
        outward_length = float(np.linalg.norm(outward))

        if outward_length < 1.0:
            outward = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        else:
            outward /= outward_length

        relative = points.astype(np.float32) - midpoint
        along = relative @ tangent
        normal = relative @ outward

        return (
            (np.abs(along) <= slit_length * 0.5 + FLOW_LOCAL_SLIT_SIDE_MARGIN_PX)
            & (normal >= -FLOW_LOCAL_SLIT_BAND_INWARD_PX)
            & (normal <= FLOW_LOCAL_SLIT_BAND_OUTWARD_PX)
        )

    @staticmethod
    def _rect_edges(mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

        return [
            np.stack([corners[index], corners[(index + 1) % 4]]).astype(np.float32)
            for index in range(4)
        ]

    def _snap_slit_to_mask_boundary(self, predicted_slit, mask):
        predicted = predicted_slit.astype(np.float32)
        predicted_vector = predicted[1] - predicted[0]
        predicted_length = float(np.linalg.norm(predicted_vector))

        if predicted_length < 1.0:
            return None

        predicted_tangent = predicted_vector / predicted_length
        predicted_midpoint = np.mean(predicted, axis=0)
        best_edge = None
        best_score = None

        for edge in self._rect_edges(mask):
            edge_vector = edge[1] - edge[0]
            edge_length = float(np.linalg.norm(edge_vector))

            if edge_length < 1.0:
                continue

            edge_tangent = edge_vector / edge_length
            parallel = abs(float(np.dot(predicted_tangent, edge_tangent)))

            if parallel < FLOW_SLIT_SNAP_MIN_PARALLEL:
                continue

            edge_midpoint = np.mean(edge, axis=0)
            midpoint_distance = float(np.linalg.norm(edge_midpoint - predicted_midpoint))

            if midpoint_distance > FLOW_SLIT_SNAP_MAX_MIDPOINT_DISTANCE_PX:
                continue

            length_ratio = edge_length / max(predicted_length, 1.0)

            if not (
                FLOW_MIN_SLIT_LENGTH_RATIO
                <= length_ratio
                <= FLOW_MAX_SLIT_LENGTH_RATIO
            ):
                continue

            score = (
                midpoint_distance
                + 85.0 * (1.0 - parallel)
                + 0.18 * abs(edge_length - predicted_length)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_edge = edge.copy()

        if best_edge is None:
            return predicted

        if np.dot(best_edge[1] - best_edge[0], predicted_vector) < 0:
            best_edge = best_edge[::-1].copy()

        return (
            (1.0 - FLOW_SLIT_SNAP_BLEND) * predicted
            + FLOW_SLIT_SNAP_BLEND * best_edge
        ).astype(np.float32)

    def refresh_from_sam(self, roi_bgr, refreshed_mask, reference_slit_points):
        """Re-anchor the flexible bag mask and slit to a recent SAM result."""
        if refreshed_mask is None or reference_slit_points is None:
            return False

        refreshed = clean_mask(refreshed_mask)
        refreshed = largest_component(refreshed)

        if refreshed is None or int(refreshed.sum()) <= 0:
            return False

        reference_slit = np.asarray(
            reference_slit_points,
            dtype=np.float32,
        ).reshape(2, 2)
        snapped_slit = self._snap_slit_to_mask_boundary(
            reference_slit,
            refreshed,
        )

        if snapped_slit is None:
            return False

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        points = self._detect_points(gray, refreshed)

        if points is None or len(points) < FLOW_MIN_GOOD_POINTS:
            return False

        self.poly_mask = refreshed.astype(np.uint8)
        self.slit_points = snapped_slit.astype(np.float32)
        self.prev_gray = gray
        self.points = points
        self.active = True
        self.frame_index = 0
        self.last_good_point_count = len(points)
        self.last_local_point_count = 0
        self.last_transform_ok = True
        self.last_transform_source = "sam_refresh"
        self.last_scale = 1.0
        self.last_rotation_deg = 0.0
        self.last_translation_px = 0.0
        self.last_inlier_ratio = 1.0
        return True

    def update(self, roi_bgr):
        if not self.active:
            return False

        current_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_index += 1

        if self.points is None or len(self.points) < FLOW_MIN_GOOD_POINTS:
            self.points = self._detect_points(self.prev_gray, self.poly_mask)

        if self.points is None or len(self.points) < FLOW_MIN_GOOD_POINTS:
            self.prev_gray = current_gray
            self.last_good_point_count = 0
            self.last_local_point_count = 0
            self.last_transform_ok = False
            self.last_transform_source = "no_points"
            return False

        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            current_gray,
            self.points,
            None,
            winSize=FLOW_WIN_SIZE,
            maxLevel=FLOW_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if next_points is None or status_forward is None:
            self.prev_gray = current_gray
            self.last_transform_ok = False
            self.last_transform_source = "forward_flow_failed"
            return False

        back_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.prev_gray,
            next_points,
            None,
            winSize=FLOW_WIN_SIZE,
            maxLevel=FLOW_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if back_points is None or status_backward is None:
            self.prev_gray = current_gray
            self.last_transform_ok = False
            self.last_transform_source = "backward_flow_failed"
            return False

        forward_ok = status_forward.reshape(-1) == 1
        backward_ok = status_backward.reshape(-1) == 1
        fb_error = np.linalg.norm(
            self.points.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1,
        )
        good = (
            forward_ok
            & backward_ok
            & (fb_error <= FLOW_FORWARD_BACKWARD_MAX_ERROR)
        )

        old_good = self.points.reshape(-1, 2)[good]
        new_good = next_points.reshape(-1, 2)[good]
        self.last_good_point_count = len(old_good)

        if len(old_good) < FLOW_MIN_GOOD_POINTS:
            self.prev_gray = current_gray
            self.points = self._detect_points(current_gray, self.poly_mask)
            self.last_local_point_count = 0
            self.last_transform_ok = False
            self.last_transform_source = "too_few_good_points"
            return False

        global_affine, global_inlier_ratio = self._estimate_similarity(old_good, new_good)

        if global_affine is None:
            self.prev_gray = current_gray
            self.points = new_good.reshape(-1, 1, 2).astype(np.float32)
            self.last_local_point_count = 0
            self.last_transform_ok = False
            self.last_transform_source = "global_transform_rejected"
            return False

        local_selection = self._local_slit_point_mask(old_good)
        local_old = old_good[local_selection]
        local_new = new_good[local_selection]
        self.last_local_point_count = len(local_old)

        local_affine = None
        local_inlier_ratio = 0.0

        if len(local_old) >= FLOW_MIN_LOCAL_SLIT_POINTS:
            local_affine, local_inlier_ratio = self._estimate_similarity(local_old, local_new)

        slit_affine = local_affine if local_affine is not None else global_affine
        transform_source = "local_slit" if local_affine is not None else "global_bag"

        updated_mask = cv2.warpAffine(
            self.poly_mask,
            global_affine,
            (ROI_WIDTH, ROI_HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)
        updated_mask = largest_component(updated_mask)

        if updated_mask is None:
            self.prev_gray = current_gray
            self.active = False
            self.last_transform_ok = False
            self.last_transform_source = "mask_lost"
            return False

        predicted_slit = self._apply_affine(self.slit_points, slit_affine).astype(np.float32)

        old_slit_length = float(np.linalg.norm(self.slit_points[1] - self.slit_points[0]))
        new_slit_length = float(np.linalg.norm(predicted_slit[1] - predicted_slit[0]))
        slit_length_ratio = new_slit_length / max(old_slit_length, 1.0)

        if not (
            FLOW_MIN_SLIT_LENGTH_RATIO
            <= slit_length_ratio
            <= FLOW_MAX_SLIT_LENGTH_RATIO
        ):
            predicted_slit = self._apply_affine(self.slit_points, global_affine).astype(np.float32)
            transform_source = "global_length_fallback"

        snapped_slit = self._snap_slit_to_mask_boundary(predicted_slit, updated_mask)

        if snapped_slit is None:
            self.prev_gray = current_gray
            self.points = new_good.reshape(-1, 1, 2).astype(np.float32)
            self.last_transform_ok = False
            self.last_transform_source = "slit_snap_failed"
            return False

        self.poly_mask = updated_mask
        self.slit_points = snapped_slit
        self.prev_gray = current_gray
        self.points = new_good.reshape(-1, 1, 2).astype(np.float32)
        self.last_transform_ok = True
        self.last_transform_source = transform_source

        scale, rotation_deg, translation_px = self._affine_properties(slit_affine)
        self.last_scale = scale
        self.last_rotation_deg = rotation_deg
        self.last_translation_px = translation_px
        self.last_inlier_ratio = (
            local_inlier_ratio if local_affine is not None else global_inlier_ratio
        )

        if (
            len(self.points) < FLOW_RESEED_POINT_COUNT
            or self.frame_index % FLOW_RESEED_INTERVAL_FRAMES == 0
        ):
            reseeded = self._detect_points(current_gray, self.poly_mask)
            if reseeded is not None and len(reseeded) >= FLOW_MIN_GOOD_POINTS:
                self.points = reseeded

        return True


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))

    if length < 1e-6:
        return None

    return vector / length


def make_slit_geometry(
    poly_mask,
    slit_points,
    baseline_slit_midpoint=None,
    baseline_slit_length=None,
    expansion_state=None,
):
    if poly_mask is None or slit_points is None:
        return None

    point_a = slit_points[0].astype(np.float32)
    point_b = slit_points[1].astype(np.float32)
    tangent = normalize_vector(point_b - point_a)
    center = mask_center(poly_mask)

    if tangent is None or center is None:
        return None

    midpoint = (point_a + point_b) * 0.5
    outward = normalize_vector(midpoint - center)

    if outward is None:
        outward = np.array(
            [-tangent[1], tangent[0]],
            dtype=np.float32,
        )

    current_slit_length = float(
        np.linalg.norm(point_b - point_a)
    )

    pose_scale = 1.0
    pull_away_px = 0.0
    lateral_motion_px = 0.0
    total_motion_px = 0.0

    outward_extra_px = 0.0
    side_extra_px = 0.0
    inward_extra_px = 0.0

    dynamic_ready = (
        DYNAMIC_CORRIDOR_ENABLE
        and baseline_slit_midpoint is not None
        and baseline_slit_length is not None
        and baseline_slit_length > 1e-6
        and expansion_state is not None
    )

    if dynamic_ready:
        pose_scale = float(
            np.clip(
                current_slit_length / float(baseline_slit_length),
                DYNAMIC_CORRIDOR_MIN_POSE_SCALE,
                DYNAMIC_CORRIDOR_MAX_POSE_SCALE,
            )
        )

        movement = (
            midpoint.astype(np.float32)
            - np.asarray(
                baseline_slit_midpoint,
                dtype=np.float32,
            )
        )

        total_motion_px = float(np.linalg.norm(movement))

        # Pulling the bag away from the opening normally moves the slit in the
        # direction opposite its outward vector. Grow most strongly in that
        # direction because the product tends to remain near the table.
        pull_away_px = max(
            0.0,
            -float(np.dot(movement, outward)),
        )

        # Sideways movement grows both ends of the corridor.
        lateral_motion_px = abs(
            float(np.dot(movement, tangent))
        )

        requested_outward_extra = min(
            DYNAMIC_CORRIDOR_MAX_EXTRA_OUTWARD_PX,
            (
                pull_away_px * DYNAMIC_CORRIDOR_PULL_FACTOR
                + total_motion_px
                * DYNAMIC_CORRIDOR_TOTAL_OUTWARD_FACTOR
            ),
        )

        requested_side_extra = min(
            DYNAMIC_CORRIDOR_MAX_EXTRA_SIDE_PX,
            (
                lateral_motion_px * DYNAMIC_CORRIDOR_LATERAL_FACTOR
                + total_motion_px
                * DYNAMIC_CORRIDOR_TOTAL_SIDE_FACTOR
            ),
        )

        requested_inward_extra = min(
            DYNAMIC_CORRIDOR_MAX_EXTRA_INWARD_PX,
            (
                total_motion_px
                * DYNAMIC_CORRIDOR_TOTAL_INWARD_FACTOR
            ),
        )

        now_monotonic = time.monotonic()
        last_update_time = expansion_state.get(
            "_last_update_time",
            None,
        )

        if last_update_time is None:
            update_dt = 1.0 / max(float(FPS), 1.0)
        else:
            update_dt = float(
                np.clip(
                    now_monotonic - float(last_update_time),
                    0.0,
                    0.12,
                )
            )

        expansion_state["_last_update_time"] = now_monotonic

        current_outward = float(
            expansion_state.get("outward_extra_px", 0.0)
        )
        current_side = float(
            expansion_state.get("side_extra_px", 0.0)
        )
        current_inward = float(
            expansion_state.get("inward_extra_px", 0.0)
        )
        current_pose_scale = float(
            expansion_state.get("pose_scale", 1.0)
        )

        if DYNAMIC_CORRIDOR_GROW_ONLY:
            target_outward = max(
                current_outward,
                float(requested_outward_extra),
            )
            target_side = max(
                current_side,
                float(requested_side_extra),
            )
            target_inward = max(
                current_inward,
                float(requested_inward_extra),
            )
            target_pose_scale = max(
                current_pose_scale,
                float(pose_scale),
            )
        else:
            target_outward = float(requested_outward_extra)
            target_side = float(requested_side_extra)
            target_inward = float(requested_inward_extra)
            target_pose_scale = float(pose_scale)

        max_outward_step = (
            DYNAMIC_CORRIDOR_OUTWARD_GROWTH_PX_PER_SECOND
            * update_dt
        )
        max_side_step = (
            DYNAMIC_CORRIDOR_SIDE_GROWTH_PX_PER_SECOND
            * update_dt
        )
        max_inward_step = (
            DYNAMIC_CORRIDOR_INWARD_GROWTH_PX_PER_SECOND
            * update_dt
        )
        max_pose_step = (
            DYNAMIC_CORRIDOR_POSE_GROWTH_PER_SECOND
            * update_dt
        )

        expansion_state["outward_extra_px"] = float(
            current_outward
            + np.clip(
                target_outward - current_outward,
                -max_outward_step,
                max_outward_step,
            )
        )
        expansion_state["side_extra_px"] = float(
            current_side
            + np.clip(
                target_side - current_side,
                -max_side_step,
                max_side_step,
            )
        )
        expansion_state["inward_extra_px"] = float(
            current_inward
            + np.clip(
                target_inward - current_inward,
                -max_inward_step,
                max_inward_step,
            )
        )
        expansion_state["pose_scale"] = float(
            current_pose_scale
            + np.clip(
                target_pose_scale - current_pose_scale,
                -max_pose_step,
                max_pose_step,
            )
        )

        expansion_state["pull_away_px"] = float(
            pull_away_px
        )
        expansion_state["lateral_motion_px"] = float(
            lateral_motion_px
        )
        expansion_state["total_motion_px"] = float(
            total_motion_px
        )

        outward_extra_px = float(
            expansion_state.get("outward_extra_px", 0.0)
        )
        side_extra_px = float(
            expansion_state.get("side_extra_px", 0.0)
        )
        inward_extra_px = float(
            expansion_state.get("inward_extra_px", 0.0)
        )
        pose_scale = float(
            expansion_state.get("pose_scale", pose_scale)
        )

    side_margin_px = (
        SLIT_SIDE_MARGIN_PX * pose_scale
        + side_extra_px
    )

    outward_distance_px = (
        SLIT_OUTWARD_DISTANCE_PX * pose_scale
        + outward_extra_px
    )

    inward_distance_px = (
        SLIT_INWARD_DISTANCE_PX * pose_scale
        + inward_extra_px
    )

    # Keep the blue contact corridor local to the actual slit. Only the yellow
    # SAM-search corridor receives the large pull-away expansion.
    contact_side_margin_px = (
        SLIT_SIDE_MARGIN_PX * pose_scale
    )
    contact_outward_distance_px = (
        SLIT_CONTACT_OUTWARD_PX * pose_scale
    )
    contact_inward_distance_px = (
        SLIT_CONTACT_INWARD_PX * pose_scale
    )

    side_a = point_a - tangent * side_margin_px
    side_b = point_b + tangent * side_margin_px

    contact_side_a = (
        point_a - tangent * contact_side_margin_px
    )
    contact_side_b = (
        point_b + tangent * contact_side_margin_px
    )

    raw_corridor_polygon = np.stack(
        [
            side_a + outward * outward_distance_px,
            side_b + outward * outward_distance_px,
            side_b - outward * inward_distance_px,
            side_a - outward * inward_distance_px,
        ]
    )

    corridor_polygon = raw_corridor_polygon

    if dynamic_ready:
        previous_polygon = expansion_state.get(
            "_smoothed_corridor_polygon",
            None,
        )

        if (
            isinstance(previous_polygon, np.ndarray)
            and previous_polygon.shape == raw_corridor_polygon.shape
        ):
            point_motion = float(
                np.max(
                    np.linalg.norm(
                        raw_corridor_polygon - previous_polygon,
                        axis=1,
                    )
                )
            )

            alpha = (
                DYNAMIC_CORRIDOR_FAST_MOTION_ALPHA
                if point_motion
                >= DYNAMIC_CORRIDOR_FAST_MOTION_THRESHOLD_PX
                else DYNAMIC_CORRIDOR_POLYGON_SMOOTH_ALPHA
            )

            corridor_polygon = (
                (1.0 - alpha) * previous_polygon
                + alpha * raw_corridor_polygon
            ).astype(np.float32)

        expansion_state["_smoothed_corridor_polygon"] = (
            corridor_polygon.copy()
        )

    contact_polygon = np.stack(
        [
            (
                contact_side_a
                + outward * contact_outward_distance_px
            ),
            (
                contact_side_b
                + outward * contact_outward_distance_px
            ),
            (
                contact_side_b
                - outward * contact_inward_distance_px
            ),
            (
                contact_side_a
                - outward * contact_inward_distance_px
            ),
        ]
    )

    corridor_mask = polygon_to_mask(corridor_polygon)
    contact_mask = polygon_to_mask(contact_polygon)

    # Always keep the true slit/contact region inside the smoothed SAM crop.
    crop_source_mask = cv2.bitwise_or(
        corridor_mask,
        contact_mask,
    )
    crop_box = mask_bounding_crop(
        crop_source_mask,
        SLIT_CROP_PADDING_PX,
    )
    crop_box = quantize_crop_box(crop_box)

    return {
        "point_a": point_a,
        "point_b": point_b,
        "midpoint": midpoint,
        "tangent": tangent,
        "outward": outward,
        "corridor_polygon": corridor_polygon,
        "contact_polygon": contact_polygon,
        "corridor_mask": corridor_mask,
        "contact_mask": contact_mask,
        "crop_box": crop_box,
        "dynamic_corridor_enabled": bool(dynamic_ready),
        "corridor_pose_scale": float(pose_scale),
        "corridor_pull_away_px": float(pull_away_px),
        "corridor_lateral_motion_px": float(
            lateral_motion_px
        ),
        "corridor_total_motion_px": float(
            total_motion_px
        ),
        "corridor_outward_extra_px": float(
            outward_extra_px
        ),
        "corridor_side_extra_px": float(side_extra_px),
        "corridor_inward_extra_px": float(
            inward_extra_px
        ),
        "corridor_outward_distance_px": float(
            outward_distance_px
        ),
        "corridor_side_margin_px": float(side_margin_px),
        "corridor_inward_distance_px": float(
            inward_distance_px
        ),
    }


def polygon_to_mask(polygon):
    mask = np.zeros((ROI_HEIGHT, ROI_WIDTH), dtype=np.uint8)
    polygon_int = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon_int, 1)
    return mask


def mask_bounding_crop(mask, padding):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return (0, 0, ROI_WIDTH, ROI_HEIGHT)

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(ROI_WIDTH, int(xs.max()) + padding + 1)
    y2 = min(ROI_HEIGHT, int(ys.max()) + padding + 1)

    width = x2 - x1
    height = y2 - y1

    if width < SLIT_CROP_MIN_SIZE_PX:
        center = (x1 + x2) // 2
        half = SLIT_CROP_MIN_SIZE_PX // 2
        x1 = max(0, center - half)
        x2 = min(ROI_WIDTH, x1 + SLIT_CROP_MIN_SIZE_PX)
        x1 = max(0, x2 - SLIT_CROP_MIN_SIZE_PX)

    if height < SLIT_CROP_MIN_SIZE_PX:
        center = (y1 + y2) // 2
        half = SLIT_CROP_MIN_SIZE_PX // 2
        y1 = max(0, center - half)
        y2 = min(ROI_HEIGHT, y1 + SLIT_CROP_MIN_SIZE_PX)
        y1 = max(0, y2 - SLIT_CROP_MIN_SIZE_PX)

    return (x1, y1, x2, y2)


def quantize_crop_box(crop_box, step_px=DYNAMIC_CORRIDOR_CROP_QUANTIZE_PX):
    """Expand a crop outward to stable step-aligned boundaries."""
    x1, y1, x2, y2 = [int(value) for value in crop_box]
    step = max(1, int(step_px))

    x1 = max(0, (x1 // step) * step)
    y1 = max(0, (y1 // step) * step)

    x2 = min(
        ROI_WIDTH,
        ((x2 + step - 1) // step) * step,
    )
    y2 = min(
        ROI_HEIGHT,
        ((y2 + step - 1) // step) * step,
    )

    if x2 <= x1:
        x2 = min(ROI_WIDTH, x1 + step)

    if y2 <= y1:
        y2 = min(ROI_HEIGHT, y1 + step)

    return (x1, y1, x2, y2)


def make_baseline_change_mask(current_roi_rgb, baseline_roi_rgb):
    if current_roi_rgb is None or baseline_roi_rgb is None:
        return None

    current_gray = cv2.cvtColor(current_roi_rgb, cv2.COLOR_RGB2GRAY)
    baseline_gray = cv2.cvtColor(baseline_roi_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
    baseline_gray = cv2.GaussianBlur(baseline_gray, (5, 5), 0)
    difference = cv2.absdiff(current_gray, baseline_gray)
    changed = (difference >= BASELINE_DIFF_THRESHOLD).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BASELINE_DIFF_OPEN_KERNEL_PX, BASELINE_DIFF_OPEN_KERNEL_PX),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BASELINE_DIFF_CLOSE_KERNEL_PX, BASELINE_DIFF_CLOSE_KERNEL_PX),
    )

    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )
    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return changed.astype(np.uint8)


def roi_edge_touch_fraction(mask, border_px=4):
    border = np.zeros_like(mask, dtype=np.uint8)
    border[:border_px, :] = 1
    border[-border_px:, :] = 1
    border[:, :border_px] = 1
    border[:, -border_px:] = 1
    return mask_overlap_fraction(mask, border)


def crop_edge_touch_fraction(
    mask,
    crop_box,
    border_px=PRODUCT_CROP_EDGE_BORDER_PX,
):
    """Return how much of a full-ROI mask touches the active SAM crop edge."""
    if crop_box is None:
        return 0.0

    x1, y1, x2, y2 = [int(value) for value in crop_box]
    x1 = max(0, min(ROI_WIDTH, x1))
    x2 = max(0, min(ROI_WIDTH, x2))
    y1 = max(0, min(ROI_HEIGHT, y1))
    y2 = max(0, min(ROI_HEIGHT, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    thickness = max(1, int(border_px))
    border = np.zeros_like(mask, dtype=np.uint8)

    border[y1:min(y2, y1 + thickness), x1:x2] = 1
    border[max(y1, y2 - thickness):y2, x1:x2] = 1
    border[y1:y2, x1:min(x2, x1 + thickness)] = 1
    border[y1:y2, max(x1, x2 - thickness):x2] = 1

    return mask_overlap_fraction(mask, border)


def slit_span_fraction(
    mask,
    slit_geometry,
    margin_px=EMERGING_SLIT_SPAN_MARGIN_PX,
):
    """Return the fraction positioned across the physical slit segment."""
    if slit_geometry is None:
        return 0.0

    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return 0.0

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]

    tangent = slit_geometry.get("tangent")

    if tangent is None:
        tangent = normalize_vector(
            slit_geometry["point_b"] - slit_geometry["point_a"]
        )

    if tangent is None:
        return 0.0

    tangent_position = relative @ tangent

    slit_length = float(
        np.linalg.norm(
            slit_geometry["point_b"] - slit_geometry["point_a"]
        )
    )
    half_span = slit_length * 0.5 + float(margin_px)
    inside = np.abs(tangent_position) <= half_span

    return float(np.count_nonzero(inside) / max(len(points), 1))


def copy_candidate(candidate):
    """Copy and latch the final product segmentation."""
    if candidate is None:
        return None

    copied = dict(candidate)

    if candidate.get("mask") is not None:
        copied["mask"] = candidate["mask"].copy()

    if candidate.get("center") is not None:
        copied["center"] = candidate["center"].copy()

    if candidate.get("mean_hsv") is not None:
        copied["mean_hsv"] = tuple(candidate["mean_hsv"])

    copied["bbox"] = tuple(candidate["bbox"])
    copied["mode"] = "latched_released_product"
    return copied


def outward_fraction_for_mask(mask, slit_geometry, margin_px=5.0):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return 0.0, 0

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]
    signed = relative @ slit_geometry["outward"]
    outward_pixels = int(np.count_nonzero(signed >= margin_px))
    return outward_pixels / max(len(points), 1), outward_pixels


def signed_slit_metrics(mask, slit_geometry):
    ys, xs = np.where(mask == 1)

    if xs.size == 0:
        return {
            "outward_fraction": 0.0,
            "inward_fraction": 0.0,
            "line_band_fraction": 0.0,
            "outward_pixels": 0,
            "min_line_distance_px": float("inf"),
            "min_signed_distance_px": float("-inf"),
            "max_signed_distance_px": float("-inf"),
            "trailing_edge_clearance_px": 0.0,
        }

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    relative = points - slit_geometry["midpoint"]
    signed = relative @ slit_geometry["outward"]

    outward = signed >= 5.0
    inward = signed <= -5.0
    line_band = np.abs(signed) <= EMERGING_LINE_BAND_PX

    min_signed_distance = float(np.min(signed))
    max_signed_distance = float(np.max(signed))

    return {
        "outward_fraction": float(np.mean(outward)),
        "inward_fraction": float(np.mean(inward)),
        "line_band_fraction": float(np.mean(line_band)),
        "outward_pixels": int(np.count_nonzero(outward)),
        "min_line_distance_px": float(np.min(np.abs(signed))),
        # Positive only when every pixel is completely on the outward side.
        # This is the clearance of the product's trailing edge from the slit.
        "min_signed_distance_px": min_signed_distance,
        "max_signed_distance_px": max_signed_distance,
        "trailing_edge_clearance_px": max(0.0, min_signed_distance),
    }


def estimate_baseline_base_hsv(roi_rgb, poly_mask):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    expanded_poly = cv2.dilate(
        poly_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        iterations=1,
    )

    sample_mask = expanded_poly == 0
    sample_mask[:12, :] = False
    sample_mask[-12:, :] = False
    sample_mask[:, :12] = False
    sample_mask[:, -12:] = False

    # The fixture/base is dark. Restricting the sample prevents bright rails,
    # labels, and cables from dominating the base appearance model.
    sample_mask &= hsv[:, :, 2] <= 145
    values = hsv[sample_mask]

    if values.shape[0] < 200:
        values = hsv[expanded_poly == 0]

    if values.shape[0] < 50:
        return None

    return (
        float(np.median(values[:, 0])),
        float(np.median(values[:, 1])),
        float(np.median(values[:, 2])),
    )


def build_baseline_static_signatures(masks, poly_mask, roi_rgb):
    signatures = []
    roi_area = ROI_WIDTH * ROI_HEIGHT

    for sam_mask in masks:
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < BASELINE_STATIC_MIN_MASK_AREA_PX:
            continue

        if area / max(roi_area, 1) > BASELINE_STATIC_MAX_MASK_AREA_RATIO:
            continue

        if mask_iou(mask, poly_mask) > 0.20:
            continue

        center = mask_center(mask)

        if center is None:
            continue

        signatures.append(
            {
                "mask": mask,
                "area": area,
                "center": center,
                "mean_hsv": masked_mean_hsv(roi_rgb, mask),
            }
        )

    return signatures


def baseline_static_match(candidate_mask, candidate_center, static_signatures):
    candidate_area = max(int(candidate_mask.sum()), 1)
    best_iou = 0.0

    for signature in static_signatures:
        signature_mask = signature["mask"]
        overlap_iou = mask_iou(candidate_mask, signature_mask)
        best_iou = max(best_iou, overlap_iou)

        intersection = int(
            np.count_nonzero(
                (candidate_mask == 1)
                & (signature_mask == 1)
            )
        )
        candidate_inside_static = (
            intersection / max(candidate_area, 1)
        )

        if (
            overlap_iou >= BASELINE_STATIC_REJECT_IOU
            or candidate_inside_static
            >= BASELINE_STATIC_REJECT_CANDIDATE_CONTAINMENT
        ):
            return True, best_iou

        if candidate_center is None:
            continue

        center_distance = float(
            np.linalg.norm(candidate_center - signature["center"])
        )
        area_ratio = candidate_area / max(signature["area"], 1)

        if (
            center_distance <= BASELINE_STATIC_REJECT_CENTER_DISTANCE_PX
            and BASELINE_STATIC_REJECT_AREA_RATIO_LOW
            <= area_ratio
            <= BASELINE_STATIC_REJECT_AREA_RATIO_HIGH
            and overlap_iou >= 0.08
        ):
            return True, best_iou

    return False, best_iou


def baseline_unchanged_fraction(
    mask,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_mask,
):
    if baseline_roi_rgb is None or baseline_poly_mask is None:
        return 0.0, 0.0

    visible_mask = (mask == 1) & (baseline_poly_mask == 0)
    candidate_area = max(int(mask.sum()), 1)
    visible_count = int(np.count_nonzero(visible_mask))
    visible_fraction = visible_count / candidate_area

    if visible_count == 0:
        return 0.0, visible_fraction

    difference = cv2.absdiff(current_roi_rgb, baseline_roi_rgb)
    difference = np.mean(difference.astype(np.float32), axis=2)
    unchanged = difference[visible_mask] <= BASELINE_UNCHANGED_PIXEL_THRESHOLD

    return float(np.mean(unchanged)), float(visible_fraction)


def candidate_height_above_local_base(depth_roi, mask, poly_mask):
    if depth_roi is None:
        return None, 0, 0, None, None

    valid_depth = (
        (depth_roi > MIN_VALID_DEPTH_MM)
        & (depth_roi < MAX_VALID_DEPTH_MM)
    )

    candidate_values = depth_roi[(mask == 1) & valid_depth].astype(np.float32)

    if candidate_values.size < EMERGING_MIN_DEPTH_SAMPLE_COUNT:
        return None, int(candidate_values.size), 0, None, None

    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            EMERGING_DEPTH_RING_INNER_PX * 2 + 1,
            EMERGING_DEPTH_RING_INNER_PX * 2 + 1,
        ),
    )
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            EMERGING_DEPTH_RING_OUTER_PX * 2 + 1,
            EMERGING_DEPTH_RING_OUTER_PX * 2 + 1,
        ),
    )

    inner = cv2.dilate(mask.astype(np.uint8), inner_kernel, iterations=1)
    outer = cv2.dilate(mask.astype(np.uint8), outer_kernel, iterations=1)
    ring = (outer == 1) & (inner == 0) & (poly_mask == 0) & valid_depth
    base_values = depth_roi[ring].astype(np.float32)

    if base_values.size < EMERGING_MIN_DEPTH_SAMPLE_COUNT:
        return (
            None,
            int(candidate_values.size),
            int(base_values.size),
            float(np.median(candidate_values)),
            None,
        )

    candidate_depth = float(np.median(candidate_values))
    base_depth = float(np.median(base_values))
    height_mm = float(base_depth - candidate_depth)

    return (
        height_mm,
        int(candidate_values.size),
        int(base_values.size),
        candidate_depth,
        base_depth,
    )


def find_fast_exit_arrival(
    roi_rgb,
    baseline_roi_rgb,
    poly_mask,
    slit_geometry,
):
    """Frame-rate detector for a product that has already left the bag.

    Returns the largest region of new, non-bag content on the outward side of
    the slit, or None. This runs on every camera frame, so unlike the SAM path
    it still sees a product that was dumped out in a few tenths of a second.
    Nothing here decides that the blob is the product; it only decides where
    SAM should look.
    """
    if (
        roi_rgb is None
        or baseline_roi_rgb is None
        or poly_mask is None
        or slit_geometry is None
    ):
        return None

    changed = make_baseline_change_mask(roi_rgb, baseline_roi_rgb)

    if changed is None:
        return None

    # Whatever optical flow still calls "bag" cannot be the product. Dilating
    # it keeps the bag's own moving boundary out of the difference.
    bag = cv2.dilate(
        poly_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (FAST_EXIT_BAG_DILATE_PX, FAST_EXIT_BAG_DILATE_PX),
        ),
        iterations=1,
    )

    changed[bag == 1] = 0

    blob = largest_component(changed)

    if blob is None:
        return None

    area = int(blob.sum())

    if area < FAST_EXIT_MIN_AREA_PX:
        return None

    metrics = signed_slit_metrics(blob, slit_geometry)

    if metrics["outward_fraction"] < FAST_EXIT_MIN_OUTWARD_FRACTION:
        return None

    center = mask_center(blob)

    if center is None:
        return None

    return {
        "mask": blob,
        "area": area,
        "center": center,
        "bbox": cv2.boundingRect(blob),
        "outward_fraction": float(metrics["outward_fraction"]),
    }


def make_fast_exit_crop(arrival):
    """Build a SAM crop around the settled arrival blob."""
    x, y, width, height = arrival["bbox"]
    padding = FAST_EXIT_CROP_PADDING_PX

    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(ROI_WIDTH, int(x + width) + padding)
    y2 = min(ROI_HEIGHT, int(y + height) + padding)

    if x2 - x1 < FAST_EXIT_CROP_MIN_SIZE_PX:
        center_x = (x1 + x2) // 2
        half = FAST_EXIT_CROP_MIN_SIZE_PX // 2
        x1 = max(0, center_x - half)
        x2 = min(ROI_WIDTH, x1 + FAST_EXIT_CROP_MIN_SIZE_PX)
        x1 = max(0, x2 - FAST_EXIT_CROP_MIN_SIZE_PX)

    if y2 - y1 < FAST_EXIT_CROP_MIN_SIZE_PX:
        center_y = (y1 + y2) // 2
        half = FAST_EXIT_CROP_MIN_SIZE_PX // 2
        y1 = max(0, center_y - half)
        y2 = min(ROI_HEIGHT, y1 + FAST_EXIT_CROP_MIN_SIZE_PX)
        y1 = max(0, y2 - FAST_EXIT_CROP_MIN_SIZE_PX)

    return (x1, y1, x2, y2)


def find_emerging_product_candidates(
    masks,
    poly_mask,
    slit_geometry,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_hsv,
    baseline_base_hsv,
    baseline_poly_mask,
    baseline_static_signatures,
    depth_roi,
    fast_exit=False,
    arrival_mask=None,
):
    """Score SAM masks as emerging-product candidates.

    With fast_exit set, the gates that require the mask to still be near the
    slit are skipped: a dumped product has already landed well past it. The
    arrival blob takes over that job -- the mask must line up with the blob
    that triggered the scan -- and every appearance, baseline, static-object
    and depth gate below still applies unchanged.
    """
    if poly_mask is None or slit_geometry is None:
        return []

    poly_area = max(int(poly_mask.sum()), 1)
    baseline_change_mask = make_baseline_change_mask(
        current_roi_rgb,
        baseline_roi_rgb,
    )
    corridor_mask = slit_geometry["corridor_mask"]
    contact_mask = slit_geometry["contact_mask"]
    candidates = []

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < EMERGING_MIN_AREA_PX:
            continue

        area_ratio_of_poly = area / poly_area

        if area_ratio_of_poly < EMERGING_MIN_AREA_RATIO_OF_POLY:
            continue

        if area_ratio_of_poly > EMERGING_MAX_AREA_RATIO_OF_POLY:
            continue

        # On the fast path this stands in for the slit-relative gates below:
        # the mask has to be the thing the frame-rate detector saw arrive, not
        # some other mask SAM happened to find in the same crop.
        arrival_overlap = 1.0

        if fast_exit and arrival_mask is not None:
            arrival_overlap = mask_overlap_fraction(mask, arrival_mask)

            if arrival_overlap < FAST_EXIT_MIN_ARRIVAL_OVERLAP:
                continue

        x, y, width, height = cv2.boundingRect(mask)

        if (
            width < EMERGING_MIN_BBOX_WIDTH_PX
            or height < EMERGING_MIN_BBOX_HEIGHT_PX
        ):
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

        if rectangularity < EMERGING_MIN_RECTANGULARITY:
            continue

        crop_edge_fraction = crop_edge_touch_fraction(
            mask,
            slit_geometry.get("crop_box"),
        )

        if (
            crop_edge_fraction
            > EMERGING_MAX_CROP_EDGE_TOUCH_FRACTION
        ):
            continue

        physical_slit_span_fraction = slit_span_fraction(
            mask,
            slit_geometry,
        )

        if (
            not fast_exit
            and physical_slit_span_fraction
            < EMERGING_MIN_SLIT_SPAN_FRACTION
        ):
            continue

        center = mask_center(mask)

        if center is None:
            continue

        static_match, static_match_iou = baseline_static_match(
            mask,
            center,
            baseline_static_signatures,
        )

        if static_match:
            continue

        unchanged_fraction, baseline_visible_fraction = (
            baseline_unchanged_fraction(
                mask,
                current_roi_rgb,
                baseline_roi_rgb,
                baseline_poly_mask,
            )
        )

        if (
            baseline_visible_fraction >= BASELINE_VISIBLE_MIN_FRACTION
            and unchanged_fraction
            >= BASELINE_UNCHANGED_REJECT_FRACTION
        ):
            continue

        poly_overlap = mask_overlap_fraction(mask, poly_mask)

        if poly_overlap > EMERGING_MAX_POLY_OVERLAP:
            continue

        outside_poly_fraction = 1.0 - poly_overlap

        if outside_poly_fraction < EMERGING_MIN_OUTSIDE_POLY_FRACTION:
            continue

        corridor_overlap = mask_overlap_fraction(mask, corridor_mask)

        if not fast_exit and corridor_overlap < EMERGING_MIN_CORRIDOR_OVERLAP:
            continue

        contact_overlap = mask_overlap_fraction(mask, contact_mask)
        slit_metrics = signed_slit_metrics(mask, slit_geometry)
        outward_fraction = slit_metrics["outward_fraction"]
        inward_fraction = slit_metrics["inward_fraction"]
        line_band_fraction = slit_metrics["line_band_fraction"]
        outward_pixels = slit_metrics["outward_pixels"]
        min_line_distance_px = slit_metrics["min_line_distance_px"]

        if outward_fraction < EMERGING_MIN_OUTWARD_FRACTION:
            continue

        if outward_pixels < EMERGING_MIN_OUTWARD_PIXELS:
            continue

        crosses_slit = (
            inward_fraction >= EMERGING_MIN_INWARD_FRACTION_FOR_CROSSING
            and line_band_fraction >= EMERGING_MIN_LINE_BAND_FRACTION
        )

        large_outside_near_slit = (
            area_ratio_of_poly
            >= EMERGING_OUTSIDE_ONLY_MIN_AREA_RATIO_OF_POLY
            and min_line_distance_px
            <= EMERGING_OUTSIDE_ONLY_MAX_LINE_DISTANCE_PX
            and line_band_fraction
            >= EMERGING_MIN_LINE_BAND_FRACTION * 0.55
        )

        if not fast_exit and not crosses_slit and not large_outside_near_slit:
            continue

        if (
            not fast_exit
            and contact_overlap < EMERGING_MIN_CONTACT_OVERLAP
            and poly_overlap < 0.01
            and not large_outside_near_slit
        ):
            continue

        if baseline_change_mask is not None:
            changed_fraction = mask_overlap_fraction(
                mask,
                baseline_change_mask,
            )
        else:
            changed_fraction = 1.0

        if changed_fraction < EMERGING_MIN_BASELINE_CHANGE_FRACTION:
            continue

        edge_touch = roi_edge_touch_fraction(mask)

        if edge_touch > EMERGING_MAX_ROI_EDGE_TOUCH_FRACTION:
            continue

        candidate_hsv = masked_mean_hsv(current_roi_rgb, mask)
        bag_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )
        base_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_base_hsv,
        )

        if (
            bag_color_similarity >= EMERGING_BAG_COLOR_REJECT_SIMILARITY
            and poly_overlap >= 0.06
        ):
            continue

        (
            height_above_base_mm,
            candidate_depth_count,
            base_depth_count,
            candidate_depth_mm,
            local_base_depth_mm,
        ) = candidate_height_above_local_base(
            depth_roi,
            mask,
            poly_mask,
        )

        depth_is_strong = (
            height_above_base_mm is not None
            and height_above_base_mm
            >= EMERGING_MIN_HEIGHT_ABOVE_BASE_MM
        )
        visual_bypass = (
            area_ratio_of_poly
            >= EMERGING_STRONG_VISUAL_BYPASS_AREA_RATIO
            and changed_fraction
            >= EMERGING_STRONG_VISUAL_BYPASS_CHANGE_FRACTION
            and base_color_similarity
            <= BASELINE_BASE_COLOR_BYPASS_MAX_SIMILARITY
            and (
                crosses_slit
                or min_line_distance_px <= 20.0
                # A landed product is nowhere near the slit line, so proximity
                # cannot be part of the bypass on the fast path. Fast-moving
                # objects are also exactly the case where stereo goes sparse
                # and depth_is_strong fails, which is what this bypasses.
                or fast_exit
            )
        )

        strong_real_slit_crossing = (
            crosses_slit
            and poly_overlap
            >= BASE_SIDE_STRONG_CROSSING_MIN_POLY_OVERLAP
            and changed_fraction
            >= BASE_SIDE_STRONG_CROSSING_MIN_CHANGE
            and depth_is_strong
        )

        base_side_like = (
            base_color_similarity
            >= BASE_SIDE_COLOR_EDGE_REJECT_SIMILARITY
            and (
                crop_edge_fraction >= 0.02
                or (
                    aspect_ratio >= BASE_SIDE_LONG_ASPECT_RATIO
                    and rectangularity
                    <= BASE_SIDE_LOW_RECTANGULARITY
                )
            )
        )

        # A landed product is detached from the bag, so the crossing override
        # for base-coloured masks can never fire on the fast path. Substitute
        # the evidence that path does have: a large change against the baseline
        # on something standing above the base surface.
        strong_fast_arrival = (
            fast_exit
            and changed_fraction >= BASE_SIDE_STRONG_CROSSING_MIN_CHANGE
            and depth_is_strong
        )

        if (
            base_side_like
            and not strong_real_slit_crossing
            and not strong_fast_arrival
        ):
            continue

        if (
            not depth_is_strong
            and base_color_similarity
            >= BASELINE_BASE_COLOR_REJECT_SIMILARITY
        ):
            continue

        if not depth_is_strong and not visual_bypass:
            continue

        duplicate = False

        for accepted in candidates:
            if mask_iou(mask, accepted["mask"]) >= EMERGING_DUPLICATE_IOU:
                duplicate = True
                break

        if duplicate:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        depth_score = 0.0

        if height_above_base_mm is not None:
            depth_score = min(max(height_above_base_mm, 0.0) / 35.0, 1.0)

        score = (
            2.6 * outward_fraction
            + 2.2 * corridor_overlap
            + 1.8 * contact_overlap
            + 1.5 * outside_poly_fraction
            + 1.4 * changed_fraction
            + 1.2 * line_band_fraction
            + 1.2 * depth_score
            + 0.7 * min(area_ratio_of_poly / 0.08, 1.0)
            + 0.5 * sam_stability
            + 0.3 * sam_iou
            - 0.8 * bag_color_similarity
            - 1.0 * base_color_similarity
            - 0.9 * unchanged_fraction
            - 0.7 * static_match_iou
        )

        if fast_exit:
            mode = "fast_exit_arrival"
        elif crosses_slit or poly_overlap > 0.03:
            mode = "sliding_through_slit"
        else:
            mode = "outside_slit"

        candidates.append(
            {
                "index": index,
                "mask": mask,
                "bbox": (x, y, width, height),
                "center": center,
                "area": area,
                "score": float(score),
                "mode": mode,
                "area_ratio_of_poly": float(area_ratio_of_poly),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect_ratio),
                "crop_edge_touch_fraction": float(
                    crop_edge_fraction
                ),
                "slit_span_fraction": float(
                    physical_slit_span_fraction
                ),
                "corridor_overlap": float(corridor_overlap),
                "contact_overlap": float(contact_overlap),
                "arrival_overlap": float(arrival_overlap),
                "poly_overlap": float(poly_overlap),
                "outside_poly_fraction": float(outside_poly_fraction),
                "outward_fraction": float(outward_fraction),
                "inward_fraction": float(inward_fraction),
                "line_band_fraction": float(line_band_fraction),
                "min_line_distance_px": float(min_line_distance_px),
                "outward_pixels": outward_pixels,
                "baseline_change_fraction": float(changed_fraction),
                "baseline_visible_fraction": float(baseline_visible_fraction),
                "baseline_unchanged_fraction": float(unchanged_fraction),
                "static_match_iou": float(static_match_iou),
                "bag_color_similarity": float(bag_color_similarity),
                "base_color_similarity": float(base_color_similarity),
                "mean_hsv": candidate_hsv,
                "height_above_base_mm": (
                    None
                    if height_above_base_mm is None
                    else float(height_above_base_mm)
                ),
                "candidate_depth_count": candidate_depth_count,
                "base_depth_count": base_depth_count,
                "candidate_depth_mm": candidate_depth_mm,
                "local_base_depth_mm": local_base_depth_mm,
                "depth_bypassed": bool(not depth_is_strong),
                "sam_iou": sam_iou,
                "sam_stability": sam_stability,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def candidate_matches_previous(candidate, previous_candidate):
    if candidate is None or previous_candidate is None:
        return False

    area_a = max(float(candidate.get("area", 0)), 1.0)
    area_b = max(float(previous_candidate.get("area", 0)), 1.0)
    area_ratio = area_a / area_b

    if not (
        PRODUCT_MATCH_MIN_AREA_RATIO
        <= area_ratio
        <= PRODUCT_MATCH_MAX_AREA_RATIO
    ):
        return False

    overlap_iou = mask_iou(
        candidate["mask"],
        previous_candidate["mask"],
    )

    if overlap_iou >= PRODUCT_MATCH_MIN_IOU:
        return True

    center_a = candidate.get("center")
    center_b = previous_candidate.get("center")

    if center_a is None or center_b is None:
        return False

    center_distance = float(np.linalg.norm(center_a - center_b))
    return center_distance <= PRODUCT_MATCH_MAX_CENTER_DISTANCE_PX



def make_product_verification_crop(candidate, slit_geometry):
    """Build a moving local crop around the last visible product mask."""
    if candidate is None:
        if slit_geometry is not None:
            return slit_geometry["crop_box"]
        return (0, 0, ROI_WIDTH, ROI_HEIGHT)

    x, y, width, height = candidate["bbox"]
    padding = FULL_RELEASE_VERIFY_CROP_PADDING_PX

    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(ROI_WIDTH, int(x + width) + padding)
    y2 = min(ROI_HEIGHT, int(y + height) + padding)

    # Keep part of the current slit in the crop so a still-partly-attached
    # product cannot disappear merely because the bag moved slightly.
    if slit_geometry is not None:
        slit_x1, slit_y1, slit_x2, slit_y2 = slit_geometry["crop_box"]
        midpoint = slit_geometry["midpoint"]
        near_x1 = max(0, int(midpoint[0]) - 90)
        near_y1 = max(0, int(midpoint[1]) - 90)
        near_x2 = min(ROI_WIDTH, int(midpoint[0]) + 91)
        near_y2 = min(ROI_HEIGHT, int(midpoint[1]) + 91)
        x1 = min(x1, max(slit_x1, near_x1))
        y1 = min(y1, max(slit_y1, near_y1))
        x2 = max(x2, min(slit_x2, near_x2))
        y2 = max(y2, min(slit_y2, near_y2))

    width_now = x2 - x1
    height_now = y2 - y1

    if width_now < FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX:
        center_x = (x1 + x2) // 2
        half = FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX // 2
        x1 = max(0, center_x - half)
        x2 = min(ROI_WIDTH, x1 + FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX)
        x1 = max(0, x2 - FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX)

    if height_now < FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX:
        center_y = (y1 + y2) // 2
        half = FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX // 2
        y1 = max(0, center_y - half)
        y2 = min(ROI_HEIGHT, y1 + FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX)
        y1 = max(0, y2 - FULL_RELEASE_VERIFY_CROP_MIN_SIZE_PX)

    return (x1, y1, x2, y2)


def candidate_reference_match(candidate_mask, candidate_center, reference_candidate):
    if reference_candidate is None or candidate_center is None:
        return False, 0.0, float("inf"), 0.0

    candidate_area = max(float(candidate_mask.sum()), 1.0)
    reference_area = max(float(reference_candidate.get("area", 0)), 1.0)
    area_ratio = candidate_area / reference_area

    if not (
        FULL_RELEASE_REFERENCE_MIN_AREA_RATIO
        <= area_ratio
        <= FULL_RELEASE_REFERENCE_MAX_AREA_RATIO
    ):
        return False, 0.0, float("inf"), area_ratio

    reference_mask = reference_candidate.get("mask")
    overlap_iou = 0.0

    if reference_mask is not None:
        overlap_iou = mask_iou(candidate_mask, reference_mask)

    reference_center = reference_candidate.get("center")
    center_distance = float("inf")

    if reference_center is not None:
        center_distance = float(
            np.linalg.norm(candidate_center - reference_center)
        )

    matched = (
        overlap_iou >= FULL_RELEASE_REFERENCE_MIN_IOU
        or center_distance
        <= FULL_RELEASE_REFERENCE_MAX_CENTER_DISTANCE_PX
    )

    return matched, overlap_iou, center_distance, area_ratio


def find_released_product_candidates(
    masks,
    reference_candidate,
    poly_mask,
    slit_geometry,
    current_roi_rgb,
    baseline_roi_rgb,
    baseline_poly_hsv,
    baseline_base_hsv,
    baseline_poly_mask,
    baseline_static_signatures,
    depth_roi,
):
    """Track the already-discovered product after it crosses the slit."""
    if reference_candidate is None or poly_mask is None or slit_geometry is None:
        return []

    poly_area = max(int(poly_mask.sum()), 1)
    baseline_change_mask = make_baseline_change_mask(
        current_roi_rgb,
        baseline_roi_rgb,
    )
    candidates = []

    for index, sam_mask in enumerate(masks):
        mask = clean_mask(sam_mask["segmentation"])
        area = int(mask.sum())

        if area < EMERGING_MIN_AREA_PX:
            continue

        area_ratio_of_poly = area / poly_area

        if area_ratio_of_poly < EMERGING_MIN_AREA_RATIO_OF_POLY * 0.70:
            continue

        if area_ratio_of_poly > EMERGING_MAX_AREA_RATIO_OF_POLY:
            continue

        x, y, width, height = cv2.boundingRect(mask)

        if (
            width < EMERGING_MIN_BBOX_WIDTH_PX
            or height < EMERGING_MIN_BBOX_HEIGHT_PX
        ):
            continue

        rectangularity = area / max(width * height, 1)
        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1),
        )

        if rectangularity < EMERGING_MIN_RECTANGULARITY:
            continue

        crop_edge_fraction = crop_edge_touch_fraction(
            mask,
            slit_geometry.get("crop_box"),
        )

        if (
            crop_edge_fraction
            > RELEASED_MAX_CROP_EDGE_TOUCH_FRACTION
        ):
            continue

        center = mask_center(mask)

        if center is None:
            continue

        matched, match_iou, center_distance, reference_area_ratio = (
            candidate_reference_match(
                mask,
                center,
                reference_candidate,
            )
        )

        if not matched:
            continue

        static_match, static_match_iou = baseline_static_match(
            mask,
            center,
            baseline_static_signatures,
        )

        if static_match:
            continue

        unchanged_fraction, baseline_visible_fraction = (
            baseline_unchanged_fraction(
                mask,
                current_roi_rgb,
                baseline_roi_rgb,
                baseline_poly_mask,
            )
        )

        if (
            baseline_visible_fraction >= BASELINE_VISIBLE_MIN_FRACTION
            and unchanged_fraction
            >= BASELINE_UNCHANGED_REJECT_FRACTION
        ):
            continue

        if baseline_change_mask is not None:
            changed_fraction = mask_overlap_fraction(
                mask,
                baseline_change_mask,
            )
        else:
            changed_fraction = 1.0

        if changed_fraction < EMERGING_MIN_BASELINE_CHANGE_FRACTION * 0.75:
            continue

        edge_touch = roi_edge_touch_fraction(mask)

        if edge_touch > EMERGING_MAX_ROI_EDGE_TOUCH_FRACTION:
            continue

        candidate_hsv = masked_mean_hsv(current_roi_rgb, mask)
        bag_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_poly_hsv,
        )
        base_color_similarity = hsv_similarity(
            candidate_hsv,
            baseline_base_hsv,
        )
        reference_color_similarity = hsv_similarity(
            candidate_hsv,
            reference_candidate.get("mean_hsv"),
        )

        poly_overlap = mask_overlap_fraction(mask, poly_mask)

        if (
            bag_color_similarity >= EMERGING_BAG_COLOR_REJECT_SIMILARITY
            and poly_overlap >= 0.06
        ):
            continue

        slit_metrics = signed_slit_metrics(mask, slit_geometry)
        contact_overlap = mask_overlap_fraction(
            mask,
            slit_geometry["contact_mask"],
        )
        core_metrics = full_release_core_metrics(
            mask,
            poly_mask,
            slit_geometry,
        )

        (
            height_above_base_mm,
            candidate_depth_count,
            base_depth_count,
            candidate_depth_mm,
            local_base_depth_mm,
        ) = candidate_height_above_local_base(
            depth_roi,
            mask,
            poly_mask,
        )

        depth_is_strong = (
            height_above_base_mm is not None
            and height_above_base_mm
            >= FULL_RELEASE_MIN_HEIGHT_ABOVE_BASE_MM
        )
        visual_table_support = (
            changed_fraction >= FULL_RELEASE_VISUAL_TABLE_MIN_CHANGE
            and base_color_similarity
            <= FULL_RELEASE_VISUAL_TABLE_MAX_BASE_COLOR_SIMILARITY
        )

        base_side_like = (
            base_color_similarity
            >= BASE_SIDE_COLOR_EDGE_REJECT_SIMILARITY
            and (
                crop_edge_fraction >= 0.025
                or (
                    aspect_ratio >= BASE_SIDE_LONG_ASPECT_RATIO
                    and rectangularity
                    <= BASE_SIDE_LOW_RECTANGULARITY
                )
            )
        )

        strong_reference_support = (
            match_iou >= 0.12
            and reference_color_similarity >= 0.55
            and depth_is_strong
        )

        if base_side_like and not strong_reference_support:
            continue

        if (
            not depth_is_strong
            and base_color_similarity
            >= BASELINE_BASE_COLOR_REJECT_SIMILARITY
        ):
            continue

        if not depth_is_strong and not visual_table_support:
            continue

        duplicate = False

        for accepted in candidates:
            if mask_iou(mask, accepted["mask"]) >= EMERGING_DUPLICATE_IOU:
                duplicate = True
                break

        if duplicate:
            continue

        sam_iou = float(sam_mask.get("predicted_iou", 0.0))
        sam_stability = float(sam_mask.get("stability_score", 0.0))
        center_score = 1.0 - min(
            center_distance / FULL_RELEASE_REFERENCE_MAX_CENTER_DISTANCE_PX,
            1.0,
        )
        area_score = 1.0 - min(abs(np.log(max(reference_area_ratio, 1e-6))) / 1.4, 1.0)
        depth_score = 0.0

        if height_above_base_mm is not None:
            depth_score = min(max(height_above_base_mm, 0.0) / 35.0, 1.0)

        score = (
            2.8 * match_iou
            + 2.2 * center_score
            + 1.4 * area_score
            + 1.4 * changed_fraction
            + 1.2 * depth_score
            + 0.8 * reference_color_similarity
            + 0.5 * sam_stability
            + 0.3 * sam_iou
            + 0.8 * (1.0 - poly_overlap)
            - 0.9 * base_color_similarity
            - 0.7 * bag_color_similarity
            - 0.7 * unchanged_fraction
            - 0.5 * static_match_iou
        )

        candidates.append(
            {
                "index": index,
                "mask": mask,
                "bbox": (x, y, width, height),
                "center": center,
                "area": area,
                "score": float(score),
                "mode": "released_product",
                "area_ratio_of_poly": float(area_ratio_of_poly),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect_ratio),
                "crop_edge_touch_fraction": float(
                    crop_edge_fraction
                ),
                "corridor_overlap": float(
                    mask_overlap_fraction(
                        mask,
                        slit_geometry["corridor_mask"],
                    )
                ),
                "contact_overlap": float(contact_overlap),
                "poly_overlap": float(poly_overlap),
                "outside_poly_fraction": float(1.0 - poly_overlap),
                "core_poly_overlap": core_metrics["core_poly_overlap"],
                "core_outward_fraction": core_metrics[
                    "core_outward_fraction"
                ],
                "core_inward_fraction": core_metrics[
                    "core_inward_fraction"
                ],
                "core_line_band_fraction": core_metrics[
                    "core_line_band_fraction"
                ],
                "core_min_line_distance_px": core_metrics[
                    "core_min_line_distance_px"
                ],
                "core_min_signed_distance_px": core_metrics[
                    "core_min_signed_distance_px"
                ],
                "core_trailing_edge_clearance_px": core_metrics[
                    "core_trailing_edge_clearance_px"
                ],
                "core_poly_gap_px": core_metrics["core_poly_gap_px"],
                "outward_fraction": float(slit_metrics["outward_fraction"]),
                "inward_fraction": float(slit_metrics["inward_fraction"]),
                "line_band_fraction": float(slit_metrics["line_band_fraction"]),
                "min_line_distance_px": float(slit_metrics["min_line_distance_px"]),
                "min_signed_distance_px": float(
                    slit_metrics["min_signed_distance_px"]
                ),
                "trailing_edge_clearance_px": float(
                    slit_metrics["trailing_edge_clearance_px"]
                ),
                "outward_pixels": int(slit_metrics["outward_pixels"]),
                "baseline_change_fraction": float(changed_fraction),
                "baseline_visible_fraction": float(baseline_visible_fraction),
                "baseline_unchanged_fraction": float(unchanged_fraction),
                "static_match_iou": float(static_match_iou),
                "bag_color_similarity": float(bag_color_similarity),
                "base_color_similarity": float(base_color_similarity),
                "reference_color_similarity": float(reference_color_similarity),
                "reference_match_iou": float(match_iou),
                "reference_center_distance_px": float(center_distance),
                "reference_area_ratio": float(reference_area_ratio),
                "mean_hsv": candidate_hsv,
                "height_above_base_mm": (
                    None
                    if height_above_base_mm is None
                    else float(height_above_base_mm)
                ),
                "candidate_depth_count": candidate_depth_count,
                "base_depth_count": base_depth_count,
                "candidate_depth_mm": candidate_depth_mm,
                "local_base_depth_mm": local_base_depth_mm,
                "depth_bypassed": bool(not depth_is_strong),
                "sam_iou": sam_iou,
                "sam_stability": sam_stability,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def candidate_center_step(candidate, previous_candidate):
    if candidate is None or previous_candidate is None:
        return None

    center_a = candidate.get("center")
    center_b = previous_candidate.get("center")

    if center_a is None or center_b is None:
        return None

    return float(np.linalg.norm(center_a - center_b))


def evaluate_full_release(candidate, previous_candidate):
    if candidate is None:
        return False, False, False, None, None

    strict_slit_clearance = (
        candidate.get("trailing_edge_clearance_px", 0.0)
        >= FULL_RELEASE_MIN_RAW_TRAILING_CLEARANCE_PX
        and candidate.get("core_poly_gap_px", 0.0)
        >= FULL_RELEASE_MIN_CORE_POLY_GAP_PX
    )

    strict_separation = (
        candidate["poly_overlap"] <= FULL_RELEASE_MAX_POLY_OVERLAP
        and candidate["inward_fraction"]
        <= FULL_RELEASE_MAX_INWARD_FRACTION
        and candidate["outward_fraction"]
        >= FULL_RELEASE_MIN_OUTWARD_FRACTION
        and strict_slit_clearance
        and (
            candidate["contact_overlap"]
            <= FULL_RELEASE_MAX_CONTACT_OVERLAP
            or candidate["min_line_distance_px"]
            >= FULL_RELEASE_MIN_LINE_DISTANCE_PX
        )
    )

    # A thin overlap between the colored outlines is not proof that the
    # physical product remains in the bag. The tolerant branch requires the
    # eroded product core to be clearly outside the eroded polymailer core.
    boundary_tolerant_separation = (
        candidate["poly_overlap"]
        <= FULL_RELEASE_MAX_RAW_POLY_OVERLAP_WITH_CLEAR_CORE
        and candidate.get("core_poly_overlap", 1.0)
        <= FULL_RELEASE_MAX_CORE_POLY_OVERLAP
        and candidate.get("core_inward_fraction", 1.0)
        <= FULL_RELEASE_MAX_CORE_INWARD_FRACTION
        and candidate.get("core_outward_fraction", 0.0)
        >= FULL_RELEASE_MIN_CORE_OUTWARD_FRACTION
        and candidate.get("core_trailing_edge_clearance_px", 0.0)
        >= FULL_RELEASE_MIN_CORE_TRAILING_CLEARANCE_PX
        and candidate.get("core_poly_gap_px", 0.0)
        >= FULL_RELEASE_MIN_CORE_POLY_GAP_PX
        and candidate.get("core_line_band_fraction", 1.0)
        <= FULL_RELEASE_MAX_CORE_LINE_BAND_FRACTION
    )

    separated = strict_separation or boundary_tolerant_separation
    candidate["strict_slit_clearance"] = bool(strict_slit_clearance)
    candidate["strict_separation"] = bool(strict_separation)
    candidate["boundary_tolerant_separation"] = bool(
        boundary_tolerant_separation
    )

    depth_on_table = (
        candidate["height_above_base_mm"] is not None
        and candidate["height_above_base_mm"]
        >= FULL_RELEASE_MIN_HEIGHT_ABOVE_BASE_MM
    )
    visual_on_table = (
        candidate["baseline_change_fraction"]
        >= FULL_RELEASE_VISUAL_TABLE_MIN_CHANGE
        and candidate["base_color_similarity"]
        <= FULL_RELEASE_VISUAL_TABLE_MAX_BASE_COLOR_SIMILARITY
    )
    on_table = depth_on_table or visual_on_table

    center_step = candidate_center_step(candidate, previous_candidate)
    area_ratio = None
    stationary = False

    if previous_candidate is not None:
        previous_area = max(float(previous_candidate.get("area", 0)), 1.0)
        area_ratio = float(candidate["area"] / previous_area)
        stationary = (
            center_step is not None
            and center_step <= FULL_RELEASE_MAX_CENTER_STEP_PX
            and FULL_RELEASE_MIN_STABLE_AREA_RATIO
            <= area_ratio
            <= FULL_RELEASE_MAX_STABLE_AREA_RATIO
        )

    return separated, on_table, stationary, center_step, area_ratio

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
    roi_view = display[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

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

    display[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = roi_view
    cv2.rectangle(
        display,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
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
        f"full_out_streak={full_out_streak}/{FULL_RELEASE_REQUIRED_STABLE_UPDATES}",
        f"selected_slit_side={selected_slit_side.upper()}",
        (
            "T=top B=bottom L=left R=right X=reset S=save Q=quit"
            if AUTO_START
            else "T=top B=bottom L=left R=right SPACE=start "
            "X=reset S=save Q=quit"
        ),
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
    frame_bgr,
    display_bgr,
    baseline_mask,
    current_poly_mask,
    slit_geometry,
    best_candidate,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = SAVE_DIR / f"moving_slit_raw_{timestamp}.jpg"
    view_path = SAVE_DIR / f"moving_slit_view_{timestamp}.png"
    cv2.imwrite(str(raw_path), frame_bgr)
    cv2.imwrite(str(view_path), display_bgr)

    print()
    print(f"Saved raw image: {raw_path}")
    print(f"Saved live view: {view_path}")

    if baseline_mask is not None:
        path = SAVE_DIR / f"baseline_polymailer_mask_{timestamp}.png"
        cv2.imwrite(str(path), baseline_mask.astype(np.uint8) * 255)
        print(f"Saved baseline mask: {path}")

    if current_poly_mask is not None:
        path = SAVE_DIR / f"tracked_polymailer_mask_{timestamp}.png"
        cv2.imwrite(str(path), current_poly_mask.astype(np.uint8) * 255)
        print(f"Saved tracked mask: {path}")

    if slit_geometry is not None:
        slit_image = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
        draw_polygon(
            slit_image,
            slit_geometry["corridor_polygon"],
            (0, 255, 255),
            2,
        )
        point_a = tuple(np.round(slit_geometry["point_a"]).astype(int))
        point_b = tuple(np.round(slit_geometry["point_b"]).astype(int))
        cv2.line(slit_image, point_a, point_b, (255, 0, 0), 5)
        path = SAVE_DIR / f"tracked_slit_{timestamp}.png"
        cv2.imwrite(str(path), slit_image)
        print(f"Saved tracked slit: {path}")

    if best_candidate is not None:
        path = SAVE_DIR / f"emerging_product_mask_{timestamp}.png"
        cv2.imwrite(str(path), best_candidate["mask"].astype(np.uint8) * 255)
        print(f"Saved product mask: {path}")


def main():
    print("Starting moving-slit product-exit test")
    print()
    print("Before SPACE:")
    print("  SAM finds the flat polymailer in the complete ROI.")
    print()
    print("After SPACE:")
    print("  Optical flow tracks the polymailer and its slit at camera FPS.")
    print("  SAM searches only a smaller moving crop around that slit.")
    print(
        "  The yellow corridor expands as the bag is pulled away "
        "or sideways."
    )
    print(
        "  Corridor growth and crop boundaries are smoothed to reduce "
        "display jitter."
    )
    print(
        "  SAM accepts one request at a time, preventing camera-frame "
        "copy backlog."
    )
    print("  Periodic SAM refreshes re-anchor the complete moving bag mask.")
    print("  Stereo depth rejects flat pieces of the black base.")
    print("  Baseline SAM masks reject objects already present before SPACE.")
    print("  Crop-edge and slit-span checks reject black base side fragments.")
    print("  A product can be detected while still touching the bag.")
    print("  After exit is seen, SAM follows the product until it is fully")
    print("  separated, above the table, and stationary for several updates.")
    print("  Once confirmed, the final green product mask stays latched.")
    print()
    print("Speed changes:")
    print(f"  full ROI SAM points per side: {FULL_SAM_POINTS_PER_SIDE}")
    print(f"  moving slit SAM points/side:  {SLIT_SAM_POINTS_PER_SIDE}")
    print(
        "  polymailer SAM refresh:      "
        f"every {POLY_REFRESH_INTERVAL_SECONDS:.2f}s"
    )
    print("  CUDA autocast:                enabled when supported")
    print("  newest-frame-only worker:     enabled")
    print(
        "  full-release verification:    "
        f"{FULL_RELEASE_REQUIRED_STABLE_UPDATES} stable SAM updates"
    )
    print()
    print("Colors:")
    print("  magenta = optical-flow tracked polymailer")
    print("  blue    = tracked moving slit")
    print("  yellow  = moving SAM search corridor")
    print("  orange  = product sliding through the slit")
    print("  cyan    = product exit seen, verifying complete release")
    print("  green   = full product is out and on the table")
    print()
    print("Keys:")
    print("  t     = choose the top edge")
    print("  b     = choose the bottom edge")
    print("  l     = choose the left edge")
    print("  r     = choose the right edge")
    if AUTO_START:
        print(
            "  (auto-start: the flat polymailer is saved and slit tracking"
        )
        print(
            "   begins on its own once warmup ends and one is found)"
        )
    else:
        print("  SPACE = save the flat polymailer and begin slit tracking")
    print("  x     = reset (restarts warmup before auto-start re-arms)")
    print("  s     = save current image and masks")
    print("  q     = quit")
    print()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 2 on device: {device_name}")
    configure_torch(device_name)

    sam2_model = build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device=device_name,
    )

    full_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=FULL_SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )

    slit_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=SLIT_SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )

    worker = LiveSAMWorker(
        full_generator,
        slit_generator,
        device_name,
    )
    worker.start()

    device = select_device()

    pipeline, rgb_queue, depth_queue = build_pipeline(device)
    pipeline.start()

    tracker = MovingSlitTracker()
    selected_slit_side = DEFAULT_SLIT_SIDE
    baseline_mask = None
    baseline_roi_rgb = None
    baseline_poly_hsv = None
    baseline_base_hsv = None
    baseline_depth_roi = None
    baseline_static_signatures = []
    prebaseline_poly = None
    prebaseline_masks = []
    baseline_slit_midpoint = None
    baseline_slit_length = None
    corridor_expansion_state = {
        "outward_extra_px": 0.0,
        "side_extra_px": 0.0,
        "inward_extra_px": 0.0,
        "pose_scale": 1.0,
        "pull_away_px": 0.0,
        "lateral_motion_px": 0.0,
        "total_motion_px": 0.0,
    }
    monitor_started_at = None
    monitor_armed = False
    slit_motion_px = 0.0
    init_fail_printed_at = 0.0
    fast_exit_arrival = None
    fast_exit_settle_frames = 0
    fast_exit_last_submit_at = 0.0
    fast_exit_triggered = False
    latest_processed_sam_frame = None
    emerging_candidates = []
    latest_candidate = None
    previous_candidate = None
    candidate_streak = 0
    exit_confirmed = False
    exit_confirmed_at = None
    verification_candidate = None
    previous_verification_candidate = None
    full_out_streak = 0
    full_out_confirmed = False
    latched_product_candidate = None
    full_release_missing_updates = 0
    first_seen_printed = False
    slit_geometry = None
    pending_poly_refresh_frame = None
    last_poly_refresh_submit_time = 0.0
    poly_refresh_count = 0
    poly_refresh_fail_count = 0
    last_poly_refresh_score = None

    frame_count = 0
    start_time = time.time()
    last_display = None
    last_frame_bgr = None

    try:
        with pipeline:
            set_ir(device)

            while pipeline.isRunning():
                rgb_message = rgb_queue.get()
                depth_message = depth_queue.get()
                frame_bgr = rgb_message.getCvFrame()
                depth_raw = depth_message.getFrame()
                depth_aligned = align_depth_to_rgb(depth_raw)
                depth_roi = depth_aligned[
                    ROI_Y1:ROI_Y2,
                    ROI_X1:ROI_X2,
                ].copy()
                last_frame_bgr = frame_bgr
                frame_count += 1
                warmed = (time.time() - start_time) >= WARMUP_SECONDS

                roi_bgr = frame_bgr[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

                if tracker.active:
                    tracker.update(roi_bgr)
                    slit_geometry = make_slit_geometry(
                        tracker.poly_mask,
                        tracker.slit_points,
                        baseline_slit_midpoint=(
                            baseline_slit_midpoint
                        ),
                        baseline_slit_length=(
                            baseline_slit_length
                        ),
                        expansion_state=(
                            corridor_expansion_state
                        ),
                    )

                    if (
                        slit_geometry is not None
                        and baseline_slit_midpoint is not None
                    ):
                        slit_motion_px = float(
                            np.linalg.norm(
                                slit_geometry["midpoint"]
                                - baseline_slit_midpoint
                            )
                        )

                    if monitor_started_at is not None:
                        monitor_armed = (
                            time.time() - monitor_started_at
                            >= EXIT_ARM_DELAY_SECONDS
                            and slit_motion_px
                            >= EXIT_ARM_MIN_SLIT_MOTION_PX
                        )

                    # Fast-exit watch. This runs at camera FPS and is
                    # deliberately NOT behind monitor_armed: a dumped product
                    # is often already on the table before the arm delay has
                    # elapsed. It is differenced against the saved baseline, so
                    # it does not need the arming heuristic to be meaningful.
                    if (
                        FAST_EXIT_ENABLE
                        and baseline_mask is not None
                        and not exit_confirmed
                    ):
                        arrival = find_fast_exit_arrival(
                            roi_rgb,
                            baseline_roi_rgb,
                            tracker.poly_mask,
                            slit_geometry,
                        )

                        if arrival is None:
                            fast_exit_arrival = None
                            fast_exit_settle_frames = 0
                        else:
                            if (
                                fast_exit_arrival is not None
                                and float(
                                    np.linalg.norm(
                                        arrival["center"]
                                        - fast_exit_arrival["center"]
                                    )
                                )
                                <= FAST_EXIT_SETTLE_MAX_CENTER_STEP_PX
                            ):
                                fast_exit_settle_frames += 1
                            else:
                                fast_exit_settle_frames = 1

                            fast_exit_arrival = arrival

                if warmed:
                    if baseline_mask is None:
                        if frame_count % FULL_SAM_INTERVAL_FRAMES == 0:
                            worker.submit(
                                frame_id=frame_count,
                                image_rgb=roi_rgb,
                                mode="full",
                                crop_box=(0, 0, ROI_WIDTH, ROI_HEIGHT),
                                metadata={"roi_rgb": roi_rgb, "depth_roi": depth_roi},
                            )
                    elif tracker.active:
                        # A settled arrival blob outranks the periodic bag
                        # refresh: the refresh only corrects drift, this is the
                        # exit itself.
                        fast_exit_due = (
                            FAST_EXIT_ENABLE
                            and not exit_confirmed
                            and not full_out_confirmed
                            and slit_geometry is not None
                            and fast_exit_arrival is not None
                            and fast_exit_settle_frames
                            >= FAST_EXIT_SETTLE_FRAMES
                            and pending_poly_refresh_frame is None
                            and time.monotonic() - fast_exit_last_submit_at
                            >= FAST_EXIT_RETRY_INTERVAL_SECONDS
                        )

                        refresh_due = (
                            POLY_REFRESH_ENABLE
                            and not fast_exit_due
                            and pending_poly_refresh_frame is None
                            and time.monotonic() - last_poly_refresh_submit_time
                            >= POLY_REFRESH_INTERVAL_SECONDS
                        )

                        if fast_exit_due:
                            crop_box = make_fast_exit_crop(fast_exit_arrival)
                            x1, y1, x2, y2 = crop_box
                            submitted = worker.submit(
                                frame_id=frame_count,
                                image_rgb=roi_rgb[y1:y2, x1:x2],
                                mode="fast_exit",
                                crop_box=crop_box,
                                metadata={
                                    "roi_rgb": roi_rgb,
                                    "depth_roi": depth_roi,
                                    "poly_mask": tracker.poly_mask,
                                    "slit_point_a": slit_geometry["point_a"],
                                    "slit_point_b": slit_geometry["point_b"],
                                    "corridor_mask": slit_geometry["corridor_mask"],
                                    "contact_mask": slit_geometry["contact_mask"],
                                    "midpoint": slit_geometry["midpoint"],
                                    "tangent": slit_geometry["tangent"],
                                    "outward": slit_geometry["outward"],
                                    "corridor_polygon": slit_geometry["corridor_polygon"],
                                    "contact_polygon": slit_geometry["contact_polygon"],
                                    "crop_box": crop_box,
                                    "reference_candidate": None,
                                    "arrival_mask": fast_exit_arrival["mask"],
                                },
                            )

                            if submitted:
                                fast_exit_last_submit_at = time.monotonic()

                        elif refresh_due:
                            refresh_crop = make_polymailer_refresh_crop(
                                tracker.poly_mask
                            )
                            x1, y1, x2, y2 = refresh_crop
                            refresh_rgb = roi_rgb[y1:y2, x1:x2]
                            submitted = worker.submit(
                                frame_id=frame_count,
                                image_rgb=refresh_rgb,
                                mode="poly_refresh",
                                crop_box=refresh_crop,
                                metadata={
                                    "roi_rgb": roi_rgb,
                                    "roi_bgr": roi_bgr,
                                    "predicted_poly_mask": tracker.poly_mask,
                                    "reference_slit_points": tracker.slit_points,
                                    "crop_box": refresh_crop,
                                },
                            )

                            if submitted:
                                pending_poly_refresh_frame = frame_count
                                last_poly_refresh_submit_time = time.monotonic()

                        elif (
                            pending_poly_refresh_frame is None
                            and slit_geometry is not None
                            # A fast exit can be confirmed before the arm delay
                            # has elapsed, and full-release verification still
                            # has to run in that case.
                            and (monitor_armed or exit_confirmed)
                            and not full_out_confirmed
                            and frame_count % SLIT_SAM_INTERVAL_FRAMES == 0
                        ):
                            if (
                                exit_confirmed
                                and not full_out_confirmed
                                and verification_candidate is not None
                            ):
                                crop_box = make_product_verification_crop(
                                    verification_candidate,
                                    slit_geometry,
                                )
                                sam_mode = "verify"
                            else:
                                crop_box = slit_geometry["crop_box"]
                                sam_mode = "slit"

                            x1, y1, x2, y2 = crop_box
                            crop_rgb = roi_rgb[y1:y2, x1:x2]
                            worker.submit(
                                frame_id=frame_count,
                                image_rgb=crop_rgb,
                                mode=sam_mode,
                                crop_box=crop_box,
                                metadata={
                                    "roi_rgb": roi_rgb,
                                    "depth_roi": depth_roi,
                                    "poly_mask": tracker.poly_mask,
                                    "slit_point_a": slit_geometry["point_a"],
                                    "slit_point_b": slit_geometry["point_b"],
                                    "corridor_mask": slit_geometry["corridor_mask"],
                                    "contact_mask": slit_geometry["contact_mask"],
                                    "midpoint": slit_geometry["midpoint"],
                                    "tangent": slit_geometry["tangent"],
                                    "outward": slit_geometry["outward"],
                                    "corridor_polygon": slit_geometry["corridor_polygon"],
                                    "contact_polygon": slit_geometry["contact_polygon"],
                                    "crop_box": crop_box,
                                    "reference_candidate": verification_candidate,
                                },
                            )


                latest_sam = worker.get_latest()

                if (
                    latest_sam is not None
                    and latest_sam["frame_id"] != latest_processed_sam_frame
                ):
                    latest_processed_sam_frame = latest_sam["frame_id"]

                    if latest_sam["error"] is None:
                        if baseline_mask is None and latest_sam["mode"] == "full":
                            result_roi_rgb = latest_sam["metadata"]["roi_rgb"]
                            prebaseline_poly = choose_polymailer_mask(
                                latest_sam["masks"],
                                result_roi_rgb,
                            )
                            prebaseline_masks = latest_sam["masks"]
                        elif (
                            baseline_mask is not None
                            and latest_sam["mode"] == "poly_refresh"
                        ):
                            metadata = latest_sam["metadata"]
                            pending_poly_refresh_frame = None
                            refresh_candidate = choose_live_polymailer_refresh_mask(
                                masks=latest_sam["masks"],
                                roi_rgb=metadata["roi_rgb"],
                                predicted_poly_mask=metadata[
                                    "predicted_poly_mask"
                                ],
                                baseline_poly_hsv=baseline_poly_hsv,
                                slit_points=metadata[
                                    "reference_slit_points"
                                ],
                                crop_box=metadata["crop_box"],
                            )

                            if refresh_candidate is not None:
                                refreshed = tracker.refresh_from_sam(
                                    roi_bgr=metadata["roi_bgr"],
                                    refreshed_mask=refresh_candidate["mask"],
                                    reference_slit_points=metadata[
                                        "reference_slit_points"
                                    ],
                                )

                                if refreshed:
                                    poly_refresh_count += 1
                                    last_poly_refresh_score = (
                                        refresh_candidate["score"]
                                    )
                                    slit_geometry = make_slit_geometry(
                                        tracker.poly_mask,
                                        tracker.slit_points,
                                        baseline_slit_midpoint=(
                                            baseline_slit_midpoint
                                        ),
                                        baseline_slit_length=(
                                            baseline_slit_length
                                        ),
                                        expansion_state=(
                                            corridor_expansion_state
                                        ),
                                    )

                                    if (
                                        poly_refresh_count == 1
                                        or poly_refresh_count
                                        % POLY_REFRESH_PRINT_EVERY
                                        == 0
                                    ):
                                        print()
                                        print("POLYMAILER SAM REFRESH APPLIED")
                                        print(
                                            "  refresh count:       "
                                            f"{poly_refresh_count}"
                                        )
                                        print(
                                            "  refresh score:       "
                                            f"{refresh_candidate['score']:.3f}"
                                        )
                                        print(
                                            "  color similarity:    "
                                            f"{refresh_candidate['color_similarity']:.3f}"
                                        )
                                        print(
                                            "  candidate in flow:   "
                                            f"{refresh_candidate['candidate_inside_predicted']:.3f}"
                                        )
                                        print(
                                            "  refresh IoU:         "
                                            f"{refresh_candidate['iou']:.3f}"
                                        )
                                        print(
                                            "  tracker source:      "
                                            f"{tracker.last_transform_source}"
                                        )
                                else:
                                    poly_refresh_fail_count += 1
                            else:
                                poly_refresh_fail_count += 1
                        elif (
                            baseline_mask is not None
                            and latest_sam["mode"]
                            in ("slit", "verify", "fast_exit")
                        ):
                            metadata = latest_sam["metadata"]
                            result_geometry = {
                                "point_a": metadata["slit_point_a"],
                                "point_b": metadata["slit_point_b"],
                                "midpoint": metadata["midpoint"],
                                "tangent": metadata.get(
                                    "tangent",
                                    normalize_vector(
                                        metadata["slit_point_b"]
                                        - metadata["slit_point_a"]
                                    ),
                                ),
                                "outward": metadata["outward"],
                                "corridor_mask": metadata["corridor_mask"],
                                "contact_mask": metadata["contact_mask"],
                                "corridor_polygon": metadata["corridor_polygon"],
                                "contact_polygon": metadata["contact_polygon"],
                                "crop_box": metadata["crop_box"],
                            }

                            if latest_sam["mode"] == "fast_exit":
                                emerging_candidates = find_emerging_product_candidates(
                                    masks=latest_sam["masks"],
                                    poly_mask=metadata["poly_mask"],
                                    slit_geometry=result_geometry,
                                    current_roi_rgb=metadata["roi_rgb"],
                                    baseline_roi_rgb=baseline_roi_rgb,
                                    baseline_poly_hsv=baseline_poly_hsv,
                                    baseline_base_hsv=baseline_base_hsv,
                                    baseline_poly_mask=baseline_mask,
                                    baseline_static_signatures=(
                                        baseline_static_signatures
                                    ),
                                    depth_roi=metadata.get("depth_roi"),
                                    fast_exit=True,
                                    arrival_mask=metadata.get("arrival_mask"),
                                )

                                latest_candidate = (
                                    emerging_candidates[0]
                                    if emerging_candidates
                                    else None
                                )

                                if latest_candidate is None:
                                    candidate_streak = 0
                                    previous_candidate = None
                                else:
                                    # Two matching scans of a product that is
                                    # already sitting still. The settle test
                                    # supplied the temporal evidence; this
                                    # guards against a single SAM fluke.
                                    if candidate_matches_previous(
                                        latest_candidate,
                                        previous_candidate,
                                    ):
                                        candidate_streak += 1
                                    else:
                                        candidate_streak = 1

                                    previous_candidate = latest_candidate

                                    if (
                                        candidate_streak
                                        >= PRODUCT_CONFIRM_REQUIRED_SAM_UPDATES
                                        and not exit_confirmed
                                    ):
                                        exit_confirmed = True
                                        exit_confirmed_at = time.time()
                                        verification_candidate = latest_candidate
                                        previous_verification_candidate = (
                                            latest_candidate
                                        )
                                        full_out_streak = 0
                                        full_out_confirmed = False
                                        latched_product_candidate = None
                                        full_release_missing_updates = 0
                                        fast_exit_triggered = True

                                        print()
                                        print("FAST EXIT DETECTED")
                                        print(
                                            "  the product was already out "
                                            "before the slit detector could "
                                            "see it cross"
                                        )
                                        print(
                                            "  arrival overlap:   "
                                            f"{latest_candidate['arrival_overlap']:.3f}"
                                        )
                                        print(
                                            "  area px:           "
                                            f"{latest_candidate['area']}"
                                        )
                                        print(
                                            "  outward fraction:  "
                                            f"{latest_candidate['outward_fraction']:.3f}"
                                        )
                                        print(
                                            "  baseline change:   "
                                            f"{latest_candidate['baseline_change_fraction']:.3f}"
                                        )
                                        print(
                                            "  base color sim:    "
                                            f"{latest_candidate['base_color_similarity']:.3f}"
                                        )
                                        print(
                                            "  height above base: "
                                            f"{latest_candidate['height_above_base_mm']}"
                                        )
                                        print(
                                            "  verifying full release..."
                                        )

                            elif latest_sam["mode"] == "slit":
                                emerging_candidates = find_emerging_product_candidates(
                                    masks=latest_sam["masks"],
                                    poly_mask=metadata["poly_mask"],
                                    slit_geometry=result_geometry,
                                    current_roi_rgb=metadata["roi_rgb"],
                                    baseline_roi_rgb=baseline_roi_rgb,
                                    baseline_poly_hsv=baseline_poly_hsv,
                                    baseline_base_hsv=baseline_base_hsv,
                                    baseline_poly_mask=baseline_mask,
                                    baseline_static_signatures=(
                                        baseline_static_signatures
                                    ),
                                    depth_roi=metadata.get("depth_roi"),
                                )

                                latest_candidate = (
                                    emerging_candidates[0]
                                    if emerging_candidates
                                    else None
                                )

                                if latest_candidate is not None:
                                    if candidate_matches_previous(
                                        latest_candidate,
                                        previous_candidate,
                                    ):
                                        candidate_streak += 1
                                    else:
                                        candidate_streak = 1

                                    previous_candidate = latest_candidate

                                    if not first_seen_printed:
                                        print()
                                        print("PRODUCT SLIDING OUT")
                                        print(f"  mode: {latest_candidate['mode']}")
                                        print(f"  score: {latest_candidate['score']:.3f}")
                                        print(
                                            "  outward fraction: "
                                            f"{latest_candidate['outward_fraction']:.3f}"
                                        )
                                        print(
                                            "  slit contact:      "
                                            f"{latest_candidate['contact_overlap']:.3f}"
                                        )
                                        print(
                                            "  bag overlap:       "
                                            f"{latest_candidate['poly_overlap']:.3f}"
                                        )
                                        print(
                                            "  area ratio:        "
                                            f"{latest_candidate['area_ratio_of_poly']:.3f}"
                                        )
                                        print(
                                            "  height above base: "
                                            f"{latest_candidate['height_above_base_mm']} mm"
                                        )
                                        print(
                                            "  static match IoU:  "
                                            f"{latest_candidate['static_match_iou']:.3f}"
                                        )
                                        print(
                                            "  base color sim:    "
                                            f"{latest_candidate['base_color_similarity']:.3f}"
                                        )
                                        first_seen_printed = True

                                    if (
                                        candidate_streak
                                        >= PRODUCT_CONFIRM_REQUIRED_SAM_UPDATES
                                        and not exit_confirmed
                                    ):
                                        exit_confirmed = True
                                        exit_confirmed_at = time.time()
                                        verification_candidate = latest_candidate
                                        previous_verification_candidate = latest_candidate
                                        full_out_streak = 0
                                        full_out_confirmed = False
                                        latched_product_candidate = None
                                        full_release_missing_updates = 0
                                        print()
                                        print("PRODUCT EXIT CONFIRMED")
                                        print(
                                            "  A separate product mask crossed the "
                                            "tracked moving slit."
                                        )
                                        print(
                                            "  Now verifying that the complete product "
                                            "is separated and resting on the table."
                                        )
                                else:
                                    candidate_streak = 0
                                    previous_candidate = None

                            else:
                                reference_candidate = metadata.get(
                                    "reference_candidate"
                                )
                                released_candidates = find_released_product_candidates(
                                    masks=latest_sam["masks"],
                                    reference_candidate=reference_candidate,
                                    poly_mask=metadata["poly_mask"],
                                    slit_geometry=result_geometry,
                                    current_roi_rgb=metadata["roi_rgb"],
                                    baseline_roi_rgb=baseline_roi_rgb,
                                    baseline_poly_hsv=baseline_poly_hsv,
                                    baseline_base_hsv=baseline_base_hsv,
                                    baseline_poly_mask=baseline_mask,
                                    baseline_static_signatures=(
                                        baseline_static_signatures
                                    ),
                                    depth_roi=metadata.get("depth_roi"),
                                )
                                emerging_candidates = released_candidates
                                latest_candidate = (
                                    released_candidates[0]
                                    if released_candidates
                                    else None
                                )

                                if latest_candidate is not None:
                                    (
                                        separated,
                                        on_table,
                                        stationary,
                                        center_step,
                                        stable_area_ratio,
                                    ) = evaluate_full_release(
                                        latest_candidate,
                                        previous_verification_candidate,
                                    )

                                    verification_candidate = latest_candidate
                                    previous_verification_candidate = latest_candidate
                                    full_release_missing_updates = 0

                                    confirm_delay_ok = (
                                        exit_confirmed_at is not None
                                        and time.time() - exit_confirmed_at
                                        >= FULL_RELEASE_MIN_CONFIRM_DELAY_SECONDS
                                    )

                                    if (
                                        separated
                                        and on_table
                                        and stationary
                                        and confirm_delay_ok
                                    ):
                                        full_out_streak += 1
                                    else:
                                        full_out_streak = max(
                                            0,
                                            full_out_streak - 1,
                                        )

                                    if (
                                        full_out_streak
                                        >= FULL_RELEASE_REQUIRED_STABLE_UPDATES
                                        and not full_out_confirmed
                                    ):
                                        full_out_confirmed = True
                                        latched_product_candidate = (
                                            copy_candidate(
                                                latest_candidate
                                            )
                                        )
                                        verification_candidate = (
                                            latched_product_candidate
                                        )
                                        previous_verification_candidate = (
                                            latched_product_candidate
                                        )
                                        print()
                                        print("FULL PRODUCT OUT AND ON TABLE")
                                        print(
                                            "  The product mask is fully separated "
                                            "from the polymailer."
                                        )
                                        print(
                                            "  The product is above the table and "
                                            "stationary."
                                        )
                                        print("  SAFE TO DISCARD POLYMAILER")
                                        print(
                                            "  final bag overlap: "
                                            f"{latest_candidate['poly_overlap']:.3f}"
                                        )
                                        print(
                                            "  final inward fraction: "
                                            f"{latest_candidate['inward_fraction']:.3f}"
                                        )
                                        print(
                                            "  final core bag overlap: "
                                            f"{latest_candidate.get('core_poly_overlap', 0.0):.3f}"
                                        )
                                        print(
                                            "  final core outward fraction: "
                                            f"{latest_candidate.get('core_outward_fraction', 0.0):.3f}"
                                        )
                                        print(
                                            "  final trailing-edge clearance: "
                                            f"{latest_candidate.get('core_trailing_edge_clearance_px', 0.0):.3f} px"
                                        )
                                        print(
                                            "  final product-to-bag core gap: "
                                            f"{latest_candidate.get('core_poly_gap_px', 0.0):.3f} px"
                                        )
                                        print(
                                            "  separation rule: "
                                            + (
                                                "strict"
                                                if latest_candidate.get(
                                                    "strict_separation",
                                                    False,
                                                )
                                                else "boundary_tolerant_core"
                                            )
                                        )
                                        print(
                                            "  final center movement: "
                                            f"{center_step} px"
                                        )
                                        print(
                                            "  final height above table: "
                                            f"{latest_candidate['height_above_base_mm']} mm"
                                        )
                                else:
                                    full_release_missing_updates += 1
                                    full_out_streak = max(
                                        0,
                                        full_out_streak - 1,
                                    )

                    else:
                        if latest_sam.get("mode") == "poly_refresh":
                            pending_poly_refresh_frame = None
                            poly_refresh_fail_count += 1
                        emerging_candidates = []

                if not warmed:
                    state = (
                        f"WARMING {max(WARMUP_SECONDS - (time.time() - start_time), 0):.1f}s"
                    )
                elif baseline_mask is None:
                    state = (
                        "AUTO-START: FINDING POLYMAILER "
                        if AUTO_START
                        else "SELECT SIDE THEN PRESS SPACE "
                    ) + f"[{selected_slit_side.upper()}]"
                elif not tracker.active:
                    state = "TRACKER LOST - PRESS R"
                elif full_out_confirmed:
                    state = "FULL PRODUCT OUT - DISCARD MAILER" + (
                        " (FAST EXIT)" if fast_exit_triggered else ""
                    )
                elif exit_confirmed:
                    if verification_candidate is not None and (
                        verification_candidate.get(
                            "core_trailing_edge_clearance_px",
                            0.0,
                        ) < FULL_RELEASE_MIN_CORE_TRAILING_CLEARANCE_PX
                        or verification_candidate.get(
                            "core_poly_gap_px",
                            0.0,
                        ) < FULL_RELEASE_MIN_CORE_POLY_GAP_PX
                    ):
                        state = "WAITING FOR PRODUCT TAIL TO CLEAR SLIT"
                    else:
                        state = "VERIFYING FULL PRODUCT RELEASE" + (
                            " (FAST EXIT)" if fast_exit_triggered else ""
                        )
                elif fast_exit_settle_frames > 0:
                    state = (
                        "FAST EXIT: OBJECT SETTLING "
                        f"({fast_exit_settle_frames}/"
                        f"{FAST_EXIT_SETTLE_FRAMES})"
                    )
                elif not monitor_armed:
                    state = (
                        "LIFT/PULL BAG TO ARM "
                        f"({slit_motion_px:.0f}/"
                        f"{EXIT_ARM_MIN_SLIT_MOTION_PX:.0f}px)"
                    )
                elif emerging_candidates:
                    state = "PRODUCT SLIDING OUT"
                else:
                    state = (
                        "TRACKING MOVING "
                        f"{selected_slit_side.upper()} SLIT"
                    )

                current_poly_mask = (
                    tracker.poly_mask
                    if tracker.active
                    else (
                        prebaseline_poly["mask"]
                        if prebaseline_poly is not None
                        else None
                    )
                )

                display_slit_geometry = slit_geometry

                if baseline_mask is None and prebaseline_poly is not None:
                    preview_slit_points = get_initial_slit_edge(
                        prebaseline_poly["mask"],
                        selected_slit_side,
                    )
                    display_slit_geometry = make_slit_geometry(
                        prebaseline_poly["mask"],
                        preview_slit_points,
                    )

                display_candidates = emerging_candidates

                if (
                    full_out_confirmed
                    and latched_product_candidate is not None
                ):
                    display_candidates = [
                        latched_product_candidate
                    ]

                display = make_live_view(
                    frame_bgr=frame_bgr,
                    current_poly_mask=current_poly_mask,
                    baseline_mask=baseline_mask,
                    slit_geometry=display_slit_geometry,
                    emerging_candidates=display_candidates,
                    latest_sam=latest_sam,
                    state=state,
                    frame_count=frame_count,
                    flow_point_count=tracker.last_good_point_count,
                    tracker_ok=tracker.last_transform_ok,
                    candidate_streak=candidate_streak,
                    exit_confirmed=exit_confirmed,
                    full_out_streak=full_out_streak,
                    full_out_confirmed=full_out_confirmed,
                    selected_slit_side=selected_slit_side,
                )

                last_display = display
                cv2.imshow("Moving Slit Product Exit Test", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                side_key_map = {
                    ord("t"): SLIT_SIDE_TOP,
                    ord("b"): SLIT_SIDE_BOTTOM,
                    ord("l"): SLIT_SIDE_LEFT,
                    ord("r"): SLIT_SIDE_RIGHT,
                }

                if key in side_key_map:
                    if baseline_mask is not None:
                        print(
                            "The slit side is locked while tracking. "
                            "Press X to reset before choosing another side."
                        )
                    else:
                        selected_slit_side = side_key_map[key]
                        print(
                            "Selected polymailer slit side: "
                            f"{selected_slit_side.upper()}"
                        )

                if key == ord("x"):
                    # Restart the warmup clock so auto-start does not instantly
                    # re-baseline whatever is in view; this leaves
                    # WARMUP_SECONDS to place the next polymailer.
                    start_time = time.time()
                    init_fail_printed_at = 0.0
                    fast_exit_arrival = None
                    fast_exit_settle_frames = 0
                    fast_exit_last_submit_at = 0.0
                    fast_exit_triggered = False
                    tracker.clear()
                    baseline_mask = None
                    baseline_roi_rgb = None
                    baseline_poly_hsv = None
                    baseline_base_hsv = None
                    baseline_depth_roi = None
                    baseline_static_signatures = []
                    prebaseline_poly = None
                    prebaseline_masks = []
                    baseline_slit_midpoint = None
                    baseline_slit_length = None
                    corridor_expansion_state = {
                        "outward_extra_px": 0.0,
                        "side_extra_px": 0.0,
                        "inward_extra_px": 0.0,
                        "pose_scale": 1.0,
                        "pull_away_px": 0.0,
                        "lateral_motion_px": 0.0,
                        "total_motion_px": 0.0,
                    }
                    monitor_started_at = None
                    monitor_armed = False
                    slit_motion_px = 0.0
                    emerging_candidates = []
                    latest_candidate = None
                    previous_candidate = None
                    candidate_streak = 0
                    exit_confirmed = False
                    exit_confirmed_at = None
                    verification_candidate = None
                    previous_verification_candidate = None
                    full_out_streak = 0
                    full_out_confirmed = False
                    latched_product_candidate = None
                    full_release_missing_updates = 0
                    first_seen_printed = False
                    slit_geometry = None
                    pending_poly_refresh_frame = None
                    last_poly_refresh_submit_time = 0.0
                    poly_refresh_count = 0
                    poly_refresh_fail_count = 0
                    last_poly_refresh_score = None
                    print(
                        "Baseline, slit tracker, and exit state cleared. "
                        f"Current selected side: {selected_slit_side.upper()}"
                    )

                # Fire the same baseline capture SPACE does, without the key,
                # as soon as warmup is over and SAM has a polymailer to lock on.
                auto_start = (
                    AUTO_START
                    and warmed
                    and baseline_mask is None
                    and prebaseline_poly is not None
                )

                if key == ord(" ") or auto_start:
                    if not warmed:
                        print("Still warming up.")
                    elif prebaseline_poly is None:
                        print("No valid polymailer is currently selected.")
                    else:
                        baseline_mask = prebaseline_poly["mask"].copy()
                        baseline_roi_rgb = roi_rgb.copy()
                        baseline_depth_roi = depth_roi.copy()
                        baseline_static_signatures = (
                            build_baseline_static_signatures(
                                prebaseline_masks,
                                baseline_mask,
                                baseline_roi_rgb,
                            )
                        )
                        baseline_poly_hsv = masked_mean_hsv(
                            baseline_roi_rgb,
                            baseline_mask,
                        )
                        baseline_base_hsv = estimate_baseline_base_hsv(
                            baseline_roi_rgb,
                            baseline_mask,
                        )

                        initialized = tracker.initialize(
                            roi_bgr,
                            baseline_mask,
                            selected_slit_side,
                        )

                        if not initialized:
                            # Auto-start retries every frame, so rate-limit the
                            # message instead of flooding the console at FPS.
                            now = time.time()

                            if now - init_fail_printed_at >= 1.0:
                                init_fail_printed_at = now
                                print(
                                    "Could not initialize optical-flow "
                                    "tracking. Keep the polymailer still; "
                                    "retrying."
                                )

                            tracker.clear()
                            baseline_mask = None
                            baseline_roi_rgb = None
                            baseline_poly_hsv = None
                            baseline_base_hsv = None
                            baseline_depth_roi = None
                            baseline_static_signatures = []
                        else:
                            slit_geometry = make_slit_geometry(
                                tracker.poly_mask,
                                tracker.slit_points,
                                baseline_slit_midpoint=(
                                    baseline_slit_midpoint
                                ),
                                baseline_slit_length=(
                                    baseline_slit_length
                                ),
                                expansion_state=(
                                    corridor_expansion_state
                                ),
                            )
                            baseline_slit_midpoint = (
                                None
                                if slit_geometry is None
                                else slit_geometry["midpoint"].copy()
                            )

                            baseline_slit_length = (
                                None
                                if slit_geometry is None
                                else float(
                                    np.linalg.norm(
                                        (
                                            slit_geometry["point_b"]
                                            - slit_geometry["point_a"]
                                        )
                                    )
                                )
                            )

                            corridor_expansion_state = {
                                "outward_extra_px": 0.0,
                                "side_extra_px": 0.0,
                                "inward_extra_px": 0.0,
                                "pose_scale": 1.0,
                                "pull_away_px": 0.0,
                                "lateral_motion_px": 0.0,
                                "total_motion_px": 0.0,
                            }

                            slit_geometry = make_slit_geometry(
                                tracker.poly_mask,
                                tracker.slit_points,
                                baseline_slit_midpoint=(
                                    baseline_slit_midpoint
                                ),
                                baseline_slit_length=(
                                    baseline_slit_length
                                ),
                                expansion_state=(
                                    corridor_expansion_state
                                ),
                            )

                            monitor_started_at = time.time()
                            monitor_armed = False
                            slit_motion_px = 0.0
                            fast_exit_arrival = None
                            fast_exit_settle_frames = 0
                            fast_exit_last_submit_at = 0.0
                            fast_exit_triggered = False
                            emerging_candidates = []
                            latest_candidate = None
                            previous_candidate = None
                            candidate_streak = 0
                            exit_confirmed = False
                            exit_confirmed_at = None
                            verification_candidate = None
                            previous_verification_candidate = None
                            full_out_streak = 0
                            full_out_confirmed = False
                            latched_product_candidate = None
                            full_release_missing_updates = 0
                            first_seen_printed = False
                            pending_poly_refresh_frame = None
                            last_poly_refresh_submit_time = time.monotonic()
                            poly_refresh_count = 0
                            poly_refresh_fail_count = 0
                            last_poly_refresh_score = None

                            print()
                            print("POLYMAILER AND MOVING SLIT SAVED")
                            print(
                                "  selected slit side: "
                                f"{selected_slit_side.upper()}"
                            )
                            print(
                                f"  polymailer area px: {int(baseline_mask.sum())}"
                            )
                            print(
                                f"  optical-flow points: {len(tracker.points)}"
                            )
                            print(
                                "  saved static SAM masks: "
                                f"{len(baseline_static_signatures)}"
                            )
                            print(
                                "  The slit line will now move with the "
                                "polymailer at camera FPS."
                            )
                            print(
                                "  SAM will search only around the moving slit."
                            )
                            print(
                                "  Detection arms only after the slit moves "
                                f"{EXIT_ARM_MIN_SLIT_MOTION_PX:.0f}px."
                            )

                if key == ord("s") and last_display is not None:
                    if (
                        full_out_confirmed
                        and latched_product_candidate is not None
                    ):
                        best = latched_product_candidate
                    else:
                        best = (
                            emerging_candidates[0]
                            if emerging_candidates
                            else None
                        )

                    save_live_result(
                        frame_bgr=last_frame_bgr,
                        display_bgr=last_display,
                        baseline_mask=baseline_mask,
                        current_poly_mask=current_poly_mask,
                        slit_geometry=slit_geometry,
                        best_candidate=best,
                    )

    finally:
        worker.stop()
        cv2.destroyAllWindows()

    print("Closed.")


if __name__ == "__main__":
    main()
