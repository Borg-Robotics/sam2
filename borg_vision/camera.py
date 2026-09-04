"""OAK-D camera ownership: depthai v3 pipeline, frame grabbing, intrinsics.

Pipeline layout, IR/exposure settings and depth->RGB alignment are identical
to the original product_detection_final.py script. Differences:
- optional mxid selection for multi-camera setups,
- output queues are maxSize=1 / non-blocking so a consumer that idles
  between detections always receives the freshest frame without
  backpressure on the device.
"""

import time
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np


@dataclass
class Frames:
    rgb: np.ndarray                    # BGR, rgb_size
    depth_class_aligned: np.ndarray    # uint16 mm, rgb_size, aligned to RGB
    depth_measure_aligned: np.ndarray  # uint16 mm, rgb_size, aligned to RGB


def resize_depth_to_rgb_size(cfg, depth):
    return cv2.resize(depth, cfg.rgb_size, interpolation=cv2.INTER_NEAREST)


def align_depth_to_rgb(cfg, depth):
    h, w = depth.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    matrix = np.float32(
        [
            [
                cfg.depth_align_scale_x,
                0,
                (1.0 - cfg.depth_align_scale_x) * cx + cfg.depth_align_x_shift_px,
            ],
            [
                0,
                cfg.depth_align_scale_y,
                (1.0 - cfg.depth_align_scale_y) * cy + cfg.depth_align_y_shift_px,
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


def apply_manual_exposure(cfg, camera_node, label):
    if not cfg.use_manual_stereo_exposure:
        return

    try:
        camera_node.initialControl.setManualExposure(
            cfg.stereo_exposure_us,
            cfg.stereo_iso,
        )
        print(f"{label}: manual exposure {cfg.stereo_exposure_us} us ISO {cfg.stereo_iso}")
    except Exception as e:
        print(f"{label}: could not set manual exposure: {e}")


def set_ir(cfg, device):
    try:
        device.setIrLaserDotProjectorIntensity(cfg.ir_laser_intensity)
        print(f"IR laser set to {cfg.ir_laser_intensity}")
    except Exception as e:
        print(f"Could not set IR laser: {e}")

    try:
        device.setIrFloodLightIntensity(cfg.ir_flood_intensity)
        print(f"IR flood set to {cfg.ir_flood_intensity}")
    except Exception as e:
        print(f"Could not set IR flood: {e}")


def get_rgb_intrinsics(cfg, device):
    try:
        calibration = device.readCalibration()
        intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            cfg.rgb_size[0],
            cfg.rgb_size[1],
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


def _apply_stereo_quality(cfg, stereo, label):
    """Preset + subpixel precision for one StereoDepth node.

    Order matters: setDefaultProfilePreset RESETS the node's config, so it
    must run before the individual setters; the caller then applies
    LR-check/subpixel/confidence, and _apply_subpixel_bits runs last."""
    name = (cfg.stereo_preset or "FAST_DENSITY").strip().upper()
    preset = getattr(dai.node.StereoDepth.PresetMode, name, None)
    if preset is None:
        print(f"Unknown stereo preset {name!r}; using FAST_DENSITY")
        preset = dai.node.StereoDepth.PresetMode.FAST_DENSITY
        name = "FAST_DENSITY"
    stereo.setDefaultProfilePreset(preset)
    print(f"{label} stereo preset: {name}")


def _apply_subpixel_bits(cfg, stereo, label):
    """Raise disparity fractional bits (finer depth steps). 0 = leave default.

    Applied AFTER setSubpixel so nothing resets it. API location varies by
    depthai version, hence the two attempts."""
    bits = int(cfg.stereo_subpixel_bits or 0)
    if bits <= 0:
        return
    try:
        stereo.initialConfig.setSubpixelFractionalBits(bits)
        print(f"{label} subpixel fractional bits: {bits}")
    except Exception:
        try:
            stereo.setSubpixelFractionalBits(bits)
            print(f"{label} subpixel fractional bits: {bits}")
        except Exception as e:
            print(f"Could not set {label} subpixel bits: {e}")


def _frame_timer():
    """Stage timer for get_frames, active only under BORG_TIME_FRAMES=1."""
    import os
    import time

    if os.environ.get("BORG_TIME_FRAMES") != "1":
        def _noop(_label=None):
            return None

        _noop.report = lambda: None
        return _noop

    start = time.time()
    marks = []
    state = {"last": start}

    def _mark(label):
        now = time.time()
        marks.append((label, now - state["last"]))
        state["last"] = now

    def _report():
        total = time.time() - start
        parts = "  ".join(f"{lbl} {dt * 1000:.0f}ms" for lbl, dt in marks)
        print(f"[frames] total {total * 1000:.0f}ms   {parts}", flush=True)

    _mark.report = _report
    return _mark



class OakCamera:
    def __init__(self, cfg, mxid=None, needs_measurement_stereo=None):
        self.cfg = cfg
        self.mxid = mxid

        # When a camera is shared across several modes (multi-mode-per-camera),
        # the pipeline must build the measurement-stereo stream if *any*
        # co-located mode needs it, regardless of which mode's cfg was used to
        # construct this camera. `needs_measurement_stereo=None` falls back to
        # cfg.needs_measurement_stereo (single-mode behaviour).
        self._needs_measurement_stereo = needs_measurement_stereo

        self._pipeline = None
        self._device = None
        self._intrinsics = None
        self._rgb_queue = None
        self._depth_class_queue = None
        self._depth_measure_queue = None

    def _build_pipeline(self):
        cfg = self.cfg

        if self.mxid:
            device = dai.Device(dai.DeviceInfo(self.mxid))
            pipeline = dai.Pipeline(device)
        else:
            pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.Camera).build()

        rgb_output = cam.requestOutput(
            size=cfg.rgb_size,
            type=dai.ImgFrame.Type.BGR888p,
            fps=cfg.fps,
        )

        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        apply_manual_exposure(cfg, left, "left stereo")
        apply_manual_exposure(cfg, right, "right stereo")

        left_class_out = left.requestOutput(
            size=cfg.rgb_size,
            type=dai.ImgFrame.Type.GRAY8,
            fps=cfg.fps,
        )

        right_class_out = right.requestOutput(
            size=cfg.rgb_size,
            type=dai.ImgFrame.Type.GRAY8,
            fps=cfg.fps,
        )

        stereo_class = pipeline.create(dai.node.StereoDepth)
        _apply_stereo_quality(cfg, stereo_class, "classification")
        stereo_class.setLeftRightCheck(cfg.use_left_right_check)
        stereo_class.setSubpixel(cfg.use_subpixel)
        _apply_subpixel_bits(cfg, stereo_class, "classification")

        try:
            stereo_class.setOutputSize(cfg.rgb_size[0], cfg.rgb_size[1])
            print(f"Classification stereo depth output size set to {cfg.rgb_size[0]}x{cfg.rgb_size[1]}.")
        except Exception as e:
            print(f"Could not set classification stereo output size: {e}")

        try:
            stereo_class.initialConfig.setConfidenceThreshold(cfg.confidence_threshold)
            print(f"Classification confidence threshold set to {cfg.confidence_threshold}.")
        except Exception as e:
            print(f"Could not set classification confidence threshold: {e}")

        left_class_out.link(stereo_class.left)
        right_class_out.link(stereo_class.right)

        self._rgb_queue = rgb_output.createOutputQueue(maxSize=1, blocking=False)
        self._depth_class_queue = stereo_class.depth.createOutputQueue(maxSize=1, blocking=False)

        # Modes that measure from a separate, lower-resolution stereo stream
        # (package, box) build a second StereoDepth node. Modes that reuse the
        # classification depth for measurement (object, clear_bag, polymailer)
        # skip it, and get_frames() falls back to the classification depth.
        needs_measurement_stereo = self._needs_measurement_stereo
        if needs_measurement_stereo is None:
            needs_measurement_stereo = cfg.needs_measurement_stereo

        if needs_measurement_stereo:
            left_measure_out = left.requestOutput(
                size=cfg.stereo_size,
                type=dai.ImgFrame.Type.GRAY8,
                fps=cfg.fps,
            )

            right_measure_out = right.requestOutput(
                size=cfg.stereo_size,
                type=dai.ImgFrame.Type.GRAY8,
                fps=cfg.fps,
            )

            stereo_measure = pipeline.create(dai.node.StereoDepth)
            _apply_stereo_quality(cfg, stereo_measure, "measurement")
            stereo_measure.setLeftRightCheck(cfg.use_left_right_check)
            stereo_measure.setSubpixel(cfg.use_subpixel)
            _apply_subpixel_bits(cfg, stereo_measure, "measurement")

            try:
                stereo_measure.initialConfig.setConfidenceThreshold(cfg.confidence_threshold)
                print(f"Measurement confidence threshold set to {cfg.confidence_threshold}.")
            except Exception as e:
                print(f"Could not set measurement confidence threshold: {e}")

            left_measure_out.link(stereo_measure.left)
            right_measure_out.link(stereo_measure.right)

            self._depth_measure_queue = stereo_measure.depth.createOutputQueue(
                maxSize=1, blocking=False
            )

        return pipeline

    def open(self):
        if self._pipeline is not None:
            return self

        self._pipeline = self._build_pipeline()
        self._pipeline.start()

        self._device = self._pipeline.getDefaultDevice()
        set_ir(self.cfg, self._device)
        self._intrinsics = get_rgb_intrinsics(self.cfg, self._device)

        return self

    def warmup(self, seconds):
        """Pump and discard frames for `seconds` so autoexposure settles.

        Camera-level (shared by every mode on this device): meant to run once
        right after open(), not before each detection.
        """
        start = time.time()
        while time.time() - start < seconds:
            self.get_frames()
        return self

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def is_running(self):
        return self._pipeline is not None and self._pipeline.isRunning()

    def get_frames(self):
        """Block until a fresh frame triple is available and return it aligned.

        Timing: set the env var BORG_TIME_FRAMES=1 to print a per-stage
        breakdown. Added 2026-08-24 -- a polymailer detect_polymailer call takes
        ~3.7 s end to end while SAM2 inference measures only ~0.36 s, so most of
        the time is elsewhere and this is where to look first. Note every mode on
        a camera pays for the measurement stereo if ANY mode on it needs one:
        polymailer reads depth_class_aligned and never touches the measurement
        stream, but camera_2 also serves box, so the resize+align below still
        runs on every polymailer call.
        """
        if self._pipeline is None:
            raise RuntimeError("OakCamera is not open; call open() first")

        cfg = self.cfg

        _t = _frame_timer()

        rgb_msg = self._rgb_queue.get()
        _t("rgb_queue.get")
        depth_class_msg = self._depth_class_queue.get()
        _t("depth_class_queue.get")

        rgb = rgb_msg.getCvFrame()
        _t("rgb.getCvFrame")

        depth_class_raw = depth_class_msg.getFrame()
        depth_class_aligned = align_depth_to_rgb(cfg, depth_class_raw)
        _t("depth_class align")

        if self._depth_measure_queue is not None:
            depth_measure_msg = self._depth_measure_queue.get()
            _t("depth_measure_queue.get")
            depth_measure_raw = depth_measure_msg.getFrame()
            depth_measure_scaled = resize_depth_to_rgb_size(cfg, depth_measure_raw)
            depth_measure_aligned = align_depth_to_rgb(cfg, depth_measure_scaled)
            _t("depth_measure resize+align")
        else:
            # No dedicated measurement stream: reuse the classification depth so
            # Frames stays the same shape for every mode.
            depth_measure_aligned = depth_class_aligned

        _t.report()

        return Frames(
            rgb=rgb,
            depth_class_aligned=depth_class_aligned,
            depth_measure_aligned=depth_measure_aligned,
        )

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")

        self._pipeline = None
        self._device = None
        self._rgb_queue = None
        self._depth_class_queue = None
        self._depth_measure_queue = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
