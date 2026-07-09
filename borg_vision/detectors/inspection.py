"""Product-inspection detector: dual OAK-D RGB capture + OpenAI verdict.

Unlike the segmentation detectors, inspection does not use SAM2, stereo depth,
barcode scanning, or TF, but it DOES need two cameras at once. It therefore does
not subclass BaseDetector (whose lifecycle is SAM2/warmup/barcode centric);
instead it mirrors that lifecycle shape (open/close/context-manager, a typed
result, save_debug) while owning TWO lightweight RGB-only depthai pipelines.

Capture is fully headless (no cv2 preview / keypresses): on `inspect()` it grabs
`capture_count` frames, alternating between the two cameras, saves them as JPEGs,
then sends all of them together to OpenAI via `inspection.run_inspection`.

`should_abort` callables are polled once per captured frame so a ROS action
server can wire goal cancellation into the capture loop.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

from ..config import InspectionConfig
from ..inspection import run_inspection
from ..results import InspectionResult

QUEUE_MAX_SIZE = 2


def make_request_id():
    return f"inspect_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _device_id(device_info):
    if hasattr(device_info, "getDeviceId"):
        return device_info.getDeviceId()
    if hasattr(device_info, "getMxId"):
        return device_info.getMxId()
    if hasattr(device_info, "deviceId"):
        return str(device_info.deviceId)
    raise RuntimeError("Could not read the OAK camera device ID.")


def resolve_camera_pair(mxid_1, mxid_2):
    """Resolve two OAK DeviceInfos by mxid (or the first two found if both None)."""
    devices = list(dai.Device.getAllAvailableDevices())

    if len(devices) < 2:
        raise RuntimeError(
            f"At least 2 OAK cameras are required, but only {len(devices)} were found."
        )

    if mxid_1 is None and mxid_2 is None:
        return devices[0], devices[1]

    if mxid_1 is None or mxid_2 is None:
        raise ValueError("Provide both mxid_1 and mxid_2, or neither.")
    if mxid_1 == mxid_2:
        raise ValueError("The two camera mxids must be different.")

    by_id = {_device_id(d): d for d in devices}
    if mxid_1 not in by_id:
        raise ValueError(f"Camera mxid not found: {mxid_1}")
    if mxid_2 not in by_id:
        raise ValueError(f"Camera mxid not found: {mxid_2}")

    return by_id[mxid_1], by_id[mxid_2]


class InspectionDetector:
    config_class = InspectionConfig
    requires_barcode = False

    def __init__(self, cfg=None, mxid_1=None, mxid_2=None):
        self.cfg = cfg if cfg is not None else self.config_class()
        self.mxid_1 = mxid_1
        self.mxid_2 = mxid_2

        self._device_info_1 = None
        self._device_info_2 = None
        self.camera_id_1 = None
        self.camera_id_2 = None

        # Per-camera (pipeline, device, rgb_queue) once opened.
        self._cams = []
        # Last returned frame sequence number per camera index, so _grab_frame
        # never returns the same buffered frame twice (avoids duplicate images).
        self._last_seq = {}

    # ----------------------------------------------------------- lifecycle

    @property
    def model_loaded(self):
        # No model to load; provided so callers can treat this like the other
        # detectors uniformly.
        return True

    def load_model(self):
        # Inspection has no local model; the OpenAI call happens at inspect().
        return

    def _build_rgb_pipeline(self, device):
        cfg = self.cfg
        pipeline = dai.Pipeline(device)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        output = cam.requestOutput(
            size=cfg.rgb_size,
            type=dai.ImgFrame.Type.NV12,
            resizeMode=dai.ImgResizeMode.CROP,
            fps=float(cfg.fps),
        )
        queue = output.createOutputQueue(maxSize=QUEUE_MAX_SIZE, blocking=False)
        return pipeline, queue

    def _open_device(self, device_info):
        if self.cfg.force_usb2:
            return dai.Device(device_info, dai.UsbSpeed.HIGH)
        return dai.Device(device_info)

    def open(self):
        if self._cams:
            return self

        self._last_seq = {}
        self._device_info_1, self._device_info_2 = resolve_camera_pair(
            self.mxid_1, self.mxid_2
        )
        self.camera_id_1 = _device_id(self._device_info_1)
        self.camera_id_2 = _device_id(self._device_info_2)

        # Open camera 1, then (optionally) wait, then camera 2 -- mirrors the
        # original script's staggered startup for USB stability.
        for index, device_info in enumerate(
            (self._device_info_1, self._device_info_2)
        ):
            if index == 1 and self.cfg.startup_delay_seconds > 0:
                time.sleep(self.cfg.startup_delay_seconds)
            device = self._open_device(device_info)
            pipeline, queue = self._build_rgb_pipeline(device)
            pipeline.start()
            self._cams.append({"device": device, "pipeline": pipeline, "queue": queue})

        return self

    def close(self):
        for cam in self._cams:
            try:
                cam["pipeline"].stop()
            except Exception as e:
                print(f"Error stopping inspection pipeline: {e}")
            try:
                cam["device"].close()
            except Exception as e:
                print(f"Error closing inspection device: {e}")
        self._cams = []
        self._last_seq = {}

    @property
    def is_open(self):
        return bool(self._cams)

    # ------------------------------------------------------------- capture

    def _grab_frame(self, cam_index):
        """Block for a genuinely fresh frame from the given camera (0 or 1).

        The DepthAI output queue is non-blocking with a small buffer, so two
        back-to-back get()s can hand back the SAME buffered ImgFrame. We skip any
        frame whose sequence number matches the last one returned for this camera
        so every saved image is a new sensor frame (no duplicate captures).
        """
        cam = self._cams[cam_index]
        while True:
            message = cam["queue"].get()
            if not isinstance(message, dai.ImgFrame):
                continue
            try:
                seq = message.getSequenceNum()
            except AttributeError:
                seq = None  # no seq API -> can't dedup, accept the frame
            if seq is not None and seq == self._last_seq.get(cam_index):
                continue  # same buffered frame as last grab -> wait for a new one
            self._last_seq[cam_index] = seq
            return message.getCvFrame()

    def capture(self, session_dir, count=None, should_abort=None,
                orientation_index=None, cam_indices=None):
        """Grab fresh frames from selected cameras and save them as JPEGs.

        For each selected camera (`cam_indices`, 0-based; defaults to all open
        cameras) `count` genuinely-fresh frames are grabbed and saved -- so with
        the default count=1 this is exactly one image per selected camera. When
        `orientation_index` is given the angle is encoded in each filename (so
        frames from different inspection orientations can be told apart and the
        files do not collide when capturing into the same session twice).

        Returns the list of saved image Paths (possibly empty if no camera is
        selected), or None if aborted.
        """
        if not self._cams:
            raise RuntimeError("InspectionDetector is not open; call open() first")

        if count is None:
            count = self.cfg.capture_count
        if cam_indices is None:
            cam_indices = list(range(len(self._cams)))

        session_dir = Path(session_dir)
        camera_ids = [self.camera_id_1, self.camera_id_2]
        quality = self.cfg.capture_jpeg_quality
        ori_tag = "" if orientation_index is None else f"ori{orientation_index:02d}_"
        saved_paths = []

        i = 0
        for cam_index in cam_indices:
            camera_dir = session_dir / f"camera_{cam_index + 1}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(count):
                if should_abort is not None and should_abort():
                    return None

                frame = self._grab_frame(cam_index)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = (
                    f"frame_{ori_tag}{i:04d}_camera{cam_index + 1}_"
                    f"{timestamp}_{camera_ids[cam_index]}.jpg"
                )
                path = camera_dir / filename
                saved = cv2.imwrite(
                    str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not saved:
                    raise RuntimeError(f"Failed to save inspection frame {i}.")
                saved_paths.append(path)
                i += 1

        return saved_paths

    def session_dir_for(self, session_id):
        """Absolute capture/analysis directory for a session id."""
        return Path(self.cfg.save_dir) / session_id

    def capture_into(self, session_id, orientation_index, count=None,
                     should_abort=None):
        """Capture frames for one orientation into a session, appending.

        Used by the granular CaptureFrames action: frames accumulate across
        calls under save_dir/<session_id>/ (the dir is NOT cleared), each tagged
        with `orientation_index`. A later analyze_session() runs one OpenAI
        inspection over every frame in the session.

        Returns the list of saved image Paths for this call, or None if aborted.
        """
        session_dir = self.session_dir_for(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        cam_indices = self.cfg.cameras_for(orientation_index, len(self._cams))
        return self.capture(
            session_dir,
            count=count,
            should_abort=should_abort,
            orientation_index=orientation_index,
            cam_indices=cam_indices,
        )

    def _session_frames(self, session_dir):
        """All captured JPEGs under a session dir, in stable sorted order."""
        session_dir = Path(session_dir)
        frames = sorted(session_dir.glob("camera_*/*.jpg"))
        return frames

    # ------------------------------------------------------------- inspect

    def inspect(self, product_name, count=None, should_abort=None,
                on_stage=None, request_id=None):
        """Capture frames from both cameras and run one OpenAI inspection.

        on_stage(stage) is called with "capturing" / "inspecting" if provided.
        Returns an InspectionResult, or None if aborted during capture.
        """
        if request_id is None:
            request_id = make_request_id()

        session_dir = Path(self.cfg.save_dir) / request_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if on_stage is not None:
            on_stage("capturing")
        image_paths = self.capture(
            session_dir, count=count, should_abort=should_abort
        )
        if image_paths is None:
            return None

        if on_stage is not None:
            on_stage("inspecting")
        run = run_inspection(
            self.cfg,
            image_paths,
            product_name,
            request_id=request_id,
            optimized_dir=session_dir / "optimized",
        )

        return InspectionResult.from_run(
            run, product_name=product_name, request_id=request_id,
            image_paths=image_paths,
        )

    def analyze_session(self, session_id, product_name, on_stage=None):
        """Run ONE OpenAI inspection over every frame already captured for a session.

        Generalizes the second half of inspect() to "all frames on disk": it does
        not capture; it globs save_dir/<session_id>/camera_*/ for the frames that
        prior capture_into() calls saved and inspects them together.

        on_stage(stage) is called with "inspecting" if provided.
        Returns an InspectionResult. Raises if the session has no frames.
        """
        session_dir = self.session_dir_for(session_id)
        image_paths = self._session_frames(session_dir)
        if not image_paths:
            raise RuntimeError(
                f"No captured frames found for session '{session_id}' "
                f"(looked under {session_dir})"
            )

        if on_stage is not None:
            on_stage("inspecting")
        run = run_inspection(
            self.cfg,
            image_paths,
            product_name,
            request_id=session_id,
            optimized_dir=session_dir / "optimized",
        )

        return InspectionResult.from_run(
            run, product_name=product_name, request_id=session_id,
            image_paths=image_paths,
        )

    # ------------------------------------------------------------- debug

    def save_debug(self, result, out_dir=None, timestamp=None):
        """Write a JSON of the verdict next to the captured images.

        The captured/optimized images are already on disk under cfg.save_dir;
        this writes an inspection_result.json summarizing the verdict and
        returns a dict of saved paths (always includes: dir, json).
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if out_dir is not None:
            save_dir = Path(out_dir)
        elif result.request_id:
            save_dir = Path(self.cfg.save_dir) / result.request_id
        else:
            save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json_path = save_dir / "inspection_result.json"
        with open(json_path, "w") as f:
            json.dump(result.to_json_dict(), f, indent=2)

        return {"dir": save_dir, "json": json_path}

    # ------------------------------------------------------- context manager

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
