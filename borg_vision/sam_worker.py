"""Background SAM2 worker for continuous (per-frame) monitoring modes.

Ported from polymailer_product_release.py (LiveSAMWorker). The monitor loop
runs optical-flow tracking at camera FPS while SAM2 inference runs at roughly
5 FPS; this worker decouples the two with a single background thread that
accepts at most one request in flight (a submit while busy is simply dropped,
so the caller always feeds the newest frame and no crop-copy backlog builds
up). Torch releases the GIL during inference, so the caller's frame loop and
any ROS executor threads keep running.
"""

import queue
import threading
import time

import cv2
import numpy as np
import torch


class LiveSAMWorker:
    """Runs full-ROI or moving-crop SAM in one background thread."""

    def __init__(self, full_generator, slit_generator, device_name,
                 roi_size, use_cuda_autocast=True):
        """
        full_generator:  SAM2AutomaticMaskGenerator for full-ROI scans
        slit_generator:  faster generator for moving slit/verify crops
        device_name:     "cuda" or "cpu"
        roi_size:        (roi_width, roi_height) masks are mapped back into
        use_cuda_autocast: try bfloat16 autocast on CUDA (falls back once on
                           failure)
        """
        self.full_generator = full_generator
        self.slit_generator = slit_generator
        self.device_name = device_name
        self.roi_size = tuple(roi_size)
        self.use_autocast = bool(
            use_cuda_autocast and device_name == "cuda"
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
        """Queue one SAM request; returns False when one is already in flight."""
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
            with self.lock:
                self.in_flight = False
            return False

        except Exception:
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
                masks = map_masks_to_full_roi(
                    local_masks, crop_box, self.roi_size
                )
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


def map_masks_to_full_roi(local_masks, crop_box, roi_size):
    roi_width, roi_height = roi_size
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
            (roi_height, roi_width),
            dtype=np.uint8,
        )
        full_mask[y1:y2, x1:x2] = local_mask

        mapped_data = dict(local_mask_data)
        mapped_data["segmentation"] = full_mask.astype(bool)
        mapped.append(mapped_data)

    return mapped
