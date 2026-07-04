"""Polymailer product-release monitoring mode (continuous, stateful).

Unlike the single-shot detectors, this mode watches the held polymailer over
many frames: it locks a baseline on the statically held bag, tracks the moving
slit with optical flow at camera FPS while SAM2 scans a moving corridor crop
in a background worker, and walks the prototype's state machine (emerging
candidate streak -> exit confirmed -> full-release verification -> latched
FULL PRODUCT OUT). The per-frame loop body is a port of main() from
polymailer_product_release.py; the interactive keys map to the monitor API:

  t/b/l/r slit side  -> start_monitoring(slit_side)
  SPACE lock baseline-> automatic after cfg.baseline_stable_updates
                        consecutive stable bag selections (the robot holds
                        the bag statically when the goal starts)
  x reset            -> reset()
  s save             -> save_event_snapshot() (called on phase transitions)
  q quit             -> stop_monitoring()
  cv2.imshow         -> render_live_view() (saved to disk, no GUI)

The caller (ROS action server or CLI) owns the frame loop:

    detector.start_monitoring("top")
    while ...:
        update = detector.process_frame(camera.get_frames())
    detector.stop_monitoring()
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from ..config import PolymailerReleaseConfig
from ..detection.package import mask_iou
from ..detection.polymailer_release import (
    build_baseline_static_signatures,
    candidate_matches_previous,
    choose_live_polymailer_refresh_mask,
    choose_polymailer_mask,
    copy_candidate,
    estimate_baseline_base_hsv,
    evaluate_full_release,
    find_emerging_product_candidates,
    find_released_product_candidates,
    make_polymailer_refresh_crop,
    make_product_verification_crop,
    make_slit_geometry,
    masked_mean_hsv,
    normalize_vector,
)
from ..detection.slit_tracker import (
    VALID_SLIT_SIDES,
    MovingSlitTracker,
    get_initial_slit_edge,
)
from ..results import PolymailerReleaseResult
from ..sam_worker import LiveSAMWorker
from ..visualization.polymailer_release import make_live_view, save_live_result
from .base import BaseDetector


def _configure_torch(device_name):
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


class ReleasePhase(str, Enum):
    FINDING_BAG = "finding_bag"
    BASELINE_LOCKED = "baseline_locked"
    TRACKING = "tracking"
    ARMED = "armed"
    EMERGING = "emerging"
    EXIT_CONFIRMED = "exit_confirmed"
    VERIFYING = "verifying"
    RELEASED = "released"
    TRACKER_LOST = "tracker_lost"


@dataclass
class MonitorUpdate:
    """One process_frame() outcome, ready to relay as action feedback."""

    phase: ReleasePhase
    slit_motion_px: float
    candidate_score: float
    released: bool
    state_text: str


class PolymailerReleaseDetector(BaseDetector):
    config_class = PolymailerReleaseConfig
    requires_barcode = False

    def __init__(self, cfg=None, mxid=None, torch_device=None, camera=None):
        super().__init__(
            cfg=cfg, mxid=mxid, torch_device=torch_device, camera=camera
        )
        self._slit_mask_generator = None
        self._device_name = None
        self._worker = None
        self._monitoring = False
        self._tracker = MovingSlitTracker(self.cfg)
        self._slit_side = self.cfg.default_slit_side
        self._latched_result = None
        self._phase = ReleasePhase.FINDING_BAG
        self._phase_history = []
        self._reset_state()

    # ----------------------------------------------------------- lifecycle

    def load_model(self):
        if self._mask_generator is not None:
            return

        device_name = self._torch_device
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_name = device_name

        print(f"Loading SAM 2 on device: {device_name}")
        _configure_torch(device_name)

        sam2_model = build_sam2(
            self.cfg.model_cfg,
            self.cfg.checkpoint,
            device=device_name,
        )

        # Two generators sharing one model: a full-ROI search for the flat/held
        # bag and a faster one for the moving slit/verify crops.
        self._mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=self.cfg.sam_points_per_side,
            pred_iou_thresh=self.cfg.sam_pred_iou_thresh,
            stability_score_thresh=self.cfg.sam_stability_score_thresh,
            min_mask_region_area=self.cfg.sam_min_mask_region_area,
        )
        self._slit_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=self.cfg.slit_sam_points_per_side,
            pred_iou_thresh=self.cfg.sam_pred_iou_thresh,
            stability_score_thresh=self.cfg.sam_stability_score_thresh,
            min_mask_region_area=self.cfg.sam_min_mask_region_area,
        )

    def detect(self, frames, barcode=None):
        raise NotImplementedError(
            "PolymailerReleaseDetector is a continuous monitor; drive it with "
            "start_monitoring()/process_frame()/stop_monitoring() instead."
        )

    # ------------------------------------------------------------- monitor

    def start_monitoring(self, slit_side=None):
        """Reset all state and start the SAM worker; phase -> FINDING_BAG."""
        if self._mask_generator is None:
            raise RuntimeError("Model not loaded; call load_model() first")

        slit_side = slit_side or self.cfg.default_slit_side
        if slit_side not in VALID_SLIT_SIDES:
            raise ValueError(
                f"Unsupported slit side: {slit_side!r} "
                f"(valid: {', '.join(VALID_SLIT_SIDES)})"
            )

        self.stop_monitoring()
        self._slit_side = slit_side
        self._tracker = MovingSlitTracker(self.cfg)
        self._reset_state()
        self._latched_result = None
        self._phase_history = []
        self._set_phase(ReleasePhase.FINDING_BAG)

        roi_size = (
            self.cfg.roi_x2 - self.cfg.roi_x1,
            self.cfg.roi_y2 - self.cfg.roi_y1,
        )
        self._worker = LiveSAMWorker(
            self._mask_generator,
            self._slit_mask_generator,
            self._device_name,
            roi_size,
            use_cuda_autocast=self.cfg.use_cuda_autocast,
        )
        self._worker.start()
        self._monitoring = True

    def stop_monitoring(self):
        """Stop the SAM worker; the latched result (if any) is kept."""
        self._monitoring = False
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def reset(self):
        """Re-arm within one monitoring session (the prototype's 'x' key)."""
        self._tracker.clear()
        self._reset_state()
        self._latched_result = None
        self._set_phase(ReleasePhase.FINDING_BAG)

    @property
    def monitoring(self):
        return self._monitoring

    @property
    def latched_result(self):
        """PolymailerReleaseResult once FULL PRODUCT OUT latched, else None."""
        return self._latched_result

    @property
    def phase(self):
        return self._phase

    @property
    def phase_history(self):
        return list(self._phase_history)

    def _set_phase(self, phase):
        if phase != self._phase:
            self._phase = phase
            self._phase_history.append((time.time(), phase.value))

    def _reset_state(self):
        # The prototype's 'x' reset block: every main()-local becomes state.
        self._baseline_mask = None
        self._baseline_roi_rgb = None
        self._baseline_poly_hsv = None
        self._baseline_base_hsv = None
        self._baseline_depth_roi = None
        self._baseline_static_signatures = []
        self._prebaseline_poly = None
        self._prebaseline_masks = []
        self._baseline_streak = 0
        self._baseline_slit_midpoint = None
        self._baseline_slit_length = None
        self._corridor_expansion_state = self._fresh_expansion_state()
        self._monitor_started_at = None
        self._monitor_armed = False
        self._slit_motion_px = 0.0
        self._latest_processed_sam_frame = None
        self._emerging_candidates = []
        self._latest_candidate = None
        self._previous_candidate = None
        self._candidate_streak = 0
        self._exit_confirmed = False
        self._exit_confirmed_at = None
        self._verification_candidate = None
        self._previous_verification_candidate = None
        self._full_out_streak = 0
        self._full_out_confirmed = False
        self._latched_product_candidate = None
        self._full_release_missing_updates = 0
        self._first_seen_printed = False
        self._slit_geometry = None
        self._pending_poly_refresh_frame = None
        self._last_poly_refresh_submit_time = 0.0
        self._poly_refresh_count = 0
        self._poly_refresh_fail_count = 0
        self._last_poly_refresh_score = None
        self._frame_count = 0
        self._latest_sam = None
        self._last_frame_bgr = None
        self._last_depth_full = None
        self._pending_phase_event = None

    @staticmethod
    def _fresh_expansion_state():
        return {
            "outward_extra_px": 0.0,
            "side_extra_px": 0.0,
            "inward_extra_px": 0.0,
            "pose_scale": 1.0,
            "pull_away_px": 0.0,
            "lateral_motion_px": 0.0,
            "total_motion_px": 0.0,
        }

    def _make_geometry(self):
        return make_slit_geometry(
            self.cfg,
            self._tracker.poly_mask,
            self._tracker.slit_points,
            baseline_slit_midpoint=self._baseline_slit_midpoint,
            baseline_slit_length=self._baseline_slit_length,
            expansion_state=self._corridor_expansion_state,
        )

    # -------------------------------------------------------- frame loop

    def process_frame(self, frames):
        """One iteration of the prototype main loop. Returns a MonitorUpdate."""
        if not self._monitoring:
            raise RuntimeError(
                "Monitor not running; call start_monitoring() first"
            )

        cfg = self.cfg
        self._frame_count += 1
        frame_bgr = frames.rgb
        self._last_frame_bgr = frame_bgr
        self._last_depth_full = frames.depth_class_aligned
        depth_roi = frames.depth_class_aligned[
            cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2
        ].copy()
        roi_bgr = frame_bgr[
            cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2
        ].copy()
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        if self._tracker.active:
            self._tracker.update(roi_bgr)
            self._slit_geometry = self._make_geometry()

            if (
                self._slit_geometry is not None
                and self._baseline_slit_midpoint is not None
            ):
                self._slit_motion_px = float(
                    np.linalg.norm(
                        self._slit_geometry["midpoint"]
                        - self._baseline_slit_midpoint
                    )
                )

            if self._monitor_started_at is not None:
                self._monitor_armed = (
                    time.time() - self._monitor_started_at
                    >= cfg.exit_arm_delay_seconds
                    and self._slit_motion_px
                    >= cfg.exit_arm_min_slit_motion_px
                )

        self._submit_sam(roi_rgb, roi_bgr, depth_roi)
        self._consume_sam(roi_bgr, roi_rgb, depth_roi)

        phase = self._resolve_phase()
        return MonitorUpdate(
            phase=phase,
            slit_motion_px=float(self._slit_motion_px),
            candidate_score=float(
                self._latest_candidate["score"]
                if self._latest_candidate is not None
                else 0.0
            ),
            released=self._full_out_confirmed,
            state_text=self._state_text(),
        )

    # ------------------------------------------------------ SAM submission

    def _submit_sam(self, roi_rgb, roi_bgr, depth_roi):
        cfg = self.cfg
        roi_width = cfg.roi_x2 - cfg.roi_x1
        roi_height = cfg.roi_y2 - cfg.roi_y1

        if self._baseline_mask is None:
            if self._frame_count % cfg.full_sam_interval_frames == 0:
                self._worker.submit(
                    frame_id=self._frame_count,
                    image_rgb=roi_rgb,
                    mode="full",
                    crop_box=(0, 0, roi_width, roi_height),
                    metadata={"roi_rgb": roi_rgb, "depth_roi": depth_roi},
                )
            return

        if not self._tracker.active:
            return

        refresh_due = (
            cfg.poly_refresh_enable
            and self._pending_poly_refresh_frame is None
            and time.monotonic() - self._last_poly_refresh_submit_time
            >= cfg.poly_refresh_interval_seconds
        )

        if refresh_due:
            refresh_crop = make_polymailer_refresh_crop(
                cfg, self._tracker.poly_mask
            )
            x1, y1, x2, y2 = refresh_crop
            refresh_rgb = roi_rgb[y1:y2, x1:x2]
            submitted = self._worker.submit(
                frame_id=self._frame_count,
                image_rgb=refresh_rgb,
                mode="poly_refresh",
                crop_box=refresh_crop,
                metadata={
                    "roi_rgb": roi_rgb,
                    "roi_bgr": roi_bgr,
                    "predicted_poly_mask": self._tracker.poly_mask,
                    "reference_slit_points": self._tracker.slit_points,
                    "crop_box": refresh_crop,
                },
            )

            if submitted:
                self._pending_poly_refresh_frame = self._frame_count
                self._last_poly_refresh_submit_time = time.monotonic()

        elif (
            self._pending_poly_refresh_frame is None
            and self._slit_geometry is not None
            and self._monitor_armed
            and not self._full_out_confirmed
            and self._frame_count % cfg.slit_sam_interval_frames == 0
        ):
            if (
                self._exit_confirmed
                and not self._full_out_confirmed
                and self._verification_candidate is not None
            ):
                crop_box = make_product_verification_crop(
                    cfg,
                    self._verification_candidate,
                    self._slit_geometry,
                )
                sam_mode = "verify"
            else:
                crop_box = self._slit_geometry["crop_box"]
                sam_mode = "slit"

            x1, y1, x2, y2 = crop_box
            crop_rgb = roi_rgb[y1:y2, x1:x2]
            self._worker.submit(
                frame_id=self._frame_count,
                image_rgb=crop_rgb,
                mode=sam_mode,
                crop_box=crop_box,
                metadata={
                    "roi_rgb": roi_rgb,
                    "depth_roi": depth_roi,
                    "poly_mask": self._tracker.poly_mask,
                    "slit_point_a": self._slit_geometry["point_a"],
                    "slit_point_b": self._slit_geometry["point_b"],
                    "corridor_mask": self._slit_geometry["corridor_mask"],
                    "contact_mask": self._slit_geometry["contact_mask"],
                    "midpoint": self._slit_geometry["midpoint"],
                    "tangent": self._slit_geometry["tangent"],
                    "outward": self._slit_geometry["outward"],
                    "corridor_polygon": self._slit_geometry["corridor_polygon"],
                    "contact_polygon": self._slit_geometry["contact_polygon"],
                    "crop_box": crop_box,
                    "reference_candidate": self._verification_candidate,
                },
            )

    # ---------------------------------------------------- SAM consumption

    def _consume_sam(self, roi_bgr, roi_rgb, depth_roi):
        latest_sam = self._worker.get_latest()
        self._latest_sam = latest_sam

        if (
            latest_sam is None
            or latest_sam["frame_id"] == self._latest_processed_sam_frame
        ):
            return

        self._latest_processed_sam_frame = latest_sam["frame_id"]

        if latest_sam["error"] is not None:
            if latest_sam.get("mode") == "poly_refresh":
                self._pending_poly_refresh_frame = None
                self._poly_refresh_fail_count += 1
            self._emerging_candidates = []
            return

        if self._baseline_mask is None and latest_sam["mode"] == "full":
            self._handle_full_result(latest_sam, roi_bgr, roi_rgb, depth_roi)
        elif (
            self._baseline_mask is not None
            and latest_sam["mode"] == "poly_refresh"
        ):
            self._handle_poly_refresh_result(latest_sam)
        elif (
            self._baseline_mask is not None
            and latest_sam["mode"] in ("slit", "verify")
        ):
            self._handle_detection_result(latest_sam)

    def _handle_full_result(self, latest_sam, roi_bgr, roi_rgb, depth_roi):
        """Find the held bag; auto-lock the baseline after a stable streak.

        This replaces the prototype's manual SPACE key: the robot already
        holds the bag statically when monitoring starts, so the baseline
        locks once cfg.baseline_stable_updates consecutive full-SAM results
        select mutually consistent bag masks.
        """
        cfg = self.cfg
        result_roi_rgb = latest_sam["metadata"]["roi_rgb"]
        previous_poly = self._prebaseline_poly
        self._prebaseline_poly = choose_polymailer_mask(
            cfg,
            latest_sam["masks"],
            result_roi_rgb,
        )
        self._prebaseline_masks = latest_sam["masks"]

        if self._prebaseline_poly is None:
            self._baseline_streak = 0
            return

        if (
            previous_poly is not None
            and mask_iou(
                self._prebaseline_poly["mask"], previous_poly["mask"]
            )
            >= cfg.baseline_stable_min_iou
        ):
            self._baseline_streak += 1
        else:
            self._baseline_streak = 1

        if self._baseline_streak >= cfg.baseline_stable_updates:
            self._lock_baseline(roi_bgr, roi_rgb, depth_roi)

    def _lock_baseline(self, roi_bgr, roi_rgb, depth_roi):
        """The prototype's SPACE block: save the baseline, start tracking."""
        cfg = self.cfg
        baseline_mask = self._prebaseline_poly["mask"].copy()

        initialized = self._tracker.initialize(
            roi_bgr,
            baseline_mask,
            self._slit_side,
        )

        if not initialized:
            print(
                "Could not initialize optical-flow tracking; retrying "
                "baseline lock on the next stable selection."
            )
            self._tracker.clear()
            self._baseline_streak = 0
            return

        self._baseline_mask = baseline_mask
        self._baseline_roi_rgb = roi_rgb.copy()
        self._baseline_depth_roi = depth_roi.copy()
        self._baseline_static_signatures = build_baseline_static_signatures(
            cfg,
            self._prebaseline_masks,
            self._baseline_mask,
            self._baseline_roi_rgb,
        )
        self._baseline_poly_hsv = masked_mean_hsv(
            self._baseline_roi_rgb,
            self._baseline_mask,
        )
        self._baseline_base_hsv = estimate_baseline_base_hsv(
            self._baseline_roi_rgb,
            self._baseline_mask,
        )

        self._baseline_slit_midpoint = None
        self._baseline_slit_length = None
        slit_geometry = self._make_geometry()
        self._baseline_slit_midpoint = (
            None if slit_geometry is None else slit_geometry["midpoint"].copy()
        )
        self._baseline_slit_length = (
            None
            if slit_geometry is None
            else float(
                np.linalg.norm(
                    slit_geometry["point_b"] - slit_geometry["point_a"]
                )
            )
        )

        self._corridor_expansion_state = self._fresh_expansion_state()
        self._slit_geometry = self._make_geometry()

        self._monitor_started_at = time.time()
        self._monitor_armed = False
        self._slit_motion_px = 0.0
        self._emerging_candidates = []
        self._latest_candidate = None
        self._previous_candidate = None
        self._candidate_streak = 0
        self._exit_confirmed = False
        self._exit_confirmed_at = None
        self._verification_candidate = None
        self._previous_verification_candidate = None
        self._full_out_streak = 0
        self._full_out_confirmed = False
        self._latched_product_candidate = None
        self._full_release_missing_updates = 0
        self._first_seen_printed = False
        self._pending_poly_refresh_frame = None
        self._last_poly_refresh_submit_time = time.monotonic()
        self._poly_refresh_count = 0
        self._poly_refresh_fail_count = 0
        self._last_poly_refresh_score = None

        self._pending_phase_event = ReleasePhase.BASELINE_LOCKED
        print()
        print("POLYMAILER BASELINE LOCKED, MOVING SLIT SAVED")
        print(f"  selected slit side:  {self._slit_side.upper()}")
        print(f"  polymailer area px:  {int(self._baseline_mask.sum())}")
        print(f"  optical-flow points: {len(self._tracker.points)}")
        print(
            "  saved static SAM masks: "
            f"{len(self._baseline_static_signatures)}"
        )
        print(
            "  Detection arms only after the slit moves "
            f"{cfg.exit_arm_min_slit_motion_px:.0f}px."
        )

    def _handle_poly_refresh_result(self, latest_sam):
        cfg = self.cfg
        metadata = latest_sam["metadata"]
        self._pending_poly_refresh_frame = None
        refresh_candidate = choose_live_polymailer_refresh_mask(
            cfg,
            masks=latest_sam["masks"],
            roi_rgb=metadata["roi_rgb"],
            predicted_poly_mask=metadata["predicted_poly_mask"],
            baseline_poly_hsv=self._baseline_poly_hsv,
            slit_points=metadata["reference_slit_points"],
            crop_box=metadata["crop_box"],
        )

        if refresh_candidate is None:
            self._poly_refresh_fail_count += 1
            return

        refreshed = self._tracker.refresh_from_sam(
            roi_bgr=metadata["roi_bgr"],
            refreshed_mask=refresh_candidate["mask"],
            reference_slit_points=metadata["reference_slit_points"],
        )

        if not refreshed:
            self._poly_refresh_fail_count += 1
            return

        self._poly_refresh_count += 1
        self._last_poly_refresh_score = refresh_candidate["score"]
        self._slit_geometry = self._make_geometry()

        if (
            self._poly_refresh_count == 1
            or self._poly_refresh_count % cfg.poly_refresh_print_every == 0
        ):
            print(
                "POLYMAILER SAM REFRESH APPLIED "
                f"(count={self._poly_refresh_count}, "
                f"score={refresh_candidate['score']:.3f}, "
                f"iou={refresh_candidate['iou']:.3f}, "
                f"source={self._tracker.last_transform_source})"
            )

    def _handle_detection_result(self, latest_sam):
        cfg = self.cfg
        metadata = latest_sam["metadata"]
        result_geometry = {
            "point_a": metadata["slit_point_a"],
            "point_b": metadata["slit_point_b"],
            "midpoint": metadata["midpoint"],
            "tangent": metadata.get(
                "tangent",
                normalize_vector(
                    metadata["slit_point_b"] - metadata["slit_point_a"]
                ),
            ),
            "outward": metadata["outward"],
            "corridor_mask": metadata["corridor_mask"],
            "contact_mask": metadata["contact_mask"],
            "corridor_polygon": metadata["corridor_polygon"],
            "contact_polygon": metadata["contact_polygon"],
            "crop_box": metadata["crop_box"],
        }

        if latest_sam["mode"] == "slit":
            self._emerging_candidates = find_emerging_product_candidates(
                cfg,
                masks=latest_sam["masks"],
                poly_mask=metadata["poly_mask"],
                slit_geometry=result_geometry,
                current_roi_rgb=metadata["roi_rgb"],
                baseline_roi_rgb=self._baseline_roi_rgb,
                baseline_poly_hsv=self._baseline_poly_hsv,
                baseline_base_hsv=self._baseline_base_hsv,
                baseline_poly_mask=self._baseline_mask,
                baseline_static_signatures=self._baseline_static_signatures,
                depth_roi=metadata.get("depth_roi"),
            )

            self._latest_candidate = (
                self._emerging_candidates[0]
                if self._emerging_candidates
                else None
            )

            if self._latest_candidate is None:
                self._candidate_streak = 0
                self._previous_candidate = None
                return

            if candidate_matches_previous(
                cfg,
                self._latest_candidate,
                self._previous_candidate,
            ):
                self._candidate_streak += 1
            else:
                self._candidate_streak = 1

            self._previous_candidate = self._latest_candidate

            if not self._first_seen_printed:
                print()
                print("PRODUCT SLIDING OUT")
                print(f"  mode:  {self._latest_candidate['mode']}")
                print(f"  score: {self._latest_candidate['score']:.3f}")
                print(
                    "  outward fraction: "
                    f"{self._latest_candidate['outward_fraction']:.3f}"
                )
                print(
                    "  height above base: "
                    f"{self._latest_candidate['height_above_base_mm']} mm"
                )
                self._first_seen_printed = True

            if (
                self._candidate_streak
                >= cfg.product_confirm_required_sam_updates
                and not self._exit_confirmed
            ):
                self._exit_confirmed = True
                self._exit_confirmed_at = time.time()
                self._verification_candidate = self._latest_candidate
                self._previous_verification_candidate = self._latest_candidate
                self._full_out_streak = 0
                self._full_out_confirmed = False
                self._latched_product_candidate = None
                self._full_release_missing_updates = 0
                self._pending_phase_event = ReleasePhase.EXIT_CONFIRMED
                print()
                print("PRODUCT EXIT CONFIRMED")
                print(
                    "  A separate product mask crossed the tracked moving "
                    "slit; now verifying full separation."
                )
            return

        # verify mode
        reference_candidate = metadata.get("reference_candidate")
        released_candidates = find_released_product_candidates(
            cfg,
            masks=latest_sam["masks"],
            reference_candidate=reference_candidate,
            poly_mask=metadata["poly_mask"],
            slit_geometry=result_geometry,
            current_roi_rgb=metadata["roi_rgb"],
            baseline_roi_rgb=self._baseline_roi_rgb,
            baseline_poly_hsv=self._baseline_poly_hsv,
            baseline_base_hsv=self._baseline_base_hsv,
            baseline_poly_mask=self._baseline_mask,
            baseline_static_signatures=self._baseline_static_signatures,
            depth_roi=metadata.get("depth_roi"),
        )
        self._emerging_candidates = released_candidates
        self._latest_candidate = (
            released_candidates[0] if released_candidates else None
        )

        if self._latest_candidate is None:
            self._full_release_missing_updates += 1
            self._full_out_streak = max(0, self._full_out_streak - 1)
            return

        (
            separated,
            on_table,
            stationary,
            center_step,
            stable_area_ratio,
        ) = evaluate_full_release(
            cfg,
            self._latest_candidate,
            self._previous_verification_candidate,
        )

        self._verification_candidate = self._latest_candidate
        self._previous_verification_candidate = self._latest_candidate
        self._full_release_missing_updates = 0

        confirm_delay_ok = (
            self._exit_confirmed_at is not None
            and time.time() - self._exit_confirmed_at
            >= cfg.full_release_min_confirm_delay_seconds
        )

        if separated and on_table and stationary and confirm_delay_ok:
            self._full_out_streak += 1
        else:
            self._full_out_streak = max(0, self._full_out_streak - 1)

        if (
            self._full_out_streak >= cfg.full_release_required_stable_updates
            and not self._full_out_confirmed
        ):
            self._full_out_confirmed = True
            self._latched_product_candidate = copy_candidate(
                self._latest_candidate
            )
            self._verification_candidate = self._latched_product_candidate
            self._previous_verification_candidate = (
                self._latched_product_candidate
            )
            self._latch_result()
            print()
            print("FULL PRODUCT OUT AND ON TABLE")
            print("  SAFE TO DISCARD POLYMAILER")
            print(
                "  final bag overlap: "
                f"{self._latched_product_candidate['poly_overlap']:.3f}"
            )
            print(
                "  final height above table: "
                f"{self._latched_product_candidate['height_above_base_mm']} mm"
            )
            print(f"  final center movement: {center_step} px")

    def _latch_result(self):
        frames = None
        intrinsics = None
        try:
            intrinsics = self.camera.intrinsics
        except Exception:
            pass

        # Rebuild a Frames-like view from the last processed frame so the
        # result can compute the product depth and save_debug can render.
        from ..camera import Frames

        if self._last_frame_bgr is not None:
            depth = (
                self._last_depth_full
                if self._last_depth_full is not None
                else np.zeros(self._last_frame_bgr.shape[:2], dtype=np.uint16)
            )
            frames = Frames(
                rgb=self._last_frame_bgr,
                depth_class_aligned=depth,
                depth_measure_aligned=depth,
            )

        self._latched_result = PolymailerReleaseResult.from_candidate(
            self.cfg,
            self._latched_product_candidate,
            frames,
            intrinsics,
            self._phase_history + [(time.time(), ReleasePhase.RELEASED.value)],
        )

    # ------------------------------------------------------------- phases

    def _resolve_phase(self):
        if self._pending_phase_event is not None:
            event = self._pending_phase_event
            self._pending_phase_event = None
            self._set_phase(event)
            return event

        if self._full_out_confirmed:
            phase = ReleasePhase.RELEASED
        elif self._baseline_mask is None:
            phase = ReleasePhase.FINDING_BAG
        elif not self._tracker.active:
            phase = ReleasePhase.TRACKER_LOST
        elif self._exit_confirmed:
            phase = ReleasePhase.VERIFYING
        elif not self._monitor_armed:
            phase = ReleasePhase.TRACKING
        elif self._emerging_candidates:
            phase = ReleasePhase.EMERGING
        else:
            phase = ReleasePhase.ARMED

        self._set_phase(phase)
        return phase

    def _state_text(self):
        cfg = self.cfg
        if self._baseline_mask is None:
            return f"FINDING HELD BAG [{self._slit_side.upper()}]"
        if self._full_out_confirmed:
            return "FULL PRODUCT OUT - DISCARD MAILER"
        if not self._tracker.active:
            return "TRACKER LOST"
        if self._exit_confirmed:
            candidate = self._verification_candidate
            if candidate is not None and (
                candidate.get("core_trailing_edge_clearance_px", 0.0)
                < cfg.full_release_min_core_trailing_clearance_px
                or candidate.get("core_poly_gap_px", 0.0)
                < cfg.full_release_min_core_poly_gap_px
            ):
                return "WAITING FOR PRODUCT TAIL TO CLEAR SLIT"
            return "VERIFYING FULL PRODUCT RELEASE"
        if not self._monitor_armed:
            return (
                "LIFT/PULL BAG TO ARM "
                f"({self._slit_motion_px:.0f}/"
                f"{cfg.exit_arm_min_slit_motion_px:.0f}px)"
            )
        if self._emerging_candidates:
            return "PRODUCT SLIDING OUT"
        return f"TRACKING MOVING {self._slit_side.upper()} SLIT"

    # ------------------------------------------------------ visualization

    def render_live_view(self, frames=None):
        """Render the prototype's annotated live view for the current state."""
        frame_bgr = frames.rgb if frames is not None else self._last_frame_bgr
        if frame_bgr is None:
            raise RuntimeError("No frame available to render")

        current_poly_mask = (
            self._tracker.poly_mask
            if self._tracker.active
            else (
                self._prebaseline_poly["mask"]
                if self._prebaseline_poly is not None
                else None
            )
        )

        display_slit_geometry = self._slit_geometry

        if self._baseline_mask is None and self._prebaseline_poly is not None:
            preview_slit_points = get_initial_slit_edge(
                self._prebaseline_poly["mask"],
                self._slit_side,
            )
            display_slit_geometry = make_slit_geometry(
                self.cfg,
                self._prebaseline_poly["mask"],
                preview_slit_points,
            )

        display_candidates = self._emerging_candidates
        if (
            self._full_out_confirmed
            and self._latched_product_candidate is not None
        ):
            display_candidates = [self._latched_product_candidate]

        return make_live_view(
            self.cfg,
            frame_bgr=frame_bgr,
            current_poly_mask=current_poly_mask,
            baseline_mask=self._baseline_mask,
            slit_geometry=display_slit_geometry,
            emerging_candidates=display_candidates,
            latest_sam=self._latest_sam,
            state=self._state_text(),
            frame_count=self._frame_count,
            flow_point_count=self._tracker.last_good_point_count,
            tracker_ok=self._tracker.last_transform_ok,
            candidate_streak=self._candidate_streak,
            exit_confirmed=self._exit_confirmed,
            full_out_streak=self._full_out_streak,
            full_out_confirmed=self._full_out_confirmed,
            selected_slit_side=self._slit_side,
        )

    def save_event_snapshot(self, frames=None, out_dir=None, tag=""):
        """Save the annotated live view + masks (the prototype's 's' key)."""
        display = self.render_live_view(frames)
        frame_bgr = frames.rgb if frames is not None else self._last_frame_bgr

        if self._full_out_confirmed and self._latched_product_candidate:
            best = self._latched_product_candidate
        else:
            best = (
                self._emerging_candidates[0]
                if self._emerging_candidates
                else None
            )

        current_poly_mask = (
            self._tracker.poly_mask if self._tracker.active else None
        )

        target = out_dir if out_dir is not None else self.cfg.save_dir
        if tag:
            target = Path(target) / tag

        return save_live_result(
            self.cfg,
            frame_bgr=frame_bgr,
            display_bgr=display,
            baseline_mask=self._baseline_mask,
            current_poly_mask=current_poly_mask,
            slit_geometry=self._slit_geometry,
            best_candidate=best,
            out_dir=target,
        )

    # ------------------------------------------------- base debug template

    def _draw_result(self, result):
        return self.render_live_view()

    def _serialize(self, result, timestamp=None):
        return {
            "timestamp": timestamp,
            "released": bool(result.released),
            "product_center_x_mm": result.product_center_x_mm,
            "product_center_y_mm": result.product_center_y_mm,
            "product_depth_mm": result.product_depth_mm,
            "height_above_base_mm": result.height_above_base_mm,
            "phase_history": result.phase_history,
        }
