"""Optical-flow tracking of the moving polymailer and its slit edge.

Ported from polymailer_product_release.py (MovingSlitTracker,
get_initial_slit_edge, transform_points_homography). All FLOW_* module
constants read from a PolymailerReleaseConfig. The tracker follows the bag
mask and the selected slit edge at camera FPS using LK optical flow with
similarity transforms only (a full projective homography could shear the bag
mask and rotate the slit across the middle of the polymailer); points close
to the physical slit get their own local transform. A periodic SAM refresh
re-anchors the mask via refresh_from_sam().
"""

import cv2
import numpy as np

from .package import largest_component
from .polymailer_release import clean_mask

# Image-relative edge of the flat polymailer that carries the cut.
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


def transform_points_homography(points, homography):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)


def get_initial_slit_edge(poly_mask, slit_side):
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


def _mask_center(mask):
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


class MovingSlitTracker:
    """Tracks the polymailer and slit without projective homography drift."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self.prev_gray = None
        self.points = None
        self.poly_mask = None
        self.slit_points = None
        self.slit_side = cfg.default_slit_side
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
        self.__init__(self.cfg)

    def initialize(self, roi_bgr, poly_mask, slit_side):
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
        self.active = (
            self.points is not None
            and len(self.points) >= self.cfg.flow_min_good_points
        )
        return self.active

    def _detect_points(self, gray, mask):
        feature_mask = cv2.erode(
            mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.flow_max_corners,
            qualityLevel=self.cfg.flow_quality_level,
            minDistance=self.cfg.flow_min_distance,
            mask=feature_mask,
            blockSize=self.cfg.flow_block_size,
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
            ransacReprojThreshold=self.cfg.flow_ransac_reproj_threshold,
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
            self.cfg.flow_min_frame_scale
            <= scale
            <= self.cfg.flow_max_frame_scale
            and abs(rotation_deg) <= self.cfg.flow_max_frame_rotation_deg
            and translation_px <= self.cfg.flow_max_frame_translation_px
            and inlier_ratio >= self.cfg.flow_min_affine_inlier_ratio
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
        center = _mask_center(self.poly_mask)

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
            (
                np.abs(along)
                <= slit_length * 0.5 + self.cfg.flow_local_slit_side_margin_px
            )
            & (normal >= -self.cfg.flow_local_slit_band_inward_px)
            & (normal <= self.cfg.flow_local_slit_band_outward_px)
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

            if parallel < self.cfg.flow_slit_snap_min_parallel:
                continue

            edge_midpoint = np.mean(edge, axis=0)
            midpoint_distance = float(
                np.linalg.norm(edge_midpoint - predicted_midpoint)
            )

            if midpoint_distance > self.cfg.flow_slit_snap_max_midpoint_distance_px:
                continue

            length_ratio = edge_length / max(predicted_length, 1.0)

            if not (
                self.cfg.flow_min_slit_length_ratio
                <= length_ratio
                <= self.cfg.flow_max_slit_length_ratio
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
            (1.0 - self.cfg.flow_slit_snap_blend) * predicted
            + self.cfg.flow_slit_snap_blend * best_edge
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

        if points is None or len(points) < self.cfg.flow_min_good_points:
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

        if self.points is None or len(self.points) < self.cfg.flow_min_good_points:
            self.points = self._detect_points(self.prev_gray, self.poly_mask)

        if self.points is None or len(self.points) < self.cfg.flow_min_good_points:
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
            winSize=tuple(self.cfg.flow_win_size),
            maxLevel=self.cfg.flow_max_level,
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
            winSize=tuple(self.cfg.flow_win_size),
            maxLevel=self.cfg.flow_max_level,
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
            & (fb_error <= self.cfg.flow_forward_backward_max_error)
        )

        old_good = self.points.reshape(-1, 2)[good]
        new_good = next_points.reshape(-1, 2)[good]
        self.last_good_point_count = len(old_good)

        if len(old_good) < self.cfg.flow_min_good_points:
            self.prev_gray = current_gray
            self.points = self._detect_points(current_gray, self.poly_mask)
            self.last_local_point_count = 0
            self.last_transform_ok = False
            self.last_transform_source = "too_few_good_points"
            return False

        global_affine, global_inlier_ratio = self._estimate_similarity(
            old_good, new_good
        )

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

        if len(local_old) >= self.cfg.flow_min_local_slit_points:
            local_affine, local_inlier_ratio = self._estimate_similarity(
                local_old, local_new
            )

        slit_affine = local_affine if local_affine is not None else global_affine
        transform_source = "local_slit" if local_affine is not None else "global_bag"

        roi_height, roi_width = self.poly_mask.shape[:2]
        updated_mask = cv2.warpAffine(
            self.poly_mask,
            global_affine,
            (roi_width, roi_height),
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

        predicted_slit = self._apply_affine(
            self.slit_points, slit_affine
        ).astype(np.float32)

        old_slit_length = float(
            np.linalg.norm(self.slit_points[1] - self.slit_points[0])
        )
        new_slit_length = float(
            np.linalg.norm(predicted_slit[1] - predicted_slit[0])
        )
        slit_length_ratio = new_slit_length / max(old_slit_length, 1.0)

        if not (
            self.cfg.flow_min_slit_length_ratio
            <= slit_length_ratio
            <= self.cfg.flow_max_slit_length_ratio
        ):
            predicted_slit = self._apply_affine(
                self.slit_points, global_affine
            ).astype(np.float32)
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
            len(self.points) < self.cfg.flow_reseed_point_count
            or self.frame_index % self.cfg.flow_reseed_interval_frames == 0
        ):
            reseeded = self._detect_points(current_gray, self.poly_mask)
            if reseeded is not None and len(reseeded) >= self.cfg.flow_min_good_points:
                self.points = reseeded

        return True
