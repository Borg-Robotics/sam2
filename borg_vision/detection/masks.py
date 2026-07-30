"""Mask primitives shared by more than one detection mode.

Deliberately dependency-light: cv2 + numpy only, and NO relative imports, so
the module can be loaded standalone by file path in tests (importing
`borg_vision` pulls torch and depthai).

`dilate_mask` lives here rather than in detection/package.py because
`masks_are_near` needs it and importing it back from `.package` would be
circular; detection/package.py re-exports it for polymailer and clear_bag.
"""

import cv2
import numpy as np


def dilate_mask(mask, radius_px):
    kernel_size = radius_px * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def convex_hull_mask(mask):
    """Filled convex hull over ALL external contours of `mask`.

    Unlike complete_min_area_rectangle (which snaps to a rotated rectangle),
    the hull follows the silhouette, so two adjacent faces of an angled box
    become one plausible box outline instead of an oversized rectangle.
    Returns None when the mask is empty.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    points = np.vstack(contours)
    hull = cv2.convexHull(points)
    completed = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(completed, hull, 1)

    return completed


def masks_are_near(mask_a, mask_b, radius_px):
    """Adjacency by dilate-and-intersect, independent of bbox geometry."""
    dilated_a = dilate_mask(mask_a, radius_px)

    return bool(np.any((dilated_a == 1) & (mask_b == 1)))


def completed_rect_metrics(union_mask, completed_mask):
    """Fill/size ratios of a min-area-rectangle completion.

    completed_fill_ratio guards against a rectangle that mostly bridges empty
    space; completed_area_ratio against one that swallows the whole ROI.
    """
    union_area = int(union_mask.sum())
    completed_area = int(completed_mask.sum())
    h, w = completed_mask.shape[:2]

    return {
        "union_area": union_area,
        "completed_area": completed_area,
        "completed_fill_ratio": float(union_area / max(completed_area, 1)),
        "completed_area_ratio": float(completed_area / max(h * w, 1)),
    }


def hull_union_metrics(mask_a, mask_b, roi_area):
    """Convex hull of two masks plus the gates the multi-face merge needs.

    Returns None when the hull cannot be built (both masks empty).
    """
    union_mask = np.logical_or(
        mask_a.astype(bool),
        mask_b.astype(bool),
    ).astype(np.uint8)

    hull_mask = convex_hull_mask(union_mask)

    if hull_mask is None:
        return None

    union_area = int(union_mask.sum())
    hull_area = int(hull_mask.sum())
    largest_area = max(int(mask_a.sum()), int(mask_b.sum()))

    return {
        "union_mask": union_mask,
        "hull_mask": hull_mask,
        "union_area": union_area,
        "hull_area": hull_area,
        "area_growth": float(hull_area / max(largest_area, 1)),
        "union_fill_ratio": float(union_area / max(hull_area, 1)),
        "hull_area_ratio": float(hull_area / max(roi_area, 1)),
    }
