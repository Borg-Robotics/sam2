"""Polymailer-mode detection logic: padded-envelope HSV mask scoring, depth
measurement and product-bulge detection.

Ported verbatim from run_polymailer()/polymailer_final.py with the module-level
constants replaced by fields of a PolymailerConfig passed as the first argument.
Generic, config-free helpers are reused from detection.package.
"""

import math

import cv2
import numpy as np
import torch

from ..visualization.polymailer import make_polymailer_depth_heatmap
from .package import (
    close_mask,
    dilate_mask,
    erode_mask,
    fit_reference_plane,
    get_mask_center,
    get_rotated_box_from_mask,
    largest_component,
    masked_gaussian_blur,
    pixel_to_camera_xy_mm,
)


def open_mask(mask, kernel_px=7, iterations=1):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )


def clean_mask(cfg, mask):
    original_area = int(mask.sum())

    if original_area <= 0:
        return mask.astype(np.uint8)

    cleaned = close_mask(
        mask,
        cfg.poly_mask_close_kernel_px,
        cfg.poly_mask_close_iterations,
    )
    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.poly_max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask.astype(np.uint8)


def get_component_near_center(mask, center):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return None

    cx, cy = center
    best_label = None
    best_score = None

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        bx = x + w / 2
        by = y + h / 2

        dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
        score = area - 2.0 * dist

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    return (labels == best_label).astype(np.uint8)


def center_depth_mm(cfg, depth_roi, mask, center_roi):
    cx, cy = center_roi
    h, w = depth_roi.shape[:2]

    radius = cfg.center_depth_radius_px

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    patch_depth = depth_roi[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    values = patch_depth[
        (patch_mask == 1)
        & (patch_depth > cfg.min_valid_depth_mm)
        & (patch_depth < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if values.size < cfg.min_center_depth_count:
        values = patch_depth[
            (patch_depth > cfg.min_valid_depth_mm)
            & (patch_depth < cfg.max_valid_depth_mm)
        ].astype(np.float32)

    if values.size < cfg.min_center_depth_count:
        return None, 0

    return float(np.median(values)), int(values.size)


def estimate_polymailer_depth_mm(cfg, polymailer_face_depth_mm):
    if polymailer_face_depth_mm is None:
        return None

    depth_mm = cfg.base_depth_mm - polymailer_face_depth_mm

    if depth_mm < 0:
        depth_mm = 0.0

    return float(depth_mm)


def estimate_polymailer_dimensions_mm(cfg, poly, intrinsics):
    empty = {
        "length_mm": None,
        "width_mm": None,
        "angle_deg": None,
        "points_full": None,
    }

    if intrinsics is None:
        return empty

    measure_mask = erode_mask(poly["mask"], cfg.poly_measure_erode_px)

    if int(measure_mask.sum()) < 100:
        measure_mask = poly["mask"].copy()

    rotated = get_rotated_box_from_mask(measure_mask)

    if rotated is None:
        return empty

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    w_px = rotated["width_px"]
    h_px = rotated["height_px"]

    # POLY_MEASURE_DEPTH_MM == BASE_DEPTH_MM in the original.
    raw_width_mm = (w_px * cfg.base_depth_mm) / fx
    raw_height_mm = (h_px * cfg.base_depth_mm) / fy

    length_mm = max(raw_width_mm, raw_height_mm) * cfg.poly_size_scale
    width_mm = min(raw_width_mm, raw_height_mm) * cfg.poly_size_scale

    points_roi = rotated["points_roi"]
    points_full = points_roi.copy()
    points_full[:, 0] += cfg.roi_x1
    points_full[:, 1] += cfg.roi_y1

    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "angle_deg": float(rotated["angle_deg"]),
        "points_full": points_full.tolist(),
    }


def estimate_product_inside_polymailer(cfg, depth_roi, poly_mask, poly_center_roi):
    inner_mask = erode_mask(poly_mask, cfg.poly_inner_erode_px)

    if int(inner_mask.sum()) < 100:
        inner_mask = poly_mask.copy()

    valid_inside = (
        (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
        & (inner_mask == 1)
    )

    inside_values = depth_roi[valid_inside].astype(np.float32)

    if inside_values.size < 100:
        return {
            "found": False,
            "reason": "not_enough_depth",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": None,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    # --- Reference surface: a fitted PLANE, not a global percentile ---------
    # The percentile this replaces assumes the mailer lies flat and level. It
    # does not -- it tilts and sags -- so a single depth value lands mid-slope,
    # and the raised half of a bare mailer then clears any bulge threshold on
    # its own. Fitting the surface per frame removes the mailer's own tilt
    # before anything is measured against it.
    #
    # The fit is trimmed rather than plain least-squares, which would be dragged
    # toward the product it is supposed to measure against and shrink the very
    # signal wanted. Ported from package mode 2026-08-18, where it fixed an empty
    # mailer reporting as a ~22% product.
    plane = fit_reference_plane(
        depth_roi,
        valid_inside,
        max_points=cfg.poly_product_max_plane_points,
        iterations=cfg.poly_product_plane_iterations,
        trim_sigma=cfg.poly_product_plane_trim_sigma,
    )

    if plane is None:
        return {
            "found": False,
            "reason": "no_reference_plane",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": None,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    surface_depth_mm = float(np.median(plane[valid_inside]))

    # Positive = raised toward the camera relative to the fitted surface.
    raw_elevation = np.where(
        valid_inside, plane - depth_roi.astype(np.float32), 0.0
    ).astype(np.float32)

    # Pixel scales derived from the mailer's own size, so one setting covers a
    # small mailer and a large one.
    poly_span_px = float(np.sqrt(max(int(poly_mask.sum()), 1)))
    smooth_px = max(2.0, poly_span_px * cfg.poly_product_smooth_frac)

    elevation, support = masked_gaussian_blur(
        raw_elevation, valid_inside, smooth_px, return_support=True
    )

    # What the blur removed is this frame's own sensor noise; the detection gate
    # is scaled to it rather than to a fixed millimetre value, so the same
    # setting holds as depth quality varies.
    high_pass = (raw_elevation - elevation)[valid_inside]
    noise_mm = float(1.4826 * np.median(np.abs(high_pass - np.median(high_pass))))
    noise_mm = max(noise_mm, 1e-3)

    inner_elevation = elevation[valid_inside]
    baseline_mm = float(
        np.percentile(inner_elevation, cfg.poly_product_baseline_percentile)
    )
    peak_mm = float(np.percentile(inner_elevation, cfg.poly_product_peak_percentile))
    dome_mm = peak_mm - baseline_mm

    if dome_mm < cfg.poly_product_min_peak_noise_multiple * noise_mm:
        return {
            "found": False,
            "reason": "no_dome_above_noise",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": surface_depth_mm,
            "product_depth_mm": None,
            "bulge_height_mm": float(dome_mm),
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    # Half-maximum contour of the dome. Being a fraction of the dome's OWN
    # height, it lands in the same relative place whatever the mailer stock --
    # padding spreads the same product into a lower, broader dome, which a fixed
    # poly_bulge_min_mm could not follow.
    threshold_mm = baseline_mm + cfg.poly_product_height_fraction * dome_mm

    # Judge on the smoothed field wherever the blur had support, not only where
    # a pixel had its own reading: the measurement stereo drops out in streaks
    # over low-texture kraft, and requiring per-pixel validity lets every streak
    # punch a notch into the product.
    well_supported = support >= cfg.poly_product_min_blur_support
    product_candidate = (
        (elevation >= threshold_mm) & (inner_mask == 1) & well_supported
    ).astype(np.uint8)

    # Open then close, at fractions of the smoothing scale. The old
    # open+close+DILATE inflated a sparse scatter into a solid blob; closing
    # after the open rejoins a dome split by a dropout streak without adding
    # area that was never measured.
    open_px = max(1, int(round(smooth_px * 0.5)))
    product_candidate = cv2.morphologyEx(
        product_candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1)),
    )
    close_px = max(1, int(round(smooth_px * cfg.poly_product_close_frac)))
    product_candidate = cv2.morphologyEx(
        product_candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)),
    )

    product_candidate[poly_mask == 0] = 0

    product_mask = get_component_near_center(product_candidate, poly_center_roi)

    if product_mask is None or int(product_mask.sum()) == 0:
        return {
            "found": False,
            "reason": "no_product_bulge",
            "mask": np.zeros_like(poly_mask, dtype=np.uint8),
            "surface_depth_mm": surface_depth_mm,
            "product_depth_mm": None,
            "bulge_height_mm": None,
            "area_ratio_of_poly": 0.0,
            "bbox": None,
            "center_roi": None,
        }

    product_area = int(product_mask.sum())
    poly_area = int(poly_mask.sum())
    area_ratio_of_poly = product_area / max(poly_area, 1)

    if area_ratio_of_poly < cfg.min_product_area_ratio_of_poly:
        found = False
        reason = "product_area_too_small"
    elif area_ratio_of_poly > cfg.max_product_area_ratio_of_poly:
        found = False
        reason = "product_area_too_large"
    else:
        found = True
        reason = "ok"

    product_values = depth_roi[
        (product_mask == 1)
        & (depth_roi > cfg.min_valid_depth_mm)
        & (depth_roi < cfg.max_valid_depth_mm)
    ].astype(np.float32)

    if product_values.size > 0:
        product_depth_mm = float(np.percentile(product_values, 20))
        bulge_height_mm = float(surface_depth_mm - product_depth_mm)
    else:
        product_depth_mm = None
        bulge_height_mm = None

    # Centre from the TOP of the dome only, not from the half-maximum mask.
    # product_mask deliberately spans down to half the dome height so it covers
    # the product's extent, but that band includes the film sloping off the
    # product, and the slope is rarely symmetric -- its centroid drifts toward
    # the shallower side and the cup lands off the product. See
    # cfg.poly_product_center_height_fraction.
    center_threshold_mm = (
        baseline_mm + cfg.poly_product_center_height_fraction * dome_mm
    )
    center_candidate = (
        (elevation >= center_threshold_mm) & (product_mask == 1) & well_supported
    ).astype(np.uint8)

    center_source = "dome_top"
    center_mask = get_component_near_center(center_candidate, poly_center_roi)

    if center_mask is None or int(center_mask.sum()) == 0:
        # Nothing survived the top slice (a flat, broad product, or a dome eaten
        # by dropouts). The half-maximum centroid is the weaker answer but it is
        # the one this code returned before, so fall back rather than fail.
        center_mask = product_mask
        center_source = "half_max_fallback"

    center_roi = get_mask_center(center_mask)

    if center_roi is None:
        center_roi = get_mask_center(product_mask)
        center_source = "half_max_fallback"

    # Extent, area and depth all still come from the half-maximum mask.
    bbox = cv2.boundingRect(product_mask.astype(np.uint8))

    return {
        "found": bool(found),
        "reason": reason,
        "mask": product_mask.astype(np.uint8),
        "surface_depth_mm": surface_depth_mm,
        "product_depth_mm": product_depth_mm,
        "bulge_height_mm": bulge_height_mm,
        "area_ratio_of_poly": float(area_ratio_of_poly),
        "bbox": bbox,
        "center_roi": center_roi,
        # "dome_top" when the centre came from the top slice, or
        # "half_max_fallback" when that slice was empty and the old
        # half-maximum centroid was used instead.
        "center_source": center_source,
        "center_area_px": int(center_mask.sum()),
    }


def choose_cut_side(cfg, poly_mask, product, intrinsics, depth_mm):
    """Pick which side to cut across (horizontal cut).

    Compares the empty space between the product and the TOP edge of the
    polymailer vs the product and the BOTTOM edge, and recommends cutting on
    whichever side has the most clearance from the product -- unless
    cfg.force_cut_side pins it to one end ("TOP"/"BOTTOM"; "" = automatic).
    """
    result = {
        "cut_side": None,
        "top_gap_px": 0,
        "bottom_gap_px": 0,
        "top_gap_mm": None,
        "bottom_gap_mm": None,
        "cut_row_full": None,
    }

    product_mask = product["mask"]

    if int(product_mask.sum()) == 0:
        return result

    poly_rows = np.where(poly_mask.any(axis=1))[0]
    prod_rows = np.where(product_mask.any(axis=1))[0]

    if poly_rows.size == 0 or prod_rows.size == 0:
        return result

    poly_top = int(poly_rows[0])
    poly_bottom = int(poly_rows[-1])
    prod_top = int(prod_rows[0])
    prod_bottom = int(prod_rows[-1])

    top_gap_px = max(prod_top - poly_top, 0)
    bottom_gap_px = max(poly_bottom - prod_bottom, 0)

    # force_cut_side overrides the clearance comparison entirely; the gaps are
    # still measured and reported either way. Grasp side follows as the
    # opposite end downstream, so forcing TOP also pins the cup to the bottom.
    forced = (getattr(cfg, "force_cut_side", "") or "").strip().upper()
    if forced == "TOP" or (forced != "BOTTOM" and top_gap_px >= bottom_gap_px):
        cut_side = "TOP"
        cut_row_roi = (poly_top + prod_top) // 2
    else:
        cut_side = "BOTTOM"
        cut_row_roi = (prod_bottom + poly_bottom) // 2

    if intrinsics is not None and depth_mm is not None:
        fy = intrinsics["fy"]
        result["top_gap_mm"] = float(top_gap_px * depth_mm / fy)
        result["bottom_gap_mm"] = float(bottom_gap_px * depth_mm / fy)

    result["cut_side"] = cut_side
    result["top_gap_px"] = int(top_gap_px)
    result["bottom_gap_px"] = int(bottom_gap_px)
    result["cut_row_full"] = int(cfg.roi_y1 + cut_row_roi)

    return result


def measure_edge_points(cfg, poly, dimensions, depth_roi, face_depth_mm, intrinsics):
    """Camera-frame X/Y/Z of the midpoints of the top and bottom lines of the
    polymailer's bounding box (the rotated outline drawn on screen), so the
    points sit centered on those edge lines."""
    points_full = dimensions.get("points_full")

    if points_full is not None and len(points_full) == 4:
        pts = sorted(points_full, key=lambda p: p[1])
        top_edge_full = (
            int(round((pts[0][0] + pts[1][0]) / 2)),
            int(round((pts[0][1] + pts[1][1]) / 2)),
        )
        bottom_edge_full = (
            int(round((pts[2][0] + pts[3][0]) / 2)),
            int(round((pts[2][1] + pts[3][1]) / 2)),
        )
    else:
        bx, by, bw, bh = poly["bbox"]
        cx_full = cfg.roi_x1 + int(bx + bw / 2)
        top_edge_full = (cx_full, cfg.roi_y1 + int(by))
        bottom_edge_full = (cx_full, cfg.roi_y1 + int(by + bh - 1))

    top_edge_roi = (top_edge_full[0] - cfg.roi_x1, top_edge_full[1] - cfg.roi_y1)
    bottom_edge_roi = (bottom_edge_full[0] - cfg.roi_x1, bottom_edge_full[1] - cfg.roi_y1)

    top_edge_depth, _ = center_depth_mm(cfg, depth_roi, poly["mask"], top_edge_roi)
    bottom_edge_depth, _ = center_depth_mm(cfg, depth_roi, poly["mask"], bottom_edge_roi)

    if top_edge_depth is None:
        top_edge_depth = face_depth_mm
    if bottom_edge_depth is None:
        bottom_edge_depth = face_depth_mm

    top_edge_x_mm, top_edge_y_mm = pixel_to_camera_xy_mm(
        top_edge_full, top_edge_depth, intrinsics
    )
    bottom_edge_x_mm, bottom_edge_y_mm = pixel_to_camera_xy_mm(
        bottom_edge_full, bottom_edge_depth, intrinsics
    )

    return {
        "top_x_mm": top_edge_x_mm,
        "top_y_mm": top_edge_y_mm,
        "top_z_mm": top_edge_depth,
        "top_point_full": top_edge_full,
        "bottom_x_mm": bottom_edge_x_mm,
        "bottom_y_mm": bottom_edge_y_mm,
        "bottom_z_mm": bottom_edge_depth,
        "bottom_point_full": bottom_edge_full,
    }


def _patch_plane(cfg, depth_roi, cx, cy, radius,
                 min_valid_px=None, min_valid_frac=None, min_depth_levels=None):
    """Fit a plane to the valid depth in a disc. Returns (rms_mm, tilt_deg, count).

    Only RELATIVE depth inside the patch matters here, so absolute stereo range
    error at working distance does not affect the result.

    The three sparsity guards default to this module's poly_grasp_* settings but
    can be passed explicitly, so a config without those fields (ObjectConfig, via
    the object-mode grasp scorer) can share this helper rather than copy it.
    """
    if min_valid_px is None:
        min_valid_px = cfg.poly_grasp_min_valid_px
    if min_valid_frac is None:
        min_valid_frac = cfg.poly_grasp_min_valid_frac
    if min_depth_levels is None:
        min_depth_levels = cfg.poly_grasp_min_depth_levels
    h, w = depth_roi.shape[:2]
    x1, x2 = max(cx - radius, 0), min(cx + radius + 1, w)
    y1, y2 = max(cy - radius, 0), min(cy + radius + 1, h)
    patch = depth_roi[y1:y2, x1:x2].astype(np.float32)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    valid = (
        disc
        & (patch > cfg.min_valid_depth_mm)
        & (patch < cfg.max_valid_depth_mm)
    )
    count = int(valid.sum())
    if count < min_valid_px:
        return None, None, count

    # Reject patches the sensor barely resolved. Without these two guards a patch
    # where stereo returned one constant value scores rms=0 -- a PERFECT flatness
    # score for the region the camera understood least. Absence of measurement is
    # not evidence of flatness.
    if count < min_valid_frac * disc.sum():
        return None, None, count
    if np.unique(patch[valid]).size < min_depth_levels:
        return None, None, count

    px = xx[valid].astype(np.float32)
    py = yy[valid].astype(np.float32)
    pz = patch[valid]

    A = np.column_stack([px - cx, py - cy, np.ones_like(px)])
    coef, *_ = np.linalg.lstsq(A, pz, rcond=None)
    rms = float(np.sqrt(np.mean((pz - A @ coef) ** 2)))
    tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    return rms, tilt, count


def score_grasp_candidates(
    cfg,
    poly,
    product,
    poly_center_full,
    edges,
    depth_roi,
    face_depth_mm,
    intrinsics,
    side=None,
):
    """Rank suction-cup grasp points in one region of the polymailer.

    `side` is "bottom" or "top" -- the END to grasp, chosen by the caller as the
    opposite of the cut side so the cup never sits on the opening the product
    slides out of. The search covers that half and centres on its geometric
    release-grasp point.

    The single release-grasp point is a fixed geometric midpoint, so when it lands
    on a crease or the shoulder of the product bulge there is nothing to fall back
    to. This scores a grid of nearby positions on surface quality and returns them
    best-first, so a failed grasp can retry somewhere sensible.

    Scored per candidate, over a disc the size of the real suction cup:
      - plane-fit RMS residual  -> roughness (creases, bulge shoulder)
      - plane tilt              -> how badly a straight-down cup would land askew
      - distance from the bulge boundary -> the dome's edge is the worst case
      - distance from the nominal pick   -> stay put unless the surface says move

    Deliberately not scored: RGB texture. Print and labels are high-contrast but
    perfectly good to grab; creases show in depth, printing does not.

    Returns a list of dicts with camera-frame x/y/z in mm, plus the raw terms, so
    the caller can see why something ranked where it did.
    """
    if side not in ("bottom", "top"):
        return []

    poly_mask = (poly["mask"] > 0).astype(np.uint8)
    product_mask = product.get("mask")
    if product_mask is None:
        product_mask = np.zeros_like(poly_mask)
    product_mask = (product_mask > 0).astype(np.uint8)

    if poly_center_full is None:
        return []

    rows_all = np.nonzero(poly_mask)[0]
    if rows_all.size == 0:
        return []
    row_min, row_max = int(rows_all.min()), int(rows_all.max())
    centre_row = poly_center_full[1] - cfg.roi_y1

    # Region gate + where the search is centred. "top"/"bottom" follow the
    # library's image-row convention (bottom = larger rows), matching cut_side.
    edge_full = edges.get(f"{side}_point_full")
    if edge_full is None:
        return []
    if side == "bottom":
        row_lo, row_hi = centre_row, float(row_max)
    else:
        row_lo, row_hi = float(row_min), centre_row
    # Anchor is set after mm_per_px is known -- see below.
    nx = ny = None

    # Scale from the mailer's own measured span, so the cup size in mm converts to
    # pixels without needing a separate calibration.
    span_px = float(row_max - row_min)
    span_mm = abs(edges.get("bottom_y_mm", 0.0) - edges.get("top_y_mm", 0.0))
    if span_px <= 0 or span_mm <= 0:
        return []
    mm_per_px = span_mm / span_px

    # Anchor the search near the mailer's outer end rather than at the geometric
    # midpoint. Grabbing further out gives a longer clear slope for the product to
    # slide down and keeps the cup well away from the bulge -- measurably better on
    # hardware. The setback stops it hugging the edge, where the film goes floppy
    # and the cup would half-overhang; w_edge and the hard margin back that up.
    if nx is None:
        edge_roi = (edge_full[0] - cfg.roi_x1, edge_full[1] - cfg.roi_y1)
        centre_roi = (poly_center_full[0] - cfg.roi_x1, centre_row)
        dx = centre_roi[0] - edge_roi[0]
        dy = centre_roi[1] - edge_roi[1]
        norm_d = math.hypot(dx, dy)
        if norm_d < 1e-6:
            nx, ny = edge_roi
        else:
            setback_px = cfg.poly_grasp_edge_setback_mm / mm_per_px
            setback_px = min(setback_px, norm_d)   # never past the mailer centre
            nx = int(round(edge_roi[0] + dx / norm_d * setback_px))
            ny = int(round(edge_roi[1] + dy / norm_d * setback_px))
    end_row = edge_roi[1]

    radius_px = max(int(round((cfg.poly_grasp_cup_diameter_mm / 2.0) / mm_per_px)), 3)
    margin_px = int(round(cfg.poly_grasp_edge_margin_mm / mm_per_px))
    max_offset_px = cfg.poly_grasp_max_offset_mm / mm_per_px

    # Distance along the mailer from the grasped end, as a BAND rather than a
    # minimum. Too near the end and the film is floppy under the cup; too far in
    # and the cup ends up over the product with a short slope for it to slide
    # down. The isotropic d_edge gate below cannot express this -- it treats the
    # side edges the same as the end -- so the two are enforced separately, which
    # is also what lets the sides be more permissive than the end.
    band_lo_px = cfg.poly_grasp_end_band_min_mm / mm_per_px
    band_hi_px = cfg.poly_grasp_end_band_max_mm / mm_per_px

    d_edge = cv2.distanceTransform(poly_mask, cv2.DIST_L2, 5)
    # Distance FROM the bulge boundary, inside or out. Landing squarely on the
    # product is fine -- it is the boundary itself, where the cup straddles the
    # dome's shoulder and rocks, that scores badly. This is a soft term only: a
    # hard "must clear the bulge" gate was tried and reverted, because it squeezed
    # candidates onto the sloping back lip and left only one or two of them.
    d_bulge = np.maximum(
        cv2.distanceTransform(1 - product_mask, cv2.DIST_L2, 5),
        cv2.distanceTransform(product_mask, cv2.DIST_L2, 5),
    )

    h, w = poly_mask.shape
    step = max(cfg.poly_grasp_grid_step_px, 1)
    cands = []
    for y in range(radius_px, h - radius_px, step):
        if y < row_lo or y > row_hi:
            continue                          # outside the requested region
        for x in range(radius_px, w - radius_px, step):
            if d_edge[y, x] < radius_px + margin_px:
                continue                      # cup fully on material
            if not (band_lo_px <= abs(y - end_row) <= band_hi_px):
                continue                      # outside the end-distance band
            # Split the offset by axis. Rows run along the mailer's length (the
            # axis bottom/top are defined on) and columns across its width, and
            # the two want opposite things: stay centred across the width, where
            # drifting sideways only moves the cup toward an edge for nothing,
            # but reach freely along the length, where getting nearer the end
            # buys a longer clear slope for the product and more distance from
            # the bulge. So they are weighted separately rather than as one
            # isotropic distance.
            d_row = float(abs(y - ny))
            d_col = float(abs(x - nx))
            offset = float(np.hypot(d_col, d_row))
            if offset > max_offset_px:
                continue
            rms, tilt, count = _patch_plane(cfg, depth_roi, x, y, radius_px)
            if rms is None:
                continue
            cands.append({
                "x_roi": x, "y_roi": y, "rms_mm": rms, "tilt_deg": tilt,
                "valid_px": count, "offset_mm": offset * mm_per_px,
                "off_col_mm": d_col * mm_per_px,
                "off_row_mm": d_row * mm_per_px,
                "bulge_dist_px": float(d_bulge[y, x]),
                "edge_dist_px": float(d_edge[y, x]),
            })

    if not cands:
        return []

    def norm(key, invert):
        v = np.array([c[key] for c in cands], dtype=np.float32)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-6:
            return np.full_like(v, 0.5)
        s = (v - lo) / (hi - lo)
        return 1.0 - s if invert else s

    s_rms = norm("rms_mm", True)
    s_tilt = norm("tilt_deg", True)
    s_bulge = norm("bulge_dist_px", False)
    s_across = norm("off_col_mm", True)   # across the width -- stay centred
    s_along = norm("off_row_mm", True)    # along the length -- weak, so it can
                                          #   reach toward the end edge
    s_edge = norm("edge_dist_px", False)

    for i, c in enumerate(cands):
        c["score"] = float(
            cfg.poly_grasp_w_rough * s_rms[i]
            + cfg.poly_grasp_w_tilt * s_tilt[i]
            + cfg.poly_grasp_w_bulge * s_bulge[i]
            + cfg.poly_grasp_w_across * s_across[i]
            + cfg.poly_grasp_w_along * s_along[i]
            + cfg.poly_grasp_w_edge * s_edge[i]
        )

    cands.sort(key=lambda c: -c["score"])

    # Non-max suppression so the list is genuinely alternative spots, not a
    # cluster of neighbouring pixels that would all fail the same way.
    kept = []
    min_sep = radius_px * 2
    for c in cands:
        if all((c["x_roi"] - k["x_roi"]) ** 2 + (c["y_roi"] - k["y_roi"]) ** 2
               > min_sep * min_sep for k in kept):
            kept.append(c)
        if len(kept) >= cfg.poly_grasp_max_candidates:
            break

    for c in kept:
        full = (c["x_roi"] + cfg.roi_x1, c["y_roi"] + cfg.roi_y1)
        # Same depth sampling as every other measured point in this module, with
        # the same face-depth fallback when the local patch is too sparse.
        depth, _ = center_depth_mm(
            cfg, depth_roi, poly["mask"], (c["x_roi"], c["y_roi"]))
        if depth is None:
            depth = face_depth_mm
        x_mm, y_mm = pixel_to_camera_xy_mm(full, depth, intrinsics)
        c["x_mm"], c["y_mm"], c["z_mm"] = x_mm, y_mm, depth
        c["point_full"] = full

    return kept


def choose_polymailer_mask(cfg, masks, roi_rgb):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    best = None

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(cfg, raw_mask)

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.min_poly_area_ratio:
            continue

        if area_ratio > cfg.max_poly_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = area / max(bw * bh, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.min_poly_rectangularity:
            continue

        if aspect_ratio > cfg.max_poly_aspect_ratio:
            continue

        masked_hsv = hsv[mask == 1]

        if masked_hsv.size == 0:
            continue

        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))

        if mean_v < cfg.min_poly_value:
            continue

        hue_score = 1.0 - min(abs(mean_h - 18.0) / 25.0, 1.0)
        sat_score = 1.0 - min(abs(mean_s - 70.0) / 90.0, 1.0)
        val_score = 1.0 - min(abs(mean_v - 170.0) / 100.0, 1.0)

        color_score = 0.50 * hue_score + 0.25 * sat_score + 0.25 * val_score

        if color_score < cfg.min_poly_color_score:
            continue

        cx = x + bw / 2
        cy = y + bh / 2

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)

        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_poly_area_ratio) / cfg.target_poly_area_ratio,
            1.0,
        )

        score = (
            cfg.poly_color_score_weight * color_score
            + cfg.poly_rect_score_weight * rectangularity
            + cfg.poly_area_score_weight * area_score
            + cfg.poly_center_score_weight * center_score
        )

        candidate = {
            "index": i,
            "score": float(score),
            "mask": mask,
            "bbox": (x, y, bw, bh),
            "area_ratio": float(area_ratio),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "center_score": float(center_score),
            "area_score": float(area_score),
            "color_score": float(color_score),
            "mean_hsv": (mean_h, mean_s, mean_v),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def run_sam2_polymailer(cfg, frame_bgr, depth_aligned, mask_generator, intrinsics):
    """Run SAM2 on the ROI, pick the best polymailer mask, measure its face
    depth + dimensions, and find any product bulge inside. Returns a result
    dict or None when no valid polymailer mask is found."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi = depth_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    poly = choose_polymailer_mask(cfg, masks, roi_rgb)

    if poly is None:
        return None

    poly_center_roi = get_mask_center(poly["mask"])

    if poly_center_roi is None:
        return None

    polymailer_face_depth_mm, depth_count = center_depth_mm(
        cfg,
        depth_roi,
        poly["mask"],
        poly_center_roi,
    )

    polymailer_depth_mm = estimate_polymailer_depth_mm(cfg, polymailer_face_depth_mm)

    dimensions = estimate_polymailer_dimensions_mm(cfg, poly, intrinsics)

    product = estimate_product_inside_polymailer(
        cfg,
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        poly_center_roi=poly_center_roi,
    )

    cut = choose_cut_side(
        cfg,
        poly_mask=poly["mask"],
        product=product,
        intrinsics=intrinsics,
        depth_mm=polymailer_face_depth_mm,
    )

    edges = measure_edge_points(
        cfg,
        poly=poly,
        dimensions=dimensions,
        depth_roi=depth_roi,
        face_depth_mm=polymailer_face_depth_mm,
        intrinsics=intrinsics,
    )

    depth_heatmap = make_polymailer_depth_heatmap(
        cfg,
        depth_roi=depth_roi,
        poly_mask=poly["mask"],
        product_mask=product["mask"],
        surface_depth_mm=product["surface_depth_mm"],
    )

    poly_center_full = (
        cfg.roi_x1 + poly_center_roi[0],
        cfg.roi_y1 + poly_center_roi[1],
    )

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        poly_center_full,
        polymailer_face_depth_mm,
        intrinsics,
    )

    # Grasp the END OPPOSITE the cut, so the cup is never sitting on the opening
    # the product has to slide out of. The cut side is already chosen from the
    # product's own clearance, so this needs no separate input -- and no caller
    # can pick the two inconsistently.
    #
    # cut_side is None when no product bulge was found, in which case there is no
    # basis for either choice and the grasp is left empty rather than guessed.
    grasp_side = {"TOP": "bottom", "BOTTOM": "top"}.get(cut["cut_side"])

    # Computed while the mailer is still closed: after cutting, the bulge mask is
    # unreliable and these inputs are gone.
    grasp_candidates = score_grasp_candidates(
        cfg,
        poly=poly,
        product=product,
        poly_center_full=poly_center_full,
        edges=edges,
        depth_roi=depth_roi,
        face_depth_mm=polymailer_face_depth_mm,
        intrinsics=intrinsics,
        side=grasp_side,
    )[:cfg.poly_grasp_max_candidates]

    # The point to pick at, then two retries. When nothing could be scored -- no
    # cut side to derive an end from, or depth too sparse to fit a plane -- fall
    # back to the mailer's own centre, which is always measured. It is a worse
    # pick than a scored one (it sits over the product bulge more often than not)
    # but it is on the mailer and it is never missing.
    best = grasp_candidates[0] if grasp_candidates else None
    grasp_x_mm = best["x_mm"] if best else center_x_mm
    grasp_y_mm = best["y_mm"] if best else center_y_mm
    grasp_z_mm = best["z_mm"] if best else polymailer_face_depth_mm

    if product["center_roi"] is not None:
        product_center_full = (
            cfg.roi_x1 + product["center_roi"][0],
            cfg.roi_y1 + product["center_roi"][1],
        )
    else:
        product_center_full = None

    product_inside_center_face_depth = product["product_depth_mm"]

    product_inside_center_x_mm, product_inside_center_y_mm = pixel_to_camera_xy_mm(
        product_center_full,
        product_inside_center_face_depth,
        intrinsics,
    )

    if product["bbox"] is not None:
        px, py, pw, ph = product["bbox"]
        product_bbox_full = (
            cfg.roi_x1 + px,
            cfg.roi_y1 + py,
            pw,
            ph,
        )
    else:
        product_bbox_full = None

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "poly": poly,
        "poly_center_roi": poly_center_roi,
        "poly_center_full": poly_center_full,
        "polymailer_face_depth_mm": polymailer_face_depth_mm,
        "polymailer_depth_mm": polymailer_depth_mm,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "depth_count": depth_count,
        "dimensions": dimensions,
        "product": product,
        "product_center_full": product_center_full,
        "product_bbox_full": product_bbox_full,
        "product_inside_center_x_mm": product_inside_center_x_mm,
        "product_inside_center_y_mm": product_inside_center_y_mm,
        "product_inside_center_face_depth": product_inside_center_face_depth,
        "cut": cut,
        "edges": edges,
        "grasp_candidates": grasp_candidates,
        "grasp_x_mm": grasp_x_mm,
        "grasp_y_mm": grasp_y_mm,
        "grasp_z_mm": grasp_z_mm,
        "grasp_side": grasp_side,
        "depth_heatmap": depth_heatmap,
        "final_output": {
            "polymailer_face_depth_mm": polymailer_face_depth_mm,
            "polymailer_depth_mm": polymailer_depth_mm,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["width_mm"],
            "angle_deg": dimensions["angle_deg"],
            "center_x_mm": center_x_mm,
            "center_y_mm": center_y_mm,
            "product_inside_found": bool(product["found"]),
            "product_inside_center_x_mm": product_inside_center_x_mm,
            "product_inside_center_y_mm": product_inside_center_y_mm,
            "product_inside_center_face_depth": product_inside_center_face_depth,
            "cut_side": cut["cut_side"],
            "top_gap_mm": cut["top_gap_mm"],
            "bottom_gap_mm": cut["bottom_gap_mm"],
            "top_edge_x_mm": edges["top_x_mm"],
            "top_edge_y_mm": edges["top_y_mm"],
            "top_edge_z_mm": edges["top_z_mm"],
            "bottom_edge_x_mm": edges["bottom_x_mm"],
            "bottom_edge_y_mm": edges["bottom_y_mm"],
            "bottom_edge_z_mm": edges["bottom_z_mm"],
            "grasp_x_mm": grasp_x_mm,
            "grasp_y_mm": grasp_y_mm,
            "grasp_z_mm": grasp_z_mm,
            "grasp_side": grasp_side,
            "grasp_candidates": grasp_candidates,
        },
    }
