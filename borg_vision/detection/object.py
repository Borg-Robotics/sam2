"""Object-mode detection logic: generic single-object segmentation + center
depth.

Ported verbatim from run_object()/object_detection_final.py with the
module-level constants replaced by fields of an ObjectConfig passed as the
first argument. Function bodies are otherwise unchanged.
"""

import cv2
import numpy as np
import torch

from .package import get_rotated_box_from_mask, pixel_to_camera_xy_mm
from .polymailer import _patch_plane


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


def clean_mask(cfg, mask):
    mask = mask.astype(np.uint8)

    if not cfg.use_mask_cleanup:
        return mask

    original_area = int(mask.sum())

    if original_area <= 0:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.mask_close_kernel_px, cfg.mask_close_kernel_px),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=cfg.mask_close_iterations,
    )

    cleaned = largest_component(cleaned)

    if cleaned is None:
        return mask

    if cfg.use_convex_hull:
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

            if hull_area <= original_area * cfg.max_cleaned_area_growth:
                return hull_mask.astype(np.uint8)

    cleaned_area = int(cleaned.sum())

    if cleaned_area <= original_area * cfg.max_cleaned_area_growth:
        return cleaned.astype(np.uint8)

    return mask


def get_mask_center(mask):
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def estimate_object_dimensions_mm(cfg, obj, depth_mm, intrinsics):
    """Object footprint and height in millimetres.

    Mirrors the box mode: length/width are the sides of the minimum-area
    rectangle scaled from pixels using the measured face depth, and height is
    how far the face stands above the base surface (cfg.base_depth_mm).
    angle_deg comes from the same rectangle fit, so it always describes the
    same edge as `length`.
    """
    empty = {
        "length_mm": None,
        "width_mm": None,
        "height_mm": None,
        "angle_deg": None,
    }

    rotated = get_rotated_box_from_mask(obj["mask"])

    if rotated is None:
        return empty

    angle_deg = float(rotated["angle_deg"])

    if depth_mm is None or intrinsics is None:
        return {**empty, "angle_deg": angle_deg}

    width_mm = (rotated["width_px"] * depth_mm) / intrinsics["fx"]
    height_px_mm = (rotated["height_px"] * depth_mm) / intrinsics["fy"]

    height_mm = cfg.base_depth_mm - depth_mm

    if height_mm < 0:
        height_mm = 0.0

    return {
        "length_mm": float(max(width_mm, height_px_mm)),
        "width_mm": float(min(width_mm, height_px_mm)),
        "height_mm": float(height_mm),
        "angle_deg": angle_deg,
    }


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


def top_face_depth_and_mask(cfg, depth_roi, mask):
    """Depth of the object's TOP face plus the submask of top-face pixels.

    The nearest coherent depth cluster inside the mask is the top face; the
    median of the WHOLE mask is not, because a standing object's mask includes
    its front face running down to the plate (see object_top_face_* in the
    config). Returns (depth_mm, count, top_mask) or (None, 0, None) when there
    is not enough valid depth to say."""
    valid = (depth_roi > cfg.min_valid_depth_mm) & (depth_roi < cfg.max_valid_depth_mm)
    inside = (mask == 1) & valid
    vals = depth_roi[inside].astype(np.float32)
    if vals.size < cfg.object_top_face_min_px:
        return None, 0, None

    # The nearest cluster must hold a REAL share of the surface before it is
    # believed to be the top face. A glossy highlight can read ~30 mm too
    # close as a compact patch (camera_2 15-23-26: 9% of the object at
    # 624 mm, the true top at 660 mm) -- a genuine top face is a large
    # fraction of the valid pixels (a standing box's measured ~30%+). Step
    # the anchor percentile deeper until the band captures enough.
    cutoff = None
    for pct in (cfg.object_top_face_percentile, 15.0, 30.0, 50.0):
        near = float(np.percentile(vals, pct))
        cand = near + cfg.object_top_face_band_mm
        share = float((vals <= cand).mean())
        if share >= cfg.object_top_face_min_frac:
            cutoff = cand
            break
    if cutoff is None:
        cutoff = float(np.percentile(vals, 50.0)) + cfg.object_top_face_band_mm

    cluster = vals[vals <= cutoff]
    if cluster.size < cfg.object_top_face_min_px:
        return None, 0, None

    top_mask = (inside & (depth_roi <= cutoff)).astype(np.uint8)
    return float(np.median(cluster)), int(cluster.size), top_mask


def candidate_height_above_base_mm(cfg, depth_roi, mask):
    """Median rise of a mask above the surface ring just outside it.

    The ring is the reference surface the candidate rests on, taken from the
    same frame, so a flat plate feature measures ~0 whatever the plate's
    absolute depth. Returns None when either side lacks valid depth -- the
    caller must treat that as unknown, not as flat.

    The valid-FRACTION check matters as much as the count: a glossy top face
    can blank the stereo out over the whole object, leaving only misaligned
    plate pixels valid inside the RGB mask -- enough pixels to pass a bare
    count, all reading base depth, measuring a 115 mm white box as 0 mm tall
    (camera_2 2026-08-31_11-24-51_rejected). If the gate cannot see most of
    the candidate's own surface it has no opinion."""
    valid = (depth_roi > cfg.min_valid_depth_mm) & (depth_roi < cfg.max_valid_depth_mm)
    mask_area = int((mask == 1).sum())
    inside = (mask == 1) & valid
    inside_count = int(inside.sum())
    if inside_count < cfg.height_gate_min_depth_count:
        return None
    if inside_count < cfg.height_gate_min_valid_frac * max(mask_area, 1):
        return None

    k = 2 * cfg.height_gate_ring_px + 1
    dilated = cv2.dilate(mask, np.ones((k, k), np.uint8))
    ring = (dilated > 0) & (mask == 0) & valid
    if int(ring.sum()) < cfg.height_gate_min_depth_count:
        return None

    base_mm = float(np.median(depth_roi[ring].astype(np.float32)))
    # Height of the TOP face, not the mask's overall median: a standing
    # object's mask is dominated by its front face at near-plate depth, and
    # the plain median would measure it as flat and reject it.
    vals = depth_roi[inside].astype(np.float32)
    near = float(np.percentile(vals, cfg.object_top_face_percentile))
    cluster = vals[vals <= near + cfg.object_top_face_band_mm]
    top_mm = float(np.median(cluster)) if cluster.size else float(np.median(vals))
    return base_mm - top_mm


def choose_best_object_mask(cfg, masks, roi_rgb, depth_roi=None):
    h, w = roi_rgb.shape[:2]
    roi_area = h * w
    roi_cx = w / 2
    roi_cy = h / 2

    best = None
    accepted = []

    if cfg.debug_print_masks:
        print()
        print("Checking SAM 2 object masks...")

    for i, m in enumerate(masks):
        raw_mask = m["segmentation"].astype(np.uint8)
        mask = clean_mask(cfg, raw_mask)

        if cfg.reject_masks_touching_roi_border:
            if mask_touches_roi_border(mask, cfg.roi_border_margin_px):
                continue

        area = int(mask.sum())
        area_ratio = area / max(roi_area, 1)

        if area_ratio < cfg.min_area_ratio:
            continue

        if area_ratio > cfg.max_area_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(mask)

        if bw <= 0 or bh <= 0:
            continue

        bbox_area = bw * bh
        rectangularity = area / max(bbox_area, 1)
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if rectangularity < cfg.min_rectangularity:
            continue

        if aspect_ratio > cfg.max_aspect_ratio:
            continue

        height_mm = None
        if depth_roi is not None:
            height_mm = candidate_height_above_base_mm(cfg, depth_roi, mask)
            if height_mm is not None and height_mm < cfg.min_height_above_base_mm:
                if cfg.debug_print_masks:
                    print(
                        f"mask={i:03d} rejected: {height_mm:.1f} mm above base "
                        f"(< {cfg.min_height_above_base_mm:.0f})"
                    )
                continue

        center = get_mask_center(mask)

        if center is None:
            continue

        cx, cy = center

        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        max_dist = np.sqrt(roi_cx ** 2 + roi_cy ** 2)
        center_score = 1.0 - min(dist / max_dist, 1.0)

        area_score = 1.0 - min(
            abs(area_ratio - cfg.target_area_ratio) / cfg.target_area_ratio,
            1.0,
        )

        sam_iou = float(m.get("predicted_iou", 0.0))
        sam_stability = float(m.get("stability_score", 0.0))

        score = (
            cfg.center_score_weight * center_score
            + cfg.area_score_weight * area_score
            + cfg.rect_score_weight * rectangularity
            + cfg.sam_iou_score_weight * sam_iou
            + cfg.sam_stability_score_weight * sam_stability
        )

        if cfg.debug_print_masks:
            print(
                f"mask={i:03d} "
                f"score={score:.3f} "
                f"area={area_ratio:.3f} "
                f"rect={rectangularity:.3f} "
                f"aspect={aspect_ratio:.2f} "
                f"center={center_score:.2f} "
                f"sam_iou={sam_iou:.2f} "
                f"stable={sam_stability:.2f} "
                f"height={'?' if height_mm is None else f'{height_mm:.1f}'} "
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
            "height_above_base_mm": height_mm,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate
        accepted.append(candidate)

    # CONTAINER VETO (operator report 2026-09-08, capture 15-13-50): an
    # opened box shell scored best -- big, centred, square -- with the real
    # object sitting INSIDE it. A container betrays itself physically: a
    # smaller candidate is nested within its mask and rises well above the
    # container's own floor. A genuine object never has raised contents.
    if depth_roi is not None and len(accepted) > 1:
        vetoed = set()
        for i, outer in enumerate(accepted):
            o_mask = outer["mask"]
            o_area = int(o_mask.sum())
            ov = depth_roi[
                (o_mask == 1)
                & (depth_roi > cfg.min_valid_depth_mm)
                & (depth_roi < cfg.max_valid_depth_mm)
            ]
            if ov.size < cfg.height_gate_min_depth_count:
                continue
            o_floor = float(np.median(ov))
            for k, inner in enumerate(accepted):
                if i == k:
                    continue
                in_mask = inner["mask"]
                in_area = int(in_mask.sum())
                if in_area > 0.6 * o_area:
                    continue
                overlap = int((o_mask & in_mask).sum())
                if overlap < 0.8 * in_area:
                    continue
                z_top, n_top, _ = top_face_depth_and_mask(cfg, depth_roi, in_mask)
                if z_top is None:
                    continue
                if (o_floor - z_top) >= cfg.container_veto_min_rise_mm:
                    vetoed.add(i)
                    if cfg.debug_print_masks:
                        print(
                            f"mask={outer['index']:03d} vetoed as CONTAINER: holds "
                            f"mask={inner['index']:03d} rising "
                            f"{o_floor - z_top:.0f} mm above its floor"
                        )
                    break
        survivors = [c for i, c in enumerate(accepted) if i not in vetoed]
        if survivors:
            best = max(survivors, key=lambda c: c["score"])

    return best



def score_object_grasp_candidates(
    cfg,
    obj,
    center_roi,
    depth_roi,
    distance_mm,
    dimensions,
    intrinsics,
):
    """Rank suction-cup grasp points on a detected object, centred on its centroid.

    The centroid is the right answer most of the time, so the search is anchored
    there and `obj_grasp_w_centre` pulls candidates back toward it. What the
    scoring buys is a ranked fallback for when the centroid happens to land on a
    crease, a label edge or a curved shoulder -- a failed grasp there otherwise
    has nowhere to retry.

    Scored per candidate over a disc the size of the real cup:
      - plane-fit RMS residual -> roughness
      - plane tilt             -> how askew a straight-down cup would land
      - distance from the centroid -> stay put unless the surface says move
      - distance from the mask edge -> prefer the cup fully supported

    No bulge term: unlike a polymailer there is no product-inside mask here.

    Returns candidates best-first with camera-frame x/y/z in mm, plus the raw
    terms so a caller can see why something ranked where it did. Empty when the
    object is too small for the cup or depth was too sparse to fit a plane --
    the caller reports that as not-scored rather than as a failure.
    """
    mask = (obj["mask"] > 0).astype(np.uint8)
    if mask.sum() == 0 or center_roi is None:
        return []

    # Pixel scale straight from the intrinsics at the measured depth: one pixel
    # spans depth/fx mm across. The polymailer scorer derives this from its edge
    # span instead, because a floppy mailer's own measured length is the more
    # trustworthy ruler there; here the object is rigid and the intrinsics are
    # exact. fx and fy differ slightly, so the tighter of the two is used -- it
    # makes the cup footprint conservative rather than optimistic.
    if distance_mm is None or intrinsics is None:
        return []
    fx, fy = intrinsics.get("fx"), intrinsics.get("fy")
    if not fx or not fy:
        return []
    mm_per_px = float(distance_mm) / float(max(fx, fy))
    if mm_per_px <= 0:
        return []

    radius_px = max(int(round((cfg.obj_grasp_cup_diameter_mm / 2.0) / mm_per_px)), 3)
    margin_px = int(round(cfg.obj_grasp_edge_margin_mm / mm_per_px))
    max_offset_px = cfg.obj_grasp_max_offset_mm / mm_per_px

    d_edge = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    nx, ny = int(center_roi[0]), int(center_roi[1])

    h, w = mask.shape
    step = max(cfg.obj_grasp_grid_step_px, 1)
    cands = []
    for y in range(radius_px, h - radius_px, step):
        for x in range(radius_px, w - radius_px, step):
            if d_edge[y, x] < radius_px + margin_px:
                continue                      # cup would overhang the object
            offset = float(np.hypot(x - nx, y - ny))
            if offset > max_offset_px:
                continue
            rms, tilt, count = _patch_plane(
                cfg, depth_roi, x, y, radius_px,
                min_valid_px=cfg.obj_grasp_min_valid_px,
                min_valid_frac=cfg.obj_grasp_min_valid_frac,
                min_depth_levels=cfg.obj_grasp_min_depth_levels)
            if rms is None:
                continue
            cands.append({
                "x_roi": x, "y_roi": y, "rms_mm": rms, "tilt_deg": tilt,
                "valid_px": count, "offset_mm": offset * mm_per_px,
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
    s_offset = norm("offset_mm", True)
    s_edge = norm("edge_dist_px", False)

    for i, c in enumerate(cands):
        c["score"] = float(
            cfg.obj_grasp_w_rough * s_rms[i]
            + cfg.obj_grasp_w_tilt * s_tilt[i]
            + cfg.obj_grasp_w_centre * s_offset[i]
            + cfg.obj_grasp_w_edge * s_edge[i]
        )

    cands.sort(key=lambda c: -c["score"])

    # Non-max suppression, so the list is genuinely alternative spots rather than
    # a cluster of neighbouring pixels that would all fail the same way. The
    # separation is configurable per mode (in cup radii).
    kept = []
    min_sep = radius_px * cfg.obj_grasp_min_sep_radii
    for c in cands:
        if all((c["x_roi"] - k["x_roi"]) ** 2 + (c["y_roi"] - k["y_roi"]) ** 2
               > min_sep * min_sep for k in kept):
            kept.append(c)
        if len(kept) >= cfg.obj_grasp_max_candidates:
            break

    for c in kept:
        full = (c["x_roi"] + cfg.roi_x1, c["y_roi"] + cfg.roi_y1)
        depth, _ = center_depth_mm(cfg, depth_roi, obj["mask"], (c["x_roi"], c["y_roi"]))
        if depth is None:
            depth = distance_mm
        x_mm, y_mm = pixel_to_camera_xy_mm(full, depth, intrinsics)
        c["x_mm"], c["y_mm"], c["z_mm"] = x_mm, y_mm, depth
        c["point_full"] = full

    return kept


def run_sam2_object_segmentation(
    cfg,
    frame_bgr,
    depth_aligned,
    mask_generator,
    intrinsics=None,
):
    """Run SAM2 on the ROI, pick the best object mask, and measure its center
    depth. Returns a result dict or None when no valid mask is found.

    With intrinsics the center is also projected into camera-frame millimetres
    (x right, y down, z = the measured distance); without them center_x_mm and
    center_y_mm come back None."""
    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()
    depth_roi = depth_aligned[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

    with torch.inference_mode():
        masks = mask_generator.generate(roi_rgb)

    obj = choose_best_object_mask(cfg, masks, roi_rgb, depth_roi=depth_roi)

    if obj is None:
        return None

    center_full = (
        cfg.roi_x1 + obj["center_roi"][0],
        cfg.roi_y1 + obj["center_roi"][1],
    )

    # TOP-FACE depth over the whole mask -- a standing object's mask includes
    # its front face down to the plate, and any centre/median sampling then
    # reports plate depth and sends the cup through the object. Falls back to
    # the old centre-patch median only when depth is too sparse to cluster.
    distance_mm, depth_count, top_mask = top_face_depth_and_mask(
        cfg, depth_roi, obj["mask"]
    )
    if distance_mm is None:
        top_mask = None
        distance_mm, depth_count = center_depth_mm(
            cfg,
            depth_roi,
            obj["mask"],
            obj["center_roi"],
        )

    dimensions = estimate_object_dimensions_mm(
        cfg,
        obj,
        distance_mm,
        intrinsics,
    )
    angle_deg = dimensions["angle_deg"]

    center_x_mm, center_y_mm = pixel_to_camera_xy_mm(
        center_full,
        distance_mm,
        intrinsics,
    )

    # The cup belongs on the TOP face. When the top-face pixels form a usable
    # region, both the grasp anchor and the scorer's playing field shrink to
    # it -- on a standing box the whole-mask centroid sits on the front face.
    grasp_mask = obj["mask"]
    grasp_anchor = obj["center_roi"]
    if top_mask is not None:
        top_region = cv2.morphologyEx(
            top_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
        )
        top_region = largest_component(top_region)
        if top_region is not None and int(top_region.sum()) >= cfg.object_top_face_min_px:
            top_center = get_mask_center(top_region)
            if top_center is not None:
                grasp_mask = top_region
                grasp_anchor = top_center

    # Suction-cup grasp. Best-first; empty when the object is too small for the
    # cup or depth was too sparse to fit a plane, in which case the grasp point
    # falls back to the anchor -- a small object is still pickable, the scorer
    # just cannot vouch for the surface it lands on.
    grasp_candidates = score_object_grasp_candidates(
        cfg,
        {"mask": grasp_mask},
        grasp_anchor,
        depth_roi,
        distance_mm,
        dimensions,
        intrinsics,
    )
    best = grasp_candidates[0] if grasp_candidates else None
    if best is not None:
        grasp_x_mm, grasp_y_mm, grasp_z_mm = best["x_mm"], best["y_mm"], best["z_mm"]
    else:
        anchor_full = (
            cfg.roi_x1 + grasp_anchor[0],
            cfg.roi_y1 + grasp_anchor[1],
        )
        grasp_x_mm, grasp_y_mm = pixel_to_camera_xy_mm(
            anchor_full, distance_mm, intrinsics
        )
        grasp_z_mm = distance_mm

    return {
        "roi_rgb": roi_rgb,
        "depth_roi": depth_roi,
        "object": obj,
        "center_full": center_full,
        "distance_mm": distance_mm,
        "depth_count": depth_count,
        "angle_deg": angle_deg,
        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "dimensions": dimensions,
        "grasp_candidates": grasp_candidates,
        "grasp_x_mm": grasp_x_mm,
        "grasp_y_mm": grasp_y_mm,
        "grasp_z_mm": grasp_z_mm,
        # Convenience scalars exposed in final_output for the result dataclass.
        "final_output": {
            "center_pixel_u": int(center_full[0]),
            "center_pixel_v": int(center_full[1]),
            "distance_mm": (
                float(distance_mm) if distance_mm is not None else None
            ),
            "depth_count": int(depth_count),
            "angle_deg": angle_deg,
            "center_x_mm": center_x_mm,
            "center_y_mm": center_y_mm,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["width_mm"],
            "height_mm": dimensions["height_mm"],
            "grasp_x_mm": grasp_x_mm,
            "grasp_y_mm": grasp_y_mm,
            "grasp_z_mm": grasp_z_mm,
            "grasp_candidates": grasp_candidates,
        },
    }
