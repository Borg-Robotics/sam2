"""Configuration for the object segmentation + center-depth mode.

ObjectConfig == BaseConfig plus the object-segmentation scoring/cleanup fields.
Every field corresponds 1:1 to a module-level constant of the original
run_object()/object_detection_final.py, with the original value as default.

Differences from the package mode captured as defaults: narrower locked ROI
(330,60)-(940,700), negative depth x-shift (-40), and a single full-resolution
stereo stream (needs_measurement_stereo = False) reused for the center depth.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ObjectConfig(BaseConfig):
    save_dir: str = "object_segmentation_depth_results"

    # Single full-res stereo; center depth comes from the classification depth.
    needs_measurement_stereo: bool = False

    # Distance to the base surface, used for height_mm. Object mode inherited
    # 695.0 from BaseConfig, but the tray this rig measures against sits at
    # 705.0 (measured median over the surface, and what box/polymailer/
    # clear_bag already use); the old value under-reported every height by 10mm.
    base_depth_mm: float = 705.0

    # Locked ROI (from the polymailer/object script).
    # SAM2 prompt grid, overriding the base 24. Measured 2026-08-24 on 10 saved
    # captures, scoring each setting's best mask against the mask the running
    # system produced:
    #
    #   pps 24 -> mean IoU 0.963, min 0.925, 10/10 over 0.9, 1.18 s
    #   pps 16 -> mean IoU 0.959, min 0.924, 10/10 over 0.9, 0.55 s
    #   pps 12 -> mean IoU 0.961, min 0.925, 10/10 over 0.9, 0.34 s
    #   pps  8 -> mean IoU 0.960, min 0.924, 10/10 over 0.9, 0.20 s
    #
    # Quality is flat at every setting while cost scales with the prompt count
    # (24^2 = 576 prompts, 12^2 = 144). Same conclusion as polymailer, which was
    # A/B'd on the live node at 1847/1633 ms (pps 24) vs 641/672 ms (pps 12).
    #
    # Confirmed A/B on the live node the same day, whole-detection timing:
    #   pps 24 -> 2760 ms then 3045 ms
    #   pps 12 ->  827 ms then  835 ms
    # 3.5x slower for quality that does not measurably differ.
    #
    # 8 measured just as well but 12 keeps some margin for an object sitting
    # unusually. The failure mode of too few prompts is the object being MISSED
    # outright, not segmented worse -- raise this back toward 24 if that happens.
    sam_points_per_side: int = 15

    roi_x1: int = 330
    roi_y1: int = 30
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (note the negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    # Center-depth sampling.
    center_depth_radius_px: int = 35
    min_center_depth_count: int = 30

    # Object mask gating & scoring.
    min_area_ratio: float = 0.008
    max_area_ratio: float = 0.35
    target_area_ratio: float = 0.08

    min_rectangularity: float = 0.18
    max_aspect_ratio: float = 7.0

    # OFF as of 2026-08-27. Objects legitimately land at the ROI edge -- a
    # product discharged near the table's far edge sat 6 px inside the old ROI
    # top, and this gate dropped its mask before any other check ran, failing
    # the detection with "No valid mask found" (capture 13-38-03_rejected).
    #
    # What it was filtering still gets filtered, just by score rather than by a
    # hard reject: on that capture the only other border-touching candidate was
    # a background blob at rectangularity 0.298 against the product's 0.694, and
    # it also sat further from target_area_ratio. Note it DOES still pass the
    # area and rectangularity gates, so it is now a competing candidate rather
    # than an excluded one -- if a background mask ever outscores a real object,
    # this is the first thing to turn back on.
    reject_masks_touching_roi_border: bool = False
    roi_border_margin_px: int = 12

    use_mask_cleanup: bool = True
    use_convex_hull: bool = True
    mask_close_kernel_px: int = 25
    mask_close_iterations: int = 2
    # Cap on how much mask cleanup (close + convex hull) may grow a mask. At
    # 2.8 the hull was free to bridge an object to whatever it sits on -- a box
    # on a pedestal came back as one 1.48x-inflated mask, which then outscored
    # the correct one because area_score rewards being closer to
    # target_area_ratio. 1.2 still allows the hull to fill small holes in a
    # mask but rejects it once it balloons.
    max_cleaned_area_growth: float = 1.2

    # Candidate height gate. The scoring below is purely 2D, so a flat feature
    # OF the plate can out-score a real object -- the moulded centre recess is
    # a perfect square at dead centre, and on camera_2 2026-08-31_10-25-52 it
    # beat an off-centre box (and shipped with height_mm=2). A candidate must
    # rise above the surface ring just outside its own mask, measured from the
    # same depth frame, so there is no fixed plate depth to calibrate and the
    # gate follows plate height or camera changes on its own. The gate is
    # skipped for a candidate when either side lacks valid depth.
    min_height_above_base_mm: float = 10.0
    height_gate_ring_px: int = 15
    height_gate_min_depth_count: int = 50
    # The gate may only REJECT when it can see at least this fraction of the
    # candidate's own surface in valid depth; below it, the gate abstains. A
    # glossy top face can blank the stereo out across the whole object while
    # misaligned plate pixels stay valid inside the RGB mask, reading base
    # depth -- without this floor that measured a 115 mm white box as 0 mm.
    height_gate_min_valid_frac: float = 0.5

    # Scoring weights.
    center_score_weight: float = 1.4
    area_score_weight: float = 1.2
    rect_score_weight: float = 1.0
    sam_iou_score_weight: float = 0.5
    sam_stability_score_weight: float = 0.5

    debug_print_masks: bool = False

    # --- Suction-cup grasp scoring -----------------------------------------
    # Where to put the cup on a detected object. The geometric centroid is the
    # obvious answer and usually the right one, so the search is centred there
    # and pulled back toward it -- but a centroid landing on a crease, a label
    # edge or the shoulder of a curved face is a failed grasp with nothing to
    # retry. Scoring a grid around it gives ranked alternatives.
    #
    # Ported from the polymailer scorer (poly_grasp_*), minus the bulge term:
    # there is no product-inside mask for a generic object.
    obj_grasp_cup_diameter_mm: float = 30.0
    obj_grasp_edge_margin_mm: float = 5.0   # keep the cup this far inside the mask
    obj_grasp_max_offset_mm: float = 40.0   # how far from the centroid to search
    obj_grasp_grid_step_px: int = 6
    obj_grasp_min_valid_px: int = 30
    obj_grasp_min_valid_frac: float = 0.60
    obj_grasp_min_depth_levels: int = 3
    # The pick plus three retries: the scorer returns one ranked list with the
    # pick at its head, so this is 1 + 3 and consumers are handed the tail.
    obj_grasp_max_candidates: int = 4
    # Spacing between kept candidates, in cup radii. 2.0 = a full cup
    # diameter apart: genuinely independent spots that cannot fail the same
    # way. (Package mode overrides this to 1.0 -- product fallbacks are
    # purely score-ranked at the operator's request.)
    obj_grasp_min_sep_radii: float = 2.0

    # Scoring weights. Centre-biased by design: w_centre dominates so the pick
    # stays at the centroid unless the surface there is measurably worse than a
    # nearby alternative. Roughness is the next strongest -- it is what actually
    # breaks a seal.
    obj_grasp_w_rough: float = 0.35
    obj_grasp_w_tilt: float = 0.15
    obj_grasp_w_centre: float = 0.45
    obj_grasp_w_edge: float = 0.05
