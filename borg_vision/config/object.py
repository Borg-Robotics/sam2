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
    # 700.0 (was 705.0, operator 2026-09-08): heights read ~5 mm large.
    base_depth_mm: float = 700.0

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

    # Measure from the MEASUREMENT stereo stream instead of the classification
    # stream. The class stream is near-blind on small or glossy objects (11-14%
    # valid over the 2026-09-01 boxes) and its fallback then reads the plate;
    # the measurement stream saw the same objects fine. Segmentation is
    # unaffected (RGB). On a camera without a dedicated measurement stream the
    # two are the same array, so this is safe everywhere.
    object_use_measurement_depth: bool = True

    # Top-face depth: measure the nearest coherent depth cluster inside the
    # mask instead of the median of everything. A standing box's mask includes
    # its FRONT face, which runs down to the plate and outnumbers the top-face
    # pixels -- the plain median then reports plate depth (13-41-11: median
    # 703 mm vs true top 635 mm) and the cup is sent through the box. The top
    # face is by definition the nearest surface: cluster = everything within
    # top_face_band_mm of the top_face_percentile'th nearest pixel.
    object_top_face_percentile: float = 5.0
    object_top_face_band_mm: float = 15.0
    object_top_face_min_px: int = 40
    # The nearest band must capture at least this fraction of the mask's
    # valid depth before it is trusted as the top face; a compact glossy
    # highlight reading ~30 mm too close holds only ~9% (2026-09-08), a real
    # top face 30%+. Below it, the anchor percentile steps deeper.
    object_top_face_min_frac: float = 0.20
    # Picks are confined to the top region only when the REST of the mask
    # carries at least this fraction of valid depth -- i.e. it is another
    # surface (a standing box's front face). Below it the rest is blank
    # stereo on a glossy flat top, and the picks keep the whole mask
    # (2026-09-10 16-00-10: 54% top region, no cup fit).
    object_top_face_rest_valid_frac: float = 0.5
    # ...AND that rest must read at least this much deeper (median) than the
    # top face. A standing box's front face drops tens of mm; speckle on a
    # flat glossy top drops ~7 mm (16-00-10) and must not shrink the picks.
    object_top_face_rest_min_drop_mm: float = 15.0
    # When the top surface covers at least this share of the mask, the mask
    # is TRIMMED to it (size/centre/angle recomputed) -- cuts attached
    # neighbours at other depths, e.g. a skirt of box floor stuck to a flat
    # product (2026-09-09 13-32-37). A standing object (top face ~30%)
    # stays whole.
    object_top_face_trim_frac: float = 0.55

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
    # 5.0 (was 10.0, lowered 2026-09-08): a bagged product's crinkly poly
    # film reads ~20 mm too DEEP in stereo -- a real bag of product inside a
    # box measured only 6-8 mm proud of the box floor and the 10 mm gate
    # rejected it (15-13-50), leaving the box shell to win. The features
    # this gate exists to reject measure 0-2 mm (plate recess), so 5 keeps
    # them out while film-wrapped contents pass.
    min_height_above_base_mm: float = 5.0
    height_gate_ring_px: int = 15
    height_gate_min_depth_count: int = 50
    # The gate may only REJECT when it can see at least this fraction of the
    # candidate's own surface in valid depth; below it, the gate abstains. A
    # glossy top face can blank the stereo out across the whole object while
    # misaligned plate pixels stay valid inside the RGB mask, reading base
    # depth -- without this floor that measured a 115 mm white box as 0 mm.
    height_gate_min_valid_frac: float = 0.5
    # Container veto: a candidate holding a smaller nested candidate whose
    # top rises at least this far above the outer mask's median depth is a
    # CONTAINER (opened box shell), not the object -- the contents win
    # (2026-09-08 15-13-50: box interior beat the bag inside it; the bag's
    # top sat ~28 mm above the box floor).
    container_veto_min_rise_mm: float = 18.0
    # Container RESCUE (no second SAM mask needed, 2026-09-10 16-20-05): the
    # chosen mask is a container when its top region holds at most this
    # fraction of it, the rest lies within container_floor_tolerance_mm of
    # base_depth (floor level) and the top rises container_veto_min_rise_mm
    # above that floor; SAM is then re-prompted on the raised island.
    container_top_max_frac: float = 0.5
    container_floor_tolerance_mm: float = 15.0

    # --- inside-box wall clearance (DetectObject.inside_box goal flag) -----
    # When the goal sets inside_box, everything the depth sees standing
    # taller than the object's top by wall_min_rise counts as a WALL, and
    # the grasp point + retries must keep wall_clearance from all of it so
    # the end-effector body fits. Clearance = gripper body radius + margin;
    # MEASURE the gripper -- this default is a placeholder.
    object_inside_box: bool = False        # set per-goal by the node
    object_wall_min_rise_mm: float = 10.0
    # GRIPPER BODY, exactly as box_mover's wall fit reads it off the robot
    # model (vacuum_gripper_base_link collision box in the cup frame):
    # x from -35.5 to +32.5 mm, y +-46 mm about the cup. The robot tries the
    # cup at the object's yaw and its three quarter turns and needs every
    # corner of that box pick.wall_fit_margin clear of every wall panel;
    # the object mode runs the SAME test per pixel (operator 2026-09-10:
    # the points we output must never fail the inside-box pick), so these
    # numbers must track box_mover. If the gripper URDF changes, change
    # them here too.
    object_effector_x_min_mm: float = -35.5
    object_effector_x_max_mm: float = 32.5
    object_effector_y_half_mm: float = 46.0
    # = box_mover pick.wall_fit_margin (0.002 m).
    object_wall_margin_mm: float = 2.0
    # Extra room on top of the robot's margin, because the robot judges
    # against wall panels built from detect_box's measurement BEFORE the cut
    # (outer size, 3 mm panels) while this mode fits its rectangle to the
    # opened box in the current image. Covers the two disagreeing by this
    # much; raise it if an inside-box pick is ever refused on a point this
    # mode reported.
    object_wall_scene_allowance_mm: float = 4.0
    # The detected box rectangle fits the box RIM, ~a few mm outside the
    # real inner wall; clearances are measured to a boundary this far inside
    # the drawn red line (operator 2026-09-10).
    object_wall_extra_inset_mm: float = 3.0
    # Scalar clearance for the DEPTH-based fallback only (no rectangle, so
    # no direction): the body's long half + margin + allowance.
    object_wall_clearance_mm: float = 52.0

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
