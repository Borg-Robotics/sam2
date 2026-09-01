"""Configuration for the polymailer (padded envelope) measurement mode.

PolymailerConfig == BaseConfig plus the polymailer HSV-scoring, mask-cleanup,
dimension and product-bulge fields. Every field corresponds 1:1 to a
module-level constant of the original run_polymailer()/polymailer_final.py,
with the original value as default.

Differences captured as defaults: narrower locked ROI (330,60)-(940,700),
negative depth x-shift (-40), base depth 705 mm, and a single full-resolution
stereo (needs_measurement_stereo = False) used for both face depth and the
product bulge.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class PolymailerConfig(BaseConfig):
    save_dir: str = "polymailer_segment_depth_heatmap_results"

    # Single full-res stereo; depth comes from the classification stream.
    needs_measurement_stereo: bool = False

    base_depth_mm: float = 705.0

    # Locked ROI.
    # SAM2 prompt grid, overriding the base 24. Measured 2026-08-24 on 8 saved
    # captures, scoring each setting's best mask against the mask the running
    # system produced:
    #
    #   pps 24 -> mean IoU 0.993, min 0.963, 8/8 over 0.9, 1.19 s
    #   pps 16 -> mean IoU 0.986, min 0.963, 8/8 over 0.9, 0.56 s
    #   pps 12 -> mean IoU 0.991, min 0.965, 8/8 over 0.9, 0.36 s
    #
    # Confirmed A/B on the live node the same day, whole-detection timing:
    #   pps 24 -> 1847 ms then 1633 ms
    #   pps 12 ->  641 ms then  672 ms
    #
    # SET TO 15 by operator preference: segmentation looked visibly better at 24
    # than at 12 on the table, and 15 is the compromise. Note the IoU benchmark
    # did NOT detect that difference (0.986-0.993 mean at every setting), which
    # suggests what improves is WHICH candidate mask the downstream scoring
    # picks, not the best mask available -- more prompts give the scorer more to
    # choose from. If that is confirmed, fixing the scoring would be cheaper than
    # paying for the extra prompts.
    #
    # Quality is flat while inference cost falls ~3x: cost scales with the prompt
    # count (24^2 = 576 prompts, 12^2 = 144), and a polymailer is a large,
    # high-contrast, well-separated object that does not need a dense grid to
    # find. Raise this back toward 24 if the mailer is ever MISSED outright --
    # the failure mode of too few prompts is no candidate at all, not a worse one.
    #
    # Polymailer only. Box detection has its own fragility (partial masks on a
    # banded top face) and was NOT measured here; leave it on the base 24.
    sam_points_per_side: int = 15

    roi_x1: int = 330
    roi_y1: int = 30
    roi_x2: int = 940
    roi_y2: int = 700

    # Depth alignment (negative x-shift vs. the package mount).
    depth_align_x_shift_px: int = -40
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    center_depth_radius_px: int = 35
    min_center_depth_count: int = 30

    # Force the cut onto one end instead of letting clearance decide.
    # "" = automatic (cut whichever end has more empty mailer between the
    # product and the edge); "TOP" or "BOTTOM" = always cut that end, in the
    # image-row convention (TOP = smaller rows; side_names in
    # vision_cameras.yaml maps these onto planner side indices). The grasp
    # end stays derived as the opposite, so forcing TOP also forces the cup
    # to the bottom end. The clearance gaps are still measured and reported
    # (top/bottom_gap_mm) -- with a forced side the cut row lands in the
    # forced end's gap however small it is, so watch those numbers if a
    # product ever rides high in the mailer.
    force_cut_side: str = ""

    # Suction-cup grasp candidate search. The single release-grasp point is a
    # fixed geometric midpoint with no notion of surface quality; these rank
    # nearby alternatives so a failed grasp has somewhere sensible to retry.
    # MEASURE the cup diameter -- the default is a placeholder and it sets both
    # the scoring disc and the edge clearance constraint.
    # Which end to grasp is NOT configured directly: it is derived as the
    # opposite of cut_side (see force_cut_side above), so the cup can never
    # end up on the end the product slides out of.

    # How far in from the end edge midpoint to aim, for bottom/top regions. The
    # search anchors here instead of at the geometric midpoint: grabbing nearer the
    # mailer's outer end leaves a longer clear slope for the product to slide down
    # and keeps the cup off the bulge. 57 mm matches the picks that worked on
    # hardware (56/59/56 mm). Note the edge points are themselves inset ~17.5 mm
    # from the physical end, so this is ~75 mm from the real end of the mailer.
    # Too small and the cup half-overhangs floppy film; too large and it drifts
    # back toward the product.
    poly_grasp_edge_setback_mm: float = 57.0

    # Two DIFFERENT diameters, do not conflate them:
    #  - cup: the active suction cup. Only this area has to be flat, so it sets
    #    the plane-fit disc used for the roughness/tilt scores.
    #  - body: the gripper's full physical footprint (UFactory xArm vacuum
    #    gripper, middle cup only -- the plate is much wider than the one cup).
    #    Nothing may collide with this, so it sets the hard clearance gates
    #    against the mailer outline and the product bulge.
    # Using the cup diameter for clearance let the gripper body strike the edge of
    # the product inside the mailer and trip the arm's force sensing on descent.
    # MEASURE BOTH -- these are placeholders.
    poly_grasp_cup_diameter_mm: float = 30.0
    # Set equal to the cup for now -- only the middle cup is in use and the plate's
    # real footprint has not been measured. Raise it to the true plate size and the
    # clearance gates tighten automatically; nothing else needs to change.
    poly_grasp_body_diameter_mm: float = 30.0
    # Clear material required between the cup's rim and the mailer outline, in ANY
    # direction. A HARD gate, not a score. This is the SIDE-edge constraint in
    # practice: distance from the grasped END is governed separately by the band
    # below, which is tighter, so this only ever bites at the sides. Kept modest
    # so the search can use the full width when the surface asks it to.
    poly_grasp_edge_margin_mm: float = 10.0

    # Distance from the grasped END, as a band. Nearer than the minimum and the
    # film is floppy under the cup; further than the maximum and the cup sits over
    # the product with a short slope for it to slide down. Measured from the end
    # edge midpoint, so it is directly comparable to the numbers read off a
    # detection: grasp_y_mm against that end's *_edge_y_mm.
    poly_grasp_end_band_min_mm: float = 40.0
    poly_grasp_end_band_max_mm: float = 70.0
    # Extra clearance the body must keep from the product bulge boundary.
    poly_grasp_bulge_margin_mm: float = 8.0
    # Hard cap on wander from the nominal pick. The search is a fallback, not a
    # replacement: it should stay where it already works unless the surface says
    # otherwise.
    poly_grasp_max_offset_mm: float = 45.0
    poly_grasp_grid_step_px: int = 6
    poly_grasp_min_valid_px: int = 30
    # Guards against scoring un-measured surface as perfectly flat: depth is
    # uint16 mm, so a patch stereo failed on comes back as one constant value and
    # fits a plane with zero residual. Require most of the disc to be valid AND
    # several distinct depth levels before believing a flatness number.
    poly_grasp_min_valid_frac: float = 0.60
    poly_grasp_min_depth_levels: int = 3
    # The pick plus three retries. The scorer returns one ranked list with the
    # pick at its head, so this is 1 + 3 -- consumers are handed the tail.
    poly_grasp_max_candidates: int = 4
    # Scoring weights. Provisional -- set these properly by logging the resulting
    # gripper_pressure per pick and seeing which term predicts a good seal.
    # Surface flatness leads. The band is already constrained by the anchor and
    # poly_grasp_max_offset_mm, so within that band the question is purely which
    # patch seals best -- and film pulled tight over the product is the flattest
    # surface on the mailer, which is why those grasps seal hardest.
    poly_grasp_w_rough: float = 0.45
    # Pull back toward the anchor, split by axis because the two directions want
    # opposite things.
    #
    # ACROSS the mailer's width: stay centred. Drifting sideways moves the cup
    # toward a side edge and buys nothing, so this is weighted hard.
    poly_grasp_w_across: float = 0.40
    # ALONG its length: the anchor (poly_grasp_edge_setback_mm) is already at the
    # distance from the end that works, so this holds the pick there. At 0.10 the
    # surface terms dragged it ~37 mm inward toward the product and let the last
    # candidate run out to within 25 mm of the edge, against the margin gate.
    poly_grasp_w_along: float = 0.30
    # Clearance from the mailer outline. Kept small: it pulls toward the interior,
    # which directly fights the edge setback. The hard margin gate is what actually
    # stops the cup overhanging; this is only a tiebreak.
    poly_grasp_w_edge: float = 0.05
    poly_grasp_w_tilt: float = 0.10
    poly_grasp_w_bulge: float = 0.05

    # Dimension measurement.
    poly_measure_erode_px: int = 8
    poly_size_scale: float = 1.04

    # Polymailer mask gating & scoring.
    min_poly_area_ratio: float = 0.04
    max_poly_area_ratio: float = 0.75
    target_poly_area_ratio: float = 0.55

    min_poly_rectangularity: float = 0.35
    max_poly_aspect_ratio: float = 3.5

    min_poly_value: float = 80
    min_poly_color_score: float = 0.30

    # Polymailer scoring weights.
    poly_color_score_weight: float = 3.0
    poly_rect_score_weight: float = 1.5
    poly_area_score_weight: float = 1.2
    poly_center_score_weight: float = 1.0

    # Mask cleanup (close kernel + max area growth).
    poly_mask_close_kernel_px: int = 21
    poly_mask_close_iterations: int = 1
    poly_max_cleaned_area_growth: float = 2.8

    # Product bulge detection inside the polymailer.
    poly_inner_erode_px: int = 22

    # ---- product-inside reference surface (ported from package mode) ------
    # A robust PLANE fitted per frame, replacing poly_surface_depth_percentile.
    # That percentile assumed the mailer lies flat and level; it tilts and sags,
    # so one depth value lands mid-slope and the raised half of a BARE mailer
    # clears any bulge threshold by itself. In package mode this was what made an
    # empty mailer report as a ~22% product.
    #
    # The fit is trimmed, not plain least-squares -- an untrimmed fit is dragged
    # toward the product it is measuring against and shrinks the signal.
    poly_product_plane_iterations: int = 5
    poly_product_plane_trim_sigma: float = 1.5
    poly_product_max_plane_points: int = 20000

    # ---- dome model ------------------------------------------------------
    # The mailer drapes over the contents, so depth shows a smooth dome with
    # sloped shoulders, not an object with edges. Every constant below is a
    # RATIO for that reason: the dome's height depends on the mailer stock
    # (padding spreads the same product into a lower, broader dome), so absolute
    # millimetres cannot transfer between stocks. This is what
    # poly_bulge_min_mm / poly_bulge_max_mm could not do.
    #
    # Smoothing and morphology scale with the mailer's own size (sqrt of its
    # area), so one setting covers a small mailer and a large one.
    poly_product_smooth_frac: float = 0.035
    poly_product_close_frac: float = 1.0
    # Cut at half the dome's own height -- the same relative place on any stock.
    poly_product_height_fraction: float = 0.5

    # Height fraction used for the product CENTER only, as a fraction of the
    # dome's own height like poly_product_height_fraction above.
    #
    # The 0.5 half-maximum contour is the right shape for the product's EXTENT
    # and for the area gates, but it includes the film sloping off the product,
    # and that slope is rarely symmetric. Taking the centroid over it pulls the
    # reported centre toward the shallower side, so the vacuum cup lands off the
    # product -- the residual error Lorenzo saw on 2026-08-24 after the plane-fit
    # rewrite fixed the gross cases.
    #
    # 0.8 keeps only the top fifth of the dome, where the surface is the product
    # itself rather than film draping away from it. Extent, area ratio, bbox and
    # depth all still come from the 0.5 mask; ONLY the centre uses this.
    #
    # Falls back to the 0.5 mask's centroid if the top slice is empty or does not
    # survive the same near-centre component selection.
    # 0.93, set 2026-08-24 (was 0.8, and 0.5 before the centre was split out).
    # On the 2026-08-24_09-27-07 capture -- dome baseline -3.8 mm, peak 8.9 mm,
    # height 12.7 mm -- this lands at ~8.0 mm, the 8-9 mm band Lorenzo asked for:
    # 6674 px in ONE component, bbox 53 x 57 mm.
    #
    # Why a fraction and not a fixed 8 mm: a padded mailer or a flatter product
    # makes a shorter dome, and a fixed millimetre cut can sit above its peak and
    # return nothing. This tracks each mailer's own dome.
    #
    # Why this high. Sweeping the threshold on that capture, the region stays
    # near-SQUARE close to the peak (17x16, 38x40, 53x57 mm) and then one axis
    # runs away -- 73x114, 82x155, 89x191 -- as the film's drape down the
    # mailer's length enters. That tail is what dragged the centroid off the
    # product. It appears between 7.5 and 7.0 mm, i.e. below ~0.89.
    #
    # NOTE the dome has no flat top: the area grows smoothly at every threshold,
    # with no plateau. So this recovers the product's CENTRE (the drape is
    # roughly symmetric about what it covers), not its outline -- the region here
    # is much smaller than the product itself.
    poly_product_center_height_fraction: float = 0.93
    poly_product_baseline_percentile: float = 25.0
    poly_product_peak_percentile: float = 99.5
    # Detection gate as a multiple of THIS frame's measured depth noise, so it
    # holds as depth quality varies rather than needing a fixed mm threshold.
    poly_product_min_peak_noise_multiple: float = 3.0
    # Minimum blur support for a pixel to be judged. The measurement stereo drops
    # out in streaks over low-texture kraft; requiring per-pixel validity let
    # every streak punch a notch into the product.
    poly_product_min_blur_support: float = 0.25

    poly_bulge_min_mm: float = 7
    poly_bulge_max_mm: float = 140

    poly_bulge_open_kernel_px: int = 7
    poly_bulge_close_kernel_px: int = 28
    poly_bulge_dilate_px: int = 5

    min_product_area_ratio_of_poly: float = 0.010
    max_product_area_ratio_of_poly: float = 0.65

    debug_print_masks: bool = False
