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

    # Suction-cup grasp candidate search. The single release-grasp point is a
    # fixed geometric midpoint with no notion of surface quality; these rank
    # nearby alternatives so a failed grasp has somewhere sensible to retry.
    # MEASURE the cup diameter -- the default is a placeholder and it sets both
    # the scoring disc and the edge clearance constraint.
    # Which end to grasp is NOT configured: it is derived as the opposite of
    # cut_side, so the cup can never end up on the end the product slides out of.

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
    poly_surface_depth_percentile: int = 75

    poly_bulge_min_mm: float = 7
    poly_bulge_max_mm: float = 140

    poly_bulge_open_kernel_px: int = 7
    poly_bulge_close_kernel_px: int = 28
    poly_bulge_dilate_px: int = 5

    min_product_area_ratio_of_poly: float = 0.010
    max_product_area_ratio_of_poly: float = 0.65

    debug_print_masks: bool = False
