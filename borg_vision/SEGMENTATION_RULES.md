# Segmentation rules

SAM 2 is class-agnostic.
`SAM2AutomaticMaskGenerator` returns dozens of unlabelled candidate masks per frame and has no notion of "box", "polymailer" or "product".
Everything that turns that mask soup into *one* selected object mask is hand-written heuristics — the **segmentation rules** documented here.

The rules live in `detection/<mode>.py`.
Every threshold is a field of the mode's dataclass in `config/<mode>.py`, so all of it is overridable per deployment without touching code (see [Tuning](#tuning)).

## Rule taxonomy

Four kinds, applied in this order:

| Kind | Question | Effect |
|---|---|---|
| **Gate** | Could this mask be the object at all? | reject outright, no score computed |
| **Score** | How much does it look like the object? | weighted sum; highest wins |
| **Repair** | Did SAM split or truncate the object? | build a *new* candidate from two masks (or swap in a larger one), re-score it, adopt only if it scores higher |
| **Classify** | *Which* object is it? | package mode only: box vs polymailer verdict |

A repair is the important idea and the least obvious one. SAM does not fail randomly — it fails structurally: a seam or a tape line splits a box in two, an angled box's lit top face and shaded side face become separate masks, a lid is returned instead of the whole box. Gates and scores cannot fix that, because the correct mask was never proposed. Repairs synthesise it.

---

## Shared pipeline

Common to every mode (`config/base.py`).

### SAM 2 generator parameters

| Field | Default | Meaning |
|---|---|---|
`sam_points_per_side` | 24 | 24×24 prompt grid over the ROI |
`sam_pred_iou_thresh` | 0.78 | drop masks SAM itself rates below this |
`sam_stability_score_thresh` | 0.82 | drop masks unstable under threshold jitter |
`sam_min_mask_region_area` | 500 | drop tiny blobs (px) |
`checkpoint` / `model_cfg` | `sam2.1_hiera_small.pt` / `sam2.1_hiera_s.yaml` | model weights and hydra config |

No crop layers, no box-NMS override, no mask-count cap: SAM 2's own defaults apply.

### ROI

SAM only ever runs on a fixed rectangle of the RGB frame (`roi_x1/y1/x2/y2`), not the full image. Package mode uses `(250,60)-(1020,700)`; every other mode `(330,60)-(940,700)`. All area ratios below are relative to the **ROI** area, not the frame.

### Depth validity

`min_valid_depth_mm = 450`, `max_valid_depth_mm = 1200`. Anything outside is treated as missing. Depth is affine-warped onto RGB with `depth_align_x_shift_px` / `y_shift` / `scale_x` / `scale_y` before any masked sampling.

### Mask cleanup (pre-gate)

Every mode closes small holes then keeps the largest connected component, and **reverts to the raw mask if cleanup grew the area beyond `max_cleaned_area_growth`** (2.5–2.8×). That guard matters: a close kernel large enough to seal a tape seam is also large enough to bridge two touching objects.

Kernels differ per mode: object 25 px ×2, box 21 px ×1, polymailer 21 px ×1, package 19 px ×1 (package path) / 21 px ×1 (box path), clear bag 17 px ×1.

---

## Common gate and score building blocks

Names differ per mode but the quantities recur:

- **area ratio** — `mask_area / roi_area`, gated by `min_*_area_ratio` / `max_*_area_ratio`.
- **rectangularity** — `mask_area / axis_aligned_bbox_area`. 1.0 = a perfectly axis-aligned rectangle. Note this penalises *rotation* as well as irregularity, which is why the repair rules below need their own separate handling for angled objects.
- **aspect ratio** — `max(w/h, h/w)`; rejects slivers.
- **center score** — `1 - min(dist_to_roi_center / roi_diagonal_half, 1)`; the object is expected near the ROI centre.
- **area score** — `1 - min(|area_ratio - target| / target, 1)`; peaks at `target_*_area_ratio`.
- **colour score** (box, polymailer) — mean HSV inside the mask against a cardboard anchor: `0.50·hue + 0.25·sat + 0.25·val`, each term a linear ramp away from its anchor.
- **contrast / texture score** (package, clear bag) — mean-RGB difference against a dilated ring outside the mask, and Laplacian variance inside it. These carry the load when colour cannot: a clear bag or a white polymailer has no distinctive hue.
- **SAM's own confidence** — `predicted_iou` and `stability_score` enter every score with a small weight (0.4–0.5).

---

## Mode: `object`

Generic "biggest sensible thing near the middle", plus its centre depth. No colour or type reasoning.

**Gates** (`detection/object.py:choose_best_object_mask`)

| Rule | Field | Default |
|---|---|---|
touches ROI border (within margin) | `reject_masks_touching_roi_border` / `roi_border_margin_px` | True / 12 px |
area ratio | `min_area_ratio` / `max_area_ratio` | 0.008 / 0.35 |
rectangularity | `min_rectangularity` | 0.18 |
aspect ratio | `max_aspect_ratio` | 7.0 |

**Score** = `1.4·center + 1.2·area + 1.0·rect + 0.5·sam_iou + 0.5·sam_stability`, peak area ratio `target_area_ratio = 0.08`.

**Cleanup** additionally applies a convex hull (`use_convex_hull = True`), still subject to the 2.8× growth guard.

**Repairs:** none.

---

## Mode: `box`

Cardboard box: HSV colour is the dominant signal (`box_color_score_weight = 2.7`, by far the largest weight anywhere in the library).

**Gates** (`detection/box.py:score_cardboard_box_candidate`)

| Rule | Field | Default |
|---|---|---|
area ratio | `min_box_area_ratio` / `max_box_area_ratio` | 0.04 / 0.75 |
rectangularity | `min_box_rectangularity` | 0.35 |
aspect ratio | `max_box_aspect_ratio` | 4.0 |
mean HSV value (brightness) | `min_box_value` | 70 |
cardboard colour score | `min_box_color_score` | 0.25 |

Colour anchors (hardcoded): hue 18 ± 30, sat 65 ± 100, val 170 ± 120.

**Score** = `2.7·color + 1.6·rect + 1.2·area + 1.0·center + 0.4·sam_iou + 0.4·sam_stability`, peak area ratio 0.45.

**Repairs** — applied in order, each adopted only on a strictly higher score:

1. **Full-mask containment override** (`select_complete_single_box_candidate`) — SAM often returns just the lid. Iteratively replace the top-scoring mask with a strictly larger mask that contains ≥ `box_full_mask_min_partial_containment` (0.72) of it, grows area ≥ `box_full_mask_min_area_growth` (1.25×), loses ≤ `box_full_mask_max_score_drop` (0.80) score, and itself passes colour ≥ 0.25 and rectangularity ≥ 0.35. Loops until stable. Reports `selection_override = "larger_containing_box_mask"`.

2. **Split-mask merge** (`build_merged_cardboard_box_candidate`) — a seam splits the box in two. Pairs of the 14 largest candidates (`box_merge_max_candidates`) with `mask_iou ≤ box_merge_max_pair_iou` (0.65, so near-duplicates are not "two halves") are tested by three geometry modes, first match wins:
   - *top/bottom* — X overlap ≥ 0.55, width ratio ≤ 1.80, X-centre offset ≤ 0.35 of the larger width, Y gap ≤ 90 px
   - *left/right* — the same four tests on the other axis
   - *rotated* (fallback, `box_merge_rotated_enable`) — for an angled box, whose fragments' bounding boxes fail both axis tests. Min-area-rect long axes within `box_merge_rotated_max_angle_diff_deg` (22°) and centre distance ≤ `box_merge_rotated_max_center_distance_ratio` (1.35) × the largest rect side.

   The pair's union is filled to its min-area rectangle. Then: area growth over the larger fragment ≥ 1.20; **rotated pairs only**, union fills ≥ 0.42 of the completed rectangle and the rectangle covers ≤ 0.78 of the ROI (the axis-aligned geometry already implies both); result rectangularity ≥ `box_merge_min_result_rectangularity` (0.72). Score bonus `+0.85`. Reports `merged_geometry_mode` and `merged_split_axis`.

3. **Multi-face merge** (`build_multiface_cardboard_box_candidate`) — an angled box shows a lit top face and a shaded side face, which SAM segments separately; the top face alone passes every rule and simply measures too small. Pairs a scoring box candidate with a candidate from a **separate loose pool** (below), requiring `mask_iou ≤ 0.20`, adjacency by dilate-and-intersect at `box_multiface_touch_dilate_px` (28 px) rather than bbox geometry, and second-face area ≥ 0.025 of the ROI. Keeps the **convex hull** of the union — a min-area rectangle around two faces at an angle overshoots badly — gated on hull growth ≥ 1.25×, union/hull fill ≥ 0.58 and hull/ROI ≤ 0.72. Score bonus `+1.10`; `selection_override = "angled_box_multiface_merge"`.

   The **loose secondary pool** (`build_secondary_face_candidates`) re-scans the raw SAM masks with geometry gates only: area ratio 0.025–0.70, rectangularity ≥ 0.08, aspect ≤ 10.0, centre score ≥ 0.12 — and **deliberately no colour gate**, because a shaded side face is not brown. These candidates can never be selected on their own; they only ever form the second half of a multi-face merge.

   Because the hull dilutes the measured cardboard colour, both merges pass the parent's colour score as a **`color_score_floor`** into the scorer, so the 0.25 colour gate cannot reject a silhouette whose parent fragment was unambiguously cardboard. The undiluted measurement is still reported as `measured_color_score`.

Note the merge score bonuses stack in favour of multi-face (+1.10) over split (+0.85) at near-equal geometry. That is intentional.

Note also that repair 2's rotated mode is bounded by `box_merge_min_result_rectangularity = 0.72`, which is measured on the *axis-aligned* completed mask — so it only rescues boxes tilted by roughly ≤ 10°. Strongly angled boxes are repair 3's job.

---

## Mode: `polymailer`

Padded envelope: same shape as `box` but flatter, and it carries a *product bulge* rather than dimensions of a rigid body.

**Gates** (`detection/polymailer.py:choose_polymailer_mask`)

| Rule | Field | Default |
|---|---|---|
area ratio | `min_poly_area_ratio` / `max_poly_area_ratio` | 0.04 / 0.75 |
rectangularity | `min_poly_rectangularity` | 0.35 |
aspect ratio | `max_poly_aspect_ratio` | 3.5 |
mean HSV value | `min_poly_value` | 80 |
colour score | `min_poly_color_score` | 0.30 |

Colour anchors are *tighter* than box mode: hue 18 ± 25, sat 70 ± 90, val 170 ± 100.

**Score** = `3.0·color + 1.5·rect + 1.2·area + 1.0·center`, peak area ratio 0.55. No SAM-confidence terms.

**Repairs:** none — but there is a second, depth-based segmentation step:

**Product bulge inside the polymailer** (`estimate_product_inside_polymailer`). Erode the polymailer mask by `poly_inner_erode_px` (22 px) to stay off the edges, take the envelope surface as the `poly_surface_depth_percentile` (75th) percentile of depth inside it, then keep pixels between `poly_bulge_min_mm` (7) and `poly_bulge_max_mm` (140) *closer* than that surface. Open 7 px → close 28 px ×2 → dilate 5 px, clip to the polymailer, keep the component near its centre, and accept if the result covers 0.010–0.65 of the polymailer area. Bulge height is the surface depth minus the 20th percentile depth inside the product mask.

---

## Mode: `clear_bag`

Transparent bag with a product inside. Colour is useless, so contrast and texture dominate (`contrast_score_weight = 2.4`), and the bag itself is found in **depth**, not in RGB.

**Product gates** (`detection/clear_bag.py:score_product_mask`)

| Rule | Field | Default |
|---|---|---|
roller-like horizontal strip | `hard_reject_roller_like` + `roller_strip_min_width_ratio` / `max_height_ratio` | True, 0.70 / 0.25 |
area ratio | `min_product_area_ratio` / `max_product_area_ratio` | 0.003 / 0.40 |
rectangularity | `min_product_rectangularity` | 0.08 |
aspect ratio | `max_product_aspect_ratio` | 10.0 |

The roller gate is a conveyor-specific reject: a wide, short strip spanning the ROI is the roller bed, never a product.

**Score** = `1.2·area + 1.0·center + 2.4·contrast + 1.0·texture + 0.5·rect + 0.4·safe_area + 0.4·sam_iou + 0.4·sam_stability`, peak area ratio 0.08. `safe_area` is the mask eroded by `safe_erode_radius_px` (12 px) — a proxy for "has a solid interior, not just a rim". A separate normalised `quality_score` (`0.25·area + 0.20·center + 0.30·contrast + 0.15·texture + 0.10·safe_area`) is reported for downstream thresholding.

**Bag detection from depth** (`detect_clear_bag_from_depth`) — not a SAM path at all:

1. Median-blur depth (5 px), estimate the conveyor base depth around the product.
2. Seed = pixels between `bag_closer_than_base_min_mm` (25) and `bag_closer_than_base_max_mm` (170) closer than base, intersected with a `bag_seed_dilate_px` (135 px) neighbourhood of the product.
3. Morphology: close 19 ×2 → dilate 7 → close 19 → erode 5.
4. Keep the component within `bag_component_keep_near_product_px` (190 px) of the product centre, union it with the product mask.
5. **Corner expansion** (`expand_bag_rect_using_corner_depth`, enabled by default) — a clear bag's flat corners return almost no depth, so the mask under-covers them. Row/column depth-activity profiles are smoothed (`bag_edge_activity_smooth_px = 21`) and the rect is grown into corners that still have ≥ `bag_corner_min_pixels` (25) of depth support in a `bag_corner_window_px` (45) window.
6. Accept if the result covers `bag_min_area_ratio`–`bag_max_area_ratio` (0.025–0.70) of the ROI.

---

## Mode: `package` (barcode-gated; the primary production mode)

The most complex mode. It runs **two independent rule sets over the same SAM masks** — the general package/polymailer rules and the exact cardboard-box rules — arbitrates between them, and then decides the package *type*.

### Path A — general package rules

`score_package_product_mask`. Colour-free; contrast and texture carry it.

| Gate | Field | Default |
|---|---|---|
roller-like strip | `roller_strip_min_width_ratio` / `max_height_ratio` | 0.72 / 0.22 |
edge strip | `edge_strip_margin_px` / `max_width_ratio` / `max_height_ratio` | 24 px / 0.16 / 0.16 |
area ratio | `min_package_area_ratio` / `max_package_area_ratio` | 0.003 / 0.70 |
rectangularity | `min_package_rectangularity` | 0.08 |
aspect ratio | `max_package_aspect_ratio` | 10.0 |
centre score | `min_package_center_score` | 0.18 |

**Score** = `1.25·area + 1.20·center + 1.30·contrast + 0.75·texture + 0.85·rect + 0.65·bbox_size + 0.45·safe_area + 0.45·sam_iou + 0.45·sam_stability`, peak area ratio 0.18. `bbox_size_score` saturates once the bbox reaches 45 % of the ROI in both axes; `safe_area_score` saturates at 3 % of the ROI.

### Path B — exact cardboard-box rules

`score_cardboard_box_mask` — the same gates, anchors and 2.7/1.6/1.2/1.0/0.4/0.4 weighting as `box` mode, run on the *package* ROI's masks (`dedicated_box_sam_enable = True` means no second SAM pass).

### Path B repairs

- **Full-box override** — replace the winning package mask with a box mask that contains ≥ `box_full_mask_min_package_containment` (0.72) of it and grows area ≥ 1.30×. Two additional hardcoded escalations: a *strong* override at containment ≥ 0.60 / growth ≥ 1.15, and an *independent* override for a mask that is convincing on its own (colour ≥ 0.70, rectangularity ≥ 0.88, area ≥ 1.45×, containment ≥ 0.55).
- **Split-mask merge** — same rule as box mode, but only the *top/bottom* axis plus the rotated fallback (this mode never had the left/right test). Thresholds identical: 0.55 / 0.35 / 1.80 / 90 px / IoU 0.65 / growth 1.20 / result rect 0.72 / bonus +0.85, rotated 22° / 1.35 / fill 0.42 / area 0.78.
- **Multi-face merge** — same geometry and gates as box mode (28 px adjacency, IoU ≤ 0.20, growth ≥ 1.25, hull/ROI ≤ 0.72, bonus +1.10), but the secondary pool is this mode's own **package candidates** (no separate loose pool is needed) and the hull is re-scored with the *package* rules. It still competes for the best-box slot and inherits the box face's `color_score` so the type override below can see the cardboard evidence.

  `box_multiface_min_union_fill_ratio` is **0.90 here, versus 0.58 in box mode**. Box mode re-scores the hull with the box rules, whose rectangularity ≥ 0.35 and aspect ≤ 4.0 gates reject an overgrown hull on their own; package mode re-scores with the permissive package rules (rectangularity ≥ 0.08, aspect ≤ 10.0), so the union-fill gate is the only thing left standing between a legitimate two-face hull and one that spikes out along a conveyor roller. Replaying 46 recorded frames: the 5 bad hulls scored 0.684–0.870 union fill, the 4 genuine angled-box repairs 0.938–0.985.

### Arbitration between A and B

`choose_best_package_mask`: a box candidate wins if it is a `similar_complete_mask` (≥ 0.90× the package area and containment ≥ 0.60) or a `full_box_contains_package` (≥ 1.10× the area and containment ≥ 0.50); otherwise the higher score wins.

### Type classification: box vs polymailer

`classify_package_type`, in three layers.

**Layer 1 — depth geometry.** Fit a plane to the depth inside the mask and score four features into
`raw_box_score = 0.35·flatness + 0.25·residual_range + 0.15·rectangularity + 0.25·center_edge`,
each a ramp between a `*_good_mm` and `*_bad_mm` pair (`box_flat_std_good/bad_mm` 6/18, `box_residual_range_good/bad_mm` 20/55, `box_rectangularity_good/bad` 0.90/0.60). A separate *polymailer signature* — centre closer than edges, full/edge depth range, side spread, each with a soft/hard threshold pair — penalises it: `box_score = raw_box_score · (1 - 0.65·poly_signature)`. Verdict `box` if `box_score ≥ box_score_threshold` (0.60).

Three earlier exits take precedence, in order: a strong polymailer signature (needs ≥ 18 % valid depth, ≥ 15 000 points, area ≥ 0.16, and either signature ≥ 0.30 or ≥ 3 of the signals firing); a `sparse_rectangular_box` (≤ 8 % valid depth but rectangularity ≥ 0.88 and area ≥ 0.12 — a glossy box reflects the projector away, so *absent* depth is itself box evidence); and a hard centre-vs-edge difference ≥ 24 mm.

**Layer 2 — segmentation override.** Strong RGB evidence can overturn the depth verdict, but only from a whitelisted source (`BOX_TYPE_SEGMENTATION_OVERRIDE_SOURCES`: `package_roi_exact_box_rules`, `cardboard_box_merged_masks`, `cardboard_box_multiface_merge`) and only above every one of `box_type_segmentation_min_score` 5.25, `min_color_score` 0.65, `min_rectangularity` 0.82, `min_area_ratio` 0.08, `min_confidence` 0.90. Reported as `segmentation_box_override_reason` — `strong_exact_box_rules`, `merged_cardboard_fragments` or `angled_box_multiface_merge`.

**Layer 3 — thin-polymailer veto.** The override is itself vetoed when the measured thickness says otherwise, because a *brown* polymailer passes the same HSV and rectangularity rules as cardboard. Three independent vetoes: thickness ≤ `box_type_polymailer_veto_max_package_depth_mm` (65 mm) with the depth classifier already saying polymailer; thickness ≤ 45 mm with polymailer score ≥ 0.55; or polymailer score ≥ 0.68 with ≥ 1 signal firing.

This is why `base_depth_mm` matters to *segmentation*, not just to measurement: thickness is `base_depth_mm - top_face_depth_mm`, so a miscalibrated base depth shifts every veto. Package mode uses **700.0 mm**; box, polymailer and clear bag use 705.0; the `BaseConfig` default is 695.0.

### Product inside the package

`detect_product_inside_polymailer` — the polymailer-mode bulge rule with package-mode fields: erode 45 px, keep pixels 8–90 mm closer than the surface, close 17 → dilate 5, accept at 0.015–0.65 of the package area with ≥ 80 valid pixels.

---

## Tuning

Every value above is a config field. Nothing needs a code change:

```python
from borg_vision import BoxConfig, get_detector

cfg = BoxConfig(box_multiface_merge_enable=False)     # or
cfg = BoxConfig.from_yaml("box.yaml")                 # partial overrides
```

```yaml
# box.yaml
min_box_color_score: 0.30
box_merge_rotated_max_angle_diff_deg: 30.0
```

Practical notes:

- Every repair rule has an `*_enable` flag. Turn one off to isolate a regression before touching thresholds.
- `debug_print_masks: true` traces every accepted candidate with its score, area, rectangularity and colour — start there, not in the thresholds.
- Loosening a *gate* changes which masks exist; loosening a *repair* threshold changes which synthetic masks get built. They fail differently: a loose gate admits background, a loose repair merges two neighbouring objects into one.
- The merge score bonuses (+0.85, +1.10) are not confidence — they exist so a correctly repaired silhouette can beat a well-scoring fragment. Raising them makes repairs win more often, including when they are wrong.

## Cross-mode duplication

`detection/box.py` and `detection/package.py` each hold their own copy of the split-mask merge and the multi-face merge. The copies are **not** interchangeable: different config field names, one vs two axis-aligned tests, different scorers with different return types, `color_score_floor` present vs absent, and a raw-mask vs package-candidate secondary pool. The shared *arithmetic* is factored into `detection/masks.py` and `utils.py`; the threshold policy above it is intentionally per-mode. Keep them in sync deliberately.
