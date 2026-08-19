# borg_vision

Importable, GUI-free detection library for Borg Robotics, built on this SAM2
fork. It drives OAK-D Pro cameras and SAM2 segmentation behind a single
mode-selectable API, so the same pipelines can be used by the ROS2
`sam2_vision` package in
[borg_cobots](https://github.com/Borg-Robotics/borg_cobots) and from the CLI.

Each mode was extracted verbatim from a validated standalone script
(`*_final.py` / `unified_detector_all_in_one.py`), with the module-level
constants replaced by a config dataclass — behavior is unchanged.

The `package` and `box` modes have since taken two mask-selection rules from the
validated `product_detection_fix.py` / `box_detection_fix.py`, both aimed at
angled boxes that SAM2 splits into separate faces (previously only half the box
was detected):

- **rotated split-mask merge** (`box_merge_rotated_*`) — a fallback pairing rule
  for `build_merged_cardboard_box_candidate`. When two box fragments fail the
  upright relationship (horizontal overlap / width ratio / centre-x / vertical
  gap), they are still merged if their rotated rectangles share an angle
  (`≤ 22°` apart) and sit close together. Rotated pairs must additionally fill
  the completed rectangle (`≥ 0.42`) and not swallow the ROI (`≤ 0.78`).
- **multi-face merge** (`box_multiface_*`) — joins a box-rule mask to a touching
  package-rule mask through their convex hull, for angled boxes showing a top
  and a side face as two separate masks. Contributes a new mask source,
  `cardboard_box_multiface_merge`, which the type classifier treats as a box
  (override reason `angled_box_multiface_merge`).

Both are on by default and gated by `*_enable` fields, so setting
`box_merge_rotated_enable: false` / `box_multiface_merge_enable: false` restores
the previous selection behavior.

Two details differ between the modes. In `package`, the multi-face partner is a
mask that passed the general package rules; in `box` there is no such scorer, so
`build_secondary_face_candidates` supplies partners using loose geometry-only
gates (`secondary_face_*`) — a box's second face is usually too dark to pass the
cardboard-colour rules. And `box` keeps its own two-axis
`pair_fragment_compatibility` (top/bottom **and** left/right splits) plus
`select_complete_single_box_candidate`, neither of which exists in the
standalone fix scripts; the rotated rule was added as a third fallback rather
than replacing them. Merged results carry `merged_geometry_mode`
(`axis_aligned` / `rotated`) so the JSON shows which rule fired.

## Modes

One camera/station = one mode. Pick a mode with `get_detector(mode)`.

| Mode | What it detects | Barcode gate | Output highlights |
|---|---|:--:|---|
| `package` | Box **or** polymailer + classification + product inside | yes | `package_type`, `confidence`, dims, center, product-inside |
| `box` | Cardboard box (HSV + geometry) | no | dims, face depth, center, angle |
| `object` | Generic single object + center depth | no | center pixel, `distance_mm` |
| `polymailer` | Polymailer + product bulge inside | no | dims, center, product-inside |
| `clear_bag` | Visible product + transparent bag around it | no | bag dims, center, product center |
| `inspection` | Return-condition verdict from **two** cameras + OpenAI | no | `result` (good/damaged/manual_review), `confidence` |

`product` is accepted as an alias of `package`.

`inspection` is the odd one out — see [Inspection mode](#inspection-mode-dual-camera--openai).

## Layout

```
borg_vision/
  config/        BaseConfig + per-mode configs (PackageConfig, BoxConfig, ...,
                 InspectionConfig — standalone, no SAM2/depth fields)
  camera.py      OakCamera + Frames (depthai v3 pipeline, depth->RGB alignment)
  barcode.py     pyzbar barcode detection + overlay
  inspection.py  OpenAI inspection logic (prompt, image optimize, API call)
  detection/     per-mode pure detection logic (run_sam2_* entry points)
  visualization/ common.py + per-mode overlays/heatmaps/JSON
  results.py     BaseResult + per-mode result dataclasses (+ InspectionResult)
  detectors/     BaseDetector + per-mode detectors (+ InspectionDetector)
  registry.py    get_detector(mode, ...) factory + get_inspection_detector()
                 + available_modes()
  cli/           run.py (generic runner) + product_detection.py (interactive)
                 + inspection.py (dual-camera / offline inspection runner)
  __main__.py    `python -m borg_vision`
```

`ProductDetectionConfig`, `ProductDetector`, `ProductDetectionResult` remain
importable as aliases of the `Package*` classes for backwards compatibility.

## Install

```bash
pip install -e ".[borg]"   # depthai, pyzbar, opencv, PyYAML on top of sam2
                            # (+ requests, Pillow for the inspection mode)
# checkpoint: checkpoints/download_ckpts.sh  (sam2.1_hiera_small.pt)
```

All modes use the same SAM2 checkpoint/config by default
(`./checkpoints/sam2.1_hiera_small.pt`, `configs/sam2.1/sam2.1_hiera_s.yaml`).
Point at a different checkpoint by setting `cfg.checkpoint` / `cfg.model_cfg`
(or the `checkpoint` / `model_cfg` fields in a YAML override).

## Usage (library)

```python
from borg_vision import get_detector

# mxid selects a camera in multi-OAK setups; cfg can be None (defaults),
# a *Config instance, or a path to a YAML file of overrides.
detector = get_detector("box", cfg=None, mxid=None)

with detector:                 # loads SAM2 model + opens the camera
    detector.warmup()          # pump frames to stabilize the sensor
    result = detector.detect_after_barcode(timeout_s=30.0)  # None on no-detection
    if result is not None:
        print(result.length_mm, result.width_mm, result.center_x_mm)
        detector.save_debug(result)   # annotated image + masks + heatmap + JSON
```

`detect_after_barcode` does the right thing per mode: barcode-gated modes
(`package`) scan for a barcode first; the others grab a frame and detect
immediately (the `timeout_s` is then ignored).

Lower-level control (manual frame grabbing, custom loops):

```python
detector.load_model()
detector.open()
frames = detector.camera.get_frames()     # Frames(rgb, depth_class_aligned, depth_measure_aligned)
result = detector.detect(frames)          # mode-specific result or None
img = detector.draw_result(result)        # annotated BGR image (no disk write)
detector.close()
```

`warmup()` and `scan_barcode()` accept a `should_abort` callable polled once
per frame — that is how the ROS2 action server wires goal cancellation in.

### Config overrides

Every tuning constant is a field on the mode's config dataclass. Override a few
in code, or load a YAML of partial overrides:

```python
from borg_vision import BoxConfig, get_detector

cfg = BoxConfig(roi_x1=350, roi_x2=930)          # or BoxConfig.from_yaml("box.yaml")
det = get_detector("box", cfg=cfg)
```

```yaml
# box.yaml — only the fields you want to change
roi_x1: 350
roi_x2: 930
base_depth_mm: 702.0
min_box_color_score: 0.30
```

Key per-mode fields (defaults equal the validated originals):

| Field | package | box | object | polymailer | clear_bag |
|---|:--:|:--:|:--:|:--:|:--:|
| ROI `roi_x1..roi_y2` | 250,60,1020,700 | 330,60,940,700 | 330,60,940,700 | 330,60,940,700 | 330,60,940,700 |
| `depth_align_x_shift_px` | +35 | -40 | -40 | -40 | -40 |
| `base_depth_mm` | 695 | 705 | n/a | 705 | 705 |
| `needs_measurement_stereo` | true | true | false | false | false |
| barcode-gated | yes | no | no | no | no |

`needs_measurement_stereo` controls the depthai pipeline: `package`/`box`
build a dedicated lower-res measurement stereo stream; the others reuse the
full-res classification depth.

## Usage (CLI)

```bash
python -m borg_vision --list                       # list modes
python -m borg_vision --mode box                   # run once, save artifacts
python -m borg_vision --mode package --mxid 14442C10B135D7D600
python -m borg_vision --mode polymailer --config overrides.yaml --out-dir /tmp/poly
```

The generic runner is headless (no display needed): load → open → warmup →
detect once → `save_debug`. For the package mode's original interactive
SPACE/q barcode-gated preview loop:

```bash
python -m borg_vision.cli.product_detection
```

## Inspection mode (dual-camera + OpenAI)

`inspection` does not fit the single-camera SAM2 pattern above. It captures
several photos of a product from **two** OAK-D cameras and sends them together
to an OpenAI vision model, which returns a return-condition verdict:

| `result` | meaning |
|---|---|
| `good` | no meaningful visible damage |
| `damaged` | clear serious visible damage |
| `manual_review` | unclear, product missing/uncertain, or not enough visible info |

Because of this it is **not** built on `BaseDetector` (no SAM2, no stereo depth,
no barcode, no TF) and has its own factory and config:

```python
from borg_vision import get_inspection_detector

# Two devices selected by mxid (both None = first two available).
detector = get_inspection_detector(cfg=None, mxid_1=None, mxid_2=None)

with detector:                              # opens BOTH OAK-D devices
    result = detector.inspect("Label Maker")   # capture N frames + call OpenAI
    if result is not None:                  # None only if aborted mid-capture
        print(result.result, result.confidence, result.summary)
        detector.save_debug(result)         # writes inspection_result.json
```

`inspect(product_name, count=None, should_abort=None, on_stage=None)` captures
`cfg.capture_count` frames (alternating cameras, headless — no preview window),
saves them as JPEGs under `cfg.save_dir/<request_id>/`, then runs one OpenAI
request over all of them. `should_abort` is polled once per captured frame for
ROS goal cancellation; `on_stage("capturing"|"inspecting")` reports progress.

Key `InspectionConfig` fields (RGB capture + OpenAI request only):

| Field | Default | Purpose |
|---|---|---|
| `capture_count` | 4 | frames to capture before inspecting |
| `rgb_size` | (1280, 720) | per-camera capture resolution |
| `startup_delay_seconds` | 2.0 | wait between opening camera 1 and 2 (USB stability) |
| `model` | `gpt-5.5` | OpenAI model (or `OPENAI_MODEL` env) |
| `max_size` / `quality` | 1600 / 72 | optimized image size/JPEG quality sent to the API |
| `timeout_s` | 180.0 | OpenAI request timeout |

The OpenAI API key is read from **`OPENAI_API_KEY`** in the environment only —
never from config. `requests` and `Pillow` are required (installed by the
`[borg]` extra / present in the borg-vision container).

### Inspection CLI

```bash
# Offline: inspect existing images (no cameras) — verifies the OpenAI path.
export OPENAI_API_KEY=sk-...
python -m borg_vision.cli.inspection --product-name "Label Maker" \
    --images test-images/IMG_0563.jpg test-images/IMG_0564.jpg

# Live: capture from two OAK-D cameras, then inspect.
python -m borg_vision.cli.inspection --product-name "Label Maker" \
    --capture --capture-count 4 --mxid-1 14442C... --mxid-2 14442C...
```

`get_detector("inspection")` raises with a redirect — always use
`get_inspection_detector()` (or the CLI) for the dual-camera mode.

## Saved artifacts

`save_debug(result)` writes a uniform set per mode under the mode's `save_dir`
(override with `out_dir=`). Every mode writes the raw RGB, an annotated result
image, a JSON of the measurements, and the mode's mask(s)/heatmap:

| Mode | files (besides `raw_rgb_*.jpg`) |
|---|---|
| `package` | `package_final_*.png`, `package_mask_*.png`, `product_inside_mask_*.png`, `product_core_mask_*.png`, `package_depth_heatmap_*.png`, `depth_measure_aligned_*.npy`, `depth_class_aligned_*.npy`, `package_final_*.json` |
| `box` | `cardboard_box_measurements_*.png`, `cardboard_box_mask_*.png`, `cardboard_box_measurements_*.json` |
| `object` | `object_segment_depth_*.png`, `object_mask_*.png`, `depth_aligned_*.npy`, `object_segment_depth_*.json` |
| `polymailer` | `polymailer_output_*.png`, `polymailer_mask_*.png`, `product_bulge_mask_*.png`, `polymailer_output_*.json` |
| `clear_bag` | `clear_bag_final_output_*.png`, `clear_bag_product_mask_*.png`, `clear_bag_mask_*.png`, `clear_bag_depth_heatmap_*.png`, `clear_bag_final_output_*.json` |

The annotated `*.png` is a side-by-side of RGB + depth (+ heatmap for modes
that produce one). `draw_result(result)` returns the same image in memory
without writing to disk.

### Product-inside detection (polymailers)

A mailer is not rigid. It drapes over whatever is inside, so depth shows a
smooth **dome with sloped shoulders**, never an object with edges — there is no
true boundary to threshold, and the same product produces a different dome
depending on the stock. Padding spreads it into a low broad mound (~6 mm on the
captures measured so far) where kraft gives a taller narrower one (~10 mm). A
fixed millimetre threshold cannot straddle that: 8 mm found the kraft mailer and
missed the padded one entirely.

So nothing in the estimate is a fixed depth. Per frame:

1. A robust plane is fitted to the mailer, **re-fitting while discarding the
   raised side** so the product cannot drag up the surface it is measured
   against. This is what removes mailer tilt, which on its own was comparable
   to the whole signal.
2. The elevation field is low-passed at a fraction of the package's own size.
   Kraft wrinkles are high-frequency and the dome is not, which is what lets the
   cut sit well below the wrinkle amplitude.
3. Dome height is read as `p99.5 − p25` of that field. The low baseline matters:
   against the median, a product covering much of the mailer pulls the baseline
   up into its own dome and looks weak.
4. The footprint is the **half-maximum contour** — a fraction of the dome's own
   height, so it lands in the same relative place on any stock — and the centre
   is the elevation-weighted centroid, which leans on the peak rather than on
   wherever the contour happened to cut.
5. Thin spurs are trimmed off (`product_inside_trim_frac`), and the result is
   closed and hole-filled into one solid region.

Step 5's trim exists because where the mailer runs off the edge of the product
it keeps sloping, and that ramp can hold above the half-max cut well past the
product while staying narrow. It cannot be rejected on height — it reaches the
same elevations the genuine far side of the dome does — so it is cut on width
instead, with a kernel scaled to the blob's own span. Opening *by
reconstruction* does not work here: the spur tapers continuously into the body
rather than joining it at a neck, so the reconstruction just re-grows it.

Step 5 is mostly not what makes it solid. The measurement stereo **drops out in
horizontal streaks over low-texture kraft** — around 8% of the mailer on a
typical capture, against ~3% for the classification stream — and requiring each
pixel to have its own depth reading let every streak carve a notch into the
product. So candidacy is judged on the smoothed field wherever the blur had
enough valid support around it (`product_inside_min_blur_support`), which
bridges a dropout narrower than the kernel. On the 11-06-08 capture that lifts
solidity from 0.86 to 0.97 and removes a truncation that had been pulling the
reported centre ~6 mm off. `product_inside_close_frac` is the backstop for a
dropout too wide to bridge: inert on good captures, but it is what keeps a dome
split by a ~60 px band from collapsing onto one half.

**Footprint vs core.** Those steps produce the *footprint* — the whole raised
region, drape shoulders included. It is deliberately generous, because a
mailer's creases and slope make the true product outline unsegmentable, and it
is a good measure of extent. But its centroid answers "where is the bulge", not
"where is the product": how far the sheet slopes off each side has nothing to do
with the product, and the two centres differ by **8-11 mm** on the captures
measured.

So the reported centre and depth come from the **core**: the footprint with the
*sloping* part removed. The product is what the sheet is resting on, and
everywhere the mailer runs off the product it descends continuously — so the
core keeps only what is flat enough to read as supported, cut at
`product_inside_core_max_slope_frac` of the slope present in that frame's own
footprint. It is saved as `product_core_mask_*.png` and drawn as a second
magenta layer inside the yellow footprint.

Measured against three hand-outlined captures (products of 140x90, 120x120 and
90x90 mm, in different positions), this roughly halved the centre error:

| estimator | mean error vs outline |
|---|---|
| footprint centroid | 52 mm |
| dome peak | 65 mm |
| height-contour core (previous) | 59 mm |
| **slope-removal core + smaller erode band** | **26 mm** |

Two things drove that. The slope removal is the larger part. The other is
`product_inside_edge_erode_frac`, cut from 0.12 to 0.05: at 0.12 the border band
was reaching into the product itself on 2 of the 3 captures, clipping the
footprint before the dome had finished falling and pulling the centre 13-20 mm
away from the clipped side.

**~26 mm looks like the floor for a depth-only method, and the search for
better has been done.** Everything below was measured against the same three
outlines; nothing shape-based separates from the rest:

| approach | mean | worst |
|---|---|---|
| footprint centroid | 47 mm | 63 mm |
| height contour (best of 0.55-0.85) | 41 mm | 57 mm |
| **slope-removal core (shipped)** | **26 mm** | 34 mm |
| height + slope combined | 33 mm | — |
| bias toward the measured steep edge | 25 mm | 30 mm |
| Hessian blob-vs-ridge | 97 mm | 164 mm |
| fusing both depth streams | 34 mm | 42 mm |
| known product size, anchored at its edge | 20 mm | 28 mm |

The strongest evidence that this is a floor rather than a tuning failure: the
**same algorithm on the two depth streams of the same scene disagrees by up to
26 mm** (11-51-07: 36 mm on measurement, 10 mm on classification, and the
ranking flips on the next capture). The differences between the top methods are
smaller than that, so choosing between them on these numbers would be fitting
noise. The ground truth itself is hand-drawn and mapped from screenshots, with
a demonstrated 12-54% size error, so its own uncertainty is comparable.

Only supplying the product's true size broke out of the pack, and it needs
information the station does not have at runtime.

Product-inside therefore reads the **classification** depth, not the
measurement depth the rest of package mode measures from. That is on principle
rather than on the table above: it needs a few-mm *relative* bulge, so
resolution and density beat absolute accuracy — full `rgb_size` against
640x400 halves depth quantisation, and 97-98% valid pixels against 91-94% is
what stops dropouts tearing the footprint.

Treat the point as an estimate, not a measurement. The product side *is*
measurable, incidentally — the footprint end that falls off more steeply was
the product's end on all three captures — but converting that into a position
still needs an extent the depth cannot supply.

Two approaches that look obvious and do NOT work, so they don't get retried:

- **A height contour** ("find the flat top"). The drape rounds every edge, so
  even lightly smoothed the surface is a dome (0.49 → 1.01 → 0.60 across a
  product), never a plateau with a step. Worse, the dome is asymmetric: a
  product spans elevations 0.77 → 1.01 → 0.60, which no symmetric contour can
  sit on correctly.
- **RGB crease texture.** The sheet is taut over the product and slack around
  it, and that is real at the footprint level (2.4-2.7x more crease energy
  outside the footprint than on it, which independently corroborates the
  footprint). Inside the footprint it is almost uniform, so it cannot localise
  the product.

If nothing in the footprint is flat enough, `core_found` is false and the centre
falls back to the footprint.

The one gate that is not self-scaling by construction is detection itself:
`product_inside_min_peak_noise_multiple` asks how many multiples of *this
frame's own measured depth noise* the dome must clear. The saved JSON reports
`dome_mm`, `noise_mm` and `dome_noise_multiple` — that ratio is the number to
read when a product is missed or invented.

**This gate wants calibrating against empty mailers**, which is the one case no
capture covers yet. An empty mailer's `dome_noise_multiple` is the
false-positive floor; the setting belongs between it and what a loaded mailer of
the same stock reports. Replay saved captures with:

```bash
python -m borg_vision.cli.tune_product_inside <capture_dir> [...] --sweep
```

which re-runs the detector over a capture's `.npy` depth dumps with no camera or
SAM2, and reports centre stability across the relative knobs.

### Diagnostics deliberately NOT in the JSON

Three blocks used to be written into `object`'s JSON and were removed once it was
confirmed nothing read them back. They were records of how a detection was set up
and why it chose the mask it did — useful while tuning, noise afterwards. The
values all still exist, just not in the result file:

| Was in JSON | Where it lives now | What it told you |
|---|---|---|
| `roi` | `cfg.roi_x1/y1/x2/y2` | The crop SAM2 runs on. Every ROI-relative pixel has `roi_x1`/`roi_y1` added back to reach full-frame coordinates, so a wrong ROI shifts every measured point. |
| `depth_alignment` | `cfg.depth_align_scale_x/y`, `cfg.depth_align_x_shift_px` | Stereo depth does not land on the RGB pixels; it is scaled ~1.25x and shifted before sampling (applied in `camera.py`). Check this first if depth looks offset from the image. |
| `selected_mask` | the annotated PNG, and `result["object"]` in memory | Why one SAM2 mask beat the others: `score` (weighted total), `area_ratio`, `rectangularity`, `aspect_ratio`, `center_score`, `bbox_roi`. This is the block to reach for when segmentation grabs the wrong thing — a merged box+tray, or a mask that swallowed the surface underneath. |

Re-add them to `visualization/object.py`'s `make_json_result` if a segmentation
problem needs diagnosing; the underlying data never went away.

`inspection` differs: the captured JPEGs (and their optimized copies) already
live under `cfg.save_dir/<request_id>/camera_{1,2}/`; `save_debug(result)` adds
`inspection_result.json` (the verdict, confidence, reasons, image paths). There
is no annotated overlay or `draw_result`.

## Units and frames

Library results are in **millimeters / degrees**, in the camera optical frame
(x right, y down, z forward) — the ROS2 layer converts to meters/radians and
transforms into the requested frame. JSON files mirror the original scripts'
schemas exactly. (The `inspection` mode is the exception: it has no geometric
output — only the verdict `result` + `confidence`.)

## Adding a new mode

> This recipe is for **SAM2 single-camera** modes. The `inspection` mode is a
> different shape — a standalone `InspectionDetector` (two cameras, OpenAI, no
> `BaseDetector`) built via `get_inspection_detector()` — so it does not follow
> these steps.

Each mode is a mechanical port of one `run_X()` in
`unified_detector_all_in_one.py`:

1. `config/<mode>.py` — a `<Mode>Config(BaseConfig)` whose fields are the
   `run_X` constants (defaults = original values).
2. `detection/<mode>.py` — the scoring/measurement functions, taking `cfg`
   first; reuse the generic helpers in `detection/package.py`
   (`close_mask`, `erode_mask`, `largest_component`,
   `pixel_to_camera_xy_mm`, ...).
3. `visualization/<mode>.py` — `draw_result(cfg, ...)` and
   `make_json_result(cfg, ...)`.
4. `results.py` — a `<Mode>Result(BaseResult)` built from `raw["final_output"]`.
5. `detectors/<mode>.py` — a `<Mode>Detector(BaseDetector)` (set `mode`/
   `requires_barcode`, implement `detect`, `_draw_result`, `_serialize`,
   `save_debug`).
6. Register it in `registry.py` and export from the `__init__.py` files.

Verify parity by comparing `python -m borg_vision --mode <mode>` JSON against
`python unified_detector_all_in_one.py --mode <mode>` on the same captured
frames.
