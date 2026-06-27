# borg_vision

Importable, GUI-free detection library for Borg Robotics, built on this SAM2
fork. It drives OAK-D Pro cameras and SAM2 segmentation behind a single
mode-selectable API, so the same pipelines can be used by the ROS2
`sam2_vision` package in
[borg_cobots](https://github.com/Borg-Robotics/borg_cobots) and from the CLI.

Each mode was extracted verbatim from a validated standalone script
(`*_final.py` / `unified_detector_all_in_one.py`), with the module-level
constants replaced by a config dataclass — behavior is unchanged.

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
| `package` | `package_final_*.png`, `package_mask_*.png`, `product_inside_mask_*.png`, `package_depth_heatmap_*.png`, `package_final_*.json` |
| `box` | `cardboard_box_measurements_*.png`, `cardboard_box_mask_*.png`, `cardboard_box_measurements_*.json` |
| `object` | `object_segment_depth_*.png`, `object_mask_*.png`, `depth_aligned_*.npy`, `object_segment_depth_*.json` |
| `polymailer` | `polymailer_output_*.png`, `polymailer_mask_*.png`, `product_bulge_mask_*.png`, `polymailer_output_*.json` |
| `clear_bag` | `clear_bag_final_output_*.png`, `clear_bag_product_mask_*.png`, `clear_bag_mask_*.png`, `clear_bag_depth_heatmap_*.png`, `clear_bag_final_output_*.json` |

The annotated `*.png` is a side-by-side of RGB + depth (+ heatmap for modes
that produce one). `draw_result(result)` returns the same image in memory
without writing to disk.

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
