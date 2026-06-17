# borg_vision

Importable, GUI-free detection library for Borg Robotics, built on this SAM2
fork. Extracted from the validated standalone script
`product_detection_final.py` (which is now a thin CLI shim over this
library), so the same pipeline can be driven by the ROS2 `sam2_vision`
package in [borg_cobots](https://github.com/Borg-Robotics/borg_cobots) and by
the interactive CLI.

## Layout

The library is organized around a shared OAK-D + SAM2 base with one subpackage
per concern, so each detection mode (package today; object/box/clear_bag/
polymailer as they are ported) plugs in its own config, detection logic,
detector and visualization.

| Path | Contents |
|---|---|
| `config/` | `BaseConfig` (shared camera/SAM2/ROI/depth fields) + per-mode configs (`PackageConfig`, …). `from_yaml()`/`from_dict()` for partial overrides. `ProductDetectionConfig` is an alias of `PackageConfig`. |
| `camera.py` | `OakCamera` — depthai v3 pipeline ownership (RGB + stereo depth, IR/exposure), optional MxID selection for multi-camera, depth→RGB alignment, `get_frames()` |
| `barcode.py` | pyzbar barcode detection + overlay |
| `detection/` | Pure detection logic per mode (SAM2 mask scoring, classification, dimensions, depth stats). `detection/package.py` holds `run_sam_package_depth_type()`. |
| `visualization/` | `common.py` (shared depth vis/heatmaps/axes) + per-mode overlays, heatmaps, save helpers, JSON result building |
| `results.py` | `BaseResult` + per-mode result dataclasses. `ProductDetectionResult` is an alias of `PackageResult`. |
| `detectors/` | `BaseDetector` (shared lifecycle: load_model/open/warmup/scan_barcode/save_debug) + per-mode detectors. `ProductDetector` is an alias of `PackageDetector`. |
| `registry.py` | `MODES` + `get_detector(mode, cfg=None, mxid=None)` factory and `available_modes()` |
| `cli/product_detection.py` | The original interactive SPACE/q loop, rebuilt on the library |

Function bodies in `detection/`/`visualization/` are verbatim ports of the
original scripts with module constants replaced by `cfg` fields — behavior is
unchanged (see the parity check in borg_cobots `docs/VISION_TESTING.md` §7).

> **Note (current state):** the multi-mode scaffolding is in place and the
> `package` mode is fully ported. `object`, `box`, `clear_bag` and `polymailer`
> are being ported from `unified_detector_all_in_one.py` into this structure;
> until then they remain registered placeholders in `registry.py`.

## Install

```bash
pip install -e ".[borg]"   # adds depthai, pyzbar, opencv, PyYAML on top of sam2
# checkpoint: checkpoints/download_ckpts.sh (sam2.1_hiera_small.pt)
```

## Usage

The mode-agnostic entry point is `get_detector(mode, ...)`:

```python
from borg_vision import get_detector

detector = get_detector("package", mxid=None)   # mxid selects a camera in multi-OAK setups
# cfg override: get_detector("package", cfg="overrides.yaml") or pass a PackageConfig instance
detector.load_model()
detector.open()
detector.warmup()
```

Or use the package class / aliases directly (equivalent):

```python
from borg_vision import ProductDetector, ProductDetectionConfig

cfg = ProductDetectionConfig()            # or ProductDetectionConfig.from_yaml("overrides.yaml")
detector = ProductDetector(cfg, mxid=None)
detector.load_model()
detector.open()
detector.warmup()

result = detector.detect_after_barcode(timeout_s=30.0)   # None on timeout / no mask
if result is not None:
    print(result.package_type, result.package_type_confidence_percent,
          result.length_mm, result.width_mm, result.center_x_mm,
          result.center_y_mm, result.top_face_depth_mm)
    detector.save_debug(result)           # same artifact set as the original script

detector.close()
```

`warmup()` and `scan_barcode()` accept a `should_abort` callable polled once
per frame — this is how the ROS2 action server wires goal cancellation in.

Units and frame: millimeters/degrees, camera frame x-right / y-down /
z-forward (= ROS optical frame convention). The ROS layer converts to
meters/radians.

## Interactive CLI

```bash
python product_detection_final.py     # unchanged UX: SPACE = scan, q = quit
```

## Adding further detectors

`clear_bag_final.py`, `polymailer_final.py`, etc. still run standalone. To
bring one into the library, follow the same pattern: constants →
`<Task>Config` dataclass, pure functions into a module, a `<Task>Detector`
facade reusing `OakCamera`, and a new `profile` value in the ROS action.
