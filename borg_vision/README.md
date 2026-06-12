# borg_vision

Importable, GUI-free detection library for Borg Robotics, built on this SAM2
fork. Extracted from the validated standalone script
`product_detection_final.py` (which is now a thin CLI shim over this
library), so the same pipeline can be driven by the ROS2 `sam2_vision`
package in [borg_cobots](https://github.com/Borg-Robotics/borg_cobots) and by
the interactive CLI.

## Layout

| Module | Contents |
|---|---|
| `config.py` | `ProductDetectionConfig` dataclass — every tuning constant of the original script (1:1, lowercased) with the validated values as defaults; `from_yaml()` for partial overrides |
| `camera.py` | `OakCamera` — depthai v3 pipeline ownership (RGB + dual stereo depth, IR/exposure), optional MxID selection for multi-camera, depth→RGB alignment, `get_frames()` |
| `barcode.py` | pyzbar barcode detection + overlay |
| `detection.py` | All pure detection logic: SAM2 mask scoring, box/polymailer classification, dimensions, depth statistics, `run_sam_package_depth_type()` |
| `visualization.py` | Result overlays, heatmaps, save helpers, JSON result building |
| `product_detector.py` | `ProductDetector` + `ProductDetectionResult` — the high-level API |
| `cli/product_detection.py` | The original interactive SPACE/q loop, rebuilt on the library |

Function bodies in `detection.py`/`visualization.py` are verbatim ports of
the original script with module constants replaced by `cfg` fields — behavior
is unchanged (see the parity check in borg_cobots
`docs/VISION_TESTING.md` §7).

## Install

```bash
pip install -e ".[borg]"   # adds depthai, pyzbar, opencv, PyYAML on top of sam2
# checkpoint: checkpoints/download_ckpts.sh (sam2.1_hiera_small.pt)
```

## Usage

```python
from borg_vision import ProductDetector, ProductDetectionConfig

cfg = ProductDetectionConfig()            # or ProductDetectionConfig.from_yaml("overrides.yaml")
detector = ProductDetector(cfg, mxid=None)  # mxid selects a camera in multi-OAK setups
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
