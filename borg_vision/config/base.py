"""Shared configuration base for all detection modes.

`BaseConfig` holds the fields common to every OAK-D + SAM2 detection mode:
camera/pipeline setup, SAM2 automatic-mask-generator parameters, ROI, depth
alignment/validity, IR and stereo exposure. Each mode subclasses this and adds
its own scoring/post-processing fields (see config/package.py, config/box.py,
etc.). The from_dict()/from_yaml() loaders live here so every mode inherits
partial-override support.
"""

import dataclasses
from dataclasses import dataclass


@dataclass
class BaseConfig:
    # ----- output -------------------------------------------------------
    save_dir: str = "detection_results"

    # ----- camera / pipeline -------------------------------------------
    fps: int = 20
    warmup_seconds: float = 5.0

    rgb_size: tuple = (1280, 720)
    stereo_size: tuple = (640, 400)

    # Whether the OAK pipeline builds a second, lower-resolution stereo
    # stream dedicated to measurement depth. Modes that measure from a
    # separate sparse-but-accurate depth map (package, box) set this True;
    # modes that reuse the classification depth (object, clear_bag,
    # polymailer) set it False and depth_measure falls back to depth_class.
    needs_measurement_stereo: bool = True

    ir_laser_intensity: float = 1.0
    ir_flood_intensity: float = 0.0
    confidence_threshold: int = 120

    use_subpixel: bool = True
    use_left_right_check: bool = True

    use_manual_stereo_exposure: bool = True
    stereo_exposure_us: int = 1000
    stereo_iso: int = 400

    # ----- SAM2 model ---------------------------------------------------
    checkpoint: str = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam_points_per_side: int = 24
    sam_pred_iou_thresh: float = 0.78
    sam_stability_score_thresh: float = 0.82
    sam_min_mask_region_area: int = 500

    # ----- ROI (detection region in the RGB frame) ---------------------
    roi_x1: int = 250
    roi_y1: int = 60
    roi_x2: int = 1020
    roi_y2: int = 700

    # ----- depth alignment (depth -> RGB) ------------------------------
    depth_align_x_shift_px: int = 35
    depth_align_y_shift_px: int = 0
    depth_align_scale_x: float = 1.25
    depth_align_scale_y: float = 1.25

    # ----- depth validity ----------------------------------------------
    min_valid_depth_mm: float = 450
    max_valid_depth_mm: float = 1200

    base_depth_mm: float = 695.0
    measurement_depth_offset_mm: float = 10.0

    # Tuple fields that from_dict() must coerce from YAML lists. Subclasses
    # that add tuple fields should extend this in from_dict via _tuple_fields().
    _TUPLE_FIELDS = ("rgb_size", "stereo_size")

    @classmethod
    def _tuple_fields(cls):
        return cls._TUPLE_FIELDS

    @classmethod
    def from_dict(cls, data):
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid

        if unknown:
            raise ValueError(
                f"Unknown {cls.__name__} keys: {sorted(unknown)}"
            )

        tuple_fields = set(cls._tuple_fields())
        kwargs = {}

        for key, value in data.items():
            if key in tuple_fields and isinstance(value, (list, tuple)):
                value = tuple(value)
            kwargs[key] = value

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path):
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)
