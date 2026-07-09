"""Configuration for the product-inspection mode.

Inspection does NOT use SAM2 or stereo depth, so it deliberately does not
subclass BaseConfig (whose fields are all SAM2/depth specific). It only carries
the dual-camera RGB capture parameters and the OpenAI request parameters. The
from_dict()/from_yaml() loaders mirror BaseConfig so callers (the ROS node, the
CLI) get the same partial-override support.

The OpenAI API key is intentionally absent: it is read from the OPENAI_API_KEY
environment variable at request time, never from config.
"""

import dataclasses
import os
from dataclasses import dataclass


@dataclass
class InspectionConfig:
    # ----- output -------------------------------------------------------
    save_dir: str = "inspection_results"

    # ----- dual-camera RGB capture -------------------------------------
    capture_count: int = 1        # fresh frames per SELECTED camera per orientation
    # Which orientations each camera captures at. None -> every camera captures
    # at every orientation (legacy behavior). Otherwise a per-camera map:
    #   {"camera_1": [0], "camera_2": "all"}
    # values are "all" or a list of 0-based orientation indices (0 = default pose).
    capture_schedule: dict = None
    rgb_size: tuple = (1280, 720)
    fps: int = 20
    startup_delay_seconds: float = 2.0
    capture_jpeg_quality: int = 95
    force_usb2: bool = False

    # ----- OpenAI request ----------------------------------------------
    model: str = "gpt-5.5"
    openai_url: str = "https://api.openai.com/v1/responses"
    max_size: int = 1600          # max optimized image dimension sent to the API
    quality: int = 72             # JPEG quality of the optimized API copies
    timeout_s: float = 180.0

    _TUPLE_FIELDS = ("rgb_size",)

    @classmethod
    def _tuple_fields(cls):
        return cls._TUPLE_FIELDS

    @classmethod
    def from_dict(cls, data):
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid

        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

        tuple_fields = set(cls._tuple_fields())
        kwargs = {}

        for key, value in data.items():
            if key in tuple_fields and isinstance(value, (list, tuple)):
                value = tuple(value)
            kwargs[key] = value

        cfg = cls(**kwargs)
        # Allow the model to be overridden by OPENAI_MODEL env when not set in cfg.
        if "model" not in data and os.environ.get("OPENAI_MODEL"):
            cfg.model = os.environ["OPENAI_MODEL"]
        return cfg

    @classmethod
    def from_yaml(cls, path):
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    # ----- capture schedule --------------------------------------------

    def cameras_for(self, orientation_index, num_cameras=2):
        """0-based device indices that capture at `orientation_index`.

        Resolves `capture_schedule` (a per-camera map keyed "camera_1".."camera_N",
        each value "all" or a list of 0-based orientation indices). A camera not
        listed captures nothing. When `capture_schedule` is None/empty every
        camera captures at every orientation (legacy behavior).

        Raises ValueError on a malformed schedule (bad camera key or a value that
        is neither "all" nor an int list).
        """
        schedule = self.capture_schedule
        if not schedule:
            return list(range(num_cameras))

        valid_keys = {f"camera_{n}" for n in range(1, num_cameras + 1)}
        selected = []
        for key, when in schedule.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid capture_schedule camera '{key}'; "
                    f"expected one of {sorted(valid_keys)}"
                )
            cam_index = int(key.split("_")[1]) - 1  # camera_1 -> 0

            if isinstance(when, str):
                if when != "all":
                    raise ValueError(
                        f"Invalid capture_schedule value for {key}: '{when}'; "
                        "expected \"all\" or a list of orientation indices"
                    )
                selected.append(cam_index)
            elif isinstance(when, (list, tuple)):
                if not all(isinstance(i, int) for i in when):
                    raise ValueError(
                        f"capture_schedule[{key}] must be a list of int indices"
                    )
                if orientation_index in when:
                    selected.append(cam_index)
            else:
                raise ValueError(
                    f"Invalid capture_schedule value for {key}: {when!r}; "
                    "expected \"all\" or a list of orientation indices"
                )

        return sorted(selected)
