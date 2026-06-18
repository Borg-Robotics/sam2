"""Mode registry and detector factory.

`get_detector(mode, ...)` is the single entry point the CLI and the ROS2 nodes
use to obtain a configured detector for a given station/mode. New modes are
registered in MODES as they are ported from unified_detector_all_in_one.py.
"""

from .config import (
    BoxConfig,
    ClearBagConfig,
    ObjectConfig,
    PackageConfig,
    PolymailerConfig,
)
from .detectors import (
    BoxDetector,
    ClearBagDetector,
    ObjectDetector,
    PackageDetector,
    PolymailerDetector,
)

#: mode name -> (detector class, config class)
MODES = {
    "package": (PackageDetector, PackageConfig),
    "object": (ObjectDetector, ObjectConfig),
    "box": (BoxDetector, BoxConfig),
    "polymailer": (PolymailerDetector, PolymailerConfig),
    "clear_bag": (ClearBagDetector, ClearBagConfig),
}

# Aliases so callers can use either the station name or the legacy "product".
MODE_ALIASES = {
    "product": "package",
}


def available_modes():
    """Return the list of registered mode names (excluding aliases)."""
    return sorted(MODES)


def resolve_mode(mode):
    """Normalize a mode/alias string to a registered mode name."""
    key = MODE_ALIASES.get(mode, mode)
    if key not in MODES:
        raise ValueError(
            f"Unknown mode {mode!r}; available: {available_modes()} "
            f"(aliases: {sorted(MODE_ALIASES)})"
        )
    return key


def config_class_for(mode):
    """Return the config dataclass for a mode (e.g. to load YAML overrides)."""
    return MODES[resolve_mode(mode)][1]


def get_detector(mode, cfg=None, mxid=None, torch_device=None, camera=None):
    """Build a detector for `mode`.

    cfg may be None (use the mode's default config), a config instance, or a
    path to a YAML file of overrides for the mode's config class.

    `camera` optionally injects a pre-built, caller-owned OakCamera so several
    detectors can share one physical device (multi-mode-per-camera). When given,
    `mxid` is ignored (the shared camera already owns the device) and the
    detector will not open/close it.
    """
    detector_class, config_class = MODES[resolve_mode(mode)]

    if cfg is None:
        cfg = config_class()
    elif isinstance(cfg, str):
        cfg = config_class.from_yaml(cfg)

    return detector_class(cfg, mxid=mxid, torch_device=torch_device, camera=camera)
