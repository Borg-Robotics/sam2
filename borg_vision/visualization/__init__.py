"""Visualization helpers.

Shared helpers live in visualization/common.py; per-mode overlays/JSON live in
visualization/<mode>.py. Everything is re-exported here so existing imports
(`from borg_vision.visualization import draw_result, ...`) keep working.
"""

from .common import (
    draw_roi_axes,
    make_depth_vis,
    make_live_depth_heatmap,
    put_text_lines,
    save_binary_mask,
)
from .package import (
    draw_result,
    draw_rotated_or_axis_box,
    make_json_result,
    make_package_depth_heatmap,
    save_accepted_masks,
)

__all__ = [
    # shared
    "draw_roi_axes",
    "make_depth_vis",
    "make_live_depth_heatmap",
    "put_text_lines",
    "save_binary_mask",
    # package mode
    "draw_result",
    "draw_rotated_or_axis_box",
    "make_json_result",
    "make_package_depth_heatmap",
    "save_accepted_masks",
]
