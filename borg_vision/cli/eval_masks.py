"""Batch-evaluate a mode's segmentation rules over a tree of images, then browse
the overlays interactively.

No camera needed. For every image found, this runs the RGB half of a mode's
pipeline -- SAM 2 on the ROI, then that mode's gate/score/repair rules -- and
writes an overlay, a JSON record and a contact sheet. That is exactly the half
the segmentation rules control, so it catches rule regressions without hardware.

What it CANNOT check: anything depth-derived (dimensions, face depth,
box-vs-polymailer type, product-inside, the clear-bag outline, and therefore
`base_depth_mm`). Those need the camera.

    # 1. run over a tree of images (recursive)
    python -m borg_vision.cli.eval_masks --mode package ~/borg-data/sam2_vision_data \
        --out /tmp/run_new

    # 2. flip through the results: arrow keys, worst-first
    python -m borg_vision.cli.eval_masks --view /tmp/run_new --sort iou

    # 3. A/B two revisions of the rules
    python -m borg_vision.cli.eval_masks --mode package DIR --out /tmp/run_old   # on the old code
    python -m borg_vision.cli.eval_masks --mode package DIR --out /tmp/run_new \
        --compare /tmp/run_old

Where a capture directory also holds the mask the deployed code selected (e.g.
`package_mask.png` next to `raw_rgb.jpg`), it is drawn in green as a reference
and scored as IoU. Treat that as a reference, not ground truth -- if the rules
just fixed a case the old code got wrong, a *low* IoU is the improvement.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import (
    BoxConfig,
    ClearBagConfig,
    ObjectConfig,
    PackageConfig,
    PolymailerConfig,
)

# Colours are BGR. Green = the mask recorded next to the image; red = this run.
RECORDED_COLOR = (0, 200, 0)
SELECTED_COLOR = (0, 0, 255)

# Image files a borg capture directory holds that are OUTPUTS, not inputs.
# Without these a `**/*.png` glob would feed masks and overlays back in.
DEFAULT_EXCLUDES = (
    "*_mask.png",
    "*_final.png",
    "*_final_output.png",
    "*_output.png",
    "*_measurements.png",
    "*_heatmap.png",
    "*_segment_depth.png",
    "*overlay*.png",
    "*sheet*.png",
)

DEFAULT_GLOBS = ("**/*.jpg", "**/*.jpeg", "**/*.png")


def _package_selector(cfg, masks, roi_rgb):
    """Mirrors run_sam_package_depth_type up to the depth stage."""
    from ..detection.package import (
        choose_best_package_mask,
        choose_cardboard_box_mask_exact,
        map_dedicated_box_candidate_to_package_roi,
    )

    dedicated = None

    if cfg.dedicated_box_sam_enable:
        dedicated = map_dedicated_box_candidate_to_package_roi(
            cfg,
            choose_cardboard_box_mask_exact(cfg, masks, roi_rgb),
        )

    best, _accepted = choose_best_package_mask(cfg, masks, roi_rgb, dedicated)

    return best


def _box_selector(cfg, masks, roi_rgb):
    from ..detection.box import choose_cardboard_box_mask

    return choose_cardboard_box_mask(cfg, masks, roi_rgb)


def _object_selector(cfg, masks, roi_rgb):
    from ..detection.object import choose_best_object_mask

    return choose_best_object_mask(cfg, masks, roi_rgb)


def _polymailer_selector(cfg, masks, roi_rgb):
    from ..detection.polymailer import choose_polymailer_mask

    return choose_polymailer_mask(cfg, masks, roi_rgb)


def _clear_bag_selector(cfg, masks, roi_rgb):
    """Product mask only -- the bag outline itself is derived from depth."""
    from ..detection.clear_bag import choose_best_product_mask

    return choose_best_product_mask(cfg, masks, roi_rgb)


MODES = {
    "package": (PackageConfig, _package_selector, "package_mask.png"),
    "box": (BoxConfig, _box_selector, "cardboard_box_mask.png"),
    "object": (ObjectConfig, _object_selector, "object_mask.png"),
    "polymailer": (PolymailerConfig, _polymailer_selector, "polymailer_mask.png"),
    "clear_bag": (ClearBagConfig, _clear_bag_selector, "clear_bag_product_mask.png"),
}

# Fields diffed by --compare. Modes populate different subsets; a key missing on
# both sides is not a difference.
COMPARED_FIELDS = (
    "source",
    "index",
    "merged_from_indices",
    "score",
    "area_ratio",
    "rectangularity",
    "color_score",
    "selection_override",
    "merged_geometry_mode",
)


# --------------------------------------------------------------------- discovery

def discover(paths, globs, excludes):
    """Every image under `paths`, recursively, minus the excluded patterns."""
    found = []

    for raw in paths:
        root = Path(raw).expanduser()

        if root.is_file():
            found.append((root.parent, root))
            continue

        if not root.is_dir():
            raise SystemExit(f"not found: {root}")

        for pattern in globs:
            found.extend((root, p) for p in root.glob(pattern) if p.is_file())

    kept = []
    skipped = 0

    for root, path in sorted(set(found)):
        if any(path.match(pattern) for pattern in excludes):
            skipped += 1
            continue

        kept.append((root, path))

    return kept, skipped


def key_for(root, path):
    """Collision-free slug from the path relative to its search root."""
    relative = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)

    return "__".join(relative.with_suffix("").parts) or path.stem


# ------------------------------------------------------------------------- model

def build_mask_generator(cfg, device=None):
    """Same construction as BaseDetector.load_model, without a camera."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam2(cfg.model_cfg, cfg.checkpoint, device=device)

    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=cfg.sam_points_per_side,
        pred_iou_thresh=cfg.sam_pred_iou_thresh,
        stability_score_thresh=cfg.sam_stability_score_thresh,
        min_mask_region_area=cfg.sam_min_mask_region_area,
    )

    return generator, device


# ---------------------------------------------------------------------- scoring

def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = int(np.logical_or(a, b).sum())

    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def summarize(info, mask_count):
    """The segmentation-only subset of a result, for diffing and sorting."""
    if info is None:
        return {"selected": False, "sam_mask_count": mask_count}

    def rounded(key):
        value = info.get(key)
        return None if value is None else round(float(value), 4)

    return {
        "selected": True,
        "sam_mask_count": mask_count,
        "source": info.get("source"),
        "index": info.get("index"),
        "merged_from_indices": info.get("merged_from_indices"),
        "score": rounded("score"),
        "area_ratio": rounded("area_ratio"),
        "rectangularity": rounded("rectangularity"),
        "aspect_ratio": rounded("aspect_ratio"),
        "color_score": rounded("color_score"),
        "selection_override": info.get("selection_override"),
        "merged_geometry_mode": info.get("merged_geometry_mode"),
        "merged_union_fill_ratio": rounded("merged_union_fill_ratio"),
        "merged_area_growth": rounded("merged_area_growth"),
        "mask_area_px": int(info["mask"].sum()),
    }


def diff_fields(new, baseline):
    changed = {}

    for field in COMPARED_FIELDS:
        if field not in baseline and field not in new:
            continue

        if new.get(field) != baseline.get(field):
            changed[field] = (baseline.get(field), new.get(field))

    return changed


# ------------------------------------------------------------------- rendering

def put_label(image, lines, scale=0.62, line_px=28, y=30):
    """Text on a translucent dark plate, so it reads over any background.

    A plate rather than a stroke halo: drawing the same string twice at
    different thicknesses renders it at different *widths*, which leaves the
    tail of the thick pass sticking out uncovered.
    """
    if not lines:
        return image

    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        for line in lines
    ]
    top = max(y - int(line_px * 0.8), 0)
    bottom = min(y + line_px * (len(lines) - 1) + int(line_px * 0.35), image.shape[0])
    right = min(10 + max(widths) + 8, image.shape[1])

    plate = image.copy()
    cv2.rectangle(plate, (0, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(plate, 0.55, image, 0.45, 0, dst=image)

    for line in lines:
        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_px

    return image


def render(roi_rgb, mask, recorded, fill=False):
    """ROI with the recorded mask outlined green and this run's outlined red.

    Outline-only by default: a filled overlay hides the cardboard seams and
    shading you need to see in order to judge whether the mask is right.
    """
    vis = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

    if fill and mask is not None:
        tint = np.zeros_like(vis)
        tint[mask.astype(bool)] = SELECTED_COLOR
        vis = cv2.addWeighted(vis, 1.0, tint, 0.25, 0)

    for m, color in ((recorded, RECORDED_COLOR), (mask, SELECTED_COLOR)):
        if m is None:
            continue

        contours, _ = cv2.findContours(
            m.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, color, 3)

    return vis


def overlay_lines(record, label):
    """The full annotation for a per-image overlay."""
    lines = [label]

    if record.get("selected"):
        lines.append(
            f"{record.get('source')}  score={record.get('score')}  "
            f"area={record.get('area_ratio')}  rect={record.get('rectangularity')}"
        )
        extra = []

        if record.get("selection_override"):
            extra.append(str(record["selection_override"]))
        if record.get("merged_geometry_mode"):
            extra.append(f"geometry={record['merged_geometry_mode']}")
        if record.get("merged_union_fill_ratio") is not None:
            extra.append(f"union_fill={record['merged_union_fill_ratio']}")
        if record.get("iou_vs_recorded") is not None:
            extra.append(f"IoU_vs_recorded={record['iou_vs_recorded']:.3f}")

        if extra:
            lines.append("  ".join(extra))
    else:
        lines.append("NO MASK SELECTED")

    return lines


def fit_text(text, max_width, scale, thickness=1):
    """Trim `text` until it fits `max_width`, so it cannot bleed into the next cell."""
    while text:
        (width, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        if width <= max_width:
            break

        text = text[:-1]

    return text


def make_cell(clean, record, label, size=(400, 333)):
    """A contact-sheet cell: downscale first, then label, so text stays legible."""
    cell = cv2.resize(clean, size)
    source = (
        (record.get("source") or "NO MASK")
        .replace("cardboard_box_", "")
        .replace("package_roi_", "")
        .replace("_rules", "")
    )
    bits = [label.removeprefix("20").removeprefix("26-"), source]
    iou = record.get("iou_vs_recorded")

    if iou is not None:
        bits.append(f"IoU {iou:.3f}")

    text = fit_text("  ".join(bits), size[0] - 20, 0.44)

    return put_label(cell, [text], scale=0.44, line_px=20, y=20)


def write_sheets(out_dir, cells, cols, rows):
    """Contact sheets, `cols`x`rows` cells each."""
    if not cells:
        return []

    cell_h, cell_w = cells[0].shape[:2]
    blank = np.zeros((cell_h, cell_w, 3), np.uint8)
    per_sheet = cols * rows
    written = []

    for start in range(0, len(cells), per_sheet):
        chunk = list(cells[start:start + per_sheet])

        while len(chunk) % cols:
            chunk.append(blank)

        grid = [
            np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)
        ]
        path = out_dir / f"sheet_{start // per_sheet:02d}.png"
        cv2.imwrite(str(path), np.vstack(grid))
        written.append(path)

    return written


# ---------------------------------------------------------------------- viewer

# cv2 returns different codes for arrows depending on backend; accept both.
KEY_PREV = {81, 2424832, 65361, ord("a"), ord("h")}
KEY_NEXT = {83, 2555904, 65363, ord("d"), ord("l")}
KEY_FIRST = {80, 2359296, 65360}
KEY_LAST = {87, 2293760, 65367}
KEY_QUIT = {27, ord("q")}


def view(run_dir, sort_by="path"):
    """Flip through a run's overlays with the arrow keys."""
    import json

    run_dir = Path(run_dir).expanduser()
    overlay_dir = run_dir / "overlay"

    if not overlay_dir.is_dir():
        raise SystemExit(f"no overlay/ directory in {run_dir}")

    items = []

    for path in sorted(overlay_dir.glob("*.png")):
        record_path = run_dir / "json" / f"{path.stem}.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        items.append((path, record))

    if not items:
        raise SystemExit(f"no overlays in {overlay_dir}")

    if sort_by == "iou":
        items.sort(key=lambda it: (
            it[1].get("iou_vs_recorded") if it[1].get("iou_vs_recorded") is not None else 2.0
        ))
    elif sort_by == "score":
        items.sort(key=lambda it: (it[1].get("score") or 0.0))
    elif sort_by == "area":
        items.sort(key=lambda it: (it[1].get("mask_area_px") or 0))
    elif sort_by == "changed":
        # Needs a run made with --compare; unchanged items sort last.
        items.sort(key=lambda it: not it[1].get("changed_vs_baseline"))

    window = f"borg_vision eval — {run_dir.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    i = 0

    print(
        f"{len(items)} overlays, sorted by {sort_by}. "
        "Left/Right or a/d = prev/next, Home/End = first/last, q/Esc = quit."
    )

    while True:
        path, record = items[i]
        frame = cv2.imread(str(path))
        banner = f"[{i + 1}/{len(items)}]  {path.stem}"

        put_label(
            frame,
            [banner],
            y=frame.shape[0] - 14,
        )

        cv2.setWindowTitle(window, f"{window}  {banner}")
        cv2.imshow(window, frame)
        key = cv2.waitKeyEx(0)

        # Closing the window with the title-bar X does not raise a key event on
        # every backend; check visibility so we exit instead of looping blind.
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key in KEY_QUIT:
            break
        if key in KEY_NEXT:
            i = (i + 1) % len(items)
        elif key in KEY_PREV:
            i = (i - 1) % len(items)
        elif key in KEY_FIRST:
            i = 0
        elif key in KEY_LAST:
            i = len(items) - 1
        elif key == -1:  # window closed
            break

    cv2.destroyAllWindows()


# ------------------------------------------------------------------------- main

def run_batch(args):
    import json

    config_class, selector, mask_artifact = MODES[args.mode]
    cfg = config_class.from_yaml(args.config) if args.config else config_class()

    globs = args.glob or list(DEFAULT_GLOBS)
    excludes = list(DEFAULT_EXCLUDES) + list(args.exclude or [])
    images, skipped = discover(args.paths, globs, excludes)

    if args.limit:
        images = images[:args.limit]

    print(f"{len(images)} images ({skipped} skipped as outputs) for mode={args.mode}")

    if not images:
        print("nothing to do — narrow or widen --glob (default: **/*.jpg,jpeg,png)")
        return 0

    out_dir = Path(args.out).expanduser()
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    generator, device = build_mask_generator(cfg, args.device)
    print(f"SAM 2 on {device}: {cfg.checkpoint}")

    baseline_dir = Path(args.compare).expanduser() / "json" if args.compare else None
    overlays = []
    cells = []
    rows = []
    changed = []
    unselected = []
    sources = {}

    for n, (root, path) in enumerate(images, 1):
        frame_bgr = cv2.imread(str(path))

        if frame_bgr is None:
            print(f"[{n}/{len(images)}] {path}: unreadable, skipped")
            continue

        key = key_for(root, path)
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = full_rgb[cfg.roi_y1:cfg.roi_y2, cfg.roi_x1:cfg.roi_x2].copy()

        # The detection modules print a running commentary; keep it out of the
        # batch log unless asked, but preserve it per image.
        trace = io.StringIO()
        redirect = (
            contextlib.nullcontext()
            if args.verbose
            else contextlib.redirect_stdout(trace)
        )

        with redirect, torch.inference_mode():
            masks = generator.generate(roi_rgb)
            info = selector(cfg, masks, roi_rgb)

        record = summarize(info, len(masks))
        record["image"] = str(path)
        record["mode"] = args.mode

        recorded = None
        recorded_path = path.parent / mask_artifact

        if recorded_path.is_file():
            candidate = cv2.imread(str(recorded_path), cv2.IMREAD_GRAYSCALE)

            if candidate is not None and candidate.shape == roi_rgb.shape[:2]:
                recorded = (candidate > 127).astype(np.uint8)

        if info is not None and recorded is not None:
            record["iou_vs_recorded"] = round(mask_iou(info["mask"], recorded), 4)

        baseline = None

        if baseline_dir is not None:
            baseline_path = baseline_dir / f"{key}.json"

            if baseline_path.is_file():
                baseline = json.loads(baseline_path.read_text())

        delta = diff_fields(record, baseline) if baseline else {}

        if baseline is not None:
            record["changed_vs_baseline"] = bool(delta)
            record["changed_fields"] = sorted(delta)

        if delta:
            changed.append((key, delta))

        if info is None:
            unselected.append(key)
        else:
            sources[record["source"]] = sources.get(record["source"], 0) + 1

        clean = render(
            roi_rgb,
            None if info is None else info["mask"],
            recorded,
            fill=args.fill,
        )

        overlay_path = out_dir / "overlay" / f"{key}.png"
        cv2.imwrite(
            str(overlay_path),
            put_label(clean.copy(), overlay_lines(record, key)),
        )
        overlays.append(overlay_path)
        # Short label: the containing directory is what identifies a capture.
        cells.append(make_cell(clean, record, path.parent.name or path.stem))

        (out_dir / "json" / f"{key}.json").write_text(json.dumps(record, indent=2))

        if args.save_trace:
            (out_dir / "json" / f"{key}.log").write_text(trace.getvalue())

        rows.append(record)

        iou = record.get("iou_vs_recorded")
        print(
            f"[{n}/{len(images)}] {key}: "
            f"{record.get('source') or 'NO MASK'} "
            f"score={record.get('score')} "
            f"iou={'n/a' if iou is None else f'{iou:.3f}'}"
            f"{'  CHANGED' if delta else ''}"
        )

        for field, (was, now) in delta.items():
            print(f"      {field}: {was} -> {now}")

    columns = [
        "image", "mode", "selected", "source", "index", "score", "area_ratio",
        "rectangularity", "aspect_ratio", "color_score", "selection_override",
        "merged_geometry_mode", "merged_union_fill_ratio", "merged_area_growth",
        "mask_area_px", "iou_vs_recorded", "sam_mask_count",
    ]

    with open(out_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    sheets = write_sheets(out_dir, cells, args.sheet_cols, args.sheet_rows)

    ious = [r["iou_vs_recorded"] for r in rows if r.get("iou_vs_recorded") is not None]

    print()
    print("=" * 64)
    print(f"{len(rows)} images, mode={args.mode} -> {out_dir}")
    print(f"  no mask selected:  {len(unselected)}")

    if baseline_dir is not None:
        print(f"  changed vs {args.compare}: {len(changed)}")

    if ious:
        print(
            f"  IoU vs recorded:   mean {sum(ious) / len(ious):.3f}  "
            f"min {min(ious):.3f}  >=0.95: {sum(1 for v in ious if v >= 0.95)}/{len(ious)}"
        )

    print("  selected sources:")

    for source, count in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {source}")

    if ious:
        print("  lowest IoU:")
        worst = sorted(
            (r for r in rows if r.get("iou_vs_recorded") is not None),
            key=lambda r: r["iou_vs_recorded"],
        )[:10]

        for r in worst:
            print(f"    {r['iou_vs_recorded']:.3f}  {Path(r['image']).parent.name}")

    for sheet in sheets:
        print(f"  sheet: {sheet}")

    print(f"  csv:   {out_dir / 'summary.csv'}")
    print()
    print(f"Browse: python -m borg_vision.cli.eval_masks --view {out_dir} --sort iou")

    if args.view_after:
        view(out_dir, args.sort)

    # A changed selection is not automatically a regression: the repair rules are
    # meant to change some. Only an outright failure to select is never better.
    return 1 if unselected else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --mode to evaluate, or --view RUNDIR to browse a finished run.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="image files or directories to search recursively "
             "(default: $BORG_VISION_DATA)",
    )
    parser.add_argument("--mode", choices=sorted(MODES), help="which rules to apply")
    parser.add_argument("--config", help="YAML of config overrides")
    parser.add_argument(
        "--glob",
        action="append",
        help=f"repeatable image pattern (default: {', '.join(DEFAULT_GLOBS)})",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="repeatable pattern to skip, on top of the saved-artifact defaults",
    )
    parser.add_argument("--limit", type=int, help="stop after N images")
    parser.add_argument("--out", help="run directory to write (default: ./eval_<mode>)")
    parser.add_argument("--compare", help="diff against another run directory")
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument(
        "--fill",
        action="store_true",
        help="tint the selected mask as well as outlining it",
    )
    parser.add_argument("--sheet-rows", type=int, default=3)
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="let the detection modules' own logging through",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="write each image's detection log next to its JSON",
    )
    parser.add_argument("--view", help="browse a finished run directory and exit")
    parser.add_argument(
        "--view-after",
        action="store_true",
        help="open the viewer when the batch finishes",
    )
    parser.add_argument(
        "--sort",
        choices=("path", "iou", "score", "area", "changed"),
        default="path",
        help="viewer ordering; 'iou' worst-first, 'changed' needs a --compare run",
    )
    args = parser.parse_args(argv)

    if args.view:
        view(args.view, args.sort)
        return 0

    if not args.mode:
        parser.error("--mode is required (or use --view RUNDIR)")

    if not args.paths:
        default = os.environ.get("BORG_VISION_DATA")

        if not default:
            parser.error("no paths given and BORG_VISION_DATA is not set")

        args.paths = [default]

    if not args.out:
        args.out = f"eval_{args.mode}"

    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
