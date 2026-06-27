"""Generic command-line runner for any borg_vision detection mode.

    python -m borg_vision --list
    python -m borg_vision --mode box
    python -m borg_vision --mode package --config overrides.yaml --mxid 14442C...

Headless single-shot flow (works without a display): load the SAM2 model, open
the camera, warm up, run one detection and save the debug artifacts
(annotated image + masks + heatmap + JSON) under the mode's save_dir.

For the package mode's original interactive SPACE/q barcode-gated loop, use:

    python -m borg_vision.cli.product_detection

The dual-camera "inspection" mode does not fit this single-camera runner; use:

    python -m borg_vision.cli.inspection --product-name "..." --capture
"""

import argparse

from ..registry import available_modes, get_detector


def run_once(mode, cfg=None, mxid=None, out_dir=None, barcode_timeout=30.0):
    """Run a single detection for `mode` and save debug artifacts.

    Returns the detector result (or None when nothing was detected).
    """
    detector = get_detector(mode, cfg=cfg, mxid=mxid)

    with detector:  # loads model + opens camera; closes on exit
        print(f"[{mode}] warming up {detector.cfg.warmup_seconds:.1f}s...")
        detector.warmup()

        print(f"[{mode}] detecting...")
        result = detector.detect_after_barcode(timeout_s=barcode_timeout)

        if result is None:
            print(f"[{mode}] no detection.")
            return None

        paths = detector.save_debug(result, out_dir=out_dir)
        print(f"[{mode}] saved artifacts:")
        for key, path in paths.items():
            if key == "result_bgr":
                continue
            print(f"    {key:14s} {path}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m borg_vision",
        description="Run a borg_vision detection mode once and save the result.",
    )
    parser.add_argument("--mode", choices=available_modes(), help="Detection mode to run")
    parser.add_argument("--list", action="store_true", help="List available modes and exit")
    parser.add_argument("--config", default=None, help="YAML file of config overrides for the mode")
    parser.add_argument("--mxid", default=None, help="OAK device MxID (multi-camera setups)")
    parser.add_argument("--out-dir", default=None, help="Override the save directory")
    parser.add_argument(
        "--barcode-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a barcode (barcode-gated modes only)",
    )
    args = parser.parse_args(argv)

    if args.list or not args.mode:
        print("Available modes:")
        for m in available_modes():
            print(f"  {m}")
        if not args.mode:
            return
        return

    run_once(
        args.mode,
        cfg=args.config,
        mxid=args.mxid,
        out_dir=args.out_dir,
        barcode_timeout=args.barcode_timeout,
    )


if __name__ == "__main__":
    main()
