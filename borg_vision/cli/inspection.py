"""Command-line runner for the product-inspection mode.

Two flows, mirroring the original integrated_oak_inspection.py:

  # Offline: inspect existing image files (no cameras needed) -- useful to
  # verify the OpenAI path end-to-end without hardware.
  python -m borg_vision.cli.inspection --product-name "Label Maker" \
      --images test-images/IMG_0563.jpg test-images/IMG_0564.jpg

  # Live: capture from two OAK-D cameras, then inspect.
  python -m borg_vision.cli.inspection --product-name "Label Maker" \
      --capture --capture-count 4 --mxid-1 14442C... --mxid-2 14442C...

Requires OPENAI_API_KEY in the environment. The verdict (good / damaged /
manual_review) plus confidence and details are printed.
"""

import argparse
import json
from pathlib import Path

from ..config import InspectionConfig
from ..inspection import run_inspection
from ..registry import get_inspection_detector
from ..results import InspectionResult
from ..detectors.inspection import make_request_id


def _print_result(result):
    print("=" * 60)
    print("BORG INSPECTION RESULT".center(60))
    print("=" * 60)
    print(f"Product    : {result.product_name}")
    print(f"Result     : {result.result.upper()}")
    print(f"Confidence : {round(result.confidence * 100)}%")
    print(f"Summary    : {result.summary}")
    if result.reasons:
        print("Reasons    :")
        for r in result.reasons:
            print(f"  - {r}")
    if result.observed_defects:
        print("Defects    :")
        for d in result.observed_defects:
            print(f"  - {d}")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m borg_vision.cli.inspection",
        description="Run a product inspection (capture from two OAK-D cameras "
                    "or inspect existing images) and print the verdict.",
    )
    parser.add_argument("--product-name", required=True,
                        help="Product context for the inspection prompt")
    parser.add_argument("--config", default=None,
                        help="YAML file of InspectionConfig overrides")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", action="store_true",
                        help="Capture from two OAK-D cameras")
    source.add_argument("--images", nargs="+",
                        help="Existing image paths to inspect instead of capturing")

    parser.add_argument("--capture-count", type=int, default=None,
                        help="Number of frames to capture (default: config)")
    parser.add_argument("--mxid-1", default=None, help="Device mxid for camera 1")
    parser.add_argument("--mxid-2", default=None, help="Device mxid for camera 2")
    parser.add_argument("--pretty", action="store_true",
                        help="Print the full result JSON instead of the report")
    args = parser.parse_args(argv)

    cfg = (
        InspectionConfig.from_yaml(args.config)
        if args.config else InspectionConfig()
    )

    if args.images:
        request_id = make_request_id()
        run = run_inspection(
            cfg, args.images, args.product_name, request_id=request_id
        )
        result = InspectionResult.from_run(
            run, product_name=args.product_name, request_id=request_id,
            image_paths=args.images,
        )
    else:
        detector = get_inspection_detector(
            cfg=cfg, mxid_1=args.mxid_1, mxid_2=args.mxid_2
        )
        with detector:
            result = detector.inspect(
                args.product_name,
                count=args.capture_count,
                on_stage=lambda stage: print(f"[inspection] {stage}..."),
            )

    if result is None:
        print("[inspection] aborted.")
        return

    if args.pretty:
        print(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
