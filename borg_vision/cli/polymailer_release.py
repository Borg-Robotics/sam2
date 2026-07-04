"""Command-line runner for the polymailer product-release monitor.

Port-parity validation tool for the polymailer_product_release.py prototype:
it opens an owned OAK-D camera and drives the library monitor API
(start_monitoring / process_frame / stop_monitoring) in a loop. With --show
it restores the prototype's live window and keys on top of the library API,
so the refactor can be compared side by side with the original script:

  t/b/l/r : choose the slit side (restarts monitoring with that side)
  x       : reset (re-find the bag, re-lock the baseline)
  s       : save the annotated live view + masks
  q       : quit

  python -m borg_vision.cli.polymailer_release --slit-side top --show
  python -m borg_vision.cli.polymailer_release --config overrides.yaml

Without --show it runs headless and prints phase transitions (what the ROS
action server relays as feedback), exiting once the release is latched.
"""

import argparse

from ..config import PolymailerReleaseConfig
from ..registry import get_detector


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m borg_vision.cli.polymailer_release",
    )
    parser.add_argument(
        "--slit-side",
        default=None,
        choices=("top", "bottom", "left", "right"),
        help="Image-relative cut edge (default: config default_slit_side)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML file of PolymailerReleaseConfig overrides",
    )
    parser.add_argument("--mxid", default=None, help="OAK-D device mxid")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the live view window with the prototype's keys",
    )
    args = parser.parse_args(argv)

    cfg = (
        PolymailerReleaseConfig.from_yaml(args.config)
        if args.config
        else PolymailerReleaseConfig()
    )

    detector = get_detector("polymailer_release", cfg=cfg, mxid=args.mxid)
    detector.load_model()

    slit_side = args.slit_side or cfg.default_slit_side

    with detector:
        print(f"Warming up for {cfg.warmup_seconds:.1f}s ...")
        detector.warmup()
        detector.start_monitoring(slit_side)
        last_phase = None

        try:
            while True:
                frames = detector.camera.get_frames()
                update = detector.process_frame(frames)

                if update.phase != last_phase:
                    print(
                        f"[{update.phase.value}] {update.state_text} "
                        f"(slit motion {update.slit_motion_px:.0f}px, "
                        f"score {update.candidate_score:.2f})"
                    )
                    last_phase = update.phase

                if args.show:
                    import cv2

                    display = detector.render_live_view(frames)
                    cv2.imshow("Polymailer Release Monitor", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    side_key_map = {
                        ord("t"): "top",
                        ord("b"): "bottom",
                        ord("l"): "left",
                        ord("r"): "right",
                    }

                    if key in side_key_map:
                        slit_side = side_key_map[key]
                        print(f"Selected slit side: {slit_side.upper()}")
                        detector.start_monitoring(slit_side)
                        last_phase = None

                    if key == ord("x"):
                        detector.reset()
                        last_phase = None
                        print("Monitor state reset.")

                    if key == ord("s"):
                        detector.save_event_snapshot(frames)
                        print(f"Saved snapshot into {cfg.save_dir}")

                elif update.released:
                    result = detector.latched_result
                    print()
                    print("RELEASE LATCHED")
                    if result is not None:
                        print(
                            "  product center (camera, mm): "
                            f"({result.product_center_x_mm}, "
                            f"{result.product_center_y_mm}, "
                            f"{result.product_depth_mm})"
                        )
                        print(
                            "  height above base: "
                            f"{result.height_above_base_mm} mm"
                        )
                    detector.save_event_snapshot(frames, tag="released")
                    break

        finally:
            detector.stop_monitoring()
            if args.show:
                import cv2

                cv2.destroyAllWindows()

    print("Closed.")


if __name__ == "__main__":
    main()
