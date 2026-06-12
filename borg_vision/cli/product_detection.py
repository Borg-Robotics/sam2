"""Interactive barcode-gated package detector (original script UX).

Keys:
  SPACE = start barcode search, then run package detection once
  q     = quit
"""

import time
from datetime import datetime

import cv2
import numpy as np

from ..barcode import detect_barcode, draw_barcode_overlay
from ..config import ProductDetectionConfig
from ..product_detector import ProductDetector
from ..visualization import (
    draw_roi_axes,
    make_depth_vis,
    make_live_depth_heatmap,
)


def main(cfg=None):
    cfg = cfg if cfg is not None else ProductDetectionConfig()

    print("Starting OAK + Barcode Gate + SAM 2 package detector")
    print("Keys:")
    print("  SPACE = start barcode search, then run package detection once")
    print("  q     = quit")
    print()
    print("FLOW:")
    print("  1. Press SPACE")
    print("  2. System searches for barcode")
    print("  3. When barcode is found, package detection runs once")
    print("  4. System waits for SPACE again")
    print()

    detector = ProductDetector(cfg)
    detector.load_model()
    detector.open()

    start_time = time.time()
    barcode_scan_active = False

    try:
        while detector.camera.is_running:
            frames = detector.camera.get_frames()
            rgb = frames.rgb
            depth_class_aligned = frames.depth_class_aligned

            elapsed = time.time() - start_time
            warmed = elapsed >= cfg.warmup_seconds

            preview_rgb = rgb.copy()
            preview_depth = make_depth_vis(cfg, depth_class_aligned)
            preview_heatmap = make_live_depth_heatmap(cfg, depth_class_aligned)

            cv2.rectangle(preview_rgb, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)
            cv2.rectangle(preview_depth, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (0, 255, 255), 3)

            draw_roi_axes(cfg, preview_rgb)
            draw_roi_axes(cfg, preview_depth)

            if not warmed:
                status = f"WARMING {cfg.warmup_seconds - elapsed:.1f}s"
            elif barcode_scan_active:
                status = "SCANNING BARCODE..."
            else:
                status = "READY - PRESS SPACE TO START"

            cv2.putText(
                preview_rgb,
                "BARCODE GATE PACKAGE DETECTOR",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview_rgb,
                status,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not warmed:
                    print("Still warming up. Wait before starting barcode scan.")
                else:
                    barcode_scan_active = True
                    print()
                    print("SPACE pressed. Searching for barcode...")

            if barcode_scan_active and warmed:
                barcode = detect_barcode(cfg, rgb)

                if barcode is not None:
                    preview_rgb = draw_barcode_overlay(preview_rgb, barcode)

                    print()
                    print(f"BARCODE FOUND: {barcode['type']} {barcode['data']}")
                    print("Running package detection...")

                    result = detector.detect(frames, barcode)

                    barcode_scan_active = False

                    if result is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        paths = detector.save_debug(result, timestamp=timestamp)

                        print()
                        print(f"Saved raw RGB:             {paths['raw_rgb']}")
                        print(f"Saved result image:        {paths['result']}")
                        print(f"Saved package mask:        {paths['mask']}")
                        print(f"Saved product inside mask: {paths['product_mask']}")
                        print(f"Saved heatmap:             {paths['heatmap']}")
                        print(f"Saved JSON:                {paths['json']}")
                        print()
                        print("Done. Press SPACE to scan the next barcode/package.")

                        cv2.imshow("Package Final Result", paths["result_bgr"])
                        cv2.waitKey(0)
                        cv2.destroyWindow("Package Final Result")

                    else:
                        print("Package detection failed. Press SPACE to try again.")

            combined_preview = np.hstack([preview_rgb, preview_depth, preview_heatmap])
            cv2.imshow("OAK Barcode Gate Package Detector", combined_preview)

    finally:
        detector.close()
        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    main()
