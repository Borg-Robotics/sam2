"""Barcode-gated package detection (box / polymailer) on an OAK-D Pro.

The implementation now lives in the importable borg_vision library
(borg_vision/), which is also used by the ROS2 integration. This script is
kept as the interactive CLI entry point with the original behavior.

The pre-refactor standalone version is preserved at
product_detection_final.py.pre_refactor_backup for parity testing.
"""

from borg_vision.cli.product_detection import main

if __name__ == "__main__":
    main()
