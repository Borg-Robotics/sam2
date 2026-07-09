"""Unit tests for InspectionConfig.cameras_for (capture schedule resolution).

Hardware-free: only exercises the pure-python schedule logic. Imports the config
module by file path so it does not pull in borg_vision/__init__ (torch/depthai).

Run:  python3 -m pytest borg_vision/tests/test_inspection_schedule.py
      python3 borg_vision/tests/test_inspection_schedule.py   # (unittest main)
"""

import importlib.util
import pathlib
import unittest

_CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "inspection.py"
_spec = importlib.util.spec_from_file_location("borg_vision_inspection_config", _CFG_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
InspectionConfig = _mod.InspectionConfig


class CamerasForTest(unittest.TestCase):
    def test_no_schedule_means_all_cameras_every_orientation(self):
        cfg = InspectionConfig()  # capture_schedule defaults to None
        self.assertEqual(cfg.cameras_for(0), [0, 1])
        self.assertEqual(cfg.cameras_for(3), [0, 1])
        self.assertEqual(cfg.cameras_for(0, num_cameras=3), [0, 1, 2])

    def test_empty_schedule_means_all_cameras(self):
        cfg = InspectionConfig(capture_schedule={})
        self.assertEqual(cfg.cameras_for(2), [0, 1])

    def test_per_camera_indices_and_all(self):
        # camera_1 only at the default pose (index 0), camera_2 everywhere.
        cfg = InspectionConfig(capture_schedule={"camera_1": [0], "camera_2": "all"})
        self.assertEqual(cfg.cameras_for(0), [0, 1])  # both at the default pose
        self.assertEqual(cfg.cameras_for(1), [1])     # camera_2 only afterwards
        self.assertEqual(cfg.cameras_for(2), [1])

    def test_omitted_camera_captures_nothing(self):
        cfg = InspectionConfig(capture_schedule={"camera_2": [1, 2]})
        self.assertEqual(cfg.cameras_for(0), [])      # neither cam scheduled here
        self.assertEqual(cfg.cameras_for(1), [1])
        self.assertEqual(cfg.cameras_for(3), [])      # index 3 not in camera_2 list

    def test_empty_list_disables_camera(self):
        cfg = InspectionConfig(capture_schedule={"camera_1": [], "camera_2": "all"})
        self.assertEqual(cfg.cameras_for(0), [1])

    def test_invalid_camera_key_raises(self):
        cfg = InspectionConfig(capture_schedule={"camera_3": [0]})
        with self.assertRaises(ValueError):
            cfg.cameras_for(0)

    def test_invalid_string_value_raises(self):
        cfg = InspectionConfig(capture_schedule={"camera_1": "every"})
        with self.assertRaises(ValueError):
            cfg.cameras_for(0)

    def test_non_int_index_list_raises(self):
        cfg = InspectionConfig(capture_schedule={"camera_1": [0, "1"]})
        with self.assertRaises(ValueError):
            cfg.cameras_for(0)


if __name__ == "__main__":
    unittest.main()
