"""Unit tests for the rotated-pair and multi-face segmentation repair rules.

Tiered by dependency, because importing `borg_vision` pulls torch and depthai:

  Tier 1  numpy only          -- utils.long_axis_angle_diff_deg, config defaults
  Tier 2  + cv2               -- detection/masks.py primitives (path-loaded)
  Tier 3  + cv2 + torch       -- detection/box.py rules (synthetic-root loaded)

Modules are loaded by file path under a synthetic `bv` root package so
borg_vision/__init__.py (which imports the detectors, hence depthai) never runs.

Run:  python3 -m pytest borg_vision/tests/test_merge_geometry.py
      python3 borg_vision/tests/test_merge_geometry.py   # (unittest main)
"""

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_standalone(name, path):
    """Load a module with no relative imports, outside any package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_optional_hardware_deps():
    """Stub the hardware-only imports the detection modules reach through.

    detection/package.py -> visualization -> barcode -> pyzbar, which is only
    needed to decode a real camera frame. Stubbing keeps these tests runnable on
    a dev box without the pyzbar/libzbar system package.
    """
    if "pyzbar" not in sys.modules:
        try:
            import pyzbar  # noqa: F401
        except ImportError:
            pyzbar = types.ModuleType("pyzbar")
            pyzbar.__path__ = []
            inner = types.ModuleType("pyzbar.pyzbar")
            inner.decode = lambda *args, **kwargs: []
            pyzbar.pyzbar = inner
            sys.modules["pyzbar"] = pyzbar
            sys.modules["pyzbar.pyzbar"] = inner


def _load(dotted):
    """Load `borg_vision.<dotted>` as `bv.<dotted>`, skipping borg_vision/__init__."""
    _stub_optional_hardware_deps()

    if "bv" not in sys.modules:
        root = types.ModuleType("bv")
        root.__path__ = [str(ROOT)]
        sys.modules["bv"] = root

    full = f"bv.{dotted}"

    if full in sys.modules:
        return sys.modules[full]

    parts = dotted.split(".")

    if len(parts) > 1:
        _load(".".join(parts[:-1]))

    target = ROOT.joinpath(*parts)

    if target.is_dir():
        spec = importlib.util.spec_from_file_location(
            full,
            target / "__init__.py",
            submodule_search_locations=[str(target)],
        )
    else:
        spec = importlib.util.spec_from_file_location(full, target.with_suffix(".py"))

    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - depends on the host env
    _HAS_CV2 = False

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the host env
    _HAS_TORCH = False


# ---------------------------------------------------------------- Tier 1

_utils = _load_standalone("bv_utils_standalone", ROOT / "utils.py")
_config = _load("config")

long_axis_angle_diff_deg = _utils.long_axis_angle_diff_deg
_long_axis_angle_deg = _utils._long_axis_angle_deg


class LongAxisAngleDiffTest(unittest.TestCase):
    def test_identical_angles(self):
        self.assertAlmostEqual(long_axis_angle_diff_deg(0.0, 0.0), 0.0)
        self.assertAlmostEqual(long_axis_angle_diff_deg(-37.5, -37.5), 0.0)

    def test_wraps_across_the_180_period(self):
        # A naive abs(a - b) would report 178 deg for two near-parallel rects.
        self.assertAlmostEqual(long_axis_angle_diff_deg(89.0, -89.0), 2.0)
        self.assertAlmostEqual(long_axis_angle_diff_deg(-80.0, 80.0), 20.0)

    def test_straddles_the_22_degree_gate(self):
        self.assertAlmostEqual(long_axis_angle_diff_deg(10.0, 30.0), 20.0)
        self.assertAlmostEqual(long_axis_angle_diff_deg(10.0, 40.0), 30.0)

    def test_long_axis_swap_is_applied_exactly_once(self):
        """The +90 deg swap belongs to _long_axis_angle_deg, not to the diff.

        get_rotated_box_from_mask already normalizes angle_deg, so feeding its
        output here must NOT re-apply the swap. Two rects describing the same
        physical direction -- one reported by cv2 on its long side (w > h,
        angle 30) and one on its short side (h > w, angle -60) -- must come out
        as a zero difference.
        """
        long_side = _long_axis_angle_deg(100.0, 40.0, 30.0)
        short_side = _long_axis_angle_deg(40.0, 100.0, -60.0)

        self.assertAlmostEqual(long_side, 30.0)
        self.assertAlmostEqual(short_side, 30.0)
        self.assertAlmostEqual(long_axis_angle_diff_deg(long_side, short_side), 0.0)


class NewConfigFieldTest(unittest.TestCase):
    def test_rotated_fallback_defaults_match_in_both_modes(self):
        for cfg in (_config.BoxConfig(), _config.PackageConfig()):
            self.assertTrue(cfg.box_merge_rotated_enable)
            self.assertAlmostEqual(cfg.box_merge_rotated_max_center_distance_ratio, 1.35)
            self.assertAlmostEqual(cfg.box_merge_rotated_max_angle_diff_deg, 22.0)
            self.assertAlmostEqual(cfg.box_merge_rotated_min_completed_fill_ratio, 0.42)
            self.assertAlmostEqual(cfg.box_merge_rotated_max_completed_area_ratio, 0.78)

    def test_multiface_defaults_match_in_both_modes(self):
        for cfg in (_config.BoxConfig(), _config.PackageConfig()):
            self.assertTrue(cfg.box_multiface_merge_enable)
            self.assertEqual(cfg.box_multiface_max_candidates, 16)
            self.assertEqual(cfg.box_multiface_touch_dilate_px, 28)
            self.assertAlmostEqual(cfg.box_multiface_max_pair_iou, 0.20)
            self.assertAlmostEqual(cfg.box_multiface_min_second_area_ratio, 0.025)
            self.assertAlmostEqual(cfg.box_multiface_min_area_growth, 1.25)
            self.assertAlmostEqual(cfg.box_multiface_max_hull_area_ratio, 0.72)
            self.assertAlmostEqual(cfg.box_multiface_score_bonus, 1.10)

    def test_multiface_union_fill_gate_is_stricter_in_package_mode(self):
        """Package mode rescores the hull with the permissive package rules, so
        the union-fill gate is the only thing stopping a hull that spikes along a
        conveyor roller. Raised to 0.90 on replay evidence; see config/package.py.
        """
        self.assertAlmostEqual(
            _config.PackageConfig().box_multiface_min_union_fill_ratio, 0.90
        )
        self.assertAlmostEqual(
            _config.BoxConfig().box_multiface_min_union_fill_ratio, 0.58
        )

    def test_secondary_face_pool_is_box_mode_only(self):
        cfg = _config.BoxConfig()
        self.assertAlmostEqual(cfg.secondary_face_max_area_ratio, 0.70)
        self.assertAlmostEqual(cfg.secondary_face_min_rectangularity, 0.08)
        self.assertAlmostEqual(cfg.secondary_face_max_aspect_ratio, 10.0)
        self.assertAlmostEqual(cfg.secondary_face_min_center_score, 0.12)

        # Package mode reuses its own package candidates instead.
        self.assertFalse(hasattr(_config.PackageConfig(), "secondary_face_max_area_ratio"))

    def test_base_depth_is_per_mode(self):
        """Package mode moved to 700 mm; nothing else may drift with it."""
        self.assertAlmostEqual(_config.PackageConfig().base_depth_mm, 700.0)
        self.assertAlmostEqual(_config.BaseConfig().base_depth_mm, 695.0)
        self.assertAlmostEqual(_config.BoxConfig().base_depth_mm, 705.0)
        self.assertAlmostEqual(_config.PolymailerConfig().base_depth_mm, 705.0)
        self.assertAlmostEqual(_config.ClearBagConfig().base_depth_mm, 705.0)

    def test_existing_merge_thresholds_are_untouched(self):
        box = _config.BoxConfig()
        self.assertAlmostEqual(box.box_merge_min_axis_overlap, 0.55)
        self.assertAlmostEqual(box.box_merge_max_center_diff_ratio, 0.35)
        self.assertAlmostEqual(box.box_merge_max_size_ratio, 1.80)
        self.assertEqual(box.box_merge_max_gap_px, 90)
        self.assertTrue(box.box_full_mask_override_enable)

        package = _config.PackageConfig()
        self.assertAlmostEqual(package.box_merge_min_horizontal_overlap, 0.55)
        self.assertAlmostEqual(package.box_merge_max_center_x_diff_ratio, 0.35)
        self.assertAlmostEqual(package.box_merge_max_width_ratio, 1.80)
        self.assertEqual(package.box_merge_max_vertical_gap_px, 90)


# ---------------------------------------------------------------- Tier 2

def _square(shape, x, y, w, h):
    mask = np.zeros(shape, np.uint8)
    mask[y:y + h, x:x + w] = 1
    return mask


@unittest.skipUnless(_HAS_CV2, "cv2 not installed")
class MaskPrimitiveTest(unittest.TestCase):
    """detection/masks.py must be loadable with no borg_vision package at all."""

    @classmethod
    def setUpClass(cls):
        cls.masks = _load_standalone(
            "bv_masks_standalone",
            ROOT / "detection" / "masks.py",
        )
        cls.package_helpers = None

    def test_hull_of_one_square_is_that_square(self):
        mask = _square((200, 300), 50, 40, 80, 60)
        hull = self.masks.convex_hull_mask(mask)
        self.assertIsNotNone(hull)
        self.assertEqual(int(hull.sum()), int(mask.sum()))

    def test_hull_bridges_the_gap_between_two_squares(self):
        mask = _square((200, 300), 40, 40, 60, 60) | _square((200, 300), 160, 40, 60, 60)
        hull = self.masks.convex_hull_mask(mask)
        self.assertGreater(int(hull.sum()), int(mask.sum()))

    def test_hull_differs_from_min_area_rect_for_a_diagonal_pair(self):
        """The reason the multi-face merge uses a hull and not a rectangle."""
        mask = _square((200, 300), 30, 20, 70, 70) | _square((200, 300), 160, 110, 70, 70)
        hull = self.masks.convex_hull_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.boxPoints(cv2.minAreaRect(np.vstack(contours)))
        rect_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(rect_mask, np.int32(rect), 1)

        self.assertLess(int(hull.sum()), int(rect_mask.sum()))

    def test_hull_of_empty_mask_is_none(self):
        self.assertIsNone(self.masks.convex_hull_mask(np.zeros((50, 50), np.uint8)))

    def test_masks_are_near_uses_the_dilate_radius(self):
        a = _square((200, 300), 40, 40, 40, 40)
        b = _square((200, 300), 100, 40, 40, 40)  # 20 px gap

        self.assertTrue(self.masks.masks_are_near(a, b, 28))
        self.assertFalse(self.masks.masks_are_near(a, b, 8))
        self.assertTrue(self.masks.masks_are_near(b, a, 28))

    def test_completed_rect_metrics_ratios(self):
        union = _square((100, 100), 10, 10, 40, 40) | _square((100, 100), 50, 10, 40, 40)
        completed = _square((100, 100), 10, 10, 80, 40)

        metrics = self.masks.completed_rect_metrics(union, completed)

        self.assertEqual(metrics["union_area"], 3200)
        self.assertEqual(metrics["completed_area"], 3200)
        self.assertAlmostEqual(metrics["completed_fill_ratio"], 1.0)
        self.assertAlmostEqual(metrics["completed_area_ratio"], 0.32)

    def test_hull_union_metrics_ratios(self):
        a = _square((100, 200), 10, 10, 40, 40)
        b = _square((100, 200), 60, 10, 40, 40)

        metrics = self.masks.hull_union_metrics(a, b, 100 * 200)

        # The hull of two side-by-side squares is their bounding rectangle.
        self.assertEqual(metrics["union_area"], 3200)
        self.assertEqual(metrics["hull_area"], 90 * 40)
        self.assertAlmostEqual(metrics["area_growth"], 3600 / 1600)
        self.assertAlmostEqual(metrics["union_fill_ratio"], 3200 / 3600)
        self.assertAlmostEqual(metrics["hull_area_ratio"], 3600 / 20000)

    def test_hull_union_metrics_none_for_empty_pair(self):
        empty = np.zeros((50, 50), np.uint8)
        self.assertIsNone(self.masks.hull_union_metrics(empty, empty, 2500))


# ---------------------------------------------------------------- Tier 3

def _rotated_rect_mask(shape, center, size, angle_deg):
    mask = np.zeros(shape, np.uint8)
    points = cv2.boxPoints((center, size, angle_deg))
    cv2.fillConvexPoly(mask, np.int32(points), 1)
    return mask


@unittest.skipUnless(_HAS_CV2 and _HAS_TORCH, "cv2/torch not installed")
class BoxMergeRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = _load("detection.box")
        cls.cfg = _config.BoxConfig()
        cls.shape = (400, 400)
        cls.roi_rgb = np.full((400, 400, 3), (170, 120, 70), np.uint8)

    def _fragment(self, mask, index):
        x, y, w, h = cv2.boundingRect(mask)
        return {
            "index": index,
            "mask": mask,
            "bbox": (x, y, w, h),
            "sam_iou": 0.9,
            "sam_stability": 0.9,
            "color_score": 0.7,
        }

    def _rotated_halves(self):
        """Two halves of one rectangle rotated 30 deg off the image axes."""
        first = _rotated_rect_mask(self.shape, (170.0, 170.0), (95.0, 60.0), 30.0)
        second = _rotated_rect_mask(self.shape, (256.6, 220.0), (95.0, 60.0), 30.0)
        return self._fragment(first, 0), self._fragment(second, 1)

    def test_axis_aligned_pair_still_reports_axis_aligned(self):
        """Guards the two-axis test kept from the pre-existing implementation."""
        first = self._fragment(_square(self.shape, 140, 100, 120, 50), 0)
        second = self._fragment(_square(self.shape, 140, 155, 120, 50), 1)

        compatibility = self.box.pair_fragment_compatibility(self.cfg, first, second)

        self.assertIsNotNone(compatibility)
        self.assertEqual(compatibility["geometry_mode"], "axis_aligned")
        self.assertEqual(compatibility["split_axis"], "top_bottom")

    def test_left_right_split_still_accepted(self):
        first = self._fragment(_square(self.shape, 100, 140, 50, 120), 0)
        second = self._fragment(_square(self.shape, 155, 140, 50, 120), 1)

        compatibility = self.box.pair_fragment_compatibility(self.cfg, first, second)

        self.assertIsNotNone(compatibility)
        self.assertEqual(compatibility["split_axis"], "left_right")

    def test_rotated_halves_are_rejected_by_the_axis_aligned_test(self):
        first, second = self._rotated_halves()
        self.assertIsNone(
            self.box.pair_fragment_compatibility(self.cfg, first, second)
        )

    def test_rotated_halves_are_admitted_by_the_fallback(self):
        first, second = self._rotated_halves()

        compatibility = self.box.rotated_pair_compatibility(self.cfg, first, second)

        self.assertIsNotNone(compatibility)
        self.assertEqual(compatibility["geometry_mode"], "rotated")
        self.assertEqual(compatibility["split_axis"], "rotated")
        self.assertLessEqual(
            compatibility["angle_diff_deg"],
            self.cfg.box_merge_rotated_max_angle_diff_deg,
        )
        self.assertLessEqual(
            compatibility["center_distance_ratio"],
            self.cfg.box_merge_rotated_max_center_distance_ratio,
        )

    def test_fallback_respects_its_enable_flag(self):
        first, second = self._rotated_halves()
        cfg = _config.BoxConfig(box_merge_rotated_enable=False)

        self.assertIsNone(self.box.rotated_pair_compatibility(cfg, first, second))

    def test_fallback_rejects_a_large_angle_difference(self):
        first = self._fragment(
            _rotated_rect_mask(self.shape, (170.0, 170.0), (95.0, 60.0), 30.0), 0
        )
        second = self._fragment(
            _rotated_rect_mask(self.shape, (256.6, 220.0), (95.0, 60.0), -30.0), 1
        )

        self.assertIsNone(
            self.box.rotated_pair_compatibility(self.cfg, first, second)
        )

    def test_fallback_rejects_distant_fragments(self):
        first = self._fragment(
            _rotated_rect_mask(self.shape, (90.0, 90.0), (95.0, 60.0), 30.0), 0
        )
        second = self._fragment(
            _rotated_rect_mask(self.shape, (320.0, 320.0), (95.0, 60.0), 30.0), 1
        )

        self.assertIsNone(
            self.box.rotated_pair_compatibility(self.cfg, first, second)
        )

    def test_color_score_floor_rescues_a_diluted_merge(self):
        """The escape hatch merges need: a hull over a shaded face is not brown."""
        roi = np.full((400, 400, 3), (60, 70, 110), np.uint8)
        mask = _square(self.shape, 120, 120, 160, 120)

        self.assertIsNone(
            self.box.score_cardboard_box_candidate(self.cfg, mask, roi)
        )

        rescued = self.box.score_cardboard_box_candidate(
            self.cfg,
            mask,
            roi,
            color_score_floor=0.7,
        )

        self.assertIsNotNone(rescued)
        self.assertAlmostEqual(rescued["color_score"], 0.7)
        self.assertLess(rescued["measured_color_score"], self.cfg.min_box_color_score)


@unittest.skipUnless(_HAS_CV2 and _HAS_TORCH, "cv2/torch not installed")
class MultifaceMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = _load("detection.box")
        cls.cfg = _config.BoxConfig()
        cls.shape = (400, 400)

    def _brown_roi_with_shaded_face(self):
        roi = np.full((400, 400, 3), (170, 120, 70), np.uint8)
        roi[100:160, 230:350] = (60, 70, 110)  # the shaded side face
        return roi

    def test_shaded_face_is_admitted_by_the_secondary_pool_only(self):
        roi = self._brown_roi_with_shaded_face()
        face_mask = _square(self.shape, 230, 100, 120, 60)

        # The cardboard rules reject it -- it is not brown enough.
        self.assertIsNone(
            self.box.score_cardboard_box_candidate(self.cfg, face_mask, roi)
        )

        pool = self.box.build_secondary_face_candidates(
            self.cfg,
            [{
                "segmentation": face_mask.astype(bool),
                "predicted_iou": 0.9,
                "stability_score": 0.9,
            }],
            roi,
        )

        self.assertEqual([item["index"] for item in pool], [0])
        self.assertNotIn("color_score", pool[0])
        self.assertNotIn("score", pool[0])

    def test_multiface_merge_joins_a_box_face_to_an_adjacent_face(self):
        roi = self._brown_roi_with_shaded_face()

        primary = self.box.score_cardboard_box_candidate(
            self.cfg,
            _square(self.shape, 100, 100, 120, 120),
            roi,
            index=0,
            sam_iou=0.9,
            sam_stability=0.9,
        )
        self.assertIsNotNone(primary)

        face_mask = _square(self.shape, 230, 100, 120, 60)
        secondary = [{
            "index": 1,
            "mask": face_mask,
            "area": int(face_mask.sum()),
            "sam_iou": 0.8,
            "sam_stability": 0.8,
        }]

        merged = self.box.build_multiface_cardboard_box_candidate(
            self.cfg,
            [primary],
            secondary,
            roi,
        )

        self.assertIsNotNone(merged)
        self.assertEqual(merged["source"], "cardboard_box_multiface_merge")
        self.assertEqual(merged["selection_override"], "angled_box_multiface_merge")
        self.assertEqual(merged["merged_from_indices"], [0, 1])
        self.assertAlmostEqual(merged["color_score"], primary["color_score"])
        self.assertAlmostEqual(
            merged["score"],
            merged["score_before_merge_bonus"] + self.cfg.box_multiface_score_bonus,
        )
        self.assertGreater(merged["area"], primary["area"])
        self.assertGreaterEqual(
            merged["merged_union_fill_ratio"],
            self.cfg.box_multiface_min_union_fill_ratio,
        )

    def test_multiface_merge_respects_its_enable_flag(self):
        roi = self._brown_roi_with_shaded_face()
        primary = self.box.score_cardboard_box_candidate(
            self.cfg,
            _square(self.shape, 100, 100, 120, 120),
            roi,
            index=0,
        )
        face_mask = _square(self.shape, 230, 100, 120, 60)

        merged = self.box.build_multiface_cardboard_box_candidate(
            _config.BoxConfig(box_multiface_merge_enable=False),
            [primary],
            [{"index": 1, "mask": face_mask, "area": int(face_mask.sum())}],
            roi,
        )

        self.assertIsNone(merged)

    def test_multiface_merge_rejects_a_far_away_face(self):
        roi = np.full((400, 400, 3), (170, 120, 70), np.uint8)
        primary = self.box.score_cardboard_box_candidate(
            self.cfg,
            _square(self.shape, 20, 20, 120, 120),
            roi,
            index=0,
        )
        face_mask = _square(self.shape, 260, 260, 120, 60)

        merged = self.box.build_multiface_cardboard_box_candidate(
            self.cfg,
            [primary],
            [{"index": 1, "mask": face_mask, "area": int(face_mask.sum())}],
            roi,
        )

        self.assertIsNone(merged)


@unittest.skipUnless(_HAS_CV2 and _HAS_TORCH, "cv2/torch not installed")
class PackageSplitMergeTest(unittest.TestCase):
    """Covers the one structural rewrite: package mode's flat chain of rejects
    became an axis_aligned_pair boolean plus a rotated fallback."""

    @classmethod
    def setUpClass(cls):
        cls.package = _load("detection.package")
        cls.cfg = _config.PackageConfig()
        cls.shape = (400, 400)
        cls.roi_rgb = np.full((400, 400, 3), (170, 120, 70), np.uint8)

    def _candidate(self, mask, index):
        result, _ = self.package.score_cardboard_box_mask(
            self.cfg,
            mask,
            self.roi_rgb,
            0.9,
            0.9,
        )
        self.assertIsNotNone(result, "test fixture must pass the box rules")
        result["index"] = index
        return result

    def _stacked_halves(self):
        return [
            self._candidate(_square(self.shape, 140, 120, 120, 60), 0),
            self._candidate(_square(self.shape, 140, 185, 120, 60), 1),
        ]

    def _tilted_halves(self):
        """Two halves of one rectangle tilted 5 deg off the image axes.

        Kept mild on purpose: complete_min_area_rectangle produces an
        axis-aligned mask, so box_merge_min_result_rectangularity (0.72, kept
        unchanged) still bounds how far this path can tilt. Strongly angled
        boxes are the multi-face merge's job, not this one.
        """
        return [
            self._candidate(
                _rotated_rect_mask(self.shape, (142.7, 195.0), (110.0, 70.0), 5.0), 0
            ),
            self._candidate(
                _rotated_rect_mask(self.shape, (257.3, 205.0), (110.0, 70.0), 5.0), 1
            ),
        ]

    def test_axis_aligned_split_still_merges(self):
        merged = self.package.build_merged_cardboard_box_candidate(
            self.cfg,
            self._stacked_halves(),
            self.roi_rgb,
        )

        self.assertIsNotNone(merged)
        self.assertEqual(merged["merged_geometry_mode"], "axis_aligned")
        self.assertEqual(merged["source"], "cardboard_box_merged_masks")
        self.assertEqual(merged["merged_from_indices"], [0, 1])

    def test_axis_aligned_split_is_unaffected_by_the_rotated_flag(self):
        """Equivalence check for the boolean rewrite: with the fallback off, the
        axis-aligned path must behave exactly as the old reject chain did."""
        halves = self._stacked_halves()

        with_flag = self.package.build_merged_cardboard_box_candidate(
            self.cfg,
            halves,
            self.roi_rgb,
        )
        without_flag = self.package.build_merged_cardboard_box_candidate(
            _config.PackageConfig(box_merge_rotated_enable=False),
            halves,
            self.roi_rgb,
        )

        self.assertIsNotNone(without_flag)
        self.assertAlmostEqual(with_flag["score"], without_flag["score"])
        self.assertEqual(with_flag["merged_geometry_mode"], "axis_aligned")
        self.assertEqual(without_flag["merged_geometry_mode"], "axis_aligned")

    def test_tilted_split_merges_only_through_the_fallback(self):
        halves = self._tilted_halves()

        merged = self.package.build_merged_cardboard_box_candidate(
            self.cfg,
            halves,
            self.roi_rgb,
        )

        self.assertIsNotNone(merged)
        self.assertEqual(merged["merged_geometry_mode"], "rotated")
        self.assertGreaterEqual(
            merged["merged_completed_fill_ratio"],
            self.cfg.box_merge_rotated_min_completed_fill_ratio,
        )
        self.assertIsNotNone(merged["merged_angle_diff_deg"])
        self.assertIsNotNone(merged["merged_center_distance_ratio"])

        # Same fragments, fallback disabled -> no merge at all.
        self.assertIsNone(
            self.package.build_merged_cardboard_box_candidate(
                _config.PackageConfig(box_merge_rotated_enable=False),
                halves,
                self.roi_rgb,
            )
        )


if __name__ == "__main__":
    unittest.main()
