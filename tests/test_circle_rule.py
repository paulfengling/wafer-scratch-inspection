"""Synthetic regression checks. These do not measure real wafer recall."""
import math
import unittest

import numpy as np

from wafer_inspection.circle_rule import RuleConfig, _circular_runs, detect_circle_scratches
from tests.fixtures import cases, scene


class CircularSupportTests(unittest.TestCase):
    def test_wraparound_is_one_arc(self):
        evidence = np.zeros(360, dtype=bool)
        evidence[350:] = True
        evidence[:30] = True
        self.assertEqual(_circular_runs(evidence, 2), [(350, 40, 1.0)])

    def test_disjoint_pieces_are_not_added(self):
        evidence = np.zeros(360, dtype=bool)
        evidence[40:60] = True
        evidence[190:210] = True
        self.assertEqual([run[1] for run in _circular_runs(evidence, 2)], [20, 20])

    def test_gap_filling_preserves_original_coverage(self):
        evidence = np.zeros(360, dtype=bool)
        evidence[40:80] = True
        evidence[50:52] = False
        self.assertEqual(_circular_runs(evidence, 2), [(40, 40, 0.95)])


class ImageDetectionTests(unittest.TestCase):
    def test_synthetic_scenes(self):
        for name, image, circles, positive in cases():
            with self.subTest(scene=name):
                result = detect_circle_scratches(image)
                self.assertEqual(bool(result.hits), positive,
                                 f"{name}: {result.to_dict()}")
                self.assertNotEqual(result.wafer['source'], 'image_center_fallback')
                self.assertLess(math.hypot(result.wafer['cx'] - 548,
                                           result.wafer['cy'] - 548), 5)
                self.assertLess(abs(result.wafer['radius'] - 515), 5)
                for cx, cy, radius in circles:
                    # Short-arc centers are intrinsically less precise than full rings.
                    self.assertTrue(any(math.hypot(hit.cx - cx, hit.cy - cy) < 16 and
                                        abs(hit.radius - radius) < 16 for hit in result.hits),
                                    f"Missing geometry {(cx, cy, radius)} in {result.to_dict()}")
                for hit in result.hits:
                    self.assertGreaterEqual(hit.arc_span_deg, 30)
                    self.assertGreaterEqual(hit.coverage, RuleConfig().min_arc_coverage)

    def test_no_boundary_is_explicitly_uncertain(self):
        result = detect_circle_scratches(np.full((256, 256, 3), 40, dtype=np.uint8))
        self.assertEqual(result.diagnostics['status'], 'uncertain')
        self.assertEqual(result.wafer['source'], 'image_center_fallback')

    def test_explicit_wafer_geometry(self):
        config = RuleConfig(wafer_center=(548, 548), wafer_radius=515,
                            max_center_offset_px=0)
        result = detect_circle_scratches(scene([(548, 548, 180, 0, 360, 130, 2)]), config)
        self.assertEqual(result.wafer['source'], 'explicit_geometry')
        self.assertTrue(result.hits)

    def test_thirty_degree_arc_and_small_full_circle(self):
        for name, image in [
            ('arc30', scene([(572, 530, 320, 70, 30, 130, 2)])),
            ('small_circle', scene([(572, 530, 40, 0, 360, 130, 2)])),
        ]:
            with self.subTest(scene=name):
                self.assertTrue(detect_circle_scratches(image).hits)

    def test_invalid_configuration_and_bit_depth(self):
        with self.assertRaises(ValueError):
            detect_circle_scratches(scene(), RuleConfig(max_gap_deg=30))
        with self.assertRaises(ValueError):
            detect_circle_scratches(np.full((100, 100), 40, dtype=np.uint16))


if __name__ == '__main__':
    unittest.main()
