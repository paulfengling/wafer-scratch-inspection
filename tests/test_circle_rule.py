"""Synthetic regression checks. These do not measure real wafer recall."""
import math
import unittest

import numpy as np

from wafer_inspection.circle_rule import CircleHit, RuleConfig, _circular_runs, detect_circle_scratches
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

    def test_weak_scratch_requires_lower_gray_difference_threshold(self):
        image = scene([(566, 535, 280, 185, 45, 60, 2)])
        self.assertFalse(detect_circle_scratches(image).hits)
        for threshold in (10.0, 0.0):
            with self.subTest(min_gray_difference=threshold):
                result = detect_circle_scratches(image, RuleConfig(min_gray_difference=threshold))
                self.assertTrue(result.hits, result.to_dict())

    def test_gray_difference_uses_local_wafer_brightness(self):
        # Shift both wafer and scratch by the same amount. Their difference,
        # rather than the absolute scratch intensity, determines the decision.
        differences = []
        for background, intensity in ((40, 100), (80, 140)):
            image = scene([(572, 530, 250, 60, 70, intensity, 3)],
                          background_level=background)
            result = detect_circle_scratches(image, RuleConfig(
                wafer_center=(548, 548), wafer_radius=515, min_gray_difference=45))
            self.assertTrue(result.hits, result.to_dict())
            best = max(result.hits, key=lambda hit: hit.score)
            difference = best.diagnostics['median_gray_difference']
            self.assertGreater(difference, 50)
            self.assertAlmostEqual(best.diagnostics['median_scratch_gray'] -
                                   best.diagnostics['median_wafer_gray'], difference, delta=3)
            differences.append(difference)
            rejected = detect_circle_scratches(image, RuleConfig(
                wafer_center=(548, 548), wafer_radius=515, min_gray_difference=75))
            self.assertFalse(rejected.hits, rejected.to_dict())
        self.assertAlmostEqual(*differences, delta=3)

    def test_bright_scratch_reports_gray_difference_and_respects_threshold(self):
        image = scene([(572, 530, 250, 0, 360, 130, 3)])
        result = detect_circle_scratches(image)
        self.assertTrue(result.hits, result.to_dict())
        best = max(result.hits, key=lambda hit: hit.arc_span_deg)
        self.assertGreater(best.diagnostics['median_scratch_gray'], 120)
        self.assertAlmostEqual(best.diagnostics['median_wafer_gray'], 40, delta=4)
        self.assertGreater(best.diagnostics['median_gray_difference'], 80)
        self.assertEqual(best.diagnostics['min_gray_difference'], 25)
        rejected = detect_circle_scratches(image, RuleConfig(min_gray_difference=110))
        self.assertFalse(rejected.hits, rejected.to_dict())

    def test_dim_sections_cannot_extend_bright_pieces_into_a_qualifying_arc(self):
        # The full 45-degree scratch is visible to the top-hat filter, but only
        # two separated 15-degree pieces pass the raw-gray difference threshold.
        image = scene([(572, 530, 320, 70, 45, 60, 2),
                       (572, 530, 320, 70, 15, 130, 2),
                       (572, 530, 320, 100, 15, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertFalse(result.hits, result.to_dict())
        unfiltered = detect_circle_scratches(image, RuleConfig(min_gray_difference=0))
        self.assertTrue(unfiltered.hits, unfiltered.to_dict())

    def test_nearby_overlapping_arcs_return_the_combined_arc(self):
        image = scene([(570, 530, 280, 30, 65, 130, 2),
                       (574, 530, 280, 80, 65, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertEqual(len(result.hits), 1, result.to_dict())
        self.assertGreaterEqual(result.hits[0].arc_span_deg, 105)
        self.assertLess(math.hypot(result.hits[0].cx - 572,
                                  result.hits[0].cy - 530), 8)

    def test_overlapping_short_arcs_form_a_qualifying_arc(self):
        image = scene([(572, 530, 320, 70, 20, 130, 2),
                       (572, 530, 320, 85, 20, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertEqual(len(result.hits), 1, result.to_dict())
        self.assertGreaterEqual(result.hits[0].arc_span_deg, 30)

    def test_single_scratch_does_not_leave_duplicate_fits(self):
        for name, start, span in (('full_circle', 0, 360),
                                  ('long_arc', 30, 140),
                                  ('wraparound_arc', 330, 90)):
            with self.subTest(scene=name):
                result = detect_circle_scratches(scene([
                    (572, 530, 280, start, span, 130, 2)]))
                self.assertEqual(len(result.hits), 1, result.to_dict())
                self.assertFalse(result.diagnostics['hit_limit_reached'], result.to_dict())

    def test_eccentric_overlapping_arcs_merge_across_zero_degrees(self):
        image = scene([(570, 530, 280, 330, 55, 130, 2),
                       (574, 530, 280, 5, 55, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertEqual(len(result.hits), 1, result.to_dict())
        self.assertGreaterEqual(result.hits[0].arc_span_deg, 88)
        self.assertFalse(result.diagnostics['hit_limit_reached'], result.to_dict())

    def test_nearby_radius_arcs_do_not_leave_duplicate_tail_fits(self):
        image = scene([(572, 530, 280, 30, 65, 130, 2),
                       (572, 530, 283, 80, 65, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertEqual(len(result.hits), 1, result.to_dict())
        self.assertGreaterEqual(result.hits[0].arc_span_deg, 110)
        self.assertFalse(result.diagnostics['hit_limit_reached'], result.to_dict())

    def test_radius_difference_at_tolerance_preserves_complete_union(self):
        # Independent short-arc fits can move slightly outside the nominal
        # five-pixel tolerance, so require the valid union without a fixed count.
        image = scene([(572, 530, 280, 30, 65, 130, 2),
                       (572, 530, 285, 80, 65, 130, 2)])
        result = detect_circle_scratches(image)
        self.assertTrue(any(hit.arc_span_deg >= 110 and
                            hit.diagnostics.get('merged_arc_count', 0) > 1
                            for hit in result.hits), result.to_dict())
        self.assertFalse(result.diagnostics['hit_limit_reached'], result.to_dict())

    def test_large_gap_and_distinct_radii_are_preserved(self):
        for name, arcs in (
                ('large_gap', [(572, 530, 250, 20, 45, 130, 2),
                               (572, 530, 250, 190, 45, 130, 2)]),
                ('distinct_radii', [(572, 530, 180, 0, 360, 130, 2),
                                    (572, 530, 280, 0, 360, 130, 2)])):
            with self.subTest(scene=name):
                result = detect_circle_scratches(scene(arcs))
                self.assertEqual(len(result.hits), 2, result.to_dict())
                if name == 'large_gap':
                    self.assertTrue(all(hit.arc_span_deg < 55 for hit in result.hits))
                else:
                    self.assertGreater(abs(result.hits[0].radius - result.hits[1].radius), 90)

    def test_new_threshold_and_tolerance_configuration_validation(self):
        nonnegative = ('min_gray_difference', 'merge_center_tolerance_px',
                       'merge_radius_tolerance_px')
        positive = ('gray_background_offset_px', 'gray_background_width_px')
        for name in nonnegative + positive:
            invalid_values = (-1, math.nan, math.inf)
            if name in positive:
                invalid_values += (0,)
            for value in invalid_values:
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    RuleConfig(**{name: value}).validate()
        for name in nonnegative:
            with self.subTest(parameter=name, value=0):
                RuleConfig(**{name: 0}).validate()
        with self.assertRaises(ValueError):
            RuleConfig(min_gray_difference=256).validate()


class ArcMergeTests(unittest.TestCase):
    @staticmethod
    def arc(start, span, center=(572, 530), radius=280, evidence_center=None):
        angles = np.deg2rad(np.arange(start, start + span, 0.5))
        ex, ey = evidence_center or center
        points = np.column_stack((ex + radius * np.cos(angles),
                                  ey + radius * np.sin(angles)))
        return CircleHit(center[0], center[1], radius, start, span, 1.0,
                         len(points) * math.radians(0.5) * radius, 0.1,
                         (0, 0, 1096, 1096), 0.95, _evidence_points=points)

    def test_overlapping_fits_are_refitted_from_joint_evidence(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits
        # Independently fitted short arcs can have biased centers. The pixel
        # evidence here lies on one real circle, allowing the union to correct it.
        hits = [self.arc(30, 65, center=(570, 530), evidence_center=(572, 530)),
                self.arc(80, 65, center=(574, 530), evidence_center=(572, 530))]
        candidates = []
        combined = self.arc(30, 115)

        def reverify(cx, cy, radius):
            candidates.append((cx, cy, radius))
            return [combined]

        merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(), reverify)
        self.assertEqual(merged, [combined])
        self.assertEqual(merge_count, 1)
        self.assertTrue(candidates)
        cx, cy, radius = candidates[-1]
        self.assertLess(math.hypot(cx - 572, cy - 530), 0.01)
        self.assertAlmostEqual(radius, 280, delta=0.01)

    def test_distinct_geometry_and_large_gaps_do_not_trigger_joint_fit(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits

        def unexpected_reverify(*candidate):
            self.fail(f'Unexpected merge candidate: {candidate}')

        for name, hits in (
                ('different_radius', [self.arc(30, 65), self.arc(80, 65, radius=300)]),
                ('different_center', [self.arc(30, 65), self.arc(80, 65, center=(590, 530))]),
                ('large_gap', [self.arc(30, 45), self.arc(190, 45)])):
            with self.subTest(case=name):
                merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(),
                                                              unexpected_reverify)
                self.assertEqual(len(merged), 2)
                self.assertEqual(merge_count, 0)

    def test_overlapping_wraparound_arcs_merge_across_zero_degrees(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits
        hits = [self.arc(350, 40), self.arc(15, 40)]
        combined = self.arc(350, 65)
        merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(),
                                                      lambda cx, cy, radius: [combined])
        self.assertEqual(merged, [combined])
        self.assertEqual(merge_count, 1)
        self.assertEqual(merged[0].start_angle_deg, 350)
        self.assertEqual(merged[0].arc_span_deg, 65)

    def test_radius_variations_within_tolerance_can_merge(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits
        hits = [self.arc(30, 65, radius=280), self.arc(80, 65, radius=282)]

        def reverify(cx, cy, radius):
            return [self.arc(30, 115, center=(cx, cy), radius=radius)]

        merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(), reverify)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merge_count, 1)
        self.assertEqual(merged[0].arc_span_deg, 115)
        for original in hits:
            self.assertLessEqual(abs(merged[0].radius - original.radius), 5)

    def test_nearby_center_chain_cannot_expand_beyond_tolerance(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits
        # Neighbor pairs are six pixels apart, but the first and last centers
        # are twelve apart. All original members must satisfy the tolerance.
        hits = [self.arc(30, 65, center=(568, 530), evidence_center=(572, 530)),
                self.arc(80, 65, center=(574, 530), evidence_center=(572, 530)),
                self.arc(130, 65, center=(580, 530), evidence_center=(572, 530))]
        candidates = []

        def reverify(cx, cy, radius):
            candidates.append((cx, cy, radius))
            span = 115 if len(candidates) == 1 else 165
            return [self.arc(30, span, center=(cx, cy), radius=radius)]

        merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(), reverify)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merge_count, 1)
        self.assertEqual(len(candidates), 1)

    def test_failed_image_reverification_keeps_original_hits(self):
        from wafer_inspection.circle_rule import _merge_overlapping_hits
        hits = [self.arc(30, 65), self.arc(80, 65)]
        merged, merge_count = _merge_overlapping_hits(hits, RuleConfig(),
                                                      lambda cx, cy, radius: [])
        self.assertEqual(merged, hits)
        self.assertEqual(merge_count, 0)


if __name__ == '__main__':
    unittest.main()
