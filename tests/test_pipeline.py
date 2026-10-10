"""Routing and coordinate checks with fake models, independent of real model quality."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from wafer_inspection.circle_rule import CircleHit, RuleResult
from wafer_inspection.pipeline import (PipelineConfig, UltralyticsRunner, YoloDetection, classwise_nms,
                                       discover_images, resolve_critical_classes,
                                       run_pipeline, tile_origins, write_png)


def fake_rule(image, config):
    hits = []
    if image[0, 0, 0] == 1:
        hits = [CircleHit(548, 548, 100, 0, 45, 1, 80, 0.1,
                          (600, 540, 660, 630), 0.8)]
    return RuleResult({'cx': 548, 'cy': 548, 'radius': 515}, hits,
                      {'status': 'ok'}, np.zeros(image.shape[:2], dtype=np.uint8))


class FakeYolo:
    names = {2: 'circle scratch', 5: 'line_scratch', 7: 'benign defect'}

    def __init__(self, detections=(), fail=False):
        self.calls = 0
        self.detections = list(detections)
        self.fail = fail

    def detect(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError('deliberate inference failure')
        return self.detections


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakePredictModel:
    """Return source-coordinate boxes, as Ultralytics does after its resizing."""
    names = {5: 'line_scratch'}

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append({'imgsz': kwargs['imgsz'], 'shape': kwargs['source'].shape,
                           'rect': kwargs['rect']})
        boxes = SimpleNamespace(xyxy=FakeTensor([[630, 635, 700, 705]]),
                                conf=FakeTensor([0.9]), cls=FakeTensor([5]))
        return [SimpleNamespace(boxes=boxes)]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / '输入图片'
        self.source.mkdir()
        for name, marker in [('circle.png', 1), ('negative.png', 2)]:
            write_png(self.source / name, np.full((1096, 1096, 3), marker, dtype=np.uint8))

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, config, runner=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_pipeline(config, rule_detector=fake_rule, yolo_runner=runner)

    def test_pending_is_not_accepted_and_rule_positive_skips_yolo(self):
        output = self.root / 'results'
        summary = self.run_case(PipelineConfig(self.source, output))
        self.assertEqual(summary['decisions'], {'reject': 1, 'review_required': 1})
        self.assertTrue((output / 'rule-based/images/circle.png').exists())
        self.assertTrue((output / 'yolo-pending/images/negative.png').exists())
        self.assertTrue(list((output / 'rule-based/crops').glob('*.png')))

    def test_yolo_only_runs_on_rule_negative_and_resolves_actual_ids(self):
        runner = FakeYolo([YoloDetection((10, 10, 100, 100), 0.7, 5, 'line_scratch')])
        summary = self.run_case(PipelineConfig(self.source, self.root / 'results'), runner)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(summary['effective_critical_classes'], (2, 5))
        self.assertEqual(summary['decisions'], {'reject': 2})

    def test_noncritical_detection_and_unknown_class_mapping(self):
        runner = FakeYolo([YoloDetection((10, 10, 100, 100), 0.7, 7, 'benign defect')])
        summary = self.run_case(PipelineConfig(self.source, self.root / 'results'), runner)
        self.assertEqual(summary['decisions']['no_critical_defect_detected'], 1)
        self.assertEqual(resolve_critical_classes({0: 'something', 1: 'else'}, None),
                         (None, 'unresolved'))
        self.assertEqual(resolve_critical_classes({0: 'circle scratch', 1: 'circle_scratch',
                                                   2: 'line scratch'}, None),
                         (None, 'unresolved'))

    def test_rule_positive_yolo_failure_retains_known_reject(self):
        runner = FakeYolo(fail=True)
        config = PipelineConfig(self.source, self.root / 'results', weights=self.root / 'fake.pt',
                                yolo_on_rule_hit=True)
        summary = self.run_case(config, runner)
        self.assertEqual(runner.calls, 2)
        self.assertEqual(summary['exit_code'], 1)
        self.assertEqual(summary['decisions'], {'reject': 1, 'review_required': 1})
        self.assertTrue((config.output / 'rule-based/images/circle.png').exists())

    def test_tile_boundaries_and_classwise_merge(self):
        origins = tile_origins(1096, 640, 0.25)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1] + 640, 1096)
        self.assertTrue(all(b - a <= 640 for a, b in zip(origins, origins[1:])))
        boxes = [YoloDetection((0, 0, 100, 100), 0.8, 2, 'circle'),
                 YoloDetection((1, 1, 101, 101), 0.7, 2, 'circle'),
                 YoloDetection((1, 1, 101, 101), 0.9, 5, 'line')]
        kept = classwise_nms(boxes, 0.45)
        self.assertEqual(len(kept), 2)
        self.assertEqual({box.class_id for box in kept}, {2, 5})

    def test_tile_input_size_matches_training_without_padding_crops(self):
        for override in (None, 960):
            with self.subTest(tile_imgsz=override):
                options = {} if override is None else {'tile_imgsz': override}
                config = PipelineConfig(self.source, self.root / 'results', tile_size=640,
                                        **options)
                config.validate()
                runner = UltralyticsRunner.__new__(UltralyticsRunner)
                runner.config = config
                runner.model = FakePredictModel()
                runner.names = runner.model.names
                detections = runner.detect(np.zeros((1096, 1096, 3), dtype=np.uint8))
                expected_tile_imgsz = 1120 if override is None else override
                self.assertEqual(runner.model.calls[0],
                                 {'imgsz': 1120, 'shape': (1120, 1120, 3), 'rect': False})
                self.assertEqual(len(runner.model.calls), 5)
                # Unpadded 640-pixel crops are scaled by the model to tile_imgsz.
                self.assertTrue(all(call == {'imgsz': expected_tile_imgsz,
                                             'shape': (640, 640, 3), 'rect': False}
                                    for call in runner.model.calls[1:]))
                edge = next(d for d in detections if d.source == 'tile:456,456')
                self.assertEqual(edge.bbox, (1086, 1091, 1096, 1096))

    def test_tile_input_size_requires_integer_stride_multiple(self):
        for invalid in (0, 31, 33, 1120.0, True):
            with self.subTest(tile_imgsz=invalid):
                config = PipelineConfig(self.source, self.root / 'results', tile_imgsz=invalid)
                with self.assertRaisesRegex(ValueError, 'tile_imgsz'):
                    config.validate()

    def test_scanner_excludes_output_and_depth_is_not_silently_truncated(self):
        output = self.source / 'results'
        write_png(output / 'rule-based/images/previous.png', np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(len(discover_images(self.source, output)), 2)
        from wafer_inspection.pipeline import read_image
        path = self.source / '16bit.png'
        write_png(path, np.full((32, 32), 2000, dtype=np.uint16))
        with self.assertRaises(ValueError):
            read_image(path)

    def test_reusing_output_is_rejected_to_avoid_stale_route_files(self):
        config = PipelineConfig(self.source, self.root / 'results')
        self.run_case(config)
        with self.assertRaises(ValueError):
            self.run_case(config)

    def test_evaluation_counts_wafers_and_retains_skipped_line(self):
        from wafer_inspection.evaluate import evaluate
        output = self.root / 'results'
        self.run_case(PipelineConfig(self.source, output), FakeYolo())
        labels = self.root / 'labels'
        labels.mkdir()
        # Multiple annotation boxes of one circle must count as one fatal wafer.
        (labels / 'circle.txt').write_text('2 0.4 0.4 0.1 0.1\n2 0.5 0.5 0.1 0.1\n'
                                          '5 0.6 0.6 0.1 0.1\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluate(output, labels, 2, 5)
        (labels / 'negative.txt').write_text('', encoding='utf-8')
        metrics = evaluate(output, labels, 2, 5)
        self.assertEqual(metrics['images'], 2)
        self.assertEqual(metrics['critical_wafer_recall'], 1.0)
        self.assertEqual(metrics['noncritical_wafer_rejection_rate'], 0.0)
        self.assertEqual(metrics['subgroups']['mixed']['total'], 1)
        self.assertEqual(metrics['type_presence']['line']['type_presence_recall'], 0.0)
        self.assertEqual(metrics['type_presence']['line']['yolo_skipped'], 1)


if __name__ == '__main__':
    unittest.main()
