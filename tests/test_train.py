"""Audit and training orchestration checks using synthetic data and a fake YOLO."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from wafer_inspection.train import (audit_dataset, build_training_plan, execute_training,
                                    main, write_dataset_snapshot)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'prepared'
        for index, split in enumerate(('train', 'val', 'test'), start=1):
            (self.dataset / 'images' / split).mkdir(parents=True)
            (self.dataset / 'labels' / split).mkdir(parents=True)
            ok, encoded = cv2.imencode('.png', np.full((64, 64, 3), 40 * index, dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(self.dataset / 'images' / split / 'sample.png'))
            (self.dataset / 'labels' / split / 'sample.txt').write_text('0 0.5 0.5 0.2 0.1\n', encoding='utf-8')
        self.data = self.dataset / 'data.yaml'
        self.data.write_text(json.dumps({'path': str(self.dataset), 'train': 'images/train', 'val': 'images/val',
                                        'test': 'images/test', 'nc': 2, 'names': ['circle scratch', 'line scratch']}), encoding='utf-8')
        self.project = self.root / 'runs'
        self.calls = []
        self.models = []

    def tearDown(self):
        self.temp.cleanup()

    def factory(self, source):
        owner = self

        class Model:
            task = 'detect'
            names = {0: 'circle scratch', 1: 'line scratch'}
            model = SimpleNamespace(yaml={'yaml_file': 'yolo11s.yaml'})

            def __init__(self):
                self.callbacks = {}
                self.ckpt = {'optimizer': {}, 'epoch': 2, 'train_args': {'data': str(owner.data.resolve()), 'imgsz': 640, 'batch': 4}}

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def train(self, **kwargs):
                owner.calls.append(('train', kwargs))
                save = Path(source).parent.parent if kwargs.get('resume') else Path(kwargs['project']) / kwargs['name']
                (save / 'weights').mkdir(parents=True, exist_ok=True)
                self.trainer = SimpleNamespace(save_dir=save, best=save / 'weights/best.pt', last=save / 'weights/last.pt',
                                               args=SimpleNamespace(imgsz=640 if kwargs.get('resume') else kwargs['imgsz']))
                self.callbacks['on_train_start'](self.trainer)
                self.trainer.best.write_bytes(b'fake test-only checkpoint')
                self.trainer.last.write_bytes(b'fake test-only checkpoint')

            def val(self, **kwargs):
                owner.calls.append(('val', kwargs))
                return SimpleNamespace(results_dict={'metrics/recall(B)': 0.75})

        model = Model()
        self.models.append(model)
        return model

    def plan(self, **kwargs):
        return build_training_plan(self.data, config={'project': str(self.project)}, **kwargs)

    def interrupted_checkpoint(self):
        save = self.project / 'interrupted'
        (save / 'weights').mkdir(parents=True)
        checkpoint = save / 'weights/last.pt'
        checkpoint.write_bytes(b'fake interrupted checkpoint')
        write_dataset_snapshot(SimpleNamespace(save_dir=save, args=SimpleNamespace(imgsz=640)), audit_dataset(self.data))
        return checkpoint

    def test_dry_run_validates_without_import_or_output(self):
        before = 'ultralytics' in sys.modules
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            status = main(['--data', str(self.data), '--project', str(self.project), '--dry-run'])
        plan = json.loads(stream.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(plan['train_kwargs']['imgsz'], 1120)
        self.assertEqual(plan['model_source'], 'yolo11s.pt')
        self.assertEqual('ultralytics' in sys.modules, before)
        self.assertFalse(self.project.exists())

    def test_training_snapshot_and_test_only_after_best_weights(self):
        plan = self.plan(evaluate_test=True)
        record = execute_training(plan, model_factory=self.factory)
        self.assertEqual([call[0] for call in self.calls], ['train', 'val'])
        self.assertEqual(self.calls[1][1]['split'], 'test')
        self.assertEqual(self.calls[1][1]['imgsz'], 1120)
        self.assertEqual(record['status'], 'completed')
        snapshot = json.loads(Path(record['dataset_snapshot']).read_text())
        self.assertEqual(snapshot['dataset_content_sha256'], plan.dataset['dataset_content_sha256'])
        self.assertTrue(Path(record['best_weights']).is_file())

    def test_resume_restores_settings_and_uses_recorded_input_size(self):
        checkpoint = self.interrupted_checkpoint()
        plan = self.plan(resume=checkpoint, resume_device='cpu', evaluate_test=True)
        self.assertEqual(plan.train_kwargs, {'resume': True, 'device': 'cpu'})
        self.assertEqual(plan.test_kwargs['imgsz'], 640)
        record = execute_training(plan, model_factory=self.factory)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(self.calls[0][1], {'resume': True, 'device': 'cpu'})

    def test_resume_rejects_same_path_with_changed_labels(self):
        checkpoint = self.interrupted_checkpoint()
        (self.dataset / 'labels/train/sample.txt').write_text('0 0.4 0.4 0.1 0.1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'content|hash|fingerprint'):
            self.plan(resume=checkpoint)

    def test_external_checkpoint_without_snapshot_requires_new_fine_tune(self):
        checkpoint = self.root / 'old.pt'
        checkpoint.write_bytes(b'fake')
        with self.assertRaisesRegex(ValueError, 'snapshot|Snapshot'):
            self.plan(resume=checkpoint)

    def test_duplicate_bytes_across_splits_and_missing_labels_are_errors(self):
        (self.dataset / 'images/val/sample.png').write_bytes((self.dataset / 'images/train/sample.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'split'):
            audit_dataset(self.data)
        (self.dataset / 'images/val/sample.png').write_bytes(b'distinct image for audit')
        (self.dataset / 'labels/val/sample.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            audit_dataset(self.data)

    def test_data_changed_after_plan_stops_before_iterations(self):
        plan = self.plan()
        (self.dataset / 'labels/train/sample.txt').write_text('0 0.3 0.3 0.1 0.1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed after planning'):
            execute_training(plan, model_factory=self.factory)
        metadata = json.loads((Path(plan.metadata_dir) / 'training_run.json').read_text())
        self.assertEqual(metadata['status'], 'error')
        self.assertFalse((self.project / 'wafer-yolo11/weights/best.pt').exists())

    def test_non_yolo11_architecture_rejected_before_train(self):
        plan = self.plan()
        def wrong_factory(source):
            model = self.factory(source)
            model.model = SimpleNamespace(yaml={'yaml_file': 'yolov8s.yaml'})
            return model
        with self.assertRaisesRegex(ValueError, 'YOLO11'):
            execute_training(plan, model_factory=wrong_factory)
        self.assertEqual(self.calls, [])

    def test_multi_device_cannot_bypass_single_process_snapshot_policy(self):
        from wafer_inspection.train import validate_single_device
        self.assertEqual(validate_single_device(0), 0)
        self.assertEqual(validate_single_device('cuda:0'), 'cuda:0')
        for device in ('0,1', '[0,1]', [0, 1], True, -1):
            with self.subTest(device=device), self.assertRaises(ValueError):
                validate_single_device(device)
        checkpoint = self.interrupted_checkpoint()
        plan = self.plan(resume=checkpoint, resume_device='cpu')
        def distributed_checkpoint(source):
            model = self.factory(source)
            model.ckpt['train_args']['device'] = '0,1'
            return model
        with self.assertRaises(ValueError):
            execute_training(plan, model_factory=distributed_checkpoint)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
