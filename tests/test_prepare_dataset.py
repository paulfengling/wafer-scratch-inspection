"""Verify box geometry, source grouping, and safe treatment of truncated arcs."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from wafer_inspection.prepare_dataset import DatasetConfig, prepare_dataset


def label(class_id, xyxy, width=1096, height=1096):
    x1, y1, x2, y2 = xyxy
    return f'{class_id} {(x1+x2)/2/width:.12f} {(y1+y2)/2/height:.12f} {(x2-x1)/width:.12f} {(y2-y1)/height:.12f}\n'


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / '原始数据'
        (self.source / 'images').mkdir(parents=True)
        (self.source / 'labels').mkdir()
        (self.source / 'data.yaml').write_text(json.dumps({'names': ['benign', 'circle scratch', 'line scratch']}), encoding='utf-8')
        for index, name in enumerate(('a', 'b', 'c'), start=1):
            image = np.full((1096, 1096, 3), index * 20, dtype=np.uint8)
            cv2.rectangle(image, (10, 10), (20, 20), (index, index, index), -1)
            ok, encoded = cv2.imencode('.png', image)
            self.assertTrue(ok)
            encoded.tofile(str(self.source / 'images' / f'{name}.png'))
            (self.source / 'labels' / f'{name}.txt').write_text('', encoding='utf-8')
        self.manifest = self.root / 'splits.csv'
        self.manifest.write_text('path,split,group_id\na.png,train,wafer_a\nb.png,val,wafer_b\nc.png,test,wafer_c\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, **kwargs):
        defaults = dict(source=str(self.source), output=str(self.root / 'prepared'), split_mode='manifest',
                        split_manifest=str(self.manifest), negative_tile_keep_prob=1.0)
        defaults.update(kwargs)
        return prepare_dataset(DatasetConfig(**defaults))

    def rows(self):
        with (self.root / 'prepared' / 'manifest.csv').open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_full_image_unchanged_only_train_tiled_and_remap_geometry(self):
        (self.source / 'labels/a.txt').write_text(label(2, (100, 120, 200, 140)) + label(0, (0, 0, 1096, 1096)), encoding='utf-8')
        result = self.run_prepare(classes=[2, 1])
        self.assertEqual(result['names'], ['line scratch', 'circle scratch'])
        self.assertEqual(result['class_id_map'], {'2': 0, '1': 1})
        rows = self.rows()
        self.assertEqual(len([r for r in rows if r['kind'] == 'tile']), 4)
        self.assertTrue(all(r['split'] == 'train' for r in rows if r['kind'] == 'tile'))
        full = next(r for r in rows if r['parent_path'] == 'a.png' and r['kind'] == 'full')
        copied = self.root / 'prepared' / full['path']
        self.assertEqual(copied.read_bytes(), (self.source / 'images/a.png').read_bytes())
        self.assertEqual(cv2.imdecode(np.fromfile(str(copied), np.uint8), cv2.IMREAD_UNCHANGED).shape[:2], (1096, 1096))
        tile = next(r for r in rows if r['kind'] == 'tile' and r['x0'] == '0' and r['y0'] == '0')
        values = list(map(float, (self.root / 'prepared' / tile['label_path']).read_text().split()))
        np.testing.assert_allclose(values, [0, 150/640, 130/640, 100/640, 20/640], atol=1e-8)
        self.assertEqual({(r['x0'], r['y0']) for r in rows if r['kind'] == 'tile'}, {('0', '0'), ('456', '0'), ('0', '456'), ('456', '456')})

    def test_truncated_large_arc_boxes_never_become_false_background(self):
        (self.source / 'labels/a.txt').write_text(label(1, (400, 400, 900, 900)), encoding='utf-8')
        result = self.run_prepare()
        self.assertEqual(result['split_summary']['train']['tile_images'], 0)
        self.assertEqual(result['split_summary']['train']['skipped_tiles'], 4)
        self.assertEqual(len(self.rows()), 3)

    def test_clip_candidates_are_excluded_from_training(self):
        (self.source / 'labels/a.txt').write_text(label(1, (400, 400, 900, 900)), encoding='utf-8')
        result = self.run_prepare(tile_label_mode='clip')
        self.assertEqual(result['split_summary']['train']['review_tiles'], 4)
        self.assertEqual(result['split_summary']['train']['tile_images'], 0)
        review_rows = [r for r in self.rows() if r['route'] == 'review']
        self.assertEqual(len(review_rows), 4)
        self.assertTrue(all(r['path'].startswith('review/') for r in review_rows))
        data = json.loads((self.root / 'prepared/data.yaml').read_text())
        self.assertEqual(data['train'], 'images/train')

    def test_one_pixel_thin_line_is_retained(self):
        (self.source / 'labels/a.txt').write_text(label(2, (100, 130, 200, 131)), encoding='utf-8')
        result = self.run_prepare()
        self.assertEqual(result['split_summary']['train']['tile_images'], 4)
        tile = next(r for r in self.rows() if r['kind'] == 'tile' and r['x0'] == '0' and r['y0'] == '0')
        self.assertAlmostEqual(float((self.root / 'prepared' / tile['label_path']).read_text().split()[4]) * 640, 1.0, places=5)

    def test_missing_label_and_invalid_label_do_not_publish_partial_output(self):
        (self.source / 'labels/a.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())
        (self.source / 'labels/a.txt').write_text('1 0.5 nan 0.2 0.2\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())

    def test_same_wafer_cannot_cross_splits(self):
        self.manifest.write_text('path,split,group_id\na.png,train,same_wafer\nb.png,val,same_wafer\nc.png,test,c\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'group|Group'):
            self.run_prepare()

    def test_renamed_identical_image_cannot_cross_splits(self):
        (self.source / 'images/b.png').write_bytes((self.source / 'images/a.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'identical|Identical|duplicate|Duplicate'):
            self.run_prepare()
        self.assertFalse((self.root / 'prepared').exists())

    def test_output_cannot_overwrite_input_or_prior_dataset(self):
        with self.assertRaises(ValueError):
            self.run_prepare(output=str(self.source / 'images/out'))
        self.run_prepare()
        original = hashlib.sha256((self.root / 'prepared/data.yaml').read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertEqual(hashlib.sha256((self.root / 'prepared/data.yaml').read_bytes()).hexdigest(), original)


if __name__ == '__main__':
    unittest.main()
