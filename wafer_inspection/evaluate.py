"""Evaluate wafer-level decisions against complete YOLO detect labels.

Usage: python evaluate.py [--run outputs/run01] [--labels labels/test]
       python -m wafer_inspection.evaluate --run outputs/run01 --labels labels/test
Use --circle-id and --line-id when model.names cannot identify the class mapping.
Empty label files represent noncritical wafers; absent labels are errors unless
--missing-is-normal is explicitly supplied. A label root must mirror the input
image relative paths. This is image-level evaluation, not box or arc IoU recall.
Detect boxes do not encode the true arc span, so this cannot evaluate the >=30
degree requirement without additional arc ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_classes(path: Path, missing_is_normal: bool = False) -> set[int]:
    if not path.is_file():
        if missing_is_normal:
            return set()
        raise ValueError(f"Missing ground truth label: {path}; use explicit empty labels for normals")
    found: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
            if class_id < 0 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
                raise ValueError
            if coords[2] <= 0 or coords[3] <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO detect label at {path}:{line_number}") from exc
        found.add(class_id)
    return found


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def evaluate(run: Path, labels: Path, circle_id: int, line_id: int,
             missing_is_normal: bool = False) -> dict:
    run = run.resolve()
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    if circle_id == line_id or min(circle_id, line_id) < 0:
        raise ValueError('circle_id and line_id must be different nonnegative IDs')
    tp = fp = fn = tn = reviews = unprocessed = 0
    subgroups = {name: {'total': 0, 'rejected': 0} for name in ('circle_only', 'line_only', 'mixed')}
    typed = {name: {'positive_images': 0, 'type_detected': 0, 'yolo_skipped': 0}
             for name in ('circle', 'line')}
    misses = []
    for item in summary['items']:
        relative = Path(item['source'])
        gt = read_classes(labels / relative.with_suffix('.txt'), missing_is_normal)
        circle = circle_id in gt
        line = line_id in gt
        actual = circle or line
        predicted = item['decision'] == 'reject'
        reviews += item['decision'] == 'review_required'
        if actual:
            tp += predicted
            fn += not predicted
            if not predicted:
                misses.append({'image': relative.as_posix(), 'decision': item['decision']})
            key = 'mixed' if circle and line else 'circle_only' if circle else 'line_only'
            subgroups[key]['total'] += 1
            subgroups[key]['rejected'] += predicted
        else:
            fp += predicted
            tn += not predicted
        record_relative = item.get('record')
        if record_relative is None or not (run / record_relative).is_file():
            unprocessed += 1
            record = {}
        else:
            record = json.loads((run / record_relative).read_text(encoding='utf-8'))
        yolo = record.get('yolo', {})
        detected_classes = {d['class_id'] for d in yolo.get('detections', [])}
        rule_circle = bool((record.get('rule') or {}).get('hits'))
        for name, present, detected in (
            ('circle', circle, rule_circle or circle_id in detected_classes),
            ('line', line, line_id in detected_classes),
        ):
            if present:
                typed[name]['positive_images'] += 1
                typed[name]['type_detected'] += detected
                typed[name]['yolo_skipped'] += not yolo.get('attempted', False)
    for values in subgroups.values():
        values['wafer_rejection_recall'] = _ratio(values['rejected'], values['total'])
    for values in typed.values():
        values['type_presence_recall'] = _ratio(values['type_detected'], values['positive_images'])
    result = {
        'images': tp + fp + fn + tn,
        'circle_id': circle_id, 'line_id': line_id,
        'critical_wafer_recall': _ratio(tp, tp + fn),
        'noncritical_wafer_rejection_rate': _ratio(fp, fp + tn),
        'counts': {'fatal_detected': tp, 'fatal_not_rejected': fn,
                   'noncritical_rejected': fp, 'noncritical_not_rejected': tn,
                   'review_required': reviews, 'missing_records': unprocessed},
        'subgroups': subgroups, 'type_presence': typed, 'misses': misses,
        'notes': [
            'Reviews and errors are retained in denominators; not rejected is not a good-wafer verdict.',
            'A mixed wafer rejected by the circle rule can still lack a line detection when YOLO was skipped.',
            'Labels must cover all critical defects. No arc-span or box-IoU performance is measured here.',
            'Noncritical means no critical labels, and can include the other benign defect classes.',
        ],
    }
    (run / 'wafer_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for both direct and module invocation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run', type=Path, required=True,
                        help='Path to the pipeline output directory (contains summary.json)')
    parser.add_argument('--labels', type=Path, required=True,
                        help='Path to the ground truth labels directory')
    parser.add_argument('--circle-id', type=int, default=None,
                        help='Class ID for circle scratch (auto-detect if not provided)')
    parser.add_argument('--line-id', type=int, default=None,
                        help='Class ID for line scratch (auto-detect if not provided)')
    parser.add_argument('--missing-is-normal', action='store_true',
                        help='Treat missing label files as noncritical wafers')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point supporting both direct execution and module invocation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if not args.run.exists():
            raise FileNotFoundError(f"Run directory not found: {args.run}")
        if not args.labels.exists():
            raise FileNotFoundError(f"Labels directory not found: {args.labels}")

        summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
        names = {int(k): v.lower().replace(' ', '').replace('_', '').replace('-', '')
                 for k, v in summary.get('model_names', {}).items()}

        def find_id(name: str, explicit: int | None) -> int:
            if explicit is not None:
                return explicit
            matches = [key for key, value in names.items() if value == name]
            if len(matches) != 1:
                raise ValueError(f'Cannot uniquely resolve {name}; supply --circle-id and --line-id')
            return matches[0]

        result = evaluate(args.run, args.labels,
                          find_id('circlescratch', args.circle_id),
                          find_id('linescratch', args.line_id),
                          args.missing_is_normal)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Evaluation error: {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
