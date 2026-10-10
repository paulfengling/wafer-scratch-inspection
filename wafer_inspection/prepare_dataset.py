"""Prepare YOLO detect data without splitting one wafer across dataset splits.

Full images retain their original dimensions and bytes. Only training images are
tiled. Bounding boxes around arcs describe enclosing rectangles, so clipping a
box is not evidence that scratch pixels are present in a crop. The default skips
every tile intersecting a truncated task box. Optional clipped candidates go to
review/, which is excluded from data.yaml and cannot enter training automatically.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("path", "label_path", "parent_path", "group_id", "split", "kind",
                   "x0", "y0", "width", "height", "route", "source_sha256")


@dataclass
class DatasetConfig:
    source: str
    output: str
    images_dir: str | None = None
    labels_dir: str | None = None
    data_yaml: str | None = None
    names: list[str] | dict[int, str] | None = None
    classes: list[int] | None = None
    split_mode: str = "random"
    group_csv: str | None = None
    split_manifest: str | None = None
    val_fraction: float = 0.2
    test_fraction: float = 0.1
    seed: int = 42
    tile_size: int = 640
    overlap: float = 0.25
    tile_label_mode: str = "conservative"
    min_bbox_px: float = 1.0
    negative_tile_keep_prob: float = 0.25
    missing_is_normal: bool = False

    def validate(self) -> None:
        if self.split_mode not in ("random", "manifest"):
            raise ValueError("split_mode must be random or manifest")
        if self.split_mode == "manifest" and not self.split_manifest:
            raise ValueError("manifest mode requires split_manifest")
        if self.split_mode == "random" and self.split_manifest:
            raise ValueError("split_manifest and random splitting are mutually exclusive; set split_mode=manifest")
        for name in ("val_fraction", "test_fraction", "overlap", "negative_tile_keep_prob"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if self.overlap >= 1:
            raise ValueError("overlap must be < 1")
        if not isinstance(self.tile_size, int) or self.tile_size < 16:
            raise ValueError("tile_size must be an integer >= 16")
        if not math.isfinite(self.min_bbox_px) or self.min_bbox_px <= 0:
            raise ValueError("min_bbox_px must be positive")
        if self.tile_label_mode not in ("conservative", "clip"):
            raise ValueError("tile_label_mode must be conservative or clip")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass
class _ImageRecord:
    image: Path
    relative: str
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float, int]]
    image_sha256: str
    label_sha256: str
    group: str = ""
    split: str = ""


def _normal_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            values = {int(k): v for k, v in names.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError("names dictionary keys must be contiguous integer IDs starting at 0") from exc
        if len(values) != len(names) or sorted(values) != list(range(len(values))):
            raise ValueError("names dictionary IDs must be contiguous starting at 0")
        names = [values[i] for i in range(len(values))]
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("names must be a non-empty list of class names or an ID-to-name dictionary")
    names = [x.strip() for x in names]
    if len({x.casefold() for x in names}) != len(names):
        raise ValueError("class names must be unique")
    return names


def _yaml_scalar(text: str) -> str:
    text = re.split(r"\s+#", text.strip(), maxsplit=1)[0].strip()
    if text[:1] in ('"', "'"):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Unsupported quoted YAML class name; install PyYAML or pass --names JSON") from exc
        if not isinstance(value, str):
            raise ValueError("class names must be strings")
        return value
    return text


def _yaml_inline_names(text: str) -> Any:
    """Read common flow-style names, including YAML's unquoted strings."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    if not ((text.startswith("[") and text.endswith("]")) or
            (text.startswith("{") and text.endswith("}"))):
        raise ValueError("Unsupported inline YAML names; pass --names JSON or install PyYAML")
    pieces, current, quote, escaped = [], [], None, False
    for char in text[1:-1]:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
            current.append(char)
        elif char == ",":
            pieces.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quote:
        raise ValueError("Unclosed quoted class name")
    if current:
        pieces.append("".join(current).strip())
    if text.startswith("["):
        return [_yaml_scalar(piece) for piece in pieces if piece]
    result = {}
    for piece in pieces:
        if not piece:
            continue
        key, separator, value = piece.partition(":")
        if not separator or not key.strip().isdigit():
            raise ValueError("Inline names dictionary needs integer class IDs")
        index = int(key.strip())
        if index in result:
            raise ValueError("Duplicate class ID in names")
        result[index] = _yaml_scalar(value)
    return result


def _read_names_yaml(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        content = json.loads(text)
        return _normal_names(content["names"])
    except json.JSONDecodeError:
        pass
    except (KeyError, TypeError) as exc:
        raise ValueError(f"No valid names in {path}") from exc
    try:
        import yaml  # Optional, not needed for the JSON-as-YAML output of this tool.
    except ImportError:
        yaml = None
    if yaml is not None:
        content = yaml.safe_load(text)
        if not isinstance(content, dict) or "names" not in content:
            raise ValueError(f"No valid names in {path}")
        return _normal_names(content["names"])
    # Restricted fallback reads only normal names lists/maps, never arbitrary YAML.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)names\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        inline = re.split(r"\s+#", match.group(2), maxsplit=1)[0].strip()
        if inline.startswith("#"):
            inline = ""
        if inline:
            try:
                parsed = json.loads(inline)
            except json.JSONDecodeError:
                parsed = _yaml_inline_names(inline)
            return _normal_names(parsed)
        indentation = len(match.group(1))
        names_list: list[str] = []
        names_dict: dict[int, str] = {}
        for following in lines[index + 1:]:
            if not following.strip() or following.lstrip().startswith("#"):
                continue
            following_indent = len(following) - len(following.lstrip())
            if following_indent <= indentation and not following.strip().startswith("-"):
                break
            value = following.strip()
            if value.startswith("-"):
                names_list.append(_yaml_scalar(value[1:].strip()))
            else:
                key, separator, scalar = value.partition(":")
                if not separator or not key.strip().isdigit():
                    raise ValueError("Unsupported YAML names block; pass --names JSON or install PyYAML")
                class_id = int(key)
                if class_id in names_dict:
                    raise ValueError("Duplicate class ID in names")
                names_dict[class_id] = _yaml_scalar(scalar)
        if names_list and names_dict:
            raise ValueError("names block mixes list and dictionary syntax")
        return _normal_names(names_list or names_dict)
    raise ValueError(f"No names in {path}; supply --names JSON or --data-yaml")


def _read_image(path: Path) -> tuple[np.ndarray, bytes]:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8; explicitly scale higher-bit-depth data first: {path}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError(f"Unsupported image channels: {path}")
    return image, raw.tobytes()


def _relative_key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Missing relative image path in CSV")
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    path = Path(value)
    if not value or path.is_absolute() or any(x == ".." for x in path.parts):
        raise ValueError(f"Manifest paths must be relative to images_dir: {value!r}")
    return value


def _read_csv_map(path: str, require_split: bool) -> dict[str, tuple[str | None, str | None]]:
    result: dict[str, tuple[str | None, str | None]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fields = set(reader.fieldnames)
        path_key = next((x for x in ("image-relative-path", "image_relative_path", "path", "image", "parent_path") if x in fields), None)
        group_key = next((x for x in ("group_id", "group") if x in fields), None)
        if path_key is None or (require_split and "split" not in fields) or (not require_split and group_key is None):
            raise ValueError("CSV requires path and split (manifest), or path and group_id (group CSV)")
        for line, row in enumerate(reader, 2):
            relative = _relative_key(row[path_key])
            if relative in result:
                raise ValueError(f"Duplicate CSV path at line {line}: {relative}")
            split = (row.get("split") or "").strip() or None
            group = (row.get(group_key) or "").strip() if group_key else None
            if require_split and split not in SPLITS:
                raise ValueError(f"Invalid split at CSV line {line}: {split!r}")
            if group_key and not group:
                raise ValueError(f"Empty group at CSV line {line}")
            result[relative] = (split, group)
    return result


def _tile_starts(length: int, size: int, overlap: float) -> list[int]:
    final = max(0, length - size)
    stride = max(1, int(round(size * (1 - overlap))))
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def _write_labels(path: Path, boxes: list[tuple[int, float, float, float, float, int]],
                  width: int, height: int, x0: float = 0, y0: float = 0) -> None:
    lines = []
    for class_id, x1, y1, x2, y2, _ in boxes:
        xc = (x1 + x2 - 2 * x0) / (2 * width)
        yc = (y1 + y2 - 2 * y0) / (2 * height)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError("Internal error: generated YOLO label is outside [0, 1]")
        lines.append(f"{class_id} {xc:.12g} {yc:.12g} {bw:.12g} {bh:.12g}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _assign_splits(records: list[_ImageRecord], config: DatasetConfig) -> dict[str, str]:
    group_map = _read_csv_map(config.group_csv, False) if config.group_csv else None
    split_map = _read_csv_map(config.split_manifest, True) if config.split_manifest else None
    known_paths = {r.relative for r in records}
    for mapping, label in ((group_map, "group_csv"), (split_map, "split_manifest")):
        if mapping is not None and set(mapping) != known_paths:
            missing, extra = sorted(known_paths - set(mapping)), sorted(set(mapping) - known_paths)
            raise ValueError(f"{label} must cover each input image exactly; missing={missing[:5]}, unknown={extra[:5]}")
    for record in records:
        explicit_group = group_map[record.relative][1] if group_map else None
        manifest_group = split_map[record.relative][1] if split_map else None
        if explicit_group and manifest_group and explicit_group != manifest_group:
            raise ValueError(f"group_csv and split_manifest disagree for {record.relative}")
        record.group = explicit_group or manifest_group or Path(record.relative).stem.casefold()
        if split_map:
            record.split = split_map[record.relative][0]
    if config.split_mode == "random":
        groups = sorted({r.group for r in records})
        random.Random(config.seed).shuffle(groups)
        n = len(groups)
        n_val, n_test = round(n * config.val_fraction), round(n * config.test_fraction)
        if config.val_fraction > 0 and n >= 2:
            n_val = max(1, n_val)
        if config.test_fraction > 0 and n >= 3:
            n_test = max(1, n_test)
        while n_val + n_test >= n:
            if n_test > 0:
                n_test -= 1
            else:
                n_val -= 1
        assignment = {group: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
                      for i, group in enumerate(groups)}
        for record in records:
            record.split = assignment[record.group]
    group_splits: dict[str, str] = {}
    for record in records:
        if record.group in group_splits and group_splits[record.group] != record.split:
            raise ValueError(f"Leakage: group {record.group!r} appears in multiple splits")
        group_splits[record.group] = record.split
    if not any(r.split == "train" for r in records):
        raise ValueError("No training images assigned")
    group_source = ("group_csv" if group_map else "split_manifest_group" if
                    split_map and all(value[1] for value in split_map.values()) else "image_stem")
    return {"group_source": group_source, "split_source": config.split_mode}


def _check_duplicate_images(records: list[_ImageRecord]) -> list[dict[str, Any]]:
    """Prevent the same source bytes crossing splits, independent of file/group IDs."""
    by_hash: dict[str, list[_ImageRecord]] = {}
    for record in records:
        by_hash.setdefault(record.image_sha256, []).append(record)
    duplicates = []
    for digest, matches in sorted(by_hash.items()):
        if len(matches) < 2:
            continue
        splits = {record.split for record in matches}
        if len(splits) > 1:
            files = "; ".join(f"{record.relative} [split={record.split}, group={record.group}]"
                              for record in matches)
            raise ValueError(f"Leakage: identical source image bytes are assigned to multiple splits "
                             f"(sha256={digest}): {files}. Fix the group CSV or split manifest; "
                             "explicit assignments are not changed automatically.")
        duplicates.append({"image_sha256": digest, "split": matches[0].split,
                           "parent_paths": [record.relative for record in matches],
                           "group_ids": [record.group for record in matches],
                           "label_sha256_by_path": {record.relative: record.label_sha256 for record in matches},
                           "conflicting_label_bytes": len({record.label_sha256 for record in matches}) > 1,
                           "action": "retained_within_same_split"})
    return duplicates


def prepare_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Build a new dataset atomically. Refuses to replace non-empty output."""
    config.validate()
    source = Path(config.source).resolve()
    output = Path(config.output).resolve()
    images_dir = Path(config.images_dir).resolve() if config.images_dir else source / "images"
    labels_dir = Path(config.labels_dir).resolve() if config.labels_dir else source / "labels"
    if not images_dir.is_dir():
        raise ValueError("source must contain an images/ directory (or supply images_dir)")
    if not labels_dir.is_dir() and not config.missing_is_normal:
        raise ValueError("source must contain labels/; missing labels are not normal unless missing_is_normal is explicit")
    if output == images_dir or output == labels_dir or output.is_relative_to(images_dir) or output.is_relative_to(labels_dir):
        raise ValueError("Output cannot be inside the input images or labels directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    original_names = (_normal_names(config.names) if config.names is not None else
                      _read_names_yaml(Path(config.data_yaml).resolve() if config.data_yaml else source / "data.yaml"))
    class_ids = list(range(len(original_names))) if config.classes is None else config.classes
    if (not class_ids or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < len(original_names) for x in class_ids)
            or len(set(class_ids)) != len(class_ids)):
        raise ValueError("classes must contain distinct valid original class IDs")
    class_map = {old: new for new, old in enumerate(class_ids)}
    names = [original_names[i] for i in class_ids]
    paths = sorted((p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.relative_to(images_dir).as_posix())
    if not paths:
        raise ValueError(f"No supported images under {images_dir}")
    records: list[_ImageRecord] = []
    audit: dict[str, Any] = {"version": 1, "source_events": [], "tile_events": [],
                             "review_policy": "review/ is excluded from data.yaml; clipped arc boxes require human review"}
    class_counts = {str(i): 0 for i in range(len(original_names))}
    source_digest = hashlib.sha256()
    source_digest.update(json.dumps(original_names, ensure_ascii=False).encode("utf-8"))
    label_owners: dict[str, str] = {}
    for image_path in paths:
        relative = image_path.relative_to(images_dir).as_posix()
        label_relative = str(Path(relative).with_suffix(".txt")).replace("\\", "/")
        # JPEG and PNG files with the same stem would share one YOLO label path.
        label_key = os.path.normcase(label_relative).replace("\\", "/")
        if label_key in label_owners:
            raise ValueError(f"Ambiguous label path for {relative} and {label_owners[label_key]}")
        label_owners[label_key] = relative
        label_path = labels_dir / label_relative
        if not label_path.is_file() and not config.missing_is_normal:
            raise ValueError(f"Missing label file (unannotated is not normal): {label_path}")
        raw_label = label_path.read_bytes() if label_path.is_file() else b""
        if not label_path.is_file():
            audit["source_events"].append({"parent_path": relative, "event": "missing_label_explicitly_assumed_normal"})
        image, raw_image = _read_image(image_path)
        height, width = image.shape[:2]
        image_hash = hashlib.sha256(raw_image).hexdigest()
        label_hash = hashlib.sha256(raw_label).hexdigest()
        source_digest.update(json.dumps([relative, image_hash, label_hash], ensure_ascii=False).encode("utf-8"))
        boxes = []
        for line_number, line in enumerate(raw_label.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected five YOLO detect fields: {label_path}:{line_number}")
            try:
                class_id = int(fields[0])
                xc, yc, bw, bh = map(float, fields[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid YOLO label: {label_path}:{line_number}") from exc
            if not 0 <= class_id < len(original_names) or not all(math.isfinite(x) for x in (xc, yc, bw, bh)):
                raise ValueError(f"Invalid class or non-finite coordinates: {label_path}:{line_number}")
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"YOLO coordinates must be normalized: {label_path}:{line_number}")
            class_counts[str(class_id)] += 1
            if class_id not in class_map:
                continue
            raw_box = ((xc - bw / 2) * width, (yc - bh / 2) * height,
                       (xc + bw / 2) * width, (yc + bh / 2) * height)
            x1, y1, x2, y2 = raw_box
            clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError(f"Empty clipped source box: {label_path}:{line_number}")
            if any(abs(a - b) > 1e-6 for a, b in zip(raw_box, clipped)):
                audit["source_events"].append({"parent_path": relative, "event": "source_box_clipped_to_image",
                                               "source_line": line_number, "original_class_id": class_id,
                                               "before_xyxy": raw_box, "after_xyxy": clipped})
            boxes.append((class_map[class_id], *clipped, line_number))
        records.append(_ImageRecord(image_path, relative, width, height, boxes, image_hash, label_hash))
    split_info = _assign_splits(records, config)
    duplicate_images = _check_duplicate_images(records)
    audit["duplicate_source_images"] = duplicate_images
    expected_label_keys = {os.path.normcase(path).replace("\\", "/") for path in label_owners}
    orphan_labels = []
    if labels_dir.is_dir():
        for path in sorted(labels_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".txt":
                continue
            relative = path.relative_to(labels_dir).as_posix()
            if os.path.normcase(relative).replace("\\", "/") not in expected_label_keys:
                orphan_labels.append({"label_path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "event": "orphan_label_no_supported_image", "action": "not_used"})
    audit["orphan_labels"] = orphan_labels
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=str(output.parent))).resolve()
    if stage.parent != output.parent or not stage.name.startswith(f".{output.name}.prepare-"):
        raise RuntimeError("Unexpected staging directory")
    manifest = []
    split_summary = {split: {"full_images": 0, "tile_images": 0, "negative_tiles": 0,
                             "label_boxes": 0, "review_tiles": 0, "skipped_tiles": 0,
                             "class_counts": {str(i): 0 for i in range(len(names))}}
                     for split in SPLITS}
    try:
        for split in SPLITS:
            (stage / "images" / split).mkdir(parents=True)
            (stage / "labels" / split).mkdir(parents=True)
        for record in records:
            split = record.split
            summary = split_summary[split]
            suffix = hashlib.sha256(record.relative.encode("utf-8")).hexdigest()[:12]
            stem = f"{Path(record.relative).stem}__{suffix}"
            full_name = stem + "__full" + record.image.suffix.lower()
            full_path = Path("images") / split / full_name
            full_label = Path("labels") / split / (Path(full_name).stem + ".txt")
            shutil.copy2(record.image, stage / full_path)
            if hashlib.sha256((stage / full_path).read_bytes()).hexdigest() != record.image_sha256:
                raise ValueError(f"Source image changed during preparation: {record.image}")
            _write_labels(stage / full_label, record.boxes, record.width, record.height)
            manifest.append({"path": full_path.as_posix(), "label_path": full_label.as_posix(),
                             "parent_path": record.relative, "group_id": record.group, "split": split,
                             "kind": "full", "x0": 0, "y0": 0, "width": record.width, "height": record.height,
                             "route": "validated", "source_sha256": record.image_sha256})
            summary["full_images"] += 1
            summary["label_boxes"] += len(record.boxes)
            for class_id, *_ in record.boxes:
                summary["class_counts"][str(class_id)] += 1
            if split != "train":
                continue
            image, image_bytes = _read_image(record.image)
            if hashlib.sha256(image_bytes).hexdigest() != record.image_sha256:
                raise ValueError(f"Source image changed during preparation: {record.image}")
            for y0 in _tile_starts(record.height, config.tile_size, config.overlap):
                for x0 in _tile_starts(record.width, config.tile_size, config.overlap):
                    width = min(config.tile_size, record.width - x0)
                    height = min(config.tile_size, record.height - y0)
                    x_end, y_end = x0 + width, y0 + height
                    tile_boxes, truncated, too_small = [], [], []
                    for class_id, x1, y1, x2, y2, source_line in record.boxes:
                        ix1, iy1, ix2, iy2 = max(x0, x1), max(y0, y1), min(x_end, x2), min(y_end, y2)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        contained = x1 >= x0 - 1e-6 and y1 >= y0 - 1e-6 and x2 <= x_end + 1e-6 and y2 <= y_end + 1e-6
                        box = (class_id, ix1, iy1, ix2, iy2, source_line)
                        if not contained:
                            truncated.append(source_line)
                        if (ix2 - ix1 + 1e-6 < config.min_bbox_px or
                                iy2 - iy1 + 1e-6 < config.min_bbox_px):
                            too_small.append(source_line)
                            continue
                        tile_boxes.append(box)
                    event = {"parent_path": record.relative, "split": split, "x0": x0, "y0": y0,
                             "width": width, "height": height, "truncated_source_lines": truncated,
                             "too_small_source_lines": too_small}
                    if too_small or (truncated and config.tile_label_mode == "conservative"):
                        event.update(event="tile_skipped", reason="too_small_intersecting_box" if too_small else "truncated_task_box")
                        audit["tile_events"].append(event)
                        summary["skipped_tiles"] += 1
                        continue
                    review = bool(truncated)
                    if not tile_boxes and not review:
                        key = f"{config.seed}:{record.relative}:{x0}:{y0}".encode("utf-8")
                        coin = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
                        if coin >= config.negative_tile_keep_prob:
                            event.update(event="tile_skipped", reason="negative_tile_sampling")
                            audit["tile_events"].append(event)
                            summary["skipped_tiles"] += 1
                            continue
                    tile_name = f"{stem}__tile_x{x0}_y{y0}.png"
                    root = Path("review") if review else Path()
                    image_relative = root / "images" / split / tile_name
                    label_relative = root / "labels" / split / (Path(tile_name).stem + ".txt")
                    (stage / image_relative).parent.mkdir(parents=True, exist_ok=True)
                    (stage / label_relative).parent.mkdir(parents=True, exist_ok=True)
                    success, encoded = cv2.imencode(".png", image[y0:y_end, x0:x_end])
                    if not success:
                        raise RuntimeError(f"PNG encoding failed for {record.relative}")
                    encoded.tofile(str(stage / image_relative))
                    _write_labels(stage / label_relative, tile_boxes, width, height, x0, y0)
                    manifest.append({"path": image_relative.as_posix(), "label_path": label_relative.as_posix(),
                                     "parent_path": record.relative, "group_id": record.group, "split": split,
                                     "kind": "tile", "x0": x0, "y0": y0, "width": width, "height": height,
                                     "route": "review" if review else "validated", "source_sha256": record.image_sha256})
                    if review:
                        event.update(event="tile_requires_review", reason="clipped_rectangle_is_not_pixel_evidence",
                                     path=image_relative.as_posix(), label_path=label_relative.as_posix())
                        audit["tile_events"].append(event)
                        summary["review_tiles"] += 1
                    else:
                        summary["tile_images"] += 1
                        summary["negative_tiles"] += int(not tile_boxes)
                        summary["label_boxes"] += len(tile_boxes)
                        for class_id, *_ in tile_boxes:
                            summary["class_counts"][str(class_id)] += 1
        with (stage / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest)
        dataset = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test",
                   "names": names, "nc": len(names)}
        warnings = []
        if split_info["group_source"] == "image_stem":
            warnings.append("Default grouping uses image stem only. For multiple views/crops from one wafer or shared lots, provide a group CSV.")
        for split in ("val", "test"):
            if not split_summary[split]["full_images"]:
                warnings.append(f"{split} is empty; evaluation on this split must be skipped.")
        if config.tile_label_mode == "clip":
            warnings.append("Clipped candidates in review/ are excluded from training and need human review.")
        if duplicate_images:
            warnings.append(f"{len(duplicate_images)} duplicate source-image groups are retained within their assigned split; see audit.json.")
        if any(record["conflicting_label_bytes"] for record in duplicate_images):
            warnings.append("Some identical source images have different label-file bytes; verify their annotations in audit.json.")
        if orphan_labels:
            warnings.append(f"{len(orphan_labels)} label files have no matching supported image and are unused; see audit.json.")
        for class_id, name in enumerate(names):
            if split_summary["train"]["class_counts"][str(class_id)] == 0:
                warnings.append(f"Training split has no labels for class {name!r}; check grouping, annotations, and selected original IDs.")
        shape_counts: dict[str, int] = {}
        for record in records:
            shape = f"{record.width}x{record.height}"
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        summary = {"version": 1, "status": "complete", "source": str(source), "output": str(output),
                   "source_images_dir": str(images_dir), "source_labels_dir": str(labels_dir),
                   "source_dataset_sha256": source_digest.hexdigest(), "source_image_count": len(records),
                   "source_group_count": len({r.group for r in records}), "source_shapes": shape_counts,
                   "duplicate_source_image_groups": len(duplicate_images),
                   "duplicate_source_extra_images": sum(len(record["parent_paths"]) - 1 for record in duplicate_images),
                   "orphan_label_count": len(orphan_labels),
                   "split_info": split_info,
                   "original_class_names": original_names, "original_class_counts": class_counts,
                   "names": names, "class_id_map": {str(k): v for k, v in class_map.items()},
                   "split_summary": split_summary, "warnings": warnings, "config": asdict(config),
                   "full_images_resized": False, "tiles_splits": ["train"],
                   "review_tiles_released_to_training": False}
        audit["source_images"] = [{"parent_path": r.relative, "image_sha256": r.image_sha256,
                                   "label_sha256": r.label_sha256, "width": r.width, "height": r.height,
                                   "group_id": r.group, "split": r.split,
                                   "task_box_count": len(r.boxes)} for r in records]
        for name, data in (("data.yaml", dataset), ("audit.json", audit), ("summary.json", summary)):
            (stage / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # The final rename publishes the complete tree, not partially written labels.
        if output.exists():
            if any(output.iterdir()):
                raise ValueError(f"Output became non-empty during preparation: {output}")
            output.rmdir()
        os.replace(stage, output)
        return summary
    except BaseException:
        # Delete only this owned temporary directory, after checking its absolute path.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.prepare-"):
            shutil.rmtree(stage)
        raise


def _names_argument(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        candidate = Path(value)
        if candidate.is_file():
            parsed = json.loads(candidate.read_text(encoding="utf-8-sig"))
            return parsed.get("names", parsed) if isinstance(parsed, dict) else parsed
        raise ValueError("--names must be a JSON list/map or a path to a JSON file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON DatasetConfig; explicit CLI flags override it")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--images-dir")
    parser.add_argument("--labels-dir")
    parser.add_argument("--data-yaml")
    parser.add_argument("--names", help="JSON list/map or path to JSON; overrides data.yaml names")
    parser.add_argument("--classes", help="Comma-separated ORIGINAL class IDs; selected IDs are remapped to 0..n-1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--split-manifest", help="CSV path,split[,group_id]; all input images must be covered")
    mode.add_argument("--random-split", action="store_true", default=None)
    parser.add_argument("--group-csv", help="CSV image-relative-path,group_id; group by wafer/lot")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--tile-label-mode", choices=("conservative", "clip"))
    parser.add_argument("--min-bbox-px", type=float)
    parser.add_argument("--negative-tile-keep-prob", type=float)
    parser.add_argument("--missing-is-normal", action="store_true", default=None)
    args = parser.parse_args(argv)
    try:
        options = json.loads(Path(args.config).read_text(encoding="utf-8-sig")) if args.config else {}
        if not isinstance(options, dict):
            raise ValueError("config must be a JSON object")
        if args.split_manifest and (args.val_fraction is not None or args.test_fraction is not None):
            raise ValueError("--split-manifest and random split fraction flags are mutually exclusive")
        for name in DatasetConfig.__dataclass_fields__:
            value = getattr(args, name, None)
            if value is not None:
                options[name] = value
        if args.names:
            options["names"] = _names_argument(args.names)
        if args.classes is not None:
            options["classes"] = [int(x.strip()) for x in args.classes.split(",") if x.strip()]
        if args.split_manifest:
            options["split_mode"] = "manifest"
        elif args.random_split:
            options.update(split_mode="random", split_manifest=None)
        if not options.get("source") or not options.get("output"):
            raise ValueError("--source and --output are required, directly or through --config")
        result = prepare_dataset(DatasetConfig(**options))
    except (ValueError, TypeError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
