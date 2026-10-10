"""Train YOLO11 detect on the prepared full-image + tile dataset.

``python -m wafer_inspection.train --data datasets/prepared/data.yaml --dry-run``
audits the prepared dataset and prints the exact constructor/train/optional test
arguments plus dataset counts and digest, without printing the full file manifest
or importing Ultralytics, opening a checkpoint, downloading a
model, or creating output files. Actual training requires ``pip install
ultralytics``. The default ``yolo11s.pt`` is downloaded by Ultralytics if absent;
use ``--model /path/to/local.pt`` to use an existing local YOLO11 detect model.

The training input size is 1120 for both prepared full images and 640-pixel tiles.
Ultralytics resizes/letterboxes each sample to that size; this is a single mixed
dataset training run, not two separately trained models. Dataset splitting and
source/wafer leakage auditing belong to prepare_dataset, before tile generation.

``--resume /path/to/last.pt`` restores checkpoint training settings, optimizer,
and epoch. Only an explicitly supplied --device can override those settings.
Fresh-run flags such as --epochs, --batch, --workers, and --seed are rejected on
resume instead of being silently ignored. --data is still required to audit and
confirm the exact dataset used by the checkpoint. Completed/stripped checkpoints
cannot resume; use --model to start a new fine-tuning run instead.
Each run writes dataset_snapshot.json to the trainer's run directory through
on_train_start, before iterations. Resume requires this snapshot and unchanged
SHA-256 hashes of every image, label, and data.yaml. External checkpoints without
this wrapper snapshot can be fine-tuned with --model, not safely resumed here.
Exact duplicate image bytes across splits are rejected; different encodings or
near-duplicate acquisitions still need source/wafer grouping during preparation.
This wrapper supports a single device only (auto/None, cpu, mps, one GPU index,
or cuda:index). Multi-device strings, lists, and distributed launch environments
are rejected because snapshot callbacks require one process. A checkpoint whose
saved training device used multiple GPUs cannot resume through this wrapper;
use --model to start a new single-device fine-tuning run.

Official API references:
https://docs.ultralytics.com/models/yolo11/
https://docs.ultralytics.com/modes/train/
https://docs.ultralytics.com/usage/cfg/
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
import uuid


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
YOLO11_DETECT_NAMES = {f"yolo11{size}.pt" for size in "nsmlx"}
DATASET_SNAPSHOT_FILENAME = "dataset_snapshot.json"
DEFAULT_CONFIG = {
    "model": "yolo11s.pt", "imgsz": 1120, "epochs": 150, "batch": -1,
    "device": None, "workers": 4, "seed": 42, "deterministic": True,
    "patience": 40, "optimizer": "auto", "cos_lr": True,
    "mosaic": 0.2, "close_mosaic": 20, "mixup": 0.0, "perspective": 0.0,
    "degrees": 0.0, "fliplr": 0.0, "flipud": 0.0, "translate": 0.02,
    "scale": 0.1, "shear": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.1,
    "rect": False, "val": True, "save": True, "plots": True,
    "project": "runs/train", "name": "wafer-yolo11",
}


@dataclass
class TrainingPlan:
    model_source: str
    train_kwargs: dict[str, Any]
    test_kwargs: dict[str, Any] | None
    dataset: dict[str, Any]
    metadata_dir: str
    resume: bool
    may_download_model: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataset_yaml(path: Path) -> dict[str, Any]:
    """prepare_dataset writes JSON, which is valid YAML and needs no YAML dependency."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "This data.yaml is not JSON-as-YAML. Install PyYAML (pip install pyyaml), "
                "or use data.yaml produced by prepare_dataset; dry-run does not require ultralytics."
            ) from exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid dataset YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("data.yaml must be an object")
    return raw


def _dataset_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        indexed: dict[int, Any] = {}
        for key, name in raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("names dictionary keys must be integer class IDs") from exc
            if isinstance(key, bool) or str(class_id) != str(key) or class_id in indexed:
                raise ValueError("names dictionary keys must be unique integer class IDs")
            indexed[class_id] = name
        if set(indexed) != set(range(len(indexed))):
            raise ValueError("names class IDs must be contiguous from 0")
        names = [indexed[i] for i in range(len(indexed))]
    elif isinstance(raw, list):
        names = raw
    else:
        raise ValueError("data.yaml names must be a list or integer-indexed dictionary")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every class needs a nonempty name; class IDs are not guessed")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    return names


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data: Path, *, evaluate_test: bool = False) -> dict[str, Any]:
    """Audit standard prepared images/{split} and labels/{split} directories.

    Labels must have five normalized detect columns. An empty label file is a
    valid background image. Missing label files are errors, not assumed background.
    This validates artifacts, not the source-family leakage policy of preparation.
    """
    data = Path(data).resolve()
    if not data.is_file():
        raise ValueError(f"Dataset YAML does not exist: {data}")
    raw = _load_dataset_yaml(data)
    names = _dataset_names(raw.get("names"))
    if "nc" in raw and (type(raw["nc"]) is not int or raw["nc"] != len(names)):
        raise ValueError("data.yaml nc must equal len(names)")
    root_value = raw.get("path", str(data.parent))
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("data.yaml path must be a nonempty absolute path")
    root = Path(root_value)
    if not root.is_absolute():
        raise ValueError("data.yaml path must be absolute; use prepare_dataset output to avoid Ultralytics relative-path ambiguity")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    splits: dict[str, Any] = {}
    seen_paths: set[Path] = set()
    seen_image_hashes: dict[str, tuple[str, Path]] = {}
    fingerprints: list[dict[str, str]] = []
    warnings: list[str] = []
    for split in ("train", "val", "test"):
        declared = raw.get(split)
        if split == "test" and declared is None and not evaluate_test:
            splits[split] = {"images": 0, "objects": 0, "class_object_counts": [0] * len(names), "declared": False}
            continue
        if not isinstance(declared, str) or not declared:
            raise ValueError(f"data.yaml {split} must be a standard image directory path")
        images_dir = (root / declared).resolve()
        expected = (root / "images" / split).resolve()
        if images_dir != expected:
            raise ValueError(f"Only prepared standard images/{split} layout is supported; got {images_dir}")
        labels_dir = (root / "labels" / split).resolve()
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise ValueError(f"Both images/{split} and labels/{split} must exist")
        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images and (split != "test" or evaluate_test):
            raise ValueError(f"Split {split} has no images")
        objects = [0] * len(names)
        background_images = 0
        label_paths: set[Path] = set()
        image_stems: set[Path] = set()
        for image in images:
            resolved_image = image.resolve()
            if resolved_image in seen_paths:
                raise ValueError(f"Image path occurs in multiple splits: {image}")
            seen_paths.add(resolved_image)
            image_hash = _sha256_file(image)
            previous = seen_image_hashes.get(image_hash)
            if previous is not None and previous[0] != split:
                raise ValueError(f"Identical image content crosses splits ({previous[0]} -> {split}): {previous[1]} and {image}")
            seen_image_hashes.setdefault(image_hash, (split, image))
            fingerprints.append({"relative_path": image.relative_to(root).as_posix(), "sha256": image_hash})
            relative = image.relative_to(images_dir)
            stem = relative.with_suffix("")
            if stem in image_stems:
                raise ValueError(f"Multiple images share one YOLO label stem: {image}")
            image_stems.add(stem)
            label = labels_dir / relative.with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label file, including required empty background labels: {label}")
            label_paths.add(label.resolve())
            label_bytes = label.read_bytes()
            fingerprints.append({"relative_path": label.relative_to(root).as_posix(), "sha256": hashlib.sha256(label_bytes).hexdigest()})
            lines = [line.strip() for line in label_bytes.decode("utf-8-sig").splitlines() if line.strip()]
            if not lines:
                background_images += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("expected five detect columns")
                    values = [float(value) for value in parts]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError("non-finite values")
                    class_value, cx, cy, width, height = values
                    class_id = int(class_value)
                    if class_id != class_value or not 0 <= class_id < len(names):
                        raise ValueError("class ID outside names mapping")
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("coordinates must be normalized; width and height must be positive")
                    if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
                        raise ValueError("box extends outside the image")
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid detect label {label}:{line_number}: {exc}") from exc
                objects[class_id] += 1
        orphans = sorted(p for p in labels_dir.rglob("*.txt") if p.resolve() not in label_paths)
        if orphans:
            raise ValueError(f"Orphan label without a supported image: {orphans[0]}")
        splits[split] = {"image_directory": str(images_dir), "label_directory": str(labels_dir),
                         "images": len(images), "objects": sum(objects), "background_images": background_images,
                         "class_object_counts": objects, "declared": True}
        if split != "test":
            absent = [names[i] for i, count in enumerate(objects) if count == 0]
            if absent:
                warnings.append(f"{split} has zero boxes for classes: {absent}")
    if splits["train"]["objects"] == 0:
        raise ValueError("Training split contains no positive detect labels")
    fingerprints.sort(key=lambda item: item["relative_path"])
    data_yaml_sha256 = _sha256_file(data)
    fingerprint_payload = {"data_yaml_sha256": data_yaml_sha256, "names": names, "nc": len(names), "files": fingerprints}
    content_digest = hashlib.sha256(json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"data_yaml": str(data), "data_yaml_sha256": data_yaml_sha256,
            "dataset_content_sha256": content_digest, "fingerprint_algorithm": "SHA-256 of file bytes and sorted relative-path manifest",
            "files": fingerprints,
            "root": str(root), "names": names, "nc": len(names), "splits": splits,
            "warnings": warnings, "source_family_leakage_audit": "Performed by prepare_dataset before tiling; not inferred by this label/path audit"}


def dataset_snapshot_path(checkpoint: Path) -> Path:
    """Standard Ultralytics checkpoints live in run/weights/{last,best}.pt."""
    checkpoint = Path(checkpoint).resolve()
    run_directory = checkpoint.parent.parent if checkpoint.parent.name.casefold() == "weights" else checkpoint.parent
    return run_directory / DATASET_SNAPSHOT_FILENAME


def verify_resume_snapshot(checkpoint: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    """Require an unchanged, content-level wrapper snapshot before restoring state."""
    snapshot_path = dataset_snapshot_path(checkpoint)
    if not snapshot_path.is_file():
        raise ValueError(f"Resume dataset snapshot is missing: {snapshot_path}. For an external or older checkpoint, use --model {checkpoint} to start a new fine-tuning run instead of --resume.")
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read resume dataset snapshot: {snapshot_path}") from exc
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError(f"Unsupported resume dataset snapshot: {snapshot_path}")
    for key in ("data_yaml", "data_yaml_sha256", "dataset_content_sha256", "names", "nc", "files"):
        if snapshot.get(key) != dataset.get(key):
            raise ValueError(f"Resume dataset content changed ({key} differs from {snapshot_path}); restore the original dataset or use --model for a new fine-tuning run")
    training_args = snapshot.get("training_args", {})
    imgsz = training_args.get("imgsz") if isinstance(training_args, dict) else None
    if type(imgsz) is not int or imgsz < 32 or imgsz % 32:
        raise ValueError("Resume snapshot is missing a valid training imgsz; use --model for a new fine-tuning run")
    return snapshot


def write_dataset_snapshot(trainer: Any, dataset: dict[str, Any]) -> Path:
    """Call at on_train_start, after trainer setup but before training iterations."""
    save_directory = getattr(trainer, "save_dir", None)
    if save_directory is None:
        raise ValueError("Trainer must expose save_dir before writing a dataset snapshot")
    save_directory = Path(save_directory).resolve()
    root = Path(dataset["root"])
    if save_directory == root or save_directory.is_relative_to(root):
        raise ValueError("Training save directory must be outside the prepared dataset")
    args = getattr(trainer, "args", None)
    imgsz = args.get("imgsz") if isinstance(args, dict) else getattr(args, "imgsz", None)
    if type(imgsz) is not int or imgsz < 32 or imgsz % 32:
        raise ValueError("Trainer must expose a valid actual training imgsz before iterations")
    snapshot = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                **{key: dataset[key] for key in ("data_yaml", "data_yaml_sha256", "dataset_content_sha256", "fingerprint_algorithm", "names", "nc", "files")},
                "training_args": {"imgsz": imgsz}}
    save_directory.mkdir(parents=True, exist_ok=True)
    path = save_directory / DATASET_SNAPSHOT_FILENAME
    if path.exists():
        existing = verify_resume_snapshot(save_directory / "weights" / "last.pt", dataset)
        if existing["training_args"]["imgsz"] != imgsz:
            raise ValueError("Existing dataset snapshot training imgsz differs; snapshot will not be overwritten")
    else:
        _save_metadata(path, snapshot)
    return path


def load_training_config(path: Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Training config must be a JSON object")
        unknown = sorted(set(raw) - config.keys())
        if unknown:
            raise ValueError(f"Unsupported training config keys: {unknown}")
        config.update(raw)
    return config


def validate_single_device(device: Any) -> str | int | None:
    """Accept only explicit single-device forms; auto selects one device, not DDP."""
    if device is None:
        return None
    if type(device) is int and device >= 0:
        return device
    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"cpu", "mps"} or re.fullmatch(r"(?:cuda:)?[0-9]+", normalized):
            return normalized
    raise ValueError("Only one training device is supported: null/auto, cpu, mps, a nonnegative GPU index, or cuda:index. Multi-device/DDP values are unsupported; start a new single-device --model run for multi-GPU checkpoints.")


def _reject_distributed_environment() -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("Distributed launch environment is unsupported; run this wrapper directly with python on one device") from exc
        if value >= 0 and key != "WORLD_SIZE" or key == "WORLD_SIZE" and value > 1:
            raise ValueError("Distributed/DDP training is unsupported by this snapshot wrapper; run directly with python on one device")


def _validate_settings(config: dict[str, Any]) -> None:
    for key in ("epochs", "imgsz", "workers", "seed", "close_mosaic", "patience"):
        if type(config[key]) is not int or config[key] < (1 if key in {"epochs", "imgsz"} else 0):
            raise ValueError(f"{key} must be {'positive' if key in {'epochs', 'imgsz'} else 'nonnegative'} integer")
    if config["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    batch = config["batch"]
    if isinstance(batch, bool) or not isinstance(batch, (int, float)) or not math.isfinite(batch):
        raise ValueError("batch must be -1, a positive integer, or a fraction in (0, 1)")
    if batch != -1 and not (0 < batch < 1 or batch >= 1 and float(batch).is_integer()):
        raise ValueError("batch must be -1, a positive integer, or a fraction in (0, 1)")
    if config["close_mosaic"] > config["epochs"]:
        raise ValueError("close_mosaic cannot exceed epochs; lower it for a short trial")
    for key in ("mosaic", "mixup", "fliplr", "flipud", "translate", "scale", "hsv_h", "hsv_s", "hsv_v"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite value in [0, 1]")
    for key, upper in (("degrees", 180), ("shear", 180), ("perspective", 0.001)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{key} must be a finite value in [0, {upper}]")
    for key in ("deterministic", "cos_lr", "rect", "val", "save", "plots"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not config["val"] or not config["save"] or config["rect"]:
        raise ValueError("This mixed full/tile recipe requires val=true, save=true and rect=false")
    if config["optimizer"] not in {"auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"}:
        raise ValueError("Unsupported optimizer for this recipe")
    if not isinstance(config["model"], str) or not config["model"]:
        raise ValueError("model must be a YOLO11 detect checkpoint")
    config["device"] = validate_single_device(config["device"])
    if not isinstance(config["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["name"]):
        raise ValueError("name must be a simple run name without path separators")
    if not isinstance(config["project"], str) or not config["project"]:
        raise ValueError("project must be a nonempty directory path")


def _resolve_model_source(raw: str) -> tuple[str, bool]:
    local = Path(raw)
    if local.is_file():
        if local.suffix.lower() != ".pt":
            raise ValueError("Training requires a PyTorch .pt YOLO11 detection checkpoint")
        return str(local.resolve()), False
    if raw in YOLO11_DETECT_NAMES:
        return raw, True
    raise ValueError("Use yolo11n/s/m/l/x.pt or an existing local YOLO11 detect .pt; other tasks and model generations are unsupported")


def build_training_plan(data: Path, *, config: dict[str, Any] | None = None,
                        resume: Path | None = None, resume_device: str | None = None,
                        evaluate_test: bool = False, metadata_dir: Path | None = None) -> TrainingPlan:
    _reject_distributed_environment()
    settings = load_training_config()
    if config:
        unknown = sorted(set(config) - settings.keys())
        if unknown:
            raise ValueError(f"Unsupported training config keys: {unknown}")
        settings.update(config)
    _validate_settings(settings)
    resume_device = validate_single_device(resume_device)
    audit = audit_dataset(data, evaluate_test=evaluate_test)
    project = Path(settings["project"]).resolve()
    root = Path(audit["root"])
    if project == root or project.is_relative_to(root):
        raise ValueError("Training project must be outside the prepared dataset directory")
    if project.exists() and not project.is_dir():
        raise ValueError("Training project must be a directory")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = Path(metadata_dir).resolve() if metadata_dir else project / "_metadata" / f"{settings['name']}-{timestamp}-{uuid.uuid4().hex[:8]}"
    if metadata == root or metadata.is_relative_to(root):
        raise ValueError("Training metadata must be outside the prepared dataset directory")
    if metadata.exists():
        raise ValueError("Metadata directory must be new; existing metadata is never overwritten")
    notes = [
        "This snapshot wrapper supports one device and one training process; multi-device/DDP configurations are rejected.",
        "1120 is the training input size for full images and prepared 640-pixel tiles; tiles are resized/letterboxed by Ultralytics.",
        "Validation uses val during training. Test is evaluated once after training only when --evaluate-test is specified.",
        "Model task and YOLO11 architecture are verified at runtime before training, not by opening .pt in dry-run.",
        "Every prepared image and label is hashed; exact image bytes shared across different splits are rejected. Source/wafer grouping must still handle near-duplicates and different encodings.",
    ]
    resume_snapshot: dict[str, Any] | None = None
    if resume is not None:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
            raise ValueError("--resume must point to an existing interrupted-run .pt checkpoint (normally last.pt)")
        resume_snapshot = verify_resume_snapshot(checkpoint, audit)
        model_source, download = str(checkpoint), False
        train_kwargs: dict[str, Any] = {"resume": True}
        if resume_device is not None:
            train_kwargs["device"] = resume_device
        notes.append("Resume restores checkpoint settings; supplied data hashes must match dataset_snapshot.json saved before the original training iterations. Fresh-run defaults are not passed to model.train.")
        notes.append(f"Optional resumed test evaluation uses checkpoint training imgsz={resume_snapshot['training_args']['imgsz']}, read from the snapshot in dry-run and verified against checkpoint train_args at runtime.")
    else:
        target = project / settings["name"]
        if target.exists():
            raise ValueError(f"Fresh training run directory already exists; choose a new --name: {target}")
        model_source, download = _resolve_model_source(settings["model"])
        train_kwargs = {key: value for key, value in settings.items() if key != "model" and value is not None}
        train_kwargs.update(data=audit["data_yaml"], project=str(project), exist_ok=False)
    test_kwargs: dict[str, Any] | None = None
    if evaluate_test:
        test_imgsz = resume_snapshot["training_args"]["imgsz"] if resume_snapshot is not None else settings["imgsz"]
        test_kwargs = {"data": audit["data_yaml"], "split": "test", "imgsz": test_imgsz,
                       "batch": 1, "project": str(metadata / "evaluation"), "name": "test", "exist_ok": False}
        device = resume_device if resume is not None else settings["device"]
        if device is not None:
            test_kwargs["device"] = device
    if download:
        notes.append(f"Actual training may download {model_source} through Ultralytics. Dry-run does not download anything.")
    return TrainingPlan(model_source, train_kwargs, test_kwargs, audit, str(metadata), resume is not None, download, notes)


def _verify_yolo11_detect(model: Any) -> None:
    if getattr(model, "task", None) != "detect":
        raise ValueError("Only YOLO11 detect checkpoints are supported")
    architecture = getattr(getattr(model, "model", None), "yaml", None)
    yaml_file = architecture.get("yaml_file", "") if isinstance(architecture, dict) else ""
    if not re.fullmatch(r"yolo11[nsmlx]?\.yaml", Path(str(yaml_file)).name.lower()):
        raise ValueError("Checkpoint architecture is not verified as standard YOLO11 detect (model.model.yaml.yaml_file); use a standard YOLO11 checkpoint")


def _verify_resume_dataset(model: Any, dataset: dict[str, Any], device: Any = None, checkpoint_path: Path | None = None) -> None:
    checkpoint = getattr(model, "ckpt", None)
    if not isinstance(checkpoint, dict) or checkpoint.get("optimizer") is None or checkpoint.get("epoch", -1) < 0:
        raise ValueError("Resume requires an interrupted checkpoint with optimizer/epoch state; start a new --model fine-tuning run for completed/stripped weights")
    args = checkpoint.get("train_args", {})
    checkpoint_data = args.get("data") if isinstance(args, dict) else None
    if not isinstance(checkpoint_data, str) or Path(checkpoint_data).resolve() != Path(dataset["data_yaml"]):
        raise ValueError("--data does not match checkpoint training data; resume cannot silently change the dataset")
    if _dataset_names(model.names) != dataset["names"]:
        raise ValueError("Checkpoint class IDs/names differ from data.yaml; resume cannot remap classes")
    if checkpoint_path is None:
        raise ValueError("Checkpoint path is required to verify its saved dataset content snapshot")
    snapshot = verify_resume_snapshot(checkpoint_path, dataset)
    if args.get("imgsz") != snapshot["training_args"]["imgsz"]:
        raise ValueError("Checkpoint training imgsz differs from the saved dataset snapshot")
    validate_single_device(args.get("device"))
    validate_single_device(device)


def _save_metadata(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def execute_training(plan: TrainingPlan, *, model_factory: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Execute a reviewed plan; a fake factory can verify wiring without training."""
    _reject_distributed_environment()
    validate_single_device(plan.train_kwargs.get("device"))
    if model_factory is None:
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Actual training requires: pip install ultralytics; use --dry-run for dependency-free auditing") from exc
        model_factory = YOLO
        dependency_version = ultralytics.__version__
    else:
        dependency_version = "injected_model_factory (not a real training verification)"
    metadata = Path(plan.metadata_dir)
    metadata.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
              "ultralytics_version": dependency_version, "plan": plan.to_dict()}
    record_path = metadata / "training_run.json"
    _save_metadata(record_path, record)
    try:
        model = model_factory(plan.model_source)
        _verify_yolo11_detect(model)
        if plan.resume:
            _verify_resume_dataset(model, plan.dataset, plan.train_kwargs.get("device"), Path(plan.model_source))
        snapshot_written: list[Path] = []

        def snapshot_before_iterations(trainer: Any) -> None:
            current = audit_dataset(Path(plan.dataset["data_yaml"]), evaluate_test=plan.test_kwargs is not None)
            if current["dataset_content_sha256"] != plan.dataset["dataset_content_sha256"]:
                raise ValueError("Dataset content changed after planning and before training iterations; rebuild the plan")
            snapshot_path = write_dataset_snapshot(trainer, plan.dataset)
            snapshot_written.append(snapshot_path)
            record["dataset_snapshot"] = str(snapshot_path)
            _save_metadata(record_path, record)

        add_callback = getattr(model, "add_callback", None)
        if not callable(add_callback):
            raise RuntimeError("Model must expose add_callback to save dataset content before training iterations")
        add_callback("on_train_start", snapshot_before_iterations)
        record["status"] = "training"
        _save_metadata(record_path, record)
        model.train(**plan.train_kwargs)
        if not snapshot_written:
            raise RuntimeError("Training did not invoke on_train_start; no dataset snapshot was saved before iterations")
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is None:
            raise RuntimeError("Training returned without a trainer save directory")
        save_dir = Path(save_dir).resolve()
        best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt")).resolve()
        last = Path(getattr(trainer, "last", save_dir / "weights" / "last.pt")).resolve()
        if not best.is_file() and not last.is_file():
            raise RuntimeError("Training returned without best.pt or last.pt; inspect Ultralytics logs")
        record.update(status="trained", save_directory=str(save_dir), best_weights=str(best) if best.is_file() else None,
                      last_weights=str(last) if last.is_file() else None)
        if plan.test_kwargs is not None:
            if not best.is_file():
                raise RuntimeError("Requested test evaluation requires best.pt selected using val")
            record["status"] = "evaluating_test"
            _save_metadata(record_path, record)
            best_model = model_factory(str(best))
            _verify_yolo11_detect(best_model)
            metrics = best_model.val(**plan.test_kwargs)
            results = getattr(metrics, "results_dict", {})
            test_metrics = {str(key): float(value) for key, value in results.items()}
            if not all(math.isfinite(value) for value in test_metrics.values()):
                raise RuntimeError("Test evaluation returned non-finite metrics; inspect validation logs")
            record["test_metrics"] = test_metrics
        record.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        _save_metadata(record_path, record)
        return record
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}", finished_utc=datetime.now(timezone.utc).isoformat())
        _save_metadata(record_path, record)
        raise


def _batch_argument(value: str) -> int | float:
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("batch must be -1, a positive integer, or a fraction in (0, 1)") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and train YOLO11 detect on the prepared full-image + 640-tile dataset.",
        epilog="Default model yolo11s.pt may download during actual training. --dry-run never imports Ultralytics or downloads weights. "
               "Use pip install ultralytics for actual training. Resume requires an unchanged dataset_snapshot.json from this wrapper "
               "and restores checkpoint settings; only --device may override them. One device only; multi-GPU/DDP is unsupported. "
               "External weights without a snapshot can use --model for new fine-tuning.",
    )
    parser.add_argument("--data", type=Path, required=True, help="data.yaml created by prepare_dataset (JSON-as-YAML)")
    parser.add_argument("--config", type=Path, help="Training JSON, e.g. training_config.json; omitted uses the same built-in defaults")
    parser.add_argument("--model", help="yolo11n/s/m/l/x.pt or existing local YOLO11 detection .pt (default yolo11s.pt)")
    parser.add_argument("--epochs", type=int, help="Fresh run epoch count (default 150)")
    parser.add_argument("--imgsz", type=int, help="Fresh run training size, multiple of 32 (default 1120)")
    parser.add_argument("--batch", type=_batch_argument, help="Fresh run batch; -1 auto, positive integer, or GPU fraction (default -1)")
    parser.add_argument("--device", help="Single device: cpu, mps, GPU index such as 0, or cuda:0; omitted auto-selects one device. Multi-GPU/DDP unsupported")
    parser.add_argument("--workers", type=int, help="Fresh run data loader workers (default 4; 0 for Windows troubleshooting)")
    parser.add_argument("--seed", type=int, help="Fresh run random seed (default 42)")
    parser.add_argument("--project", help="Fresh run output root outside dataset (default runs/train)")
    parser.add_argument("--name", help="Fresh run unique directory name (default wafer-yolo11)")
    parser.add_argument("--mosaic", type=float, help="Fresh run mosaic probability (default 0.2)")
    parser.add_argument("--close-mosaic", type=int, help="Disable mosaic in last N epochs (default 20)")
    parser.add_argument("--degrees", type=float, help="Rotation range in degrees, choose based on acquisition geometry (default 0)")
    parser.add_argument("--fliplr", type=float, help="Horizontal flip probability, if physically valid (default 0)")
    parser.add_argument("--flipud", type=float, help="Vertical flip probability, if physically valid (default 0)")
    parser.add_argument("--resume", type=Path, help="Interrupted last.pt plus wrapper dataset_snapshot.json; restores settings and requires unchanged image/label content")
    parser.add_argument("--metadata-dir", type=Path, help="Optional new metadata directory outside dataset; default unique directory under project/_metadata")
    parser.add_argument("--evaluate-test", action="store_true", help="After training only, evaluate best.pt once on the nonempty test split")
    parser.add_argument("--dry-run", action="store_true", help="Audit data and print exact arguments; do not import YOLO, download, train or write outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_training_config(args.config)
        fresh_keys = ("model", "epochs", "imgsz", "batch", "workers", "seed", "project", "name", "mosaic", "close_mosaic", "degrees", "fliplr", "flipud")
        if args.resume is not None:
            incompatible = ["--" + key.replace("_", "-") for key in fresh_keys if getattr(args, key) is not None]
            if args.config is not None:
                incompatible.append("--config")
            if incompatible:
                raise ValueError(f"Resume restores checkpoint settings; these fresh-run options cannot override it: {incompatible}. Only --device can override.")
        for key in (*fresh_keys, "device"):
            value = getattr(args, key)
            if value is not None:
                settings[key] = value
        plan = build_training_plan(args.data, config=settings, resume=args.resume, resume_device=args.device,
                                   evaluate_test=args.evaluate_test, metadata_dir=args.metadata_dir)
        printed_plan = plan.to_dict()
        printed_plan["dataset"]["fingerprinted_files"] = len(printed_plan["dataset"].pop("files"))
        print(json.dumps(printed_plan, ensure_ascii=False, indent=2, allow_nan=False))
        if args.dry_run:
            return 0
        record = execute_training(plan)
        print(f"Training metadata: {Path(plan.metadata_dir) / 'training_run.json'}")
        print(f"Best weights: {record.get('best_weights')}")
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
