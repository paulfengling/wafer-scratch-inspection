"""CLI and conservative routing for wafer scratch inspection.

Install the core dependencies with ``pip install -r requirements.txt``.
YOLO is optional: install it separately with ``pip install ultralytics`` and
provide an existing local detection model via ``--weights``. No model is
downloaded automatically. Model class IDs must come from your own training
configuration. If --critical-classes is omitted, only exact, unique matches of
circle scratch and line scratch in model.names (ignoring spaces, underscores,
hyphens, and case) identify the two critical classes. Other labels stay unresolved.

A negative image without a YOLO model is pending, not accepted. Even a
``no_defect_detected`` decision is an observation from this configured pipeline,
not a guarantee of a good wafer. Rule scores are geometric ranking measures,
not calibrated probabilities. Default rule hits skip all YOLO inference;
use ``--yolo-on-rule-hit`` when other defects also need to be recorded.

``--tile-size`` controls the original-pixel crop size; ``--tile-imgsz`` controls
YOLO's tile input size independently. The default 640-pixel crops are resized
to 1120 by YOLO, matching the mixed full-image/tile training recipe.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import cv2
import numpy as np

from . import __version__
from .circle_rule import RuleConfig, detect_circle_scratches

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class PipelineConfig:
    source: Path
    output: Path
    rule: RuleConfig | None = None
    weights: Path | None = None
    critical_classes: tuple[int, ...] | None = None
    class_ids: tuple[int, ...] | None = None
    imgsz: int = 1120
    confidence: float = 0.15
    nms_iou: float = 0.45
    tile_size: int = 0
    tile_overlap: float = 0.25
    max_det: int = 1000
    device: str | None = None
    yolo_on_rule_hit: bool = False
    tile_imgsz: int = 1120

    def validate(self, *, check_weights: bool = True) -> None:
        self.source = Path(self.source).resolve()
        self.output = Path(self.output).resolve()
        if not self.source.exists():
            raise ValueError(f"Source does not exist: {self.source}")
        if self.source.is_file() and self.source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {self.source.suffix}")
        if not (self.source.is_file() or self.source.is_dir()):
            raise ValueError("Source must be an image file or directory")
        if self.output.exists() and not self.output.is_dir():
            raise ValueError("Output must be a directory")
        if self.source == self.output or self.source.is_relative_to(self.output):
            raise ValueError("Output cannot equal or contain the source")
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("Output must be empty or new; choose a new run directory to avoid mixing previous routing results")
        if isinstance(self.imgsz, bool) or self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if not math.isfinite(self.nms_iou) or not 0 < self.nms_iou <= 1:
            raise ValueError("nms_iou must be between 0 and 1")
        if isinstance(self.tile_size, bool) or self.tile_size < 0:
            raise ValueError("tile_size must be zero or a multiple of 32 at least 64")
        if self.tile_size and (self.tile_size < 64 or self.tile_size % 32):
            raise ValueError("tile_size must be zero or a multiple of 32 at least 64")
        if type(self.tile_imgsz) is not int or self.tile_imgsz < 32 or self.tile_imgsz % 32:
            raise ValueError("tile_imgsz must be a positive integer multiple of 32")
        if not math.isfinite(self.tile_overlap) or not 0 <= self.tile_overlap < 1:
            raise ValueError("tile_overlap must be in [0, 1)")
        if isinstance(self.max_det, bool) or self.max_det < 1:
            raise ValueError("max_det must be positive")
        for name in ("critical_classes", "class_ids"):
            value = getattr(self, name)
            if value is not None:
                if not value or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value):
                    raise ValueError(f"{name} must be nonempty nonnegative integer IDs")
                setattr(self, name, tuple(dict.fromkeys(value)))
        if self.class_ids is not None and self.critical_classes is not None:
            if not set(self.critical_classes).issubset(self.class_ids):
                raise ValueError("class_ids must include every critical class ID")
        if self.weights is not None:
            self.weights = Path(self.weights).resolve()
            if check_weights and not self.weights.is_file():
                raise ValueError(f"YOLO weights must be an existing local file: {self.weights}")
        if self.yolo_on_rule_hit and self.weights is None:
            raise ValueError("--yolo-on-rule-hit requires --weights")
        if self.rule is None:
            self.rule = RuleConfig()
        self.rule.validate()


@dataclass
class YoloDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    source: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_image(path: Path) -> np.ndarray:
    """OpenCV imread does not reliably handle Windows Unicode paths."""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8; explicitly normalize {image.dtype} input before inspection: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image channels: {image.shape}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode PNG: {path}")
    encoded.tofile(str(path))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def discover_images(source: Path, output: Path) -> list[Path]:
    """Exclude an output tree inside source, including resolved symlink paths."""
    if source.is_file():
        return [source]
    output = output.resolve()
    return sorted(
        (p for p in source.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
         and not p.resolve().is_relative_to(output)),
        key=lambda p: str(p.relative_to(source)).casefold(),
    )


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Place a final tile at the far edge; no trailing pixels are discarded."""
    if length <= tile_size:
        return [0]
    step = max(1, int(round(tile_size * (1 - overlap))))
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def classwise_nms(detections: list[YoloDetection], iou_threshold: float) -> list[YoloDetection]:
    """Merge full image and tile detections only within the same class."""
    kept: list[YoloDetection] = []
    by_class: dict[int, list[YoloDetection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)
    for class_id in sorted(by_class):
        candidates = sorted(by_class[class_id], key=lambda d: d.confidence, reverse=True)
        while candidates:
            winner, *others = candidates
            kept.append(winner)
            x1, y1, x2, y2 = winner.bbox
            winner_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            candidates = []
            for candidate in others:
                a1, b1, a2, b2 = candidate.bbox
                intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
                area = max(0.0, a2 - a1) * max(0.0, b2 - b1)
                union = winner_area + area - intersection
                if union <= 0 or intersection / union <= iou_threshold:
                    candidates.append(candidate)
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def _model_names(names: Any) -> dict[int, str]:
    if isinstance(names, (tuple, list)):
        mapping = dict(enumerate(names))
    elif isinstance(names, dict):
        mapping = names
    else:
        raise ValueError("YOLO model must provide a class names list or dictionary")
    normalized: dict[int, str] = {}
    for key, name in mapping.items():
        class_id = int(key)
        if str(class_id) != str(key) or class_id < 0 or not isinstance(name, str) or not name:
            raise ValueError(f"Invalid YOLO class mapping entry: {key!r}")
        normalized[class_id] = name
    if not normalized:
        raise ValueError("YOLO model class mapping is empty")
    return normalized


def _validate_class_mapping(names: dict[int, str], config: PipelineConfig) -> None:
    for requested in (config.critical_classes, config.class_ids):
        if requested is not None:
            missing = sorted(set(requested) - names.keys())
            if missing:
                raise ValueError(f"Requested class IDs are absent from model.names: {missing}")


def resolve_critical_classes(names: dict[int, str], explicit: tuple[int, ...] | None) -> tuple[tuple[int, ...] | None, str]:
    if explicit is not None:
        return explicit, "explicit"
    normalized = {key: name.lower().replace(" ", "").replace("_", "").replace("-", "") for key, name in names.items()}
    circles = [key for key, name in normalized.items() if name == "circlescratch"]
    lines = [key for key, name in normalized.items() if name == "linescratch"]
    if len(circles) == len(lines) == 1 and circles[0] != lines[0]:
        return (circles[0], lines[0]), "model_names_unique_match"
    return None, "unresolved"


class UltralyticsRunner:
    """Lazy model loading; direct and tiled inference return original coordinates."""

    def __init__(self, config: PipelineConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO requires the optional dependency: pip install ultralytics") from exc
        self.config = config
        self.model = YOLO(str(config.weights))
        if getattr(self.model, "task", None) != "detect":
            raise ValueError("YOLO weights must be for object detection")
        self.names = _model_names(self.model.names)
        _validate_class_mapping(self.names, config)

    def _predict_patch(self, patch: np.ndarray, x: int, y: int, size: int, source: str,
                       *, pad_to_size: bool = False) -> list[YoloDetection]:
        height, width = patch.shape[:2]
        # Preserve native pixels for full-image inference. Tiles must remain
        # unpadded here so YOLO scales 640-pixel crops to tile_imgsz, as in training.
        # YOLO returns boxes in the supplied image's coordinates after resizing.
        model_input = patch
        if pad_to_size:
            model_input = cv2.copyMakeBorder(
                patch, 0, max(0, size - height), 0, max(0, size - width),
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
        kwargs: dict[str, Any] = {
            "source": model_input, "imgsz": size, "rect": False,
            "conf": self.config.confidence, "iou": self.config.nms_iou,
            "max_det": self.config.max_det, "agnostic_nms": False, "verbose": False,
        }
        if self.config.class_ids is not None:
            kwargs["classes"] = list(self.config.class_ids)
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        results = self.model.predict(**kwargs)
        if len(results) != 1 or results[0].boxes is None:
            raise RuntimeError("YOLO did not return one valid detection result")
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        if xyxy.ndim != 2 or xyxy.shape[1] != 4 or len(xyxy) != len(confidence) or len(xyxy) != len(classes):
            raise ValueError("Malformed YOLO detection tensors")
        found: list[YoloDetection] = []
        for bbox, score, raw_class in zip(xyxy, confidence, classes):
            if not np.isfinite(bbox).all() or not math.isfinite(float(score)) or not math.isfinite(float(raw_class)):
                raise ValueError("YOLO returned a non-finite detection")
            class_id = int(raw_class)
            if float(raw_class) != class_id or class_id not in self.names:
                raise ValueError(f"Unknown YOLO class ID: {raw_class}")
            if not 0 <= float(score) <= 1:
                raise ValueError("YOLO confidence must be between 0 and 1")
            x1, y1, x2, y2 = map(float, bbox)
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            found.append(YoloDetection((x1 + x, y1 + y, x2 + x, y2 + y), float(score), class_id, self.names[class_id], source))
        return found

    def detect(self, image: np.ndarray) -> list[YoloDetection]:
        config = self.config
        detections = self._predict_patch(image, 0, 0, config.imgsz, "full", pad_to_size=True)
        if config.tile_size:
            height, width = image.shape[:2]
            for y in tile_origins(height, config.tile_size, config.tile_overlap):
                for x in tile_origins(width, config.tile_size, config.tile_overlap):
                    patch = image[y:y + config.tile_size, x:x + config.tile_size]
                    detections.extend(self._predict_patch(patch, x, y, config.tile_imgsz, f"tile:{x},{y}"))
        return classwise_nms(detections, config.nms_iou)


def _check_detections(detections: list[YoloDetection], names: dict[int, str]) -> None:
    for detection in detections:
        if detection.class_id not in names:
            raise ValueError(f"Unknown YOLO class ID: {detection.class_id}")
        if len(detection.bbox) != 4 or not all(math.isfinite(float(x)) for x in detection.bbox):
            raise ValueError("Invalid YOLO box coordinates")
        if not math.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
            raise ValueError("Invalid YOLO confidence")


def make_overlay(image: np.ndarray, hits: list[Any], detections: list[YoloDetection]) -> np.ndarray:
    canvas = image.copy()
    for i, hit in enumerate(hits, start=1):
        center = (int(round(hit.cx)), int(round(hit.cy)))
        radius = max(1, int(round(hit.radius)))
        cv2.ellipse(canvas, center, (radius, radius), 0, hit.start_angle_deg, hit.start_angle_deg + hit.arc_span_deg, (0, 220, 255), 2)
        cv2.drawMarker(canvas, center, (0, 220, 255), cv2.MARKER_CROSS, 12, 1)
        x1, y1, x2, y2 = map(lambda n: int(round(n)), hit.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 1)
        label = f"rule circle {i}: {hit.arc_span_deg:.1f} deg"
        cv2.putText(canvas, label, (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    for detection in detections:
        x1, y1, x2, y2 = map(lambda n: int(round(n)), detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 90, 255), 2)
        label = f"YOLO {detection.class_id}: {detection.class_name} {detection.confidence:.3f}"
        cv2.putText(canvas, label, (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 90, 255), 1, cv2.LINE_AA)
    return canvas


def _save_artifacts(config: PipelineConfig, relative: Path, bucket: str, path: Path,
                    image: np.ndarray | None, hits: list[Any], detections: list[YoloDetection],
                    record: dict[str, Any]) -> None:
    root = config.output / bucket
    original = root / "images" / relative
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, original)
    artifact_paths: dict[str, Any] = {"image": str(original.relative_to(config.output))}
    if image is not None:
        overlay = root / "overlays" / relative.parent / (relative.name + ".png")
        write_png(overlay, make_overlay(image, hits, detections))
        artifact_paths["overlay"] = str(overlay.relative_to(config.output))
        if hits:
            height, width = image.shape[:2]
            crop_paths = []
            for i, hit in enumerate(hits, start=1):
                x1, y1, x2, y2 = hit.bbox
                left, top = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
                right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
                if right > left and bottom > top:
                    crop = root / "crops" / relative.parent / (relative.name + f".circle-{i:03d}.png")
                    write_png(crop, image[top:bottom, left:right])
                    crop_paths.append(str(crop.relative_to(config.output)))
            artifact_paths["circle_crops"] = crop_paths
    record_path = root / "records" / relative.parent / (relative.name + ".json")
    artifact_paths["record"] = str(record_path.relative_to(config.output))
    record["artifacts"] = artifact_paths
    write_json(record_path, record)


def run_pipeline(config: PipelineConfig, *, rule_detector: Callable[..., Any] | None = None,
                 yolo_runner: Any = None) -> dict[str, Any]:
    """Process a dataset. Inject a rule detector/runner to test routing without a model.

    An injected runner exposes ``names`` and ``detect(image) -> list[YoloDetection]``.
    It does not demonstrate the performance of any trained YOLO model.
    """
    config.validate(check_weights=yolo_runner is None)
    detector = rule_detector or detect_circle_scratches
    images = discover_images(config.source, config.output)
    if not images:
        raise ValueError("No supported images found in source (the output tree is excluded)")
    config.output.mkdir(parents=True, exist_ok=True)
    startup_error: str | None = None
    names: dict[int, str] = {}
    critical_classes = config.critical_classes
    critical_mapping_source = "explicit" if critical_classes is not None else "model_not_loaded"
    if config.weights is not None or yolo_runner is not None:
        try:
            if yolo_runner is None:
                yolo_runner = UltralyticsRunner(config)
            names = _model_names(yolo_runner.names)
            _validate_class_mapping(names, config)
            critical_classes, critical_mapping_source = resolve_critical_classes(names, config.critical_classes)
            if config.class_ids is not None and critical_classes is not None:
                if not set(critical_classes).issubset(config.class_ids):
                    raise ValueError("class_ids must include the critical IDs resolved from model.names")
            print(f"YOLO model.names: {json.dumps(names, ensure_ascii=False)}", flush=True)
            if critical_classes is None:
                print("Critical IDs unresolved: specify --critical-classes using model.names. YOLO decisions will require review.", flush=True)
            else:
                print(f"Critical IDs ({critical_mapping_source}): {','.join(map(str, critical_classes))}", flush=True)
        except Exception as exc:
            startup_error = f"YOLO initialization failed: {type(exc).__name__}: {exc}"
            yolo_runner = None
    count: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for index, path in enumerate(images, start=1):
        relative = Path(path.name) if config.source.is_file() else path.relative_to(config.source)
        image: np.ndarray | None = None
        hits: list[Any] = []
        detections: list[YoloDetection] = []
        record: dict[str, Any] = {
            "schema_version": 1, "source": str(path), "relative_source": relative.as_posix(),
            "status": "error", "decision": "review_required", "critical_reject": None,
            "rule": None, "yolo": {"attempted": False, "detections": []},
            "critical_classes": critical_classes, "critical_mapping_source": critical_mapping_source,
            "errors": [], "warnings": [],
        }
        bucket = "errors"
        try:
            image = read_image(path)
            record["image_size"] = {"width": image.shape[1], "height": image.shape[0]}
            if image.shape[:2] != (1096, 1096):
                record["warnings"].append("Input differs from the expected 1096 x 1096 size; validate geometry settings")
            rule_result = detector(image, config.rule)
            hits = list(rule_result.hits)
            record["rule"] = rule_result.to_dict()
            rule_status = rule_result.diagnostics.get("status", "uncertain")
            record["rule_score_semantics"] = "Geometric ranking measure, not a probability"
            if hits:
                bucket = "rule-based"
                record.update(status="rule_defect_detected", decision="reject", critical_reject=True)
                count["rule_positive"] += 1
            else:
                count["rule_negative"] += 1
            if rule_status != "ok":
                record["warnings"].append(f"Rule detection status is {rule_status}; negative rule result requires review")
            run_yolo = not hits or config.yolo_on_rule_hit
            if not run_yolo:
                record["yolo"]["skip_reason"] = "rule_positive_default"
            if run_yolo:
                if startup_error:
                    raise RuntimeError(startup_error)
                if yolo_runner is None:
                    bucket = "yolo-pending"
                    record["status"] = "yolo_pending"
                    record["yolo"]["pending_reason"] = "weights_not_supplied"
                    record["warnings"].append("No YOLO weights supplied; second-stage detection is pending")
                    count["yolo_pending"] += 1
                else:
                    record["yolo"]["attempted"] = True
                    count["yolo_attempted"] += 1
                    detections = list(yolo_runner.detect(image))
                    _check_detections(detections, names)
                    record["yolo"]["detections"] = [d.to_dict() for d in detections]
                    if detections:
                        count["yolo_positive"] += 1
                    else:
                        count["yolo_negative"] += 1
                    if not hits:
                        bucket = "yolo-detected" if detections else "undetected"
                        record["status"] = "yolo_defect_detected" if detections else "no_defect_detected"
                        if critical_classes is None:
                            record["warnings"].append("Critical IDs unresolved; specify critical_classes to determine critical reject from YOLO")
                        elif any(d.class_id in critical_classes for d in detections):
                            record.update(decision="reject", critical_reject=True)
                        elif rule_status == "ok":
                            record.update(
                                decision="no_critical_defect_detected" if detections else "no_defect_detected",
                                critical_reject=False,
                            )
            if record["decision"] in {"no_defect_detected", "no_critical_defect_detected"}:
                record["warnings"].append("Detection result is not a guarantee of a good wafer")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            record["errors"].append(error)
            if not hits:
                bucket = "errors"
                record.update(status="error", decision="review_required", critical_reject=None)
        try:
            _save_artifacts(config, relative, bucket, path, image, hits, detections, record)
        except Exception as exc:
            record["errors"].append(f"Artifact write failed: {type(exc).__name__}: {exc}")
            if not hits:
                record.update(status="error", decision="review_required", critical_reject=None)
            # Persist the failure when possible; summary also records it.
            try:
                write_json(config.output / "errors" / "records" / relative.parent / (relative.name + ".json"), record)
            except Exception:
                pass
        count["processed"] += 1
        count[f"bucket:{bucket}"] += 1
        decisions[record["decision"]] += 1
        if record["errors"]:
            count["images_with_errors"] += 1
            errors.append({"source": str(path), "errors": record["errors"]})
        items.append({"source": relative.as_posix(), "bucket": bucket,
                      "status": record["status"], "decision": record["decision"],
                      "record": record.get("artifacts", {}).get("record")})
        print(f"[{index}/{len(images)}] {relative}: {record['status']} / {record['decision']}", flush=True)
    summary = {
        "schema_version": 1, "software_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": asdict(config), "model_names": names,
        "effective_critical_classes": critical_classes,
        "critical_mapping_source": critical_mapping_source,
        "critical_class_mapping_status": "invalid" if startup_error else "validated" if names and critical_classes is not None else "unresolved" if names else "model_not_loaded",
        "images_found": len(images), "counts": dict(count), "decisions": dict(decisions),
        "startup_error": startup_error, "errors": errors, "items": items,
        "limitation": "No defect detected is not a guarantee of a good wafer; thresholds require validation on real data.",
        "exit_code": 1 if errors or startup_error else 0,
    }
    write_json(config.output / "summary.json", summary)
    return summary


def load_rule_config(path: Path | None) -> RuleConfig:
    if path is None:
        return RuleConfig()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Rule config JSON must be an object of RuleConfig fields")
    allowed = {field.name for field in fields(RuleConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown rule config fields: {unknown}; allowed: {sorted(allowed)}")
    config = RuleConfig(**raw)
    config.validate()
    return config


def _parse_class_ids(raw: str) -> tuple[int, ...]:
    try:
        parts = raw.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise ValueError
        return tuple(dict.fromkeys(int(part.strip()) for part in parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative class IDs, e.g. 2,5") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based circle scratch detection, then optional YOLO on rule-negative images.",
        epilog="Install optional YOLO: pip install ultralytics. Example existing YOLO11 model: --weights best.pt. "
               "No defect detected does not guarantee a good wafer; rule scores are not probabilities. "
               "Use a dedicated image input directory and a new output directory per run. "
               "Output inside source is excluded from scanning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="Image file or recursive directory, including Unicode paths")
    parser.add_argument("--output", type=Path, required=True, help="New or empty artifact directory per run; cannot equal or contain source")
    parser.add_argument("--rule-config", type=Path, help="JSON object containing RuleConfig dataclass fields")
    parser.add_argument("--print-rule-config", action="store_true", help="Print validated effective rule config before processing")
    parser.add_argument("--weights", type=Path, help="Existing local Ultralytics YOLO detection weights; omit to queue rule-negative images")
    parser.add_argument("--critical-classes", type=_parse_class_ids, help="Actual circle/line class IDs, e.g. 2,5; overrides unique automatic model.names matching")
    parser.add_argument("--class-ids", type=_parse_class_ids, help="Optional YOLO class filter; omitted means all model classes")
    parser.add_argument("--imgsz", type=int, default=1120, help="Full image inference size, multiple of 32; 1096 inputs are padded to 1120")
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO confidence threshold, tune on validation data for recall")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Same-class NMS IoU threshold for full and tile predictions")
    parser.add_argument("--tile-size", type=int, default=0, help="Optional original-pixel tile crop size in addition to full image; 640 is a starting point")
    parser.add_argument("--tile-imgsz", type=int, default=1120, help="YOLO input size for each tile, independent of tile-size; multiple of 32, matching 1120 training by default")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="Tile overlap fraction in [0, 1)")
    parser.add_argument("--max-det", type=int, default=1000, help="YOLO maximum detections per input patch")
    parser.add_argument("--device", help="Ultralytics device selection, e.g. cpu or 0")
    parser.add_argument("--yolo-on-rule-hit", action="store_true", help="Also inspect rule-positive images for line and other defects; requires weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rule = load_rule_config(args.rule_config)
        config = PipelineConfig(
            source=args.source, output=args.output, rule=rule, weights=args.weights,
            critical_classes=args.critical_classes, class_ids=args.class_ids,
            imgsz=args.imgsz, confidence=args.confidence, nms_iou=args.nms_iou,
            tile_size=args.tile_size, tile_imgsz=args.tile_imgsz, tile_overlap=args.tile_overlap, max_det=args.max_det,
            device=args.device, yolo_on_rule_hit=args.yolo_on_rule_hit,
        )
        config.validate()
        if args.print_rule_config:
            print(json.dumps(asdict(rule), ensure_ascii=False, indent=2, default=_json_default))
        summary = run_pipeline(config)
        print(f"Summary: {config.output / 'summary.json'}")
        print(json.dumps({"counts": summary["counts"], "decisions": summary["decisions"]}, ensure_ascii=False))
        return summary["exit_code"]
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"Configuration/output error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
