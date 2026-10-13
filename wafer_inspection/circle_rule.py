"""OpenCV baseline for bright, approximately concentric wafer scratch arcs.

The angular span and the fraction of that span supported by image evidence are
different measurements.  A 35 degree arc never needs to cover most of a circle.
Angles follow image coordinates: zero points right and positive angles go down.
This is a configurable baseline; its defaults are not validated production limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class RuleConfig:
    min_arc_deg: float = 30.0
    min_radius_px: float = 12.0
    max_radius_px: float | None = None
    max_center_offset_px: float = 48.0
    edge_margin_px: float = 8.0
    angular_step_deg: float = 0.5
    radial_tolerance_px: float = 2.5
    max_radial_residual_px: float = 2.0
    min_line_residual_px: float = 0.75
    min_circle_line_gain: float = 0.25
    min_tangent_alignment: float = 0.8
    min_tangent_consistency: float = 0.8
    min_tensor_coherence: float = 0.3
    tensor_sigma_px: float = 2.0
    max_gap_deg: float = 2.0
    min_arc_coverage: float = 0.72
    min_support_pixels: float = 4.0
    min_contrast: float = 6.0
    threshold_mad_multiplier: float = 4.0
    brightness_threshold: float | None = None
    min_gray_difference: float = 25.0
    gray_background_offset_px: float = 6.0
    gray_background_width_px: float = 3.0
    merge_center_tolerance_px: float = 8.0
    merge_radius_tolerance_px: float = 5.0
    top_hat_sizes: tuple[int, ...] = (9, 17, 31)
    center_grid_step_px: float = 16.0
    max_components: int = 128
    max_grid_candidates: int = 256
    max_radius_candidates_per_center: int = 16
    max_hits: int = 64
    min_wafer_confidence: float = 0.6
    wafer_center: tuple[float, float] | None = None
    wafer_radius: float | None = None

    def validate(self) -> None:
        positive = ("min_radius_px", "angular_step_deg", "radial_tolerance_px",
                    "max_radial_residual_px", "min_support_pixels", "min_contrast",
                    "center_grid_step_px", "tensor_sigma_px",
                    "gray_background_offset_px", "gray_background_width_px")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not 0 < self.min_arc_deg <= 360:
            raise ValueError("min_arc_deg must be in (0, 360]")
        if not 0 < self.min_arc_coverage <= 1:
            raise ValueError("min_arc_coverage must be in (0, 1]")
        if not 0 <= self.min_wafer_confidence <= 1:
            raise ValueError("min_wafer_confidence must be in [0, 1]")
        if not 0 <= self.min_circle_line_gain < 1:
            raise ValueError("min_circle_line_gain must be in [0, 1)")
        for name in ("min_tangent_alignment", "min_tangent_consistency", "min_tensor_coherence"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("max_center_offset_px", "edge_margin_px", "max_gap_deg",
                     "threshold_mad_multiplier", "min_line_residual_px",
                     "min_gray_difference", "merge_center_tolerance_px",
                     "merge_radius_tolerance_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_gray_difference > 255:
            raise ValueError("min_gray_difference must be in [0, 255]")
        if self.angular_step_deg > 5 or self.max_gap_deg >= self.min_arc_deg:
            raise ValueError("angular_step_deg must be <= 5 and max_gap_deg < min_arc_deg")
        for name in ("max_components", "max_grid_candidates",
                     "max_radius_candidates_per_center", "max_hits"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.top_hat_sizes or any(not isinstance(k, int) or k < 3 or k % 2 == 0
                                         for k in self.top_hat_sizes):
            raise ValueError("top_hat_sizes must contain odd integers >= 3")
        if self.max_radius_px is not None and (not math.isfinite(self.max_radius_px)
                                              or self.max_radius_px < self.min_radius_px):
            raise ValueError("max_radius_px must be >= min_radius_px")
        if self.brightness_threshold is not None and (not math.isfinite(self.brightness_threshold)
                                                      or self.brightness_threshold <= 0):
            raise ValueError("brightness_threshold must be positive when specified")
        if (self.wafer_center is None) != (self.wafer_radius is None):
            raise ValueError("wafer_center and wafer_radius must be specified together")
        if self.wafer_center is not None:
            if len(self.wafer_center) != 2 or not all(math.isfinite(v) for v in self.wafer_center):
                raise ValueError("wafer_center must contain two finite coordinates")
            if not math.isfinite(self.wafer_radius) or self.wafer_radius <= 0:
                raise ValueError("wafer_radius must be positive")


@dataclass
class CircleHit:
    cx: float
    cy: float
    radius: float
    start_angle_deg: float
    arc_span_deg: float
    coverage: float
    support_pixels: float
    radial_residual_px: float
    bbox: tuple[int, int, int, int]
    score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _evidence_points: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("_evidence_points", None)
        return result


@dataclass
class RuleResult:
    wafer: dict[str, Any]
    hits: list[CircleHit]
    diagnostics: dict[str, Any]
    enhanced: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"wafer": self.wafer, "hits": [hit.to_dict() for hit in self.hits],
                "diagnostics": self.diagnostics}


def _gray_u8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8; explicitly scale higher-bit-depth input before detection")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if min(gray.shape) < 32:
        raise ValueError("image is too small for wafer geometry estimation")
    return np.ascontiguousarray(gray)


def _fit_circle(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Centered algebraic initialization, followed by robust geometric fitting."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 6:
        return None
    origin = points.mean(axis=0)
    q = points - origin
    a = np.column_stack((2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))))
    b = (q * q).sum(axis=1)
    try:
        solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or singular[-1] < singular[0] * 1e-8:
        return None
    center = solution[:2] + origin
    radius_squared = float(solution[2] + (solution[:2] ** 2).sum())
    if radius_squared <= 0:
        return None
    radius = math.sqrt(radius_squared)
    for _ in range(6):
        delta = center - points
        distance = np.linalg.norm(delta, axis=1)
        residual = distance - radius
        scale = max(0.5, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
        jacobian = np.column_stack((delta / np.maximum(distance[:, None], 1e-6),
                                    -np.ones(len(points))))
        try:
            update = np.linalg.lstsq(jacobian * np.sqrt(weights[:, None]),
                                     -residual * np.sqrt(weights), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(update)):
            return None
        center += update[:2]
        radius += update[2]
        if np.linalg.norm(update) < 1e-3:
            break
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    if radius <= 0 or not np.all(np.isfinite([*center, radius])):
        return None
    return float(center[0]), float(center[1]), float(radius), float(np.median(residual))


def _line_residual(points: np.ndarray) -> float:
    """Median orthogonal residual of the best TLS line through an arc's evidence."""
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    try:
        _, _, directions = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.median(np.abs(centered @ directions[-1])))


def _locate_wafer(gray: np.ndarray, config: RuleConfig) -> dict[str, Any]:
    h, w = gray.shape
    if config.wafer_center is not None:
        cx, cy = config.wafer_center
        return {"cx": float(cx), "cy": float(cy), "radius": float(config.wafer_radius),
                "confidence": 1.0, "source": "explicit_geometry"}
    # Blurring suppresses sparse bright scratches before separating wafer/background.
    smooth = cv2.GaussianBlur(gray, (0, 0), 7.0)
    threshold, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alternatives = [otsu, cv2.bitwise_not(otsu)]
    border = np.concatenate((smooth[0], smooth[-1], smooth[:, 0], smooth[:, -1]))
    border_level = float(np.median(border))
    # A low-contrast wafer can occupy enough area that Otsu cuts inside its texture.
    for sign in (1, -1):
        alternatives.append(((sign * (smooth.astype(np.float32) - border_level)) >
                              max(3.0, float(np.std(border)) * 2)).astype(np.uint8) * 255)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for foreground in alternatives:
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = float(cv2.contourArea(contour))
            if area < 0.2 * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if x == 0 and y == 0 and bw == w and bh == h:
                continue  # Outer image rectangle is not evidence of a wafer circle.
            points = contour.reshape(-1, 2)[::max(1, len(contour) // 1000)]
            fitted = _fit_circle(points)
            if fitted is None:
                continue
            cx, cy, radius, residual = fitted
            if not (0.3 * min(h, w) <= radius <= 0.8 * max(h, w)):
                continue
            if math.hypot(cx - w / 2, cy - h / 2) > 0.22 * min(h, w):
                continue
            fill = area / (math.pi * radius * radius)
            if not 0.65 < fill < 1.15 or residual > 0.035 * radius:
                continue
            confidence = min(0.99, max(0.1, fill) * math.exp(-residual / max(2.0, radius * 0.01)))
            record = {"cx": cx, "cy": cy, "radius": radius, "confidence": confidence,
                      "source": "foreground_circle_fit"}
            candidates.append((confidence, record))
    if candidates:
        result = max(candidates, key=lambda item: item[0])[1]
        # Thresholding the strongly blurred wafer can move its contour outward.
        # Refine at the original image's boundary gradient, requiring the outer
        # side to match background so a nearby bright scratch is not selected.
        search = max(12, min(28, int(result["radius"] * 0.04)))
        radii = np.arange(result["radius"] - search, result["radius"] + search + 1,
                          dtype=np.float32)
        angles = np.arange(360, dtype=np.float32) * (math.pi / 180)
        cs, sn = np.cos(angles), np.sin(angles)
        boundary_gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
        profiles, valid = _remap(boundary_gray, np.ones_like(gray, dtype=np.uint8) * 255,
                                (result["cx"], result["cy"]), radii, cs, sn)
        profiles = profiles.astype(np.float32)
        outside = np.median(profiles[-4:], axis=0)
        inside = np.median(profiles[:4], axis=0)
        contrast = inside - outside
        sign = np.sign(contrast)
        derivative = -np.diff(profiles, axis=0) * sign[None, :]
        after_indices = np.minimum(np.arange(len(radii) - 1) + 7, len(radii) - 1)
        matches_background = (np.abs(profiles[after_indices] - outside[None, :]) <=
                              np.maximum(5.0, np.abs(contrast)[None, :] * 0.25))
        usable = (valid[:-1] & valid[1:] & valid[after_indices] & matches_background &
                  (np.abs(contrast)[None, :] > 5))
        derivative[~usable] = -1
        best_indices = derivative.argmax(axis=0)
        best_values = derivative[best_indices, np.arange(360)]
        supported = best_values > 1.5
        if supported.sum() >= 180:
            boundary_radii = radii[best_indices[supported]] + 0.5
            points = np.column_stack((result["cx"] + boundary_radii * cs[supported],
                                      result["cy"] + boundary_radii * sn[supported]))
            refined = _fit_circle(points)
            if refined is not None:
                cx, cy, radius, residual = refined
                if (abs(radius - result["radius"]) <= search and
                        math.hypot(cx - result["cx"], cy - result["cy"]) <= search and
                        residual < max(2.0, radius * 0.01)):
                    result.update(cx=cx, cy=cy, radius=radius,
                                  boundary_refinement="radial_gradient_with_background_check")
        return result
    return {"cx": (w - 1) / 2, "cy": (h - 1) / 2, "radius": min(h, w) / 2,
            "confidence": 0.25, "source": "image_center_fallback"}


def _circular_runs(support: np.ndarray, max_gap_bins: int) -> list[tuple[int, int, float]]:
    """Fill only short internal gaps and return wrap-safe (start, bin_count, coverage)."""
    original = support.astype(bool)
    n = len(original)
    if not np.any(original):
        return []
    closed = original.copy()
    true_indices = np.flatnonzero(original)
    for start, end in zip(true_indices, np.roll(true_indices, -1)):
        gap = int((end - start - 1) % n)
        if 0 < gap <= max_gap_bins:
            closed[(start + np.arange(1, gap + 1)) % n] = True
    if np.all(closed):
        return [(0, n, float(original.mean()))]
    starts = np.flatnonzero(closed & ~np.roll(closed, 1))
    runs = []
    for start in starts:
        length = 0
        while length < n and closed[(start + length) % n]:
            length += 1
        indices = (start + np.arange(length)) % n
        runs.append((int(start), length, float(original[indices].mean())))
    return runs


def _component_candidates(binary: np.ndarray, wafer: dict[str, Any], config: RuleConfig,
                          maximum_radius: float) -> list[tuple[float, float, float]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    indices = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 12]
    indices.sort(key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    candidates = []
    for index in indices[:config.max_components]:
        x, y, width, height, area = stats[index]
        yy, xx = np.nonzero(labels[y:y + height, x:x + width] == index)
        stride = max(1, len(xx) // 1200)
        fitted = _fit_circle(np.column_stack((xx[::stride] + x, yy[::stride] + y)))
        if fitted is None:
            continue
        cx, cy, radius, residual = fitted
        if (config.min_radius_px <= radius <= maximum_radius and
                math.hypot(cx - wafer["cx"], cy - wafer["cy"]) <= config.max_center_offset_px and
                residual <= config.radial_tolerance_px * 2):
            candidates.append((cx, cy, radius))
    return candidates


def _remap(response: np.ndarray, mask: np.ndarray, center: tuple[float, float],
           radii: np.ndarray, cosines: np.ndarray, sines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xmap = (center[0] + radii[:, None] * cosines[None, :]).astype(np.float32)
    ymap = (center[1] + radii[:, None] * sines[None, :]).astype(np.float32)
    samples = cv2.remap(response, xmap, ymap, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(mask, xmap, ymap, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return samples, valid


def _radial_peaks(counts: np.ndarray, minimum: float, limit: int, separation: int) -> list[int]:
    # Greedy radial NMS works for broad peaks and does not collapse separate rings.
    remaining = counts.copy().astype(float)
    peaks = []
    for _ in range(limit):
        index = int(np.argmax(remaining))
        if remaining[index] < minimum:
            break
        peaks.append(index)
        remaining[max(0, index - separation):index + separation + 1] = -1
    return peaks


def _ridge_radii(samples: np.ndarray, radii: np.ndarray) -> np.ndarray:
    # A white ridge often has a flat maximum several pixels wide. Taking its first
    # maximum biases all fitted radii toward the inner stroke edge. Use the radial
    # centroid of the upper half of the response instead.
    weights = np.maximum(samples - samples.max(axis=0, keepdims=True) * 0.5, 0)
    weight_sum = weights.sum(axis=0)
    return ((weights * radii[:, None]).sum(axis=0) /
            np.maximum(weight_sum, 1e-6))


def _gray_contrast(gray: np.ndarray, mask: np.ndarray, center: tuple[float, float],
                   ridge_radii: np.ndarray, cosines: np.ndarray, sines: np.ndarray,
                   config: RuleConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original ridge gray minus the local wafer median on both radial sides.

    Background bands stay outside the radial support strip. Samples outside the
    usable wafer never become background; at least one full band's worth of
    valid samples is required, including when the other side approaches an edge.
    """
    offset = max(config.gray_background_offset_px, config.radial_tolerance_px + 1.0)
    side_offsets = offset + np.arange(math.ceil(config.gray_background_width_px), dtype=np.float32)
    offsets = np.concatenate((-side_offsets, side_offsets))
    sample_radii = np.vstack((ridge_radii, ridge_radii[None, :] + offsets[:, None]))
    xmap = (center[0] + sample_radii * cosines[None, :]).astype(np.float32)
    ymap = (center[1] + sample_radii * sines[None, :]).astype(np.float32)
    samples = cv2.remap(gray, xmap, ymap, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    valid = (cv2.remap(mask, xmap, ymap, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT) > 0) & (sample_radii > 0)
    count = valid[1:].sum(axis=0)
    ordered = np.sort(np.where(valid[1:], samples[1:], np.inf), axis=0)
    columns = np.arange(len(ridge_radii))
    lower = np.maximum(0, (count - 1) // 2)
    upper = count // 2
    wafer_gray = (ordered[lower, columns] + ordered[upper, columns]) * 0.5
    usable = valid[0] & (count >= len(side_offsets))
    difference = np.where(usable, samples[0] - wafer_gray, -np.inf)
    return samples[0], wafer_gray, difference


def _verify_candidate(response: np.ndarray, mask: np.ndarray, wafer: dict[str, Any],
                      candidate: tuple[float, float, float], config: RuleConfig,
                      threshold: float, maximum_radius: float, cosines: np.ndarray,
                      sines: np.ndarray, angular_step: float, source: str,
                      tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
                      gray: np.ndarray) -> list[CircleHit]:
    cx, cy, radius = candidate
    tolerance = config.radial_tolerance_px
    # Subpixel radial sampling finds the ridge across the estimated circle, then
    # a geometric fit refines the circle center independently of the wafer center.
    radii = radius + np.arange(-tolerance * 2, tolerance * 2 + 0.25, 0.5, dtype=np.float32)
    radii = radii[radii > 0]
    samples, valid = _remap(response, mask, (cx, cy), radii, cosines, sines)
    samples = np.where(valid, samples, 0)
    best_index = samples.argmax(axis=0)
    best_values = samples[best_index, np.arange(samples.shape[1])]
    ridge_radii = _ridge_radii(samples, radii)
    _, _, gray_difference = _gray_contrast(gray, mask, (cx, cy), ridge_radii,
                                           cosines, sines, config)
    support = best_values >= threshold
    if config.min_gray_difference > 0:
        support &= gray_difference >= config.min_gray_difference
    gap_bins = int(math.floor(config.max_gap_deg / angular_step + 1e-6))
    initial_runs = _circular_runs(support, gap_bins)
    hits: list[CircleHit] = []
    for start, length, initial_coverage in initial_runs:
        if length * angular_step + 1e-6 < config.min_arc_deg or initial_coverage < config.min_arc_coverage:
            continue
        indices = (start + np.arange(length)) % len(cosines)
        supported = indices[support[indices]]
        chosen_radii = ridge_radii[supported]
        points = np.column_stack((cx + chosen_radii * cosines[supported],
                                  cy + chosen_radii * sines[supported]))
        fitted = _fit_circle(points)
        if fitted is None:
            continue
        fx, fy, fr, residual = fitted
        if config.max_center_offset_px == 0:
            # Exact fixed-center operation is useful with calibrated wafer geometry;
            # unconstrained fitting would otherwise reject harmless subpixel drift.
            fx, fy = wafer["cx"], wafer["cy"]
            point_radii = np.linalg.norm(points - np.array([fx, fy]), axis=1)
            fr = float(np.median(point_radii))
            residual = float(np.median(np.abs(point_radii - fr)))
        if (not config.min_radius_px <= fr <= maximum_radius or
                math.hypot(fx - wafer["cx"], fy - wafer["cy"]) > config.max_center_offset_px or
                residual > config.max_radial_residual_px):
            continue
        # Revalidate the refined circle with a narrower strip so a line, crossing
        # spot cluster, or poorly fitted curved component cannot pass by fitting alone.
        check_radii = fr + np.arange(-tolerance, tolerance + 0.25, 0.5, dtype=np.float32)
        final_samples, final_valid = _remap(response, mask, (fx, fy), check_radii, cosines, sines)
        final_samples = np.where(final_valid, final_samples, 0)
        final_index = final_samples.argmax(axis=0)
        final_values = final_samples[final_index, np.arange(final_samples.shape[1])]
        final_ridge_radii = _ridge_radii(final_samples, check_radii)
        scratch_gray, wafer_gray, final_gray_difference = _gray_contrast(
            gray, mask, (fx, fy), final_ridge_radii, cosines, sines, config)
        final_support = final_values >= threshold
        if config.min_gray_difference > 0:
            final_support &= final_gray_difference >= config.min_gray_difference
        for final_start, final_length, coverage in _circular_runs(final_support, gap_bins):
            span = min(360.0, final_length * angular_step)
            if span + 1e-6 < config.min_arc_deg or coverage < config.min_arc_coverage:
                continue
            arc_indices = (final_start + np.arange(final_length)) % len(cosines)
            evidence_indices = arc_indices[final_support[arc_indices]]
            support_pixels = len(evidence_indices) * math.radians(angular_step) * fr
            if support_pixels < config.min_support_pixels:
                continue
            # This residual checks the local ridge positions on this particular arc.
            distances = final_ridge_radii[evidence_indices] - fr
            arc_residual = float(np.median(np.abs(distances)))
            if arc_residual > config.max_radial_residual_px:
                continue
            evidence_radii = final_ridge_radii[evidence_indices]
            evidence_points = np.column_stack((fx + evidence_radii * cosines[evidence_indices],
                                               fy + evidence_radii * sines[evidence_indices]))
            line_residual = _line_residual(evidence_points)
            circle_residual = float(np.median(np.abs(evidence_radii - fr)))
            line_gain = 1.0 - circle_residual / max(line_residual, 1e-6)
            # A small tangent segment can fit the polar tolerance strip. Require
            # observable curvature and a material improvement over a straight line.
            if line_residual < config.min_line_residual_px or line_gain < config.min_circle_line_gain:
                continue
            # Neighborhood gradient energy remains meaningful at a white ridge's
            # flat center. Its dominant gradient must point approximately radially,
            # so the local ridge direction agrees with the fitted circle tangent.
            xmap = evidence_points[:, 0].astype(np.float32)[None, :]
            ymap = evidence_points[:, 1].astype(np.float32)[None, :]
            txx, txy, tyy = [cv2.remap(t, xmap, ymap, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT).ravel() for t in tensors]
            trace = txx + tyy
            cc, ss = cosines[evidence_indices], sines[evidence_indices]
            radial_energy = cc * cc * txx + 2 * cc * ss * txy + ss * ss * tyy
            alignment = radial_energy / np.maximum(trace, 1e-6)
            coherence = np.sqrt((txx - tyy) ** 2 + 4 * txy * txy) / np.maximum(trace, 1e-6)
            tangent_ok = ((alignment >= config.min_tangent_alignment) &
                          (coherence >= config.min_tensor_coherence))
            tangent_consistency = float(tangent_ok.mean())
            if tangent_consistency < config.min_tangent_consistency:
                continue
            xs = fx + fr * cosines[arc_indices]
            ys = fy + fr * sines[arc_indices]
            padding = math.ceil(tolerance + 2)
            h, w = response.shape
            bbox = (max(0, int(math.floor(xs.min())) - padding),
                    max(0, int(math.floor(ys.min())) - padding),
                    min(w, int(math.ceil(xs.max())) + padding + 1),
                    min(h, int(math.ceil(ys.max())) + padding + 1))
            strength = float(np.median(final_values[evidence_indices]))
            gray_indices = evidence_indices[np.isfinite(final_gray_difference[evidence_indices])]
            gray_diagnostics = {name: float(np.median(values[gray_indices])) if len(gray_indices) else None
                                for name, values in (("median_scratch_gray", scratch_gray),
                                                     ("median_wafer_gray", wafer_gray),
                                                     ("median_gray_difference", final_gray_difference))}
            score = coverage * math.exp(-arc_residual / max(tolerance, 1)) * min(1.0, strength / (threshold * 2))
            hits.append(CircleHit(fx, fy, fr, final_start * angular_step, span, coverage,
                                  support_pixels, arc_residual, bbox, float(score),
                                  {"candidate_source": source, "median_response": strength,
                                   "fit_residual_px": residual, "angle_step_deg": angular_step,
                                   "center_offset_px": math.hypot(fx - wafer["cx"], fy - wafer["cy"]),
                                   "line_residual_px": line_residual, "circle_line_gain": line_gain,
                                   "tangent_consistency": tangent_consistency,
                                   "median_tangent_alignment": float(np.median(alignment)),
                                   **gray_diagnostics,
                                   "min_gray_difference": config.min_gray_difference,
                                   "max_gap_deg": config.max_gap_deg},
                                  _evidence_points=evidence_points))
    return hits


def _spatial_arc_overlap(hit: CircleHit, old: CircleHit, tolerance: float) -> float:
    """How much of a proposed arc is already explained by an accepted arc?

    Short arcs have poorly conditioned centers. Comparing centers alone leaves
    many differently fitted circles describing the very same scratch pixels.
    """
    if hit._evidence_points is not None:
        x = hit._evidence_points[:, 0] - old.cx
        y = hit._evidence_points[:, 1] - old.cy
    else:
        count = max(12, int(math.ceil(hit.arc_span_deg / 2)))
        angles = np.deg2rad(hit.start_angle_deg + np.linspace(0, hit.arc_span_deg, count, endpoint=False))
        x = hit.cx + hit.radius * np.cos(angles) - old.cx
        y = hit.cy + hit.radius * np.sin(angles) - old.cy
    near_radius = np.abs(np.hypot(x, y) - old.radius) <= tolerance
    relative_angles = (np.rad2deg(np.arctan2(y, x)) - old.start_angle_deg) % 360
    on_old_arc = relative_angles < old.arc_span_deg + 1e-6
    return float(np.mean(near_radius & on_old_arc))


def _arc_interval(hit: CircleHit, center: tuple[float, float]) -> tuple[float, float]:
    """Project the whole arc interval into a common center's angular frame."""
    if hit.arc_span_deg >= 360 - 1e-6:
        return 0.0, 360.0
    angles = np.deg2rad(hit.start_angle_deg + np.linspace(
        0, hit.arc_span_deg, max(2, int(math.ceil(hit.arc_span_deg / 5)) + 1)))
    x = hit.cx + hit.radius * np.cos(angles) - center[0]
    y = hit.cy + hit.radius * np.sin(angles) - center[1]
    projected = np.rad2deg(np.unwrap(np.arctan2(y, x)))
    start = float(projected[0] % 360)
    return start, start + float(projected[-1] - projected[0])


def _arc_gap(a: CircleHit, b: CircleHit) -> float:
    a_start, a_end = _arc_interval(a, (a.cx, a.cy))
    b_start, b_end = _arc_interval(b, (a.cx, a.cy))
    return min(max(0.0, max(a_start, b_start + shift) - min(a_end, b_end + shift))
               for shift in (-360, 0, 360))


def _arc_is_contained(hit: CircleHit, old: CircleHit, angular_tolerance: float) -> bool:
    """Duplicate removal may discard a contained fit, but never a long new tail."""
    if old.arc_span_deg >= 360 - 1e-6:
        return True
    if hit._evidence_points is not None and len(hit._evidence_points) > 1:
        delta = hit._evidence_points - np.array([old.cx, old.cy])
        angles = np.sort(np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) % 360)
        gaps = np.diff(np.concatenate((angles, angles[:1] + 360)))
        gap_index = int(np.argmax(gaps))
        start = float(angles[(gap_index + 1) % len(angles)])
        end = start + 360 - float(gaps[gap_index])
    else:
        start, end = _arc_interval(hit, (old.cx, old.cy))
    return any(start + shift >= old.start_angle_deg - angular_tolerance and
               end + shift <= old.start_angle_deg + old.arc_span_deg + angular_tolerance
               for shift in (-360, 0, 360))


def _evidence_arc_overlap(hit: CircleHit, old: CircleHit, tolerance: float) -> float:
    """Compare actual ridge points when a common circle approximates two arcs."""
    if (hit._evidence_points is None or old._evidence_points is None or
            not len(hit._evidence_points) or not len(old._evidence_points)):
        return 0.0
    near_count = 0
    old_points = old._evidence_points
    for start in range(0, len(hit._evidence_points), 128):
        points = hit._evidence_points[start:start + 128]
        squared_distance = ((points[:, None, 0] - old_points[None, :, 0]) ** 2 +
                            (points[:, None, 1] - old_points[None, :, 1]) ** 2)
        near_count += int(np.count_nonzero(squared_distance.min(axis=1) <= tolerance ** 2))
    return near_count / len(hit._evidence_points)


def _merge_overlapping_hits(hits: list[CircleHit], config: RuleConfig,
                            reverify: Callable[[float, float, float], list[CircleHit]]) -> tuple[list[CircleHit], int]:
    """Fit and reverify unions before duplicate suppression can discard tails.

    Each pair of original members must remain within the configured geometry
    tolerances, preventing a chain of nearby centers from drifting arbitrarily.
    Failed unions keep the independently verified arcs.
    """
    groups: list[tuple[CircleHit, list[CircleHit]]] = []
    merge_count = 0
    spatial_tolerance = max(config.radial_tolerance_px, config.max_radial_residual_px)

    def compatible(a: CircleHit, b: CircleHit) -> bool:
        return (math.hypot(a.cx - b.cx, a.cy - b.cy) <= config.merge_center_tolerance_px and
                abs(a.radius - b.radius) <= config.merge_radius_tolerance_px)

    for hit in hits:
        current, members = hit, [hit]
        index = 0
        while index < len(groups):
            old, old_members = groups[index]
            if (not all(compatible(a, b) for a in members for b in old_members) or
                    _arc_gap(current, old) > config.max_gap_deg + 1e-6):
                index += 1
                continue
            # Fully contained duplicate evidence cannot extend this arc. Leave
            # it for the inexpensive duplicate suppression below.
            if _spatial_arc_overlap(current, old, spatial_tolerance) >= 0.98:
                index += 1
                continue
            joint_members = old_members + members
            evidence = [member._evidence_points for member in joint_members]
            if any(points is None for points in evidence):
                index += 1
                continue
            points = np.unique(np.round(np.concatenate(evidence), 3), axis=0)
            fitted = _fit_circle(points)
            if fitted is None:
                index += 1
                continue
            cx, cy, radius, residual = fitted
            if config.max_center_offset_px == 0 and config.wafer_center is not None:
                cx, cy = config.wafer_center
                point_radii = np.linalg.norm(points - np.array([cx, cy]), axis=1)
                radius = float(np.median(point_radii))
                residual = float(np.median(np.abs(point_radii - radius)))
            if (residual > config.max_radial_residual_px or
                    any(math.hypot(cx - member.cx, cy - member.cy) > config.merge_center_tolerance_px or
                        abs(radius - member.radius) > config.merge_radius_tolerance_px
                        for member in joint_members)):
                index += 1
                continue
            proposals = [proposal for proposal in reverify(cx, cy, radius)
                         if all(compatible(proposal, member) and
                                _spatial_arc_overlap(member, proposal, spatial_tolerance) >= 0.9
                                for member in joint_members)]
            if not proposals:
                index += 1
                continue
            current = max(proposals, key=lambda value: (value.arc_span_deg * value.coverage, value.score))
            members = joint_members
            current.diagnostics.update(merged_arc_count=len(members),
                                       merge_fit_residual_px=residual,
                                       merge_center_tolerance_px=config.merge_center_tolerance_px,
                                       merge_radius_tolerance_px=config.merge_radius_tolerance_px)
            groups.pop(index)
            merge_count += 1
            index = 0
        groups.append((current, members))
    return [hit for hit, _ in groups], merge_count


def detect_circle_scratches(image: np.ndarray, config: RuleConfig | None = None) -> RuleResult:
    """Detect bright scratch circles/arcs, returning geometry and independent arc evidence.

    A miss means only "the rule did not find a qualifying arc" and is suitable for
    secondary model inference. It must not be interpreted as an OK wafer verdict.
    """
    started = time.perf_counter()
    config = config or RuleConfig()
    config.validate()
    gray = _gray_u8(image)
    wafer = _locate_wafer(gray, config)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    usable_radius = max(0.0, wafer["radius"] - config.edge_margin_px)
    mask = (((xx - wafer["cx"]) ** 2 + (yy - wafer["cy"]) ** 2) <= usable_radius ** 2).astype(np.uint8) * 255
    maximum_radius = config.max_radius_px if config.max_radius_px is not None else usable_radius
    diagnostics: dict[str, Any] = {"status": "uncertain" if (wafer["source"] == "image_center_fallback" or
                                                           wafer["confidence"] < config.min_wafer_confidence) else "ok",
                                  "wafer_source": wafer["source"], "wafer_confidence": wafer["confidence"],
                                  "angle_convention": "0 degrees right; positive clockwise in image coordinates",
                                  "min_arc_deg": config.min_arc_deg, "edge_margin_px": config.edge_margin_px,
                                  "max_center_offset_px": config.max_center_offset_px,
                                  "min_radius_px": config.min_radius_px, "max_radius_px": maximum_radius,
                                  "min_gray_difference": config.min_gray_difference,
                                  "gray_background_offset_px": max(config.gray_background_offset_px,
                                                                    config.radial_tolerance_px + 1.0),
                                  "gray_background_width_px": config.gray_background_width_px,
                                  "merge_center_tolerance_px": config.merge_center_tolerance_px,
                                  "merge_radius_tolerance_px": config.merge_radius_tolerance_px}
    response = np.zeros_like(gray)
    for size in config.top_hat_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
    response[mask == 0] = 0
    values = response[mask > 0].astype(np.float32)
    if not len(values) or maximum_radius < config.min_radius_px:
        diagnostics.update(status="uncertain", reason="empty_usable_wafer_or_radius_range",
                           elapsed_seconds=time.perf_counter() - started)
        return RuleResult(wafer, [], diagnostics, response)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = (config.brightness_threshold if config.brightness_threshold is not None else
                 max(config.min_contrast, median + config.threshold_mad_multiplier * 1.4826 * mad))
    binary = ((response >= threshold) & (mask > 0)).astype(np.uint8)
    diagnostics.update(threshold=float(threshold), response_median=median, response_mad=mad,
                       bright_pixel_count=int(binary.sum()))
    angle_count = int(math.ceil(360.0 / config.angular_step_deg))
    angular_step = 360.0 / angle_count
    angles = np.arange(angle_count, dtype=np.float32) * math.radians(angular_step)
    cosines, sines = np.cos(angles), np.sin(angles)
    direct = _component_candidates(binary, wafer, config, maximum_radius)
    diagnostics["component_candidate_count"] = len(direct)
    candidates: list[tuple[float, float, float, str]] = [(cx, cy, radius, "component_circle_fit")
                                                     for cx, cy, radius in direct]
    # The polar search also catches fragmented rings that never form a component.
    # It searches a disk of centers rather than assuming one common wafer center.
    offset = config.max_center_offset_px
    if offset == 0:
        offsets = np.array([0.0])
    else:
        steps = int(math.ceil(offset / config.center_grid_step_px))
        offsets = np.linspace(-offset, offset, 2 * steps + 1)
    centers = [(wafer["cx"] + dx, wafer["cy"] + dy)
               for dy in offsets for dx in offsets if math.hypot(dx, dy) <= offset + 1e-6]
    # Candidate component centers get full polar searches too: one connected ring
    # may reveal the shared center of several disconnected arcs at other radii.
    for cx, cy, radius in direct:
        if all(math.hypot(cx - x, cy - y) > 3 for x, y in centers):
            centers.append((cx, cy))
    radius_values = np.arange(math.ceil(config.min_radius_px), math.floor(maximum_radius) + 1,
                              dtype=np.float32)
    radial_band = 2 * int(math.ceil(config.radial_tolerance_px)) + 1
    coarse: list[tuple[float, float, float, float]] = []
    minimum_bins = config.min_arc_deg / angular_step * config.min_arc_coverage
    for cx, cy in centers:
        samples, valid = _remap(response, mask, (cx, cy), radius_values, cosines, sines)
        evidence = ((samples >= threshold) & valid).astype(np.uint8)
        band_support = cv2.dilate(evidence, np.ones((radial_band, 1), dtype=np.uint8)) > 0
        counts = band_support.sum(axis=1)
        window_bins = max(1, int(math.ceil(config.min_arc_deg / angular_step)))
        wrapped = np.concatenate((band_support, band_support[:, :window_bins]), axis=1).astype(np.float32)
        local_counts = cv2.boxFilter(wrapped, cv2.CV_32F, (window_bins, 1),
                                    anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
        local_max = local_counts[:, :angle_count].max(axis=1)
        # Local support is primary; total support is only a small tie breaker.
        ranking = local_max + 0.05 * counts / angle_count * window_bins
        ranking[local_max < minimum_bins] = -1
        peaks = _radial_peaks(ranking, minimum_bins, config.max_radius_candidates_per_center,
                              int(math.ceil(config.radial_tolerance_px * 2)))
        for index in peaks:
            qualifying = [(start, length, coverage) for start, length, coverage in
                          _circular_runs(band_support[index], int(config.max_gap_deg / angular_step))
                          if length * angular_step + 1e-6 >= config.min_arc_deg and
                          coverage >= config.min_arc_coverage]
            if not qualifying:
                continue
            priority = max(length * coverage for _, length, coverage in qualifying)
            coarse.append((priority, cx, cy, float(radius_values[index])))
    # Suppress neighboring coarse seeds to spend refinement on distinct geometry.
    selected: list[tuple[float, float, float]] = []
    for priority, cx, cy, radius in sorted(coarse, reverse=True):
        if any(math.hypot(cx - x, cy - y) <= config.center_grid_step_px * 0.8 and
               abs(radius - r) <= config.radial_tolerance_px * 2 for x, y, r in selected):
            continue
        selected.append((cx, cy, radius))
        if len(selected) >= config.max_grid_candidates:
            break
    candidates.extend((cx, cy, radius, "polar_center_search") for cx, cy, radius in selected)
    diagnostics.update(center_search_count=len(centers), grid_candidate_count=len(coarse),
                       refined_grid_candidate_count=len(selected))
    all_hits: list[CircleHit] = []
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tensors = tuple(cv2.GaussianBlur(term, (0, 0), config.tensor_sigma_px)
                    for term in (gx * gx, gx * gy, gy * gy))
    for cx, cy, radius, source in candidates:
        all_hits.extend(_verify_candidate(response, mask, wafer, (cx, cy, radius), config,
                                          float(threshold), maximum_radius, cosines, sines,
                                          angular_step, source, tensors, gray))
    def reverify(cx: float, cy: float, radius: float) -> list[CircleHit]:
        return _verify_candidate(response, mask, wafer, (cx, cy, radius), config,
                                 float(threshold), maximum_radius, cosines, sines,
                                 angular_step, "merged_arc_fit", tensors, gray)

    ordered_hits = sorted(all_hits, key=lambda value: (value.arc_span_deg >= 350, value.score,
                                                       value.arc_span_deg * value.coverage), reverse=True)
    merged_hits, merge_count = _merge_overlapping_hits(ordered_hits, config, reverify)
    # Keep separated arcs and radii, remove only overlapping geometric duplicates
    # after unions have been independently checked against the image.
    kept: list[CircleHit] = []
    duplicate_count = 0
    # Verified unions must explain their shorter constituent fits first. For
    # unmerged partial arcs, prefer evidence quality over an inflated fitted span.
    def rank(value: CircleHit) -> tuple[bool, bool, float, float, float]:
        merged = value.diagnostics.get("merged_arc_count", 0) > 1
        support = value.arc_span_deg * value.coverage
        return value.arc_span_deg >= 350, merged, support if merged else 0.0, value.score, support

    duplicate_tolerance = max(3.0, config.radial_tolerance_px * 1.5)
    for hit in sorted(merged_hits, key=rank, reverse=True):
        duplicate = any(_arc_is_contained(hit, old, max(config.max_gap_deg, angular_step)) and
                        (_spatial_arc_overlap(hit, old, duplicate_tolerance) >= 0.8 or
                         _evidence_arc_overlap(hit, old, duplicate_tolerance) >= 0.98)
                        for old in kept)
        if not duplicate:
            kept.append(hit)
        else:
            duplicate_count += 1
        if len(kept) >= config.max_hits:
            break
    diagnostics.update(hit_count=len(kept), elapsed_seconds=time.perf_counter() - started,
                       raw_hit_count=len(all_hits), duplicate_hit_count=duplicate_count,
                       merged_arc_count=merge_count,
                       hit_limit_reached=len(kept) >= config.max_hits,
                       candidate_limit_reached=len(selected) >= config.max_grid_candidates)
    return RuleResult(wafer, kept, diagnostics, response)
