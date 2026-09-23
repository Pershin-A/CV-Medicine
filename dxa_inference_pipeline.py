#!/usr/bin/env python
"""Единый inference/export pipeline для DXA QC.

Примеры:

    # Проверить загрузку двух checkpoint и собрать единый bundle
    python dxa_inference_pipeline.py \
        --project-root "C:\\Users\\Андрей\\Desktop\\Хакатон" \
        --export-bundle "C:\\Users\\Андрей\\Desktop\\Хакатон\\prototype_output\\dxa_qc_pipeline_bundle.pt"

    # Обработать один DICOM единым bundle
    python dxa_inference_pipeline.py \
        --bundle "C:\\Users\\Андрей\\Desktop\\Хакатон\\prototype_output\\dxa_qc_pipeline_bundle.pt" \
        --input "C:\\path\\image.dcm" \
        --output "C:\\path\\result.json"

    # Обработать папку; результат сохраняется как CSV
    python dxa_inference_pipeline.py \
        --bundle "C:\\Users\\Андрей\\Desktop\\Хакатон\\prototype_output\\dxa_qc_pipeline_bundle.pt" \
        --input "C:\\path\\dicom_folder" \
        --output "C:\\path\\qc_results.csv"

SAM не включается в bundle из-за размера. Для проверки подвздошных костей передай
--use-sam --sam-checkpoint PATH. Без SAM результат позвоночника будет явно помечен
как неполный, но anatomy-router и остальные доступные проверки продолжат работу.

Загружай только собственные/доверенные .pt checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
    import numpy as np
    import pandas as pd
    import pydicom
    import torch
    import torch.nn as nn
    from PIL import Image
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        gaussian_filter1d,
        label as ndi_label,
        map_coordinates,
    )
    from torchvision import transforms as T
    from torchvision.models import resnet18
except ImportError as exc:
    raise SystemExit(
        "Не установлены зависимости. Выполни:\n"
        "pip install pydicom pylibjpeg pylibjpeg-libjpeg numpy pandas scipy "
        "pillow torch torchvision opencv-python-headless"
    ) from exc

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


PIPELINE_FORMAT_VERSION = 1
DEFAULT_CLASS_TO_IDX = {"SPINE": 0, "LEG_LEFT": 1, "LEG_RIGHT": 2}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PipelineConfig:
    image_size: int = 224
    background_quantile: float = 0.01
    hip_roi_top_mm: float = 30.0
    hip_roi_bottom_mm: float = 30.0
    hip_roi_right_mm: float = 20.0
    spine_bright_fraction: float = 0.10
    spine_segments: int = 7
    spine_max_tilt_deg: float = 5.0
    t12_visible_fraction_low: float = 0.30
    t12_visible_fraction_high: float = 0.75
    iliac_top_y_low: float = 0.50
    iliac_top_y_high: float = 0.90
    iliac_area_low: float = 0.003
    iliac_area_high: float = 0.25
    iliac_max_pair_iou: float = 0.20
    sam_model_type: str = "vit_b"


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Совместимая загрузка простого checkpoint на новых и старых версиях PyTorch."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint должен быть dict: {path}")
    return payload


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _build_resnet(n_outputs: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_outputs)
    return model


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def read_dicom_image(path: Path) -> tuple[Any, np.ndarray]:
    ds = pydicom.dcmread(path, force=True)
    arr = np.squeeze(np.asarray(ds.pixel_array))
    if arr.ndim != 2:
        raise ValueError(f"Ожидается один 2D DXA-кадр, получена форма {arr.shape}")
    arr = np.asarray(apply_modality_lut(arr, ds), dtype=np.float32)
    try:
        arr = np.asarray(apply_voi_lut(arr, ds), dtype=np.float32)
    except Exception:
        pass
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("В DICOM нет конечных значений PixelData")
    lo, hi = np.nanpercentile(arr[finite], [0.5, 99.5])
    if hi <= lo:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    image = np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 1 - image
    return ds, image.astype(np.float32)


def preprocess_dicom(path: Path, q: float) -> tuple[Any, np.ndarray, np.ndarray, float]:
    ds, image = read_dicom_image(path)
    positive = image[image > 0]
    threshold = float(np.quantile(positive, q)) if len(positive) else 0.0
    filtered = image.copy()
    filtered[filtered < threshold] = 0
    return ds, image, filtered, threshold


def pixel_spacing_mm(ds: Any) -> Optional[tuple[float, float]]:
    for name in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        value = getattr(ds, name, None)
        if value is None or len(value) < 2:
            continue
        try:
            row_mm, col_mm = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            continue
        if row_mm > 0 and col_mm > 0:
            return row_mm, col_mm
    return None


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi_label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def build_hip_bone_mask(filtered: np.ndarray) -> np.ndarray:
    positive = filtered[filtered > 0]
    if not len(positive):
        return np.zeros_like(filtered, dtype=bool)
    threshold = float(np.quantile(positive, 0.62))
    mask = filtered >= threshold
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
    ).astype(bool)
    mask = binary_fill_holes(binary_closing(mask, iterations=2))
    return largest_component(mask)


def _column_envelope(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.flatnonzero(mask.any(axis=0))
    if not len(xs):
        raise ValueError("Пустая маска кости")
    top = np.array([np.flatnonzero(mask[:, x])[0] for x in xs], dtype=int)
    bottom = np.array([np.flatnonzero(mask[:, x])[-1] for x in xs], dtype=int)
    return xs, top, bottom


def _right_min_after_max(xs: np.ndarray, distances: np.ndarray) -> tuple[int, int]:
    i_max = int(np.argmax(distances))
    right_indices = (
        np.arange(i_max + 1, len(xs))
        if i_max + 1 < len(xs)
        else np.array([i_max])
    )
    i_min = int(right_indices[np.argmin(distances[right_indices])])
    return i_max, i_min


def find_hip_roi_boundaries(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    height, _ = mask.shape
    xs, top_y, bottom_y = _column_envelope(mask)
    top_max_i, top_boundary_i = _right_min_after_max(xs, top_y.astype(float))
    bottom_distance = height - 1 - bottom_y
    bottom_max_i, bottom_boundary_i = _right_min_after_max(
        xs, bottom_distance.astype(float)
    )
    right_x = int(xs[-1])
    right_y = int(np.median(np.flatnonzero(mask[:, right_x])))
    return {
        "right_boundary": (right_x, right_y),
        "top_max_anchor": (int(xs[top_max_i]), int(top_y[top_max_i])),
        "top_boundary": (int(xs[top_boundary_i]), int(top_y[top_boundary_i])),
        "bottom_max_anchor": (int(xs[bottom_max_i]), int(bottom_y[bottom_max_i])),
        "bottom_boundary": (
            int(xs[bottom_boundary_i]),
            int(bottom_y[bottom_boundary_i]),
        ),
    }


def evaluate_hip_roi(
    ds: Any,
    image: np.ndarray,
    filtered: np.ndarray,
    config: PipelineConfig,
) -> dict[str, Any]:
    mask = build_hip_bone_mask(filtered)
    boundaries = find_hip_roi_boundaries(mask)
    spacing = pixel_spacing_mm(ds)
    height, width = image.shape
    result: dict[str, Any] = {
        "pixel_spacing_available": spacing is not None,
        "bone_mask_area_fraction": float(mask.mean()),
    }
    for name, (x, y) in boundaries.items():
        result[f"{name}_x_px"] = int(x)
        result[f"{name}_y_px"] = int(y)
    if spacing is None:
        result.update(
            {
                "top_margin_mm": np.nan,
                "bottom_margin_mm": np.nan,
                "right_margin_mm": np.nan,
                "top_margin_ok": None,
                "bottom_margin_ok": None,
                "right_margin_ok": None,
                "hip_roi_ok": None,
            }
        )
        return result

    row_mm, col_mm = spacing
    top_margin = boundaries["top_boundary"][1] * row_mm
    bottom_margin = (height - 1 - boundaries["bottom_boundary"][1]) * row_mm
    right_margin = (width - 1 - boundaries["right_boundary"][0]) * col_mm
    result.update(
        {
            "row_spacing_mm": row_mm,
            "col_spacing_mm": col_mm,
            "top_margin_mm": float(top_margin),
            "bottom_margin_mm": float(bottom_margin),
            "right_margin_mm": float(right_margin),
            "top_margin_ok": bool(top_margin >= config.hip_roi_top_mm),
            "bottom_margin_ok": bool(bottom_margin >= config.hip_roi_bottom_mm),
            "right_margin_ok": bool(right_margin >= config.hip_roi_right_mm),
        }
    )
    result["hip_roi_ok"] = bool(
        result["top_margin_ok"]
        and result["bottom_margin_ok"]
        and result["right_margin_ok"]
    )
    return result


def crop_spine_roi(
    image: np.ndarray,
    x_range: tuple[float, float] = (0.20, 0.80),
    y_range: tuple[float, float] = (0.04, 0.96),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape
    x0, x1 = int(width * x_range[0]), int(width * x_range[1])
    y0, y1 = int(height * y_range[0]), int(height * y_range[1])
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def top_fraction_mask(image: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(image, dtype=np.float32).ravel()
    n_keep = max(1, int(round(len(flat) * fraction)))
    indices = np.argpartition(flat, -n_keep)[-n_keep:]
    mask = np.zeros(len(flat), dtype=bool)
    mask[indices] = True
    return mask.reshape(image.shape)


def row_centers_of_mass(
    roi: np.ndarray, bright_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = roi * bright_mask.astype(np.float32)
    row_mass = weights.sum(axis=1)
    xs = np.arange(roi.shape[1], dtype=np.float32)
    x_center = np.full(roi.shape[0], np.nan, dtype=np.float32)
    valid = row_mass > 0
    x_center[valid] = (
        (weights[valid] * xs[None, :]).sum(axis=1) / row_mass[valid]
    )
    if valid.sum() < 2:
        x_center[:] = roi.shape[1] / 2
    else:
        x_center = np.interp(
            np.arange(len(x_center)), np.where(valid)[0], x_center[valid]
        ).astype(np.float32)
    confidence = row_mass / max(float(row_mass.max()), 1e-6)
    confidence = np.where(valid, np.maximum(confidence, 0.05), 0.01)
    return x_center.astype(np.float64), confidence.astype(np.float64)


def regression_prefix(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    values = {
        "w": w,
        "wy": w * y,
        "wx": w * x,
        "wyy": w * y * y,
        "wyx": w * y * x,
        "wxx": w * x * x,
    }
    return {key: np.r_[0.0, np.cumsum(value)] for key, value in values.items()}


def interval_line_fit(
    prefix: dict[str, np.ndarray], start: int, end: int
) -> tuple[float, float, float]:
    def total(name: str) -> float:
        return float(prefix[name][end] - prefix[name][start])

    sw, sy, sx = total("w"), total("wy"), total("wx")
    syy, syx, sxx = total("wyy"), total("wyx"), total("wxx")
    if sw <= 1e-8 or end - start < 2:
        return np.inf, 0.0, 0.0
    determinant = sw * syy - sy * sy
    if abs(determinant) < 1e-10:
        slope = 0.0
        intercept = sx / sw
    else:
        slope = (sw * syx - sy * sx) / determinant
        intercept = (sx - slope * sy) / sw
    sse = sxx - intercept * sx - slope * syx
    return max(float(sse), 0.0), float(intercept), float(slope)


def fit_piecewise_spine_axis(
    x_center: np.ndarray,
    confidence: np.ndarray,
    full_height: int,
    segment_count: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    height = len(x_center)
    y = np.arange(height, dtype=np.float64)
    prefix = regression_prefix(y, x_center, confidence)
    n_breaks = segment_count - 1
    min_gap = max(2, int(math.ceil(full_height / segment_count)))
    edge_min = max(3, int(round(0.02 * height)))
    candidate_step = 2 if height > 500 else 1

    @lru_cache(None)
    def solve(k: int, start: int) -> tuple[float, tuple[int, ...]]:
        if k == 0:
            cost, _, _ = interval_line_fit(prefix, start, height)
            return cost, ()
        first_min_length = edge_min if start == 0 else min_gap
        low = start + first_min_length
        high = height - edge_min - (k - 1) * min_gap
        if low > high:
            return np.inf, ()
        best_cost, best_breaks = np.inf, ()
        candidates = list(range(low, high + 1, candidate_step))
        if high not in candidates:
            candidates.append(high)
        for breakpoint in candidates:
            left_cost, _, _ = interval_line_fit(prefix, start, breakpoint)
            if not np.isfinite(left_cost):
                continue
            right_cost, right_breaks = solve(k - 1, breakpoint)
            current = left_cost + right_cost
            if current < best_cost:
                best_cost = current
                best_breaks = (breakpoint,) + right_breaks
        return best_cost, best_breaks

    _, breaks = solve(n_breaks, 0)
    if len(breaks) != n_breaks:
        raise RuntimeError("Не удалось построить семь сегментов оси позвоночника")
    bounds = np.r_[0, np.asarray(breaks, dtype=int), height]
    axis_x = np.zeros(height, dtype=np.float64)
    segments: list[dict[str, Any]] = []
    names = ["T12", "L1", "L2", "L3", "L4", "L5", "SACRUM"]
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        _, intercept, slope = interval_line_fit(prefix, int(start), int(end))
        yy = np.arange(start, end, dtype=np.float64)
        axis_x[start:end] = intercept + slope * yy
        segments.append(
            {
                "segment_index": index,
                "segment_name": names[index] if index < len(names) else f"segment_{index}",
                "y_top": int(start),
                "y_bottom": int(end - 1),
                "vertical_angle": math.degrees(math.atan(slope)),
            }
        )
    return axis_x, pd.DataFrame(segments)


def analyze_spine_geometry(
    image: np.ndarray, filtered: np.ndarray, config: PipelineConfig
) -> dict[str, Any]:
    bright_mask = top_fraction_mask(image, config.spine_bright_fraction)
    roi, (_, y0, _, y1) = crop_spine_roi(filtered)
    bright_roi = bright_mask[y0:y1, int(image.shape[1] * 0.20):int(image.shape[1] * 0.80)]
    x_center, confidence = row_centers_of_mass(roi, bright_roi)
    axis_x, segments = fit_piecewise_spine_axis(
        x_center, confidence, image.shape[0], config.spine_segments
    )
    top = np.array([axis_x[0], 0.0])
    bottom = np.array([axis_x[-1], float(len(axis_x) - 1)])
    spine_angle = math.degrees(
        math.atan2(bottom[0] - top[0], bottom[1] - top[1])
    )
    scoliosis_proxy = abs(
        float(segments.iloc[0]["vertical_angle"] - segments.iloc[-1]["vertical_angle"])
    )
    return {
        "spine_axis_angle_deg": float(spine_angle),
        "scoliosis_proxy_deg": float(scoliosis_proxy),
        "segments": segments,
    }


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / max(union, 1))


class DXAInferencePipeline:
    """Загружает модели один раз и выполняет единый DXA QC inference."""

    def __init__(
        self,
        anatomy_checkpoint: Optional[Path] = None,
        metal_checkpoint: Optional[Path] = None,
        bundle_path: Optional[Path] = None,
        sam_checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        metal_threshold_override: Optional[float] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sam_checkpoint = Path(sam_checkpoint) if sam_checkpoint else None
        self._sam_predictor = None
        self.health_warnings: list[str] = []

        if bundle_path:
            payload = _torch_load(Path(bundle_path), self.device)
            self._load_bundle(payload)
            self.model_source = str(bundle_path)
        else:
            if anatomy_checkpoint is None or metal_checkpoint is None:
                raise ValueError(
                    "Нужны оба checkpoint либо один --bundle: anatomy_checkpoint и metal_checkpoint"
                )
            self._load_separate(Path(anatomy_checkpoint), Path(metal_checkpoint))
            self.model_source = f"{anatomy_checkpoint}; {metal_checkpoint}"

        if metal_threshold_override is not None:
            if not 0 < metal_threshold_override < 1:
                raise ValueError("metal_threshold_override должен находиться между 0 и 1")
            self.metal_threshold = float(metal_threshold_override)
            self.health_warnings.append("Используется ручной override порога metal-head")

        if self.metal_threshold < 1e-3:
            self.health_warnings.append(
                "Порог metal-head экстремально мал; metal_score нельзя трактовать как "
                "калиброванную клиническую вероятность"
            )

        self.transform = T.Compose(
            [
                T.Resize((self.config.image_size, self.config.image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _load_anatomy(self, payload: dict[str, Any]) -> None:
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("В anatomy checkpoint отсутствует state_dict")
        mapping = payload.get("class_to_idx", DEFAULT_CLASS_TO_IDX)
        if set(mapping) != set(DEFAULT_CLASS_TO_IDX) or sorted(mapping.values()) != [0, 1, 2]:
            raise ValueError(f"Неожиданное class_to_idx anatomy-router: {mapping}")
        model = _build_resnet(len(mapping))
        model.load_state_dict(state, strict=True)
        self.anatomy_model = model.to(self.device).eval()
        self.class_to_idx = {str(key): int(value) for key, value in mapping.items()}

    def _load_metal(self, payload: dict[str, Any]) -> None:
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("В metal checkpoint отсутствует state_dict")
        model = _build_resnet(1)
        model.load_state_dict(state, strict=True)
        self.metal_model = model.to(self.device).eval()
        threshold = float(payload.get("threshold", 0.5))
        if not 0 < threshold < 1:
            raise ValueError(f"Некорректный threshold в metal checkpoint: {threshold}")
        self.metal_threshold = threshold
        self.metal_metadata = {
            key: value for key, value in payload.items() if key != "state_dict"
        }

    def _load_separate(self, anatomy_path: Path, metal_path: Path) -> None:
        if not anatomy_path.exists():
            raise FileNotFoundError(f"Anatomy checkpoint не найден: {anatomy_path}")
        if not metal_path.exists():
            raise FileNotFoundError(f"Metal checkpoint не найден: {metal_path}")
        self._load_anatomy(_torch_load(anatomy_path, self.device))
        self._load_metal(_torch_load(metal_path, self.device))

    def _load_bundle(self, payload: dict[str, Any]) -> None:
        if payload.get("format_version") != PIPELINE_FORMAT_VERSION:
            raise ValueError(
                f"Неподдерживаемая версия bundle: {payload.get('format_version')}"
            )
        if "config" in payload:
            self.config = PipelineConfig(**payload["config"])
        self._load_anatomy(payload["anatomy"])
        self._load_metal(payload["metal"])

    def export_bundle(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format_version": PIPELINE_FORMAT_VERSION,
            "config": asdict(self.config),
            "anatomy": {
                "state_dict": _cpu_state_dict(self.anatomy_model),
                "class_to_idx": self.class_to_idx,
            },
            "metal": {
                "state_dict": _cpu_state_dict(self.metal_model),
                **self.metal_metadata,
                "threshold": float(self.metal_threshold),
            },
        }
        torch.save(bundle, output_path)
        return output_path

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "READY",
            "device": str(self.device),
            "model_source": self.model_source,
            "anatomy_classes": self.class_to_idx,
            "metal_threshold": float(self.metal_threshold),
            "metal_score_is_calibrated_probability": False,
            "sam_checkpoint_available": bool(
                self.sam_checkpoint and self.sam_checkpoint.exists()
            ),
            "warnings": self.health_warnings,
        }

    def _tensor(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        return self.transform(pil).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_anatomy(self, image: np.ndarray) -> dict[str, Any]:
        probabilities = torch.softmax(self.anatomy_model(self._tensor(image)), 1)[0]
        values = probabilities.detach().cpu().numpy()
        index = int(np.argmax(values))
        idx_to_class = {value: key for key, value in self.class_to_idx.items()}
        return {
            "anatomical_region": idx_to_class[index],
            "anatomy_confidence": float(values[index]),
            "anatomy_scores": {
                idx_to_class[i]: float(values[i]) for i in range(len(values))
            },
        }

    @torch.no_grad()
    def predict_metal(self, image: np.ndarray) -> dict[str, Any]:
        logit = self.metal_model(self._tensor(image)).squeeze()
        score = float(torch.sigmoid(logit).detach().cpu())
        return {
            "metal_score": score,
            "metal_score_is_calibrated_probability": False,
            "metal_threshold": float(self.metal_threshold),
            "metal_pred": int(score >= self.metal_threshold),
        }

    def _get_sam_predictor(self):
        if self._sam_predictor is not None:
            return self._sam_predictor
        if self.sam_checkpoint is None or not self.sam_checkpoint.exists():
            raise FileNotFoundError("SAM checkpoint не задан или не найден")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Для SAM установи segment-anything: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc
        sam = sam_model_registry[self.config.sam_model_type](
            checkpoint=str(self.sam_checkpoint)
        ).to(self.device)
        self._sam_predictor = SamPredictor(sam)
        return self._sam_predictor

    @staticmethod
    def _weighted_prompt(
        image: np.ndarray, x0: int, x1: int, y0: int, y1: int
    ) -> tuple[float, float]:
        crop = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        weights = np.maximum(crop - np.quantile(crop, 0.65), 0)
        if weights.sum() == 0:
            return (x0 + x1) / 2, (y0 + y1) / 2
        return (
            float((xx * weights).sum() / weights.sum()),
            float((yy * weights).sum() / weights.sum()),
        )

    def segment_iliac_bones(self, image: np.ndarray) -> dict[str, Any]:
        height, width = image.shape
        rgb = np.repeat((image * 255).astype(np.uint8)[..., None], 3, axis=2)
        predictor = self._get_sam_predictor()
        predictor.set_image(rgb)
        regions = {
            "left_image": (
                int(0.03 * width), int(0.48 * width),
                int(0.55 * height), int(0.98 * height),
            ),
            "right_image": (
                int(0.52 * width), int(0.97 * width),
                int(0.55 * height), int(0.98 * height),
            ),
        }
        masks: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        for name, (x0, x1, y0, y1) in regions.items():
            point = self._weighted_prompt(image, x0, x1, y0, y1)
            candidates, scores, _ = predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            zone = np.zeros((height, width), dtype=bool)
            zone[y0:y1, x0:x1] = True
            best = None
            for mask, sam_score in zip(candidates, scores):
                area = float(mask.mean())
                overlap = float((mask & zone).sum() / max(mask.sum(), 1))
                selection_score = (
                    float(sam_score) + 0.6 * overlap - 0.5 * max(0, area - 0.25)
                )
                in_area = self.config.iliac_area_low <= area <= self.config.iliac_area_high
                if in_area and (best is None or selection_score > best[0]):
                    best = (selection_score, mask, float(sam_score))
            if best is None:
                index = int(np.argmax(scores))
                best = (float(scores[index]), candidates[index], float(scores[index]))
            mask = best[1]
            ys, xs = np.where(mask)
            if len(ys):
                top_y = int(ys.min())
                top_x = float(np.median(xs[ys == top_y]))
            else:
                top_x = top_y = np.nan
            masks[name] = mask
            rows.append(
                {
                    "side": name,
                    "sam_score": best[2],
                    "area_fraction": float(mask.mean()),
                    "top_x_px": top_x,
                    "top_y_px": top_y,
                    "top_y_norm": float(top_y / height) if np.isfinite(top_y) else np.nan,
                }
            )
        table = pd.DataFrame(rows)
        pair_iou = _mask_iou(masks["left_image"], masks["right_image"])
        return {
            "table": table,
            "pair_iou": pair_iou,
            "pair_distinct": bool(pair_iou <= self.config.iliac_max_pair_iou),
        }

    def evaluate_spine_coverage(
        self, image: np.ndarray, geometry: dict[str, Any]
    ) -> dict[str, Any]:
        iliac = self.segment_iliac_bones(image)
        segments = geometry["segments"]
        heights = (segments["y_bottom"] - segments["y_top"] + 1).to_numpy(float)
        lumbar_reference = float(np.median(heights[1:5]))
        t12_fraction = float(heights[0] / max(lumbar_reference, 1e-6))
        t12_ok = bool(
            self.config.t12_visible_fraction_low
            <= t12_fraction
            <= self.config.t12_visible_fraction_high
        )
        table = iliac["table"]
        finite = bool(
            len(table) == 2
            and np.isfinite(table[["top_x_px", "top_y_px", "top_y_norm"]]).all().all()
        )
        top_range_ok = bool(
            finite
            and table["top_y_norm"].between(
                self.config.iliac_top_y_low, self.config.iliac_top_y_high
            ).all()
        )
        area_range_ok = bool(
            len(table) == 2
            and table["area_fraction"].between(
                self.config.iliac_area_low, self.config.iliac_area_high
            ).all()
        )
        top_ok = bool(top_range_ok and iliac["pair_distinct"])
        area_ok = bool(area_range_ok and iliac["pair_distinct"])
        result: dict[str, Any] = {
            "t12_visible_fraction": t12_fraction,
            "t12_half_visible_ok": t12_ok,
            "iliac_pair_iou": iliac["pair_iou"],
            "iliac_pair_distinct": iliac["pair_distinct"],
            "iliac_top_points_ok": top_ok,
            "iliac_area_ok_experimental": area_ok,
            "spine_coverage_ok": bool(t12_ok and top_ok),
        }
        for _, row in table.iterrows():
            prefix = str(row["side"])
            for column in (
                "sam_score", "area_fraction", "top_x_px", "top_y_px", "top_y_norm"
            ):
                result[f"{prefix}_{column}"] = float(row[column])
        return result

    def process_image(
        self,
        path: Path,
        use_sam: bool = False,
        allow_partial_quality: bool = False,
    ) -> dict[str, Any]:
        path = Path(path)
        started = time.perf_counter()
        result: dict[str, Any] = {
            "path_to_study": str(path.parent),
            "path_to_image": str(path),
            "study_uid": "",
            "image_uid": "",
            "anatomical_region": "UNKNOWN",
            "quality_class": "UNKNOWN",
            "partial_quality_class": "UNKNOWN",
            "violation_type": "",
            "qc_complete": False,
            "checks_run": "",
            "checks_skipped": "",
            "processing_status": "Failure",
            "time_of_processing": 0.0,
        }
        checks_run: list[str] = []
        checks_skipped: list[str] = []
        violations: list[str] = []
        try:
            ds, image, filtered, background_threshold = preprocess_dicom(
                path, self.config.background_quantile
            )
            result["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
            result["image_uid"] = str(getattr(ds, "SOPInstanceUID", ""))
            result["background_threshold"] = background_threshold

            anatomy = self.predict_anatomy(image)
            result.update(anatomy)
            checks_run.append("ANATOMY_ROUTER")
            region = anatomy["anatomical_region"]

            if region == "SPINE":
                geometry = analyze_spine_geometry(image, filtered, self.config)
                result["spine_axis_angle_deg"] = geometry["spine_axis_angle_deg"]
                result["scoliosis_proxy_deg"] = geometry["scoliosis_proxy_deg"]
                checks_run.extend(["SPINE_ALIGNMENT", "SCOLIOSIS_PROXY_SCORE"])
                if abs(geometry["spine_axis_angle_deg"]) > self.config.spine_max_tilt_deg:
                    violations.append("SPINE_ALIGNMENT")
                checks_skipped.extend(
                    ["SCOLIOSIS_DECISION", "SPINE_ARTIFACT", "SPINE_ROI"]
                )
                if use_sam:
                    try:
                        coverage = self.evaluate_spine_coverage(image, geometry)
                        result.update(coverage)
                        checks_run.append("SPINE_COVERAGE")
                        if not coverage["t12_half_visible_ok"]:
                            violations.append("T12_COVERAGE")
                        if not coverage["iliac_top_points_ok"]:
                            violations.append("ILIAC_CREST_COVERAGE")
                    except Exception as exc:
                        checks_skipped.append("SPINE_COVERAGE")
                        result["sam_warning"] = f"{type(exc).__name__}: {exc}"
                else:
                    checks_skipped.append("SPINE_COVERAGE")

            elif region in {"LEG_LEFT", "LEG_RIGHT"}:
                roi = evaluate_hip_roi(ds, image, filtered, self.config)
                result.update(roi)
                if roi["hip_roi_ok"] is None:
                    checks_skipped.append("HIP_ROI_NO_PIXEL_SPACING")
                else:
                    checks_run.append("HIP_ROI")
                    if not roi["hip_roi_ok"]:
                        violations.append("HIP_ROI")
                checks_skipped.append("HIP_POSITION_ROTATION_AUTO")
                metal = self.predict_metal(image)
                result.update(metal)
                checks_run.append("METAL")
                if metal["metal_pred"]:
                    violations.append("METAL")
            else:
                raise ValueError(f"Неизвестный anatomical_region: {region}")

            result["checks_run"] = ";".join(dict.fromkeys(checks_run))
            result["checks_skipped"] = ";".join(dict.fromkeys(checks_skipped))
            result["qc_complete"] = not bool(checks_skipped)
            result["violation_type"] = ";".join(dict.fromkeys(violations))
            result["partial_quality_class"] = "VIOLATION" if violations else "QUALITY"
            if violations:
                result["quality_class"] = "VIOLATION"
            elif result["qc_complete"] or allow_partial_quality:
                result["quality_class"] = "QUALITY"
            result["processing_status"] = "Success"
        except Exception as exc:
            result["checks_run"] = ";".join(dict.fromkeys(checks_run))
            result["checks_skipped"] = ";".join(dict.fromkeys(checks_skipped))
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
        finally:
            result["time_of_processing"] = float(time.perf_counter() - started)
        return _json_safe(result)

    def smoke_test(self, dicom_path: Path, use_sam: bool = False) -> dict[str, Any]:
        result = self.process_image(dicom_path, use_sam=use_sam)
        return {
            "passed": result.get("processing_status") == "Success",
            "anatomical_region": result.get("anatomical_region"),
            "anatomy_confidence": result.get("anatomy_confidence"),
            "metal_score": result.get("metal_score"),
            "quality_class": result.get("quality_class"),
            "qc_complete": result.get("qc_complete"),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
        }

    def process_folder(
        self,
        folder: Path,
        use_sam: bool = False,
        allow_partial_quality: bool = False,
    ) -> pd.DataFrame:
        folder = Path(folder)
        candidates = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {"", ".dcm", ".dicom", ".ima"}
        )
        if not candidates:
            raise FileNotFoundError(f"В папке не найдены DICOM-кандидаты: {folder}")
        rows = [
            self.process_image(
                path,
                use_sam=use_sam,
                allow_partial_quality=allow_partial_quality,
            )
            for path in candidates
        ]
        return pd.DataFrame(rows)


def _default_output(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "qc_results.csv"
    return input_path.with_suffix(input_path.suffix + ".qc.json")


def _save_result(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, pd.DataFrame):
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        output_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DXA unified QC inference pipeline")
    parser.add_argument("--project-root", type=Path, help="Корень проекта с prototype_output")
    parser.add_argument("--anatomy-checkpoint", type=Path)
    parser.add_argument("--metal-checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path, help="Единый bundle вместо двух checkpoint")
    parser.add_argument("--export-bundle", type=Path, help="Сохранить единый bundle")
    parser.add_argument("--input", type=Path, help="Один DICOM или папка")
    parser.add_argument("--output", type=Path, help="JSON для файла или CSV для папки")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--metal-threshold", type=float, help="Override checkpoint threshold")
    parser.add_argument("--allow-partial-quality", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    anatomy_checkpoint = args.anatomy_checkpoint
    metal_checkpoint = args.metal_checkpoint
    if args.bundle is None:
        if args.project_root is not None:
            output_dir = args.project_root / "prototype_output"
            anatomy_checkpoint = anatomy_checkpoint or output_dir / "resnet18_anatomy3.pt"
            metal_checkpoint = metal_checkpoint or output_dir / "resnet18_metal_binary.pt"
        if anatomy_checkpoint is None or metal_checkpoint is None:
            raise SystemExit(
                "Укажи --bundle либо --project-root, либо оба отдельных checkpoint"
            )

    pipeline = DXAInferencePipeline(
        anatomy_checkpoint=anatomy_checkpoint,
        metal_checkpoint=metal_checkpoint,
        bundle_path=args.bundle,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        metal_threshold_override=args.metal_threshold,
    )
    print(json.dumps(pipeline.health_check(), ensure_ascii=False, indent=2))

    if args.export_bundle:
        exported = pipeline.export_bundle(args.export_bundle)
        print(f"Bundle сохранён: {exported}")

    if args.input is not None:
        input_path = args.input
        if not input_path.exists():
            raise SystemExit(f"Input не найден: {input_path}")
        if input_path.is_dir():
            result = pipeline.process_folder(
                input_path,
                use_sam=args.use_sam,
                allow_partial_quality=args.allow_partial_quality,
            )
        else:
            result = pipeline.process_image(
                input_path,
                use_sam=args.use_sam,
                allow_partial_quality=args.allow_partial_quality,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        output_path = args.output or _default_output(input_path)
        _save_result(result, output_path)
        print(f"Результат сохранён: {output_path}")

    if args.input is None and args.export_bundle is None:
        print("Модели успешно загружены. Для smoke test передай --input PATH_TO_DICOM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
