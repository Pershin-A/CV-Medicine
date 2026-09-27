"""Read-only geometry filters shared by the original and augmented labeler."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


FILTER_OPTIONS = {
    "Все изображения": "all",
    "Позвоночник: 0 подвздошных точек": "spine_iliac_0",
    "Позвоночник: 1 подвздошная точка": "spine_iliac_1",
    "Позвоночник: 0 или 1 подвздошная точка": "spine_iliac_incomplete",
    "Позвоночник: нет межпозвоночных линий": "spine_no_disc_lines",
    "Бедро: нет ROI": "hip_no_roi",
    "Бедро: нет опорных точек": "hip_landmarks_0",
    "Бедро: не все 3 опорные точки": "hip_landmarks_incomplete",
    "Бедро: нет разметки малого вертела": "hip_no_trochanter",
    "Бедро: нет обоих контуров малого вертела": "hip_incomplete_traces",
    "Любая неполная геометрия": "any_incomplete",
}


def _relative(value) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def augmentation_index(manifest_path: Path | None) -> dict[str, dict]:
    """Map DATA_ROOT-relative augmented image names to region and JSON path."""
    if manifest_path is None or not manifest_path.is_file():
        return {}
    root = manifest_path.parent.resolve()
    entries = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image = _relative(row.get("image_path", ""))
            geometry = _relative(row.get("geometry_path", ""))
            if not image.startswith("images/") or not geometry:
                continue
            target = (root / geometry).resolve()
            if not target.is_relative_to(root):
                continue
            entries[image.removeprefix("images/")] = {
                "region": row.get("region", ""),
                "geometry_path": str(target),
            }
    return entries


def _sidecar(output_root: Path, relative_path: str) -> Path:
    digest = hashlib.sha256(_relative(relative_path).encode("utf-8")).hexdigest()
    return output_root / "geometry" / f"{digest}.json"


def geometry_path_for(relative_path: str, output_root: Path,
                      augmented: dict[str, dict]) -> Path | None:
    """Prefer a user's saved edit, then the generated starter annotation."""
    sidecar = _sidecar(output_root, relative_path)
    if sidecar.is_file():
        return sidecar
    entry = augmented.get(_relative(relative_path))
    if entry:
        path = Path(entry["geometry_path"])
        if path.is_file():
            return path
    return None


def summarize_geometry(geometry: dict | None, region: str) -> dict:
    spine = (geometry or {}).get("spine") or {}
    hip = (geometry or {}).get("hip") or {}
    crests = spine.get("iliac_crests") or {}
    landmarks = hip.get("landmarks") or {}
    traces = hip.get("lesser_trochanter_traces") or {}
    trochanter = bool(traces.get("trochanter") or hip.get("lesser_trochanter") or
                       hip.get("lesser_trochanter_pixels"))
    adjacent = bool(traces.get("adjacent_bone"))
    return {
        "region": region,
        "iliac_count": sum(crests.get(key) is not None
                           for key in ("image_left", "image_right")),
        "disc_count": len(spine.get("disc_lines") or []),
        "landmark_count": sum(landmarks.get(key) is not None for key in
                              ("greater_trochanter", "femoral_neck", "ischial_bone")),
        "roi_present": hip.get("roi_box") is not None,
        "trochanter_present": trochanter,
        "both_traces": bool(traces.get("trochanter")) and adjacent,
    }


def build_summaries(relative_paths: list[str], output_root: Path,
                    labels: dict, augmented: dict[str, dict]) -> dict[str, dict]:
    summaries = {}
    for rel in relative_paths:
        annotation = labels.get(rel, {})
        category = str(annotation.get("label", "")).upper()
        region = ("SPINE" if category == "SPINE" else
                  "LEG" if category == "LEG" else
                  augmented.get(rel, {}).get("region", ""))
        path = geometry_path_for(rel, output_root, augmented)
        geometry = None
        if path is not None:
            try:
                geometry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                geometry = None
        summaries[rel] = summarize_geometry(geometry, region)
    return summaries


def matches_filter(summary: dict, option: str) -> bool:
    if option == "all":
        return True
    region = summary["region"]
    spine = region == "SPINE"
    hip = region in ("LEG", "LEG_LEFT", "LEG_RIGHT")
    if option == "spine_iliac_0":
        return spine and summary["iliac_count"] == 0
    if option == "spine_iliac_1":
        return spine and summary["iliac_count"] == 1
    if option == "spine_iliac_incomplete":
        return spine and summary["iliac_count"] < 2
    if option == "spine_no_disc_lines":
        return spine and summary["disc_count"] == 0
    if option == "hip_no_roi":
        return hip and not summary["roi_present"]
    if option == "hip_landmarks_0":
        return hip and summary["landmark_count"] == 0
    if option == "hip_landmarks_incomplete":
        return hip and summary["landmark_count"] < 3
    if option == "hip_no_trochanter":
        return hip and not summary["trochanter_present"]
    if option == "hip_incomplete_traces":
        return hip and not summary["both_traces"]
    if option == "any_incomplete":
        return ((spine and (summary["iliac_count"] < 2 or summary["disc_count"] == 0)) or
                (hip and (not summary["roi_present"] or summary["landmark_count"] < 3 or
                          not summary["trochanter_present"])))
    raise ValueError(f"Unknown annotation filter: {option}")
