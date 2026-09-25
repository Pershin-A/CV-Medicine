"""Validated DXA annotation schema and atomic JSON sidecar persistence.

Coordinates are expressed in original decoded DICOM pixel space, not rendered preview
pixels. The origin is top-left; x increases rightwards, y downwards.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

SCHEMA_VERSION = 1


def is_newer_canvas_revision(payload, last_revision: int | float) -> bool:
    """Accept only newer complete snapshots from the browser component.

    The transport-only revision is deliberately excluded by validate_geometry,
    so the on-disk annotation schema stays backward-compatible.
    """
    if not isinstance(payload, dict):
        return False
    revision = payload.get("_client_revision")
    return (
        isinstance(revision, (int, float))
        and not isinstance(revision, bool)
        and math.isfinite(revision)
        and revision > last_revision
    )


def empty_geometry(width: int, height: int) -> dict:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Image dimensions must be positive")
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": "original_dicom_pixels_top_left",
        "image_width": int(width),
        "image_height": int(height),
        "spine": {
            "disc_lines": [],
            "iliac_crests": {"image_left": None, "image_right": None},
            "foreign_objects": [],
        },
        "hip": {
            "landmarks": {
                "greater_trochanter": None,
                "femoral_neck": None,
                "ischial_bone": None,
            },
            "lesser_trochanter": None,
            # Freehand paths are kept separately from the original v5.2 polygon.
            # This preserves every existing annotation after an application update.
            "lesser_trochanter_traces": {
                "trochanter": [],
                "adjacent_bone": [],
            },
            "roi_box": None,
        },
        # Threshold is applied to the 8-bit display image, after the application's
        # 1st/99th-percentile normalization and MONOCHROME1 inversion, if any.
        # It is a per-image manual display annotation, NOT a clinical DXA cutoff.
        "image_view": {"mode": "original", "threshold_8bit": 128},
        "complete": {"spine": False, "hip": False},
    }


def _as_point(raw, width: int, height: int):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("A point must contain exactly [x, y]")
    x, y = [float(v) for v in raw]
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Point coordinates must be finite")
    if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
        raise ValueError(f"Point [{x}, {y}] lies outside {width}x{height}")
    return [round(x, 2), round(y, 2)]


def _as_bbox(raw, width: int, height: int):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("A box must contain [x1, y1, x2, y2]")
    p1 = _as_point(raw[:2], width, height)
    p2 = _as_point(raw[2:], width, height)
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("A bounding box must have nonzero width and height")
    return [x1, y1, x2, y2]


def _as_id(value):
    txt = str(value or "")[:64]
    return txt if txt and all(c.isalnum() or c in "_-" for c in txt) else ""


def validate_geometry(raw: dict | None, width: int, height: int) -> dict:
    """Validate and canonicalize potentially untrusted browser payloads."""
    width, height = int(width), int(height)
    out = empty_geometry(width, height)
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise ValueError("Geometry must be a JSON object")
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError("Unsupported geometry schema version")
    if int(raw.get("image_width", width)) != width or int(raw.get("image_height", height)) != height:
        raise ValueError("Stored annotation image dimensions do not match DICOM")
    spine = raw.get("spine") or {}
    hip = raw.get("hip") or {}
    if not isinstance(spine, dict) or not isinstance(hip, dict):
        raise ValueError("spine and hip must be JSON objects")
    lines = spine.get("disc_lines") or []
    objects = spine.get("foreign_objects") or []
    if not isinstance(lines, list) or not isinstance(objects, list) or len(lines) > 32 or len(objects) > 128:
        raise ValueError("Too many lines or foreign objects")
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("points"), list) or len(line["points"]) != 2:
            raise ValueError("Each disc line must have two endpoints")
        p = [_as_point(v, width, height) for v in line["points"]]
        if math.dist(p[0], p[1]) < 1:
            raise ValueError("Disc line is too short")
        out["spine"]["disc_lines"].append({"id": _as_id(line.get("id")), "points": p})
    crests = spine.get("iliac_crests") or {}
    for name in ("image_left", "image_right"):
        out["spine"]["iliac_crests"][name] = _as_point(crests.get(name), width, height)
    allowed_kinds = {"metal", "clothing", "other"}
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("Foreign object must be a JSON object")
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in allowed_kinds:
            raise ValueError("Invalid foreign object kind")
        out["spine"]["foreign_objects"].append({
            "id": _as_id(item.get("id")), "kind": kind,
            "bbox": _as_bbox(item.get("bbox"), width, height),
        })
        if out["spine"]["foreign_objects"][-1]["bbox"] is None:
            raise ValueError("Foreign-object bounding box is missing")
    landmarks = hip.get("landmarks") or {}
    for name in out["hip"]["landmarks"]:
        out["hip"]["landmarks"][name] = _as_point(landmarks.get(name), width, height)
    poly = hip.get("lesser_trochanter")
    if poly is not None:
        if not isinstance(poly, list) or not (3 <= len(poly) <= 256):
            raise ValueError("Lesser trochanter polygon needs 3–256 vertices")
        pts = [_as_point(v, width, height) for v in poly]
        area = abs(sum(
            pts[i][0] * pts[(i + 1) % len(pts)][1]
            - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts))
        )) / 2
        if area < 1:
            raise ValueError("Lesser trochanter polygon has zero area")
        out["hip"]["lesser_trochanter"] = pts
    out["hip"]["roi_box"] = _as_bbox(hip.get("roi_box"), width, height)
    traces = hip.get("lesser_trochanter_traces") or {}
    if not isinstance(traces, dict):
        raise ValueError("Lesser-trochanter traces must be an object")
    for name in ("trochanter", "adjacent_bone"):
        strokes = traces.get(name) or []
        if not isinstance(strokes, list) or len(strokes) > 24:
            raise ValueError("A freehand trace layer must contain at most 24 strokes")
        for stroke in strokes:
            if not isinstance(stroke, dict) or not isinstance(stroke.get("points"), list):
                raise ValueError("Each freehand stroke must have an id and points")
            points = stroke["points"]
            if not 2 <= len(points) <= 2048:
                raise ValueError("Each freehand stroke needs 2–2048 points")
            normalized = [_as_point(p, width, height) for p in points]
            if sum(math.dist(a, b) for a, b in zip(normalized, normalized[1:])) < 1:
                raise ValueError("A freehand stroke is too short")
            out["hip"]["lesser_trochanter_traces"][name].append({
                "id": _as_id(stroke.get("id")), "points": normalized,
            })
    view = raw.get("image_view") or {}
    if not isinstance(view, dict):
        raise ValueError("Image view settings must be an object")
    mode = view.get("mode", "original")
    if mode not in ("original", "threshold"):
        raise ValueError("Unknown image display mode")
    threshold = view.get("threshold_8bit", 128)
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold):
        raise ValueError("Threshold must be a finite number between 0 and 255")
    if not 0 <= threshold <= 255 or int(threshold) != threshold:
        raise ValueError("Threshold must be an integer between 0 and 255")
    out["image_view"] = {"mode": mode, "threshold_8bit": int(threshold)}
    complete = raw.get("complete") or {}
    for name in ("spine", "hip"):
        out["complete"][name] = bool(complete.get(name, False))
    return out


def geometry_sidecar(output_root: Path, relative_path: str) -> Path:
    """Filename uses sha256 of source path: no PHI/path components in output filename."""
    digest = hashlib.sha256(relative_path.replace('\\', '/').encode('utf-8')).hexdigest()
    return Path(output_root) / "geometry" / f"{digest}.json"


def read_geometry(path: Path, width: int, height: int) -> dict:
    if not Path(path).exists():
        return empty_geometry(width, height)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_geometry(raw, width, height)


def atomic_write_json(path: Path, value: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_geometry(path: Path, geometry: dict, relative_path: str, annotator: str):
    """Save canonical geometry and append a complete snapshot to JSONL history."""
    path = Path(path)
    record = {
        "relative_path": relative_path,
        "annotator": annotator,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "geometry": geometry,
    }
    atomic_write_json(path, record["geometry"])
    history = path.parent / "geometry_history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def geometry_counts(geometry: dict) -> dict:
    traces = geometry["hip"].get("lesser_trochanter_traces") or {}
    t_count = len(traces.get("trochanter") or [])
    b_count = len(traces.get("adjacent_bone") or [])
    return {
        "disc_lines": len(geometry["spine"]["disc_lines"]),
        "iliac_points": sum(v is not None for v in geometry["spine"]["iliac_crests"].values()),
        "foreign_objects": len(geometry["spine"]["foreign_objects"]),
        "hip_landmarks": sum(v is not None for v in geometry["hip"]["landmarks"].values()),
        "lesser_trochanter": int(geometry["hip"]["lesser_trochanter"] is not None or t_count > 0),
        "trochanter_traces": t_count,
        "bone_contour_traces": b_count,
        "hip_roi": int(geometry["hip"]["roi_box"] is not None),
        "threshold_8bit": geometry.get("image_view", {}).get("threshold_8bit", 128),
    }
