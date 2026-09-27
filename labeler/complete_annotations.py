"""Synchronize completion flags in labels.csv, geometry JSON and DICOM copies.

Default is read-only. Apply only after making a separate folder backup.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pydicom

from labeler.geometry import validate_geometry


PRIVATE_GROUP = 0x0011
PRIVATE_CREATOR = "DXA_MANUAL_LABELER"
GEOMETRY_OFFSET = 0x09


def _read_embedded(ds):
    block = ds.private_block(PRIVATE_GROUP, PRIVATE_CREATOR, create=False)
    if block is None or block.get_tag(GEOMETRY_OFFSET) not in ds:
        raise ValueError("DICOM geometry private tag is missing")
    value = ds[block.get_tag(GEOMETRY_OFFSET)].value
    if isinstance(value, bytes):
        value = value.decode("utf-8").rstrip("\x00")
    return json.loads(value)


def _atomic_write(path: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_dicom(path: Path, ds, geometry: dict) -> None:
    block = ds.private_block(PRIVATE_GROUP, PRIVATE_CREATOR, create=False)
    element = ds[block.get_tag(GEOMETRY_OFFSET)]
    # Implicit-VR DICOM decodes unknown private elements as UN (bytes), even
    # though the labeler originally wrote this one as UT text.
    element.VR = "UT"
    element.value = json.dumps(geometry, ensure_ascii=True, separators=(",", ":"))
    original_pixels = hashlib.sha256(ds.PixelData).digest()
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".dcm", dir=path.parent)
    os.close(fd)
    try:
        pydicom.dcmwrite(temporary, ds, enforce_file_format=True)
        check = pydicom.dcmread(temporary, force=True)
        if _read_embedded(check) != geometry:
            raise ValueError(f"DICOM roundtrip lost geometry: {path}")
        if hashlib.sha256(check.PixelData).digest() != original_pixels:
            raise ValueError(f"DICOM roundtrip changed pixel data: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _hip_has_all_required_objects(geometry: dict) -> bool:
    hip = geometry["hip"]
    traces = hip["lesser_trochanter_traces"]
    return (all(point is not None for point in hip["landmarks"].values())
            and hip["roi_box"] is not None
            and bool(traces["trochanter"])
            and bool(traces["adjacent_bone"]))


def complete(root: Path, apply: bool = False, allow_partial_hip: bool = False) -> dict:
    root = root.resolve()
    registry = root / "labels.csv"
    csv_bytes = registry.read_bytes()
    with registry.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not {"relative_path", "output_path", "geometry_path"}.issubset(fields):
        raise ValueError("labels.csv lacks required path columns")
    for field in ("geometry_spine_complete", "geometry_hip_complete"):
        if field not in fields:
            fields.append(field)
    planned = []
    exceptions = []
    for index, row in enumerate(rows, 1):
        region = row.get("label")
        if region not in ("SPINE", "LEG"):
            exceptions.append({"index": index, "reason": "unknown_region"})
            continue
        geo_relative = Path(row.get("geometry_path") or "")
        dicom_relative = Path(row.get("output_path") or row["relative_path"])
        geometry_path = (root / geo_relative).resolve()
        dicom_path = (root / dicom_relative).resolve()
        if (not geometry_path.is_relative_to(root) or
                not dicom_path.is_relative_to(root)):
            raise ValueError(f"Path escapes annotation folder at row {index}")
        if not geometry_path.is_file() or not dicom_path.is_file():
            exceptions.append({"index": index, "reason": "missing_file"})
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        ds = pydicom.dcmread(dicom_path, force=True)
        geometry = validate_geometry(geometry, int(ds.Columns), int(ds.Rows))
        if validate_geometry(_read_embedded(ds), int(ds.Columns), int(ds.Rows)) != geometry:
            raise ValueError(f"DICOM and JSON disagree at row {index}")
        key = "spine" if region == "SPINE" else "hip"
        if geometry["complete"][key] and row.get(f"geometry_{key}_complete") == "1":
            continue
        if region == "SPINE" and not 4 <= len(geometry["spine"]["disc_lines"]) <= 7:
            exceptions.append({"index": index, "reason": "missing_spine_lines"})
            continue
        if region == "LEG" and not _hip_has_all_required_objects(geometry) and not allow_partial_hip:
            exceptions.append({"index": index, "reason": "partial_hip_annotation"})
            continue
        geometry["complete"][key] = True
        row[f"geometry_{key}_complete"] = "1"
        planned.append((index, geometry_path, dicom_path, geometry, ds))
    report = {"root": str(root), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "rows": len(rows), "planned_updates": len(planned),
              "updated_indices": [item[0] for item in planned],
              "exceptions": exceptions, "applied": apply}
    if not apply:
        return report
    if hashlib.sha256(registry.read_bytes()).digest() != hashlib.sha256(csv_bytes).digest():
        raise RuntimeError("labels.csv changed during review; retry later")
    for _, geometry_path, dicom_path, geometry, ds in planned:
        # A rerun can safely finish an interrupted update; CSV is committed last.
        _atomic_dicom(dicom_path, ds, geometry)
        _atomic_write(geometry_path, (json.dumps(geometry, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(registry, ("\ufeff" + buffer.getvalue()).encode("utf-8"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "Размеченные")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-partial-hip", action="store_true")
    args = parser.parse_args()
    print(json.dumps(complete(args.root, args.apply, args.allow_partial_hip),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
