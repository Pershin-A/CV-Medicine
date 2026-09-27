"""Audit DICOM geometry metadata and nominal-scale ROI agreement."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pydicom

from dxa_project.augmentation.core import hip_roi_margins_mm, prepare_geometry
from dxa_project.augmentation.generate import SCANNER_NOMINAL_SPACING_MM, _dicom_spacing

GEOMETRY_TAGS = (
    "PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing",
    "PixelAspectRatio", "DetectorElementPhysicalSize",
    "FieldOfViewDimensions", "SpatialResolution", "ExposedArea",
)


def audit(workspace: Path) -> dict:
    report = {
        "scanner_nominal_mm_row_col": SCANNER_NOMINAL_SPACING_MM,
        "folders": {},
        "roi_reference_comparison": {},
    }
    for name in ("Исследования", "Размеченные"):
        root = workspace / name
        files = sorted(root.rglob("*.dcm"))
        spacing = Counter()
        tag_presence = Counter()
        dimensions = Counter()
        for path in files:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            value, basis = _dicom_spacing(ds)
            spacing[basis] += 1
            for keyword in GEOMETRY_TAGS:
                if getattr(ds, keyword, None) is not None:
                    tag_presence[keyword] += 1
            dimensions[(int(ds.Rows), int(ds.Columns))] += 1
        report["folders"][name] = {
            "dicom_count": len(files),
            "spacing_basis_counts": dict(spacing),
            "metadata_tag_presence": dict(tag_presence),
            "distinct_dimensions": len(dimensions),
            "min_rows": min((size[0] for size in dimensions), default=None),
            "max_rows": max((size[0] for size in dimensions), default=None),
            "min_cols": min((size[1] for size in dimensions), default=None),
            "max_cols": max((size[1] for size in dimensions), default=None),
        }
    labels_path = workspace / "Размеченные" / "labels.csv"
    with (workspace / "dxa_project" / "outputs" / "manifest.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        reference_by_path = {
            row["relative_path"].replace("\\", "/"): row
            for row in csv.DictReader(stream)
        }
    checked, mismatch = 0, []
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            if row.get("label", "").upper() != "LEG":
                continue
            side = row.get("side", "").upper()
            if side not in ("LEFT", "RIGHT"):
                continue
            reference_row = reference_by_path.get(
                row["relative_path"].replace("\\", "/"), {})
            reference = reference_row.get(f"{side.lower()}_hip_roi", "")
            if reference not in ("0", "1", "0.0", "1.0"):
                continue
            path = workspace / "Размеченные" / row.get("geometry_path", "")
            if not path.is_file():
                continue
            geometry = json.loads(path.read_text(encoding="utf-8"))
            geometry = prepare_geometry(geometry, "LEG_" + side)
            if geometry["hip"]["roi_box"] is None:
                continue
            margins = hip_roi_margins_mm(
                geometry, side, SCANNER_NOMINAL_SPACING_MM)
            nominal_bad = int(
                margins["top"] < 30 or margins["bottom"] < 30 or
                margins["lateral"] < 20)
            checked += 1
            if nominal_bad != int(float(reference)):
                mismatch.append({
                    "label_row": index, "relative_path": row["relative_path"],
                    "reference_violation": int(float(reference)),
                    "nominal_violation": nominal_bad,
                    "margins_cm": {key: round(value / 10, 3)
                                   for key, value in margins.items()},
                })
    report["roi_reference_comparison"] = {
        "checked": checked,
        "mismatch_count": len(mismatch),
        "mismatches": mismatch,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "outputs" / "spacing_audit.json")
    args = parser.parse_args()
    result = audit(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "folders": result["folders"],
        "roi_reference_comparison": {
            key: value for key, value in result["roi_reference_comparison"].items()
            if key != "mismatches"},
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
