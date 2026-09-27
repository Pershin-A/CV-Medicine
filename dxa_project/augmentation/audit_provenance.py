"""Verify that reference rows and annotated copies point to the original DICOMs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import warnings

import pydicom


def audit(workspace: Path) -> dict:
    with (workspace / "Размеченные" / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    with (workspace / "dxa_project" / "outputs" / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    by_path = {row["relative_path"].replace("\\", "/"): row for row in manifest}
    report = {"label_rows": len(labels), "manifest_rows": len(manifest),
              "unique_label_paths": len({r["relative_path"] for r in labels}),
              "unique_manifest_paths": len(by_path), "pixel_equal": 0,
              "byte_equal": 0, "uid_equal": 0, "dimension_equal": 0,
              "manifest_path_correct": 0, "manifest_uid_correct": 0,
              "manifest_dimension_correct": 0, "geometry_dimension_correct": 0,
              "problems": []}
    for index, row in enumerate(labels, 1):
        rel = row["relative_path"].replace("\\", "/")
        original = workspace / "Исследования" / rel
        annotated = workspace / "Размеченные" / rel
        reference = by_path.get(rel)
        issues = []
        if reference is None or not original.is_file() or not annotated.is_file():
            report["problems"].append({"row": index, "path": rel,
                                      "issue": "missing reference or DICOM"})
            continue
        if Path(reference["path"]).resolve() == original.resolve():
            report["manifest_path_correct"] += 1
        else:
            issues.append("manifest source path")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
            a = pydicom.dcmread(str(original), force=True)
            b = pydicom.dcmread(str(annotated), force=True)
        if hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(annotated.read_bytes()).digest():
            report["byte_equal"] += 1
        if a.PixelData == b.PixelData:
            report["pixel_equal"] += 1
        else:
            issues.append("pixel data")
        if (str(a.StudyInstanceUID), str(a.SeriesInstanceUID), str(a.SOPInstanceUID)) == (str(b.StudyInstanceUID), str(b.SeriesInstanceUID), str(b.SOPInstanceUID)):
            report["uid_equal"] += 1
        else:
            issues.append("copy UIDs")
        if (int(a.Rows), int(a.Columns)) == (int(b.Rows), int(b.Columns)):
            report["dimension_equal"] += 1
        else:
            issues.append("copy dimensions")
        if str(a.StudyInstanceUID) == reference["study_uid"] and str(a.SOPInstanceUID) == reference["image_uid"]:
            report["manifest_uid_correct"] += 1
        else:
            issues.append("manifest UIDs")
        if int(a.Rows) == int(reference["rows"]) and int(a.Columns) == int(reference["columns"]):
            report["manifest_dimension_correct"] += 1
        else:
            issues.append("manifest dimensions")
        geometry_path = workspace / "Размеченные" / row.get("geometry_path", "")
        if geometry_path.is_file():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            if (geometry.get("image_height"), geometry.get("image_width")) == (int(a.Rows), int(a.Columns)):
                report["geometry_dimension_correct"] += 1
            else:
                issues.append("geometry dimensions")
        if issues:
            report["problems"].append({"row": index, "path": rel, "issues": issues})
    return report


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[2]
    result = audit(workspace)
    output = workspace / "dxa_project" / "outputs" / "roi_provenance_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v if k != "problems" else v[:20]
                      for k, v in result.items()}, ensure_ascii=False, indent=2))
