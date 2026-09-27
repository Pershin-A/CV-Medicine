"""Count lesser trochanter masks on the annotated originals."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dxa_project.augmentation.core import build_lesser_trochanter_mask


REVIEW_ROWS = [24, 37, 54, 77, 78, 79, 81, 96, 104, 112, 119, 142,
               188, 192, 197, 202, 206, 213, 226, 227, 229, 258, 259,
               262, 263, 266, 276, 286, 293, 410, 455]


def audit(workspace: Path) -> dict:
    with (workspace / "Размеченные" / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    rows = []
    filled = 0
    with_traces = 0
    for index, row in enumerate(labels, 1):
        if row["label"] != "LEG":
            continue
        geometry = json.loads((workspace / "Размеченные" / row["geometry_path"]).read_text(encoding="utf-8"))
        layers = geometry["hip"]["lesser_trochanter_traces"]
        if not (layers["trochanter"] and layers["adjacent_bone"]):
            continue
        with_traces += 1
        area = int(build_lesser_trochanter_mask(
            geometry, "LEG_" + row["side"].upper()).sum())
        filled += area > 0
        rows.append({"number": index, "area_px": area,
                     "relative_path": row["relative_path"]})
    return {"with_both_traces": with_traces, "filled": filled,
            "zero": with_traces - filled,
            "review_rows": [r for r in rows if r["number"] in REVIEW_ROWS],
            "all_rows": rows}


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[2]
    result = audit(workspace)
    output = workspace / "dxa_project" / "outputs" / "trochanter_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "all_rows"},
                     ensure_ascii=False, indent=2))
