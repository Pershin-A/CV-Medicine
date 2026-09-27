import csv
import json
from pathlib import Path

import pytest

from labeler.merge_annotation_folders import build_plan, merge, parse_ranges, safe_relative


def make_folder(root: Path, marker: str, studies=("A", "A", "B", "B", "C")):
    root.mkdir()
    (root / "geometry").mkdir()
    rows = []
    for index, study in enumerate(studies, 1):
        relative = f"{study}/scan_{index}.dcm"
        file = root / relative
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(f"DICOM {marker} {index}".encode())
        geometry = f"geometry/scan_{index}.json"
        (root / geometry).write_text(json.dumps({"from": marker, "index": index}),
                                     encoding="utf-8")
        rows.append({"relative_path": relative, "output_path": relative,
                     "label": f"{marker}_{index}", "geometry_path": geometry,
                     "geometry_spine_complete": "1"})
    with (root / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_image_ranges_copy_matching_dicom_row_and_geometry(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "merged"))
    make_folder(first, "ONE")
    make_folder(second, "TWO")
    report = merge(first, second, output, parse_ranges(["2-3,5"]))
    assert report["from_first"] == 3
    assert report["from_second"] == 2
    rows = read_rows(output / "labels.csv")
    assert [row["label"] for row in rows] == ["TWO_1", "ONE_2", "ONE_3", "TWO_4", "ONE_5"]
    for index, row in enumerate(rows, 1):
        marker = "ONE" if index in (2, 3, 5) else "TWO"
        assert (output / row["output_path"]).read_bytes() == f"DICOM {marker} {index}".encode()
        assert json.loads((output / row["geometry_path"]).read_text())["from"] == marker
    assert read_rows(output / "merge_sources.csv")[0]["origin"] == "second"
    assert (first / "labels.csv").exists() and (second / "labels.csv").exists()


def test_study_ranges_select_all_scans_in_selected_study(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    make_folder(first, "ONE")
    make_folder(second, "TWO")
    plan, _, report = build_plan(first, second, parse_ranges(["2"]), "study")
    assert report["first_units"] == 3
    assert [item["origin"] for item in plan] == ["second", "second", "first", "first", "second"]


def test_dry_run_writes_nothing_and_empty_output_is_allowed(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "merged"))
    make_folder(first, "ONE")
    make_folder(second, "TWO")
    report = merge(first, second, output, parse_ranges(["1"]), dry_run=True)
    assert report["total_scans"] == 5
    assert not output.exists()
    output.mkdir()
    merge(first, second, output, parse_ranges(["1"]), dry_run=True)
    merge(first, second, output, parse_ranges(["1"]))
    assert (output / "labels.csv").exists()
    with pytest.raises(FileExistsError):
        merge(first, second, output, parse_ranges(["1"]))


@pytest.mark.parametrize("value", ["../x.dcm", "/tmp/x.dcm", "C:/x.dcm", "a//b.dcm", "a/./b.dcm"])
def test_unsafe_paths_rejected(value):
    with pytest.raises(ValueError):
        safe_relative(value, "relative_path")


def test_missing_selected_file_prevents_output(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "merged"))
    make_folder(first, "ONE")
    make_folder(second, "TWO")
    (first / "A" / "scan_1.dcm").unlink()
    with pytest.raises(FileNotFoundError):
        merge(first, second, output, parse_ranges(["1"]))
    assert not output.exists()


def test_range_validation(tmp_path):
    with pytest.raises(ValueError):
        parse_ranges(["0-3"])
    with pytest.raises(ValueError):
        parse_ranges(["4-2"])
    first, second = tmp_path / "first", tmp_path / "second"
    make_folder(first, "ONE")
    make_folder(second, "TWO")
    with pytest.raises(ValueError, match="exceeds"):
        build_plan(first, second, parse_ranges(["6"]))
