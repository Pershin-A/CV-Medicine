import csv
import json

from annotation_filters import (
    augmentation_index, build_summaries, geometry_path_for, matches_filter,
    summarize_geometry,
)


def test_missing_structure_filters_distinguish_spine_and_hip():
    spine = summarize_geometry(
        {"spine": {"iliac_crests": {"image_left": [2, 3], "image_right": None},
                   "disc_lines": []}}, "SPINE")
    assert matches_filter(spine, "spine_iliac_1")
    assert matches_filter(spine, "spine_no_disc_lines")
    assert matches_filter(spine, "any_incomplete")
    assert not matches_filter(spine, "hip_no_roi")
    hip = summarize_geometry(
        {"hip": {"roi_box": None,
                 "landmarks": {"greater_trochanter": [2, 3],
                               "femoral_neck": None, "ischial_bone": None},
                 "lesser_trochanter_traces": {
                     "trochanter": [{"id": "t", "points": [[1, 1], [2, 2]]}],
                     "adjacent_bone": []}}}, "LEG_RIGHT")
    assert matches_filter(hip, "hip_no_roi")
    assert matches_filter(hip, "hip_landmarks_incomplete")
    assert matches_filter(hip, "hip_incomplete_traces")
    assert not matches_filter(hip, "hip_no_trochanter")


def test_augmented_starter_geometry_and_saved_edit_priority(tmp_path):
    (tmp_path / "geometry").mkdir()
    source = tmp_path / "geometry" / "starter.json"
    source.write_text(json.dumps({"spine": {"iliac_crests": {
        "image_left": None, "image_right": None}, "disc_lines": []}}),
        encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "geometry_path", "region"])
        writer.writeheader()
        writer.writerow({"image_path": "images/SPINE/example.dcm",
                         "geometry_path": "geometry/starter.json", "region": "SPINE"})
    augmented = augmentation_index(manifest)
    rel = "SPINE/example.dcm"
    assert geometry_path_for(rel, tmp_path, augmented) == source
    summaries = build_summaries([rel], tmp_path, {}, augmented)
    assert matches_filter(summaries[rel], "spine_iliac_0")
    edited = tmp_path / "geometry" / (
        __import__("hashlib").sha256(rel.encode()).hexdigest() + ".json")
    edited.write_text(json.dumps({"spine": {"iliac_crests": {
        "image_left": [4, 5], "image_right": None}, "disc_lines": []}}),
        encoding="utf-8")
    assert geometry_path_for(rel, tmp_path, augmented) == edited
    summaries = build_summaries([rel], tmp_path, {}, augmented)
    assert matches_filter(summaries[rel], "spine_iliac_1")
