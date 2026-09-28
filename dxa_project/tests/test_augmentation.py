"""Synthetic checks only: never reads or writes real hackathon images."""
import random
import csv
import json

import numpy as np
import pytest

from dxa_project.augmentation.core import (Transform, hip_position_ok, hip_roi_ok,
                               lesser_trochanter_between_area, prepare_geometry,
                               physical_axis_angle, pixel_axis_angle,
                               spine_axis_angle, spine_position_ok,
                               transform_geometry, transformed_spacing, warp_image)
from dxa_project.augmentation.generate import (_check_group, _hip_variant,
                                               _labels, _roi_proxy_spacing,
                                               _source_spacing, _spine_variant,
                                               _write_dicom, generate)
from labeler.geometry import empty_geometry, validate_geometry


def spine_fixture():
    g = empty_geometry(300, 300)
    for idx, y in enumerate((22, 67, 112, 157, 202, 247)):
        g["spine"]["disc_lines"].append({"id": f"d{idx}",
                                           "points": [[90, y], [210, y]]})
    g["spine"]["iliac_crests"] = {"image_left": [40, 265],
                                    "image_right": [260, 265]}
    g["spine"]["foreign_objects"] = [
        {"id": "a", "kind": "metal", "bbox": [5, 5, 15, 15]}]
    g["complete"]["spine"] = True
    return validate_geometry(g, 300, 300)


def hip_fixture():
    g = empty_geometry(300, 300)
    g["hip"]["landmarks"] = {
        "greater_trochanter": [70, 100], "femoral_neck": [150, 115],
        "ischial_bone": [220, 170]}
    g["hip"]["roi_box"] = [40, 45, 235, 255]
    g["hip"]["lesser_trochanter"] = [[55, 120], [70, 120], [70, 135], [55, 135]]
    g["complete"]["hip"] = True
    return validate_geometry(g, 300, 300)


def test_spine_positive_and_top_crop_removes_disc_and_artifact():
    g = spine_fixture()
    assert spine_position_ok(g)
    t = Transform(300, 300, 1.5, 0, 150, 115)
    assert t.covers_output()
    moved, info = transform_geometry(g, t)
    assert not spine_position_ok(moved)
    assert info["dropped"]["disc_lines"] >= 1
    assert info["dropped"]["foreign_objects"] == 1
    assert len(moved["spine"]["disc_lines"]) < 6


def test_side_crop_removes_crest_and_keeps_original_dimensions():
    g = spine_fixture()
    t = Transform(300, 300, 1.55, 0, 195, 150)
    assert t.covers_output()
    moved, _ = transform_geometry(g, t)
    assert moved["image_width"] == 300 and moved["image_height"] == 300
    assert moved["spine"]["iliac_crests"]["image_left"] is None
    assert not spine_position_ok(moved)


def test_axis_uses_brightness_and_rotation_label_boundary():
    g = spine_fixture()
    y, x = np.mgrid[:300, :300]
    image = (1000 + 500 * np.exp(-((x - 150) / 25) ** 2)).astype(np.uint16)
    angle = spine_axis_angle(image, g)
    assert angle is not None and abs(angle) < 1
    for desired, expected in ((3, 0), (8, 1), (-12, 1)):
        t = Transform(300, 300, 1.4, -desired)
        assert t.covers_output()
        moved_image = warp_image(image, t)
        moved_geometry, _ = transform_geometry(g, t)
        actual = spine_axis_angle(moved_image, moved_geometry)
        assert actual is not None
        assert int(abs(actual) > 5) == expected


def test_spine_reflection_changes_axis_sign_and_swaps_crest_names():
    g = spine_fixture()
    g["spine"]["iliac_crests"] = {"image_left": [35, 260],
                                      "image_right": [240, 265]}
    y, x = np.mgrid[:300, :300]
    image = (1000 + 500 * np.exp(-((x - (145 + .12 * (y - 150))) / 25) ** 2)).astype(np.uint16)
    original = spine_axis_angle(image, g)
    t = Transform(300, 300, reflect_x=True)
    assert t.covers_output()
    moved_image = warp_image(image, t)
    moved, _ = transform_geometry(g, t, region="SPINE")
    assert moved["spine"]["iliac_crests"]["image_left"][0] == pytest.approx(59)
    assert moved["spine"]["iliac_crests"]["image_right"][0] == pytest.approx(264)
    reflected = spine_axis_angle(moved_image, moved)
    assert original is not None and reflected is not None
    assert reflected == pytest.approx(-original, abs=1.5)


def test_spine_target_angle_ranges_and_normalization():
    rng = random.Random(91)
    g = spine_fixture()
    for group in ("positive", "negative_axis_or_roi"):
        for _ in range(100):
            t, variant = _spine_variant(g, group, rng, 9.0)
            target = float(variant.rsplit("_", 1)[-1])
            intended = (-9.0 if t.reflect_x else 9.0) - t.angle_deg
            assert target == pytest.approx(intended, abs=.01)
            assert t.physical_spacing_mm == (1.05, .6)
            if group == "positive":
                assert -4 <= target <= 4
            else:
                assert -10 <= target <= -6 or 6 <= target <= 10


def test_physical_angle_roundtrip_uses_anisotropic_spacing():
    for angle in (-15, -6, -4, 0, 4, 6, 15):
        pixel = pixel_axis_angle(angle, (1.05, .6))
        assert physical_axis_angle(pixel, (1.05, .6)) == pytest.approx(angle)
    assert pixel_axis_angle(15, (1.05, .6)) > 15


def test_physical_rotation_preserves_physical_lengths():
    transform = Transform(300, 300, angle_deg=15,
                          physical_spacing_mm=(1.05, .6))
    for vector in ((20, 0), (0, 20), (13, 27)):
        before = np.linalg.norm(np.asarray(vector) * (.6, 1.05))
        after = np.linalg.norm((transform.matrix @ vector) * (.6, 1.05))
        assert after == pytest.approx(before)


def test_hip_roi_and_position_checked_together():
    g = hip_fixture()
    spacing = _roi_proxy_spacing(g, "LEFT")
    assert hip_position_ok(g)
    assert hip_roi_ok(g, True, "LEFT", spacing)
    t = Transform(300, 300, 1.45, 0, 174, 150)
    assert t.covers_output()
    moved, info = transform_geometry(g, t)
    new_spacing = transformed_spacing(spacing, t)
    assert not hip_position_ok(moved)
    assert not hip_roi_ok(moved, info["roi_fully_visible"], "LEFT", new_spacing)
    assert _check_group("LEG_LEFT", "negative_position",
                        {"hip_position": 1, "hip_roi": 1})


def test_missing_spacing_is_unknown_not_positive():
    g = hip_fixture()
    assert hip_roi_ok(g, True, "LEFT", None) is None
    assert not _check_group("LEG_LEFT", "positive",
                            {"hip_position": 0, "hip_roi": None})


def test_no_extrapolation_for_rotation():
    t = Transform(300, 300, 1.02, 20)
    assert not t.covers_output()
    with np.testing.assert_raises(ValueError):
        warp_image(np.zeros((300, 300), dtype=np.uint16), t)


def test_spine_lines_extend_to_frame_and_seven_lines_are_allowed():
    g = spine_fixture()
    g["spine"]["disc_lines"].append({"id": "d6", "points": [[90, 292], [210, 292]]})
    prepared = prepare_geometry(g, "SPINE")
    assert spine_position_ok(prepared)
    assert g["spine"]["disc_lines"][0]["points"] == [[90, 22], [210, 22]]
    assert all(line["points"][0][0] == 0 and line["points"][1][0] == 299
               for line in prepared["spine"]["disc_lines"])


def test_roi_arbitrary_edge_is_extended_and_can_be_clipped():
    for region, side, arbitrary_index in (("LEG_LEFT", "LEFT", 0),
                                          ("LEG_RIGHT", "RIGHT", 2)):
        g = hip_fixture()
        prepared = prepare_geometry(g, region)
        assert prepared["hip"]["roi_box"][arbitrary_index] == (0 if side == "LEFT" else 299)
        assert g["hip"]["roi_box"] == [40, 45, 235, 255]
        spacing = _roi_proxy_spacing(prepared, side)
        assert hip_roi_ok(prepared, True, side, spacing)
        t = Transform(300, 300, 1.12)
        moved, info = transform_geometry(prepared, t, region=region)
        assert info["roi_fully_visible"]
        assert moved["hip"]["roi_box"][arbitrary_index] == (0 if side == "LEFT" else 299)


def test_trochanter_pixels_need_overlap_in_height_not_curve_crossings():
    g = hip_fixture()
    g["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [{"id": "t", "points": [[100, 40], [118, 60], [100, 80]]}],
        "adjacent_bone": [{"id": "b", "points": [[98, 30], [98, 90]]}],
    }
    prepared = prepare_geometry(g, "LEG_LEFT")
    area, crossings = lesser_trochanter_between_area(prepared)
    assert crossings == 0 and area > 0
    assert len(prepared["hip"]["lesser_trochanter_pixels"]) == area
    assert g["hip"]["lesser_trochanter_pixels"] == []
    zoom = Transform(300, 300, 2, 0, 150, 150)
    assert zoom.covers_output()
    moved, _ = transform_geometry(prepared, zoom, region="LEG_LEFT")
    assert 0 < len(moved["hip"]["lesser_trochanter_pixels"]) < area * 4
    assert moved["hip"]["lesser_trochanter_partial"]
    assert moved["hip"]["lesser_trochanter_traces"] != {}
    g["hip"]["lesser_trochanter_traces"]["adjacent_bone"][0]["points"] = [[90, 160], [98, 200]]
    assert lesser_trochanter_between_area(g) == (0, 0)


def test_rotation_copies_source_if_pixels_remain_otherwise_violation():
    g = hip_fixture()
    g["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [{"id": "t", "points": [[96, 40], [78, 60], [96, 80]]}],
        "adjacent_bone": [{"id": "b", "points": [[98, 30], [98, 90]]}],
    }
    prepared = prepare_geometry(g, "LEG_RIGHT")
    source = {"right_hip_rotation": "0.0"}
    image = np.zeros((300, 300), dtype=np.uint16)
    labels = _labels("LEG_RIGHT", prepared, {"roi_fully_visible": True},
                     image, (1, 1), source)
    assert labels["hip_rotation"] == 0
    prepared["hip"]["lesser_trochanter_pixels"] = []
    prepared["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [], "adjacent_bone": []}
    labels = _labels("LEG_RIGHT", prepared, {"roi_fully_visible": True},
                     image, (1, 1), source)
    assert labels["hip_rotation"] == 1


def test_inward_contour_has_no_trochanter_pixels_but_keeps_contour():
    g = hip_fixture()
    g["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [{"id": "t", "points": [[96, 40], [90, 60], [96, 80]]}],
        "adjacent_bone": [{"id": "b", "points": [[98, 30], [98, 90]]}],
    }
    prepared = prepare_geometry(g, "LEG_LEFT")
    assert prepared["hip"]["lesser_trochanter_pixels"] == []
    assert prepared["hip"]["lesser_trochanter_mask_ready"]
    moved, _ = transform_geometry(prepared, Transform(300, 300, 2, 0, 150, 150),
                                  region="LEG_LEFT")
    assert moved["hip"]["lesser_trochanter_pixels"] == []
    assert moved["hip"]["lesser_trochanter_traces"]["trochanter"]
    assert _labels("LEG_LEFT", moved, {"roi_fully_visible": True},
                   np.zeros((300, 300)), (1.05 / 2, 0.6 / 2),
                   {"left_hip_rotation": "0"})["hip_rotation"] == 1


def test_small_interior_widening_is_filled_without_two_local_minima():
    g = hip_fixture()
    g["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [{"id": "t", "points": [[97, 40], [100, 55],
                                           [100, 60], [97, 75]]}],
        "adjacent_bone": [{"id": "b", "points": [[95, 35], [95, 80]]}],
    }
    prepared = prepare_geometry(g, "LEG_LEFT")
    rows = [point[1] for point in prepared["hip"]["lesser_trochanter_pixels"]]
    assert rows and 39 <= min(rows) < 55 and 60 < max(rows) <= 75


def test_partial_visible_trochanter_forces_rotation_violation():
    g = hip_fixture()
    g["hip"]["lesser_trochanter_traces"] = {
        "trochanter": [{"id": "t", "points": [[100, 40], [118, 60], [100, 80]]}],
        "adjacent_bone": [{"id": "b", "points": [[98, 30], [98, 90]]}],
    }
    prepared = prepare_geometry(g, "LEG_LEFT")
    moved, _ = transform_geometry(prepared, Transform(300, 300, 2, 0, 150, 150),
                                  region="LEG_LEFT")
    assert len(moved["hip"]["lesser_trochanter_pixels"]) > 0
    assert moved["hip"]["lesser_trochanter_partial"]
    prepared_again = prepare_geometry(moved, "LEG_LEFT")
    assert prepared_again["hip"]["lesser_trochanter_pixels"] == moved["hip"]["lesser_trochanter_pixels"]
    assert prepared_again["hip"]["lesser_trochanter_partial"]
    labels = _labels("LEG_LEFT", moved, {"roi_fully_visible": True},
                     np.zeros((300, 300)), (1.05 / 2, 0.6 / 2),
                     {"left_hip_rotation": "0"})
    assert labels["hip_rotation"] == 1


def test_nominal_spacing_and_strict_axis_intervals():
    from pydicom.dataset import Dataset
    spacing, basis = _source_spacing(Dataset())
    assert spacing == (1.05, 0.6)
    assert basis == "scanner_nominal_user_supplied"
    assert _source_spacing(Dataset(), allow_nominal=False) == (None, "missing")
    for angle in (-10, -6, 6, 10):
        assert _check_group("SPINE", "negative_axis_or_roi",
                            {"spine_position": 0, "spine_axis": 1,
                             "spine_axis_angle_deg": angle})
    for angle in (-5, 4.6, 4.1, 5):
        assert not _check_group("SPINE", "negative_axis_or_roi",
                                {"spine_position": 0, "spine_axis": 0,
                                 "spine_axis_angle_deg": angle})
    for angle in (-20, -15, -5, 5, 15, 20):
        assert not _check_group("SPINE", "negative_axis_or_roi",
                                {"spine_position": 0, "spine_axis": 1,
                                 "spine_axis_angle_deg": angle})


def test_candidate_sampler_reaches_each_requested_group():
    rng = random.Random(17)
    spine = spine_fixture()
    y, x = np.mgrid[:300, :300]
    image = (1000 + 500 * np.exp(-((x - 150) / 25) ** 2)).astype(np.uint16)
    hip = hip_fixture()
    hip_image = np.zeros((300, 300), dtype=np.uint16)
    for region, geometry, pixels, spacing in (
        ("SPINE", spine, image, None),
        ("LEG_LEFT", hip, hip_image, _roi_proxy_spacing(hip, "LEFT")),
    ):
        for group in ("positive", "negative_position", "negative_axis_or_roi"):
            successes = 0
            for _ in range(150):
                if region == "SPINE":
                    t, _ = _spine_variant(geometry, group, rng, 0)
                else:
                    t, _ = _hip_variant(geometry, group, rng, "LEFT")
                if not t.covers_output():
                    continue
                moved, info = transform_geometry(geometry, t, region=region)
                labels = _labels(region, moved, info, warp_image(pixels, t),
                                 transformed_spacing(spacing, t), {"spine_artifact": "0"})
                successes += _check_group(region, group, labels)
            assert successes >= 2, (region, group, successes)


def test_synthetic_dicom_roundtrip(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    if not hasattr(pydicom, "Dataset"):
        pytest.skip("pydicom is not available in this sandbox")
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    ds = pydicom.Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.PatientName = "Synthetic"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = ds.Columns = 100
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    image = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    t = Transform(100, 100, 1.2)
    expected = warp_image(image, t)
    destination = tmp_path / "synthetic.dcm"
    _write_dicom(ds, expected, t, destination, "synthetic-seed",
                 {"hip": {"roi_box": [0, 2, 10, 20]}},
                 {"hip_roi": 1}, "LEG_LEFT")
    reread = pydicom.dcmread(destination)
    np.testing.assert_array_equal(reread.pixel_array, expected)
    assert reread.pixel_array.shape == image.shape
    assert tuple(float(v) for v in reread.PixelSpacing) == pytest.approx((0.5 / 1.2,) * 2)
    assert reread.SOPInstanceUID != ds.SOPInstanceUID
    block = reread.private_block(0x0011, "DXA_MANUAL_LABELER")
    assert json.loads(reread[block.get_tag(0x09)].value)["hip"]["roi_box"] == [0, 2, 10, 20]
    assert json.loads(reread[block.get_tag(0x0A)].value)["hip_roi"] == 1


def test_end_to_end_synthetic_generation(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    if not hasattr(pydicom, "Dataset"):
        pytest.skip("pydicom is not available in this sandbox")
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    labels_root = tmp_path / "Размеченные"
    geometry_dir = labels_root / "geometry"
    geometry_dir.mkdir(parents=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_rows, label_rows = [], []
    y, x = np.mgrid[:300, :300]
    for index, (region, side, geometry) in enumerate((
        ("SPINE", "", spine_fixture()),
        ("LEG", "LEFT", hip_fixture()),
        ("LEG", "RIGHT", hip_fixture()),
    )):
        ds = pydicom.Dataset()
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = ds.Columns = 300
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        pixels = (1000 + 500 * np.exp(-((x - 150) / 25) ** 2)).astype(np.uint16)
        ds.PixelData = pixels.tobytes()
        source_file = source_dir / f"image_{index}.dcm"
        pydicom.dcmwrite(source_file, ds, enforce_file_format=True)
        geometry_file = geometry_dir / f"image_{index}.json"
        geometry_file.write_text(json.dumps(geometry), encoding="utf-8")
        rel = f"study_{index}/image.dcm"
        manifest_rows.append({"relative_path": rel, "path": str(source_file),
                              "study_uid": ds.StudyInstanceUID, "rows": 300,
                              "columns": 300, "spine_artifact": "0"})
        label_rows.append({"relative_path": rel, "label": region, "side": side,
                           "geometry_path": f"geometry/{geometry_file.name}"})
    manifest_file = tmp_path / "manifest.csv"
    for path, rows in ((manifest_file, manifest_rows),
                       (labels_root / "labels.csv", label_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output = tmp_path / "augmented_dataset"
    report = generate(tmp_path, manifest_file, output,
                      target_per_group={"positive": 1, "negative_position": 1,
                                        "negative_axis_or_roi": 1},
                      max_attempts_per_image=150,workers=2)
    assert report["total_generated"] == 9, report
    with (output / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row["region"] for row in rows} == {"SPINE", "LEG_LEFT", "LEG_RIGHT"}
    for row in rows:
        assert (output / row["image_path"]).is_file()
        assert (output / row["geometry_path"]).is_file()
        assert pydicom.dcmread(output / row["image_path"]).pixel_array.shape == (300, 300)
