import csv
import json

import numpy as np
import pydicom
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage

from labeler.complete_annotations import complete
from labeler.geometry import empty_geometry


def _dicom(path, geometry):
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = ds.Columns = 100
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = np.arange(10000, dtype=np.uint16)
    ds.PixelData = pixels.tobytes()
    block = ds.private_block(0x0011, "DXA_MANUAL_LABELER", create=True)
    ds.add_new(block.get_tag(0x09), "UT", json.dumps(geometry, ensure_ascii=True))
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return pixels.tobytes()


def test_completion_flags_update_json_csv_and_dicom(tmp_path):
    root = tmp_path / "Размеченные"
    (root / "geometry").mkdir(parents=True)
    rows = []
    for index, region in enumerate(("SPINE", "LEG")):
        g = empty_geometry(100, 100)
        if region == "SPINE":
            for n, y in enumerate((10, 25, 40, 55, 70)):
                g["spine"]["disc_lines"].append({"id": str(n), "points": [[20, y], [80, y]]})
        else:
            g["hip"]["landmarks"] = {"greater_trochanter": [20, 20],
                                         "femoral_neck": [40, 25], "ischial_bone": [70, 40]}
            g["hip"]["roi_box"] = [10, 10, 85, 80]
            g["hip"]["lesser_trochanter_traces"] = {
                "trochanter": [{"id": "t", "points": [[20, 40], [30, 40]]}],
                "adjacent_bone": [{"id": "b", "points": [[20, 45], [30, 45]]}]}
        gp = root / "geometry" / f"{index}.json"
        gp.write_text(json.dumps(g), encoding="utf-8")
        image = root / f"{index}.dcm"
        expected_pixels = _dicom(image, g)
        rows.append({"relative_path": image.name, "output_path": image.name,
                     "geometry_path": f"geometry/{index}.json", "label": region,
                     "geometry_spine_complete": "0", "geometry_hip_complete": "0"})
    with (root / "labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert complete(root)["planned_updates"] == 2
    assert complete(root, apply=True)["planned_updates"] == 2
    assert complete(root)["planned_updates"] == 0
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        updated = list(csv.DictReader(stream))
    assert updated[0]["geometry_spine_complete"] == "1"
    assert updated[1]["geometry_hip_complete"] == "1"
    for index, region in enumerate(("spine", "hip")):
        geometry = json.loads((root / "geometry" / f"{index}.json").read_text())
        ds = pydicom.dcmread(root / f"{index}.dcm")
        block = ds.private_block(0x0011, "DXA_MANUAL_LABELER", create=False)
        raw = ds[block.get_tag(0x09)].value
        if isinstance(raw, bytes):
            raw = raw.decode().rstrip("\x00")
        assert json.loads(raw) == geometry
        assert geometry["complete"][region]
        assert ds.PixelData == expected_pixels
