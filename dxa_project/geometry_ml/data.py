"""Original DICOMs and aligned spatial targets; no synthetic augmentation."""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image, ImageDraw
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from torch.utils.data import Dataset

from dxa_project.augmentation.core import prepare_geometry


REGIONS = ("SPINE", "LEG_LEFT", "LEG_RIGHT")
LANDMARKS = ("greater_trochanter", "femoral_neck", "ischial_bone")
CRESTS = ("image_left", "image_right")


@dataclass(frozen=True)
class Record:
    relative_path: str
    source_path: Path
    geometry_path: Path
    study: str
    region: str
    spacing_mm: tuple[float,float] = (1.05,.6)
    source_id: str = ''


def load_records(root: Path) -> list[Record]:
    with (root / "Размеченные" / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = {r["relative_path"].replace("\\", "/"): r for r in csv.DictReader(f)}
    with (root / "dxa_project" / "outputs" / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    records = []
    for row in manifest:
        rel = row["relative_path"].replace("\\", "/")
        label = labels.get(rel)
        if label is None:
            continue
        region = ("SPINE" if label["label"] == "SPINE" else
                  "LEG_" + label["side"] if label["label"] == "LEG" else "UNKNOWN")
        if region not in REGIONS:
            continue
        source = root / "Исследования" / rel
        geometry = root / "Размеченные" / label["geometry_path"]
        if not source.is_file() or not geometry.is_file():
            raise FileNotFoundError(rel)
        records.append(Record(rel, source, geometry,
                              row["reference_study_uid"], region))
    return records


def load_augmented_records(root: Path, augmented_root: Path,
                           originals: list[Record],annotations_root: Path | None = None) -> list[Record]:
    """Load generated DICOMs, retaining each source's original study group."""
    by_source = {record.relative_path: record.study for record in originals}
    with (augmented_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    edits={}
    if annotations_root is not None:
        with (annotations_root/'labels.csv').open(encoding='utf-8-sig',newline='') as f:
            edits={r['relative_path'].replace('\\','/'):r for r in csv.DictReader(f)}
    for row in rows:
        relative = row["source_relative_path"].replace("\\", "/")
        study = by_source.get(relative)
        if study is None:
            raise ValueError(f"Augmentation has an unknown source: {relative}")
        region = row["region"]
        if region not in REGIONS:
            raise ValueError(f"Unexpected augmented region: {region}")
        image = augmented_root / row["image_path"]
        geometry = augmented_root / row["geometry_path"]
        edit=edits.get(row['image_path'].removeprefix('images/'))
        if edit and edit.get('geometry_path'):
            geometry=annotations_root/edit['geometry_path']
        if not image.is_file() or not geometry.is_file():
            raise FileNotFoundError(f"Augmented image or geometry missing: {image}, {geometry}")
        records.append(Record(f"aug/{row['image_path']}", image, geometry, study, region,
                              (float(row['row_spacing_mm']),float(row['col_spacing_mm'])),relative))
    return records


def read_dicom_image(path: Path) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
        ds = pydicom.dcmread(str(path), force=True)
    raw = np.squeeze(ds.pixel_array)
    if raw.ndim != 2:
        raise ValueError(f"Expected a two-dimensional DICOM: {path}")
    image = np.asarray(apply_modality_lut(raw, ds), dtype=np.float32)
    try:
        image = np.asarray(apply_voi_lut(image, ds), dtype=np.float32)
    except (ValueError, TypeError):
        pass
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError(f"No finite pixels: {path}")
    lo, hi = np.percentile(image[finite], (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(image[finite].min()), float(image[finite].max())
    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 1 - image
    return image


def _gaussian(size: int, x: float, y: float, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)


class DxaDataset(Dataset):
    def __init__(self, records: list[Record], size: int = 256,
                 router_mode: bool = False):
        self.records = records
        self.size = size
        self.router_mode = router_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_dicom_image(record.source_path)
        h, w = image.shape
        if self.router_mode:
            resized=Image.fromarray((image*255).astype(np.uint8)).resize((self.size,self.size),Image.Resampling.BILINEAR)
            rgb=torch.from_numpy(np.repeat(np.asarray(resized,dtype=np.float32)[None],3,axis=0)/255)
            rgb=(rgb-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            return rgb,{'region':torch.tensor(REGIONS.index(record.region)),
                        'relative_path':record.relative_path,'study':record.study,'width':w,'height':h}
        geometry = json.loads(record.geometry_path.read_text(encoding="utf-8"))
        if (geometry["image_width"], geometry["image_height"]) != (w, h):
            raise ValueError(f"Geometry and DICOM dimensions differ: {record.relative_path}")
        geometry = prepare_geometry(geometry, record.region)
        flipped = record.region == "LEG_RIGHT" and not self.router_mode
        if flipped:
            image = np.fliplr(image).copy()
        scale = min(self.size / w, self.size / h)
        dw, dh = round(w * scale), round(h * scale)
        left, top = (self.size - dw) // 2, (self.size - dh) // 2
        canvas = Image.new("L", (self.size, self.size))
        pil = Image.fromarray((image * 255).astype(np.uint8))
        if self.router_mode:
            canvas.paste(pil.resize((self.size, self.size), Image.Resampling.BILINEAR))
        else:
            canvas.paste(pil.resize((dw, dh), Image.Resampling.BILINEAR), (left, top))
        rgb = torch.from_numpy(np.repeat(np.asarray(canvas, dtype=np.float32)[None], 3, axis=0) / 255)
        rgb = ((rgb - torch.tensor([.485, .456, .406])[:, None, None]) /
               torch.tensor([.229, .224, .225])[:, None, None])

        def point(p):
            x = w - 1 - p[0] if flipped else p[0]
            return left + x * scale, top + p[1] * scale

        s = self.size
        line_image = Image.new("L", (s, s))
        draw = ImageDraw.Draw(line_image)
        crests = np.zeros((2, s, s), dtype=np.float32)
        crest_present = np.zeros(2, dtype=np.float32)
        for line in geometry["spine"]["disc_lines"]:
            a, b = [point(p) for p in line["points"]]
            draw.line((*a, *b), fill=255, width=2)
        for k, name in enumerate(CRESTS):
            key = CRESTS[1-k] if flipped else name
            p = geometry["spine"]["iliac_crests"][key]
            if p is not None:
                x, y = point(p)
                crests[k] = _gaussian(s, x, y)
                crest_present[k] = 1
        boxes = []
        for obj in geometry["spine"]["foreign_objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            a, b = point((x1, y1)), point((x2, y2))
            boxes.append([min(a[0], b[0]), min(a[1], b[1]),
                          max(a[0], b[0]), max(a[1], b[1])])

        hip_points = np.zeros((3, s, s), dtype=np.float32)
        hip_present = np.zeros(3, dtype=np.float32)
        for k, name in enumerate(LANDMARKS):
            p = geometry["hip"]["landmarks"][name]
            if p is not None:
                x, y = point(p)
                hip_points[k] = _gaussian(s, x, y)
                hip_present[k] = 1
        roi = np.zeros(3, dtype=np.float32)
        roi_present = 0.0
        box = geometry["hip"]["roi_box"]
        if box is not None:
            x1, y1, x2, y2 = box
            lateral = w - 1 - x1 if flipped else x2
            roi = np.asarray(((top + y1 * scale) / s,
                              (top + y2 * scale) / s,
                              (left + lateral * scale) / s), dtype=np.float32)
            roi_present = 1.0
        trochanter = Image.new("L", (s, s))
        td = ImageDraw.Draw(trochanter)
        for p in geometry["hip"].get("lesser_trochanter_pixels", []):
            x, y = point(p)
            td.point((round(x), round(y)), fill=255)
        return rgb, {
            "region": torch.tensor(REGIONS.index(record.region)),
            "line": torch.from_numpy(np.asarray(line_image, dtype=np.float32)[None] / 255),
            "crest": torch.from_numpy(crests),
            "crest_present": torch.from_numpy(crest_present),
            "artifact_boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "hip_points": torch.from_numpy(hip_points),
            "hip_present": torch.from_numpy(hip_present),
            "roi": torch.from_numpy(roi),
            "roi_present": torch.tensor(roi_present),
            "trochanter": torch.from_numpy(np.asarray(trochanter, dtype=np.float32)[None] / 255),
            "relative_path": record.relative_path,
            "study": record.study,
            "width": w, "height": h, "scale": scale, "pad_left": left,
            "pad_top": top, "flipped": flipped,
            "spacing_mm":record.spacing_mm,
        }


def collate(items):
    images, targets = zip(*items)
    return torch.stack(images), list(targets)
