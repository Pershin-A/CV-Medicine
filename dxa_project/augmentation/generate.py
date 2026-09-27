"""Generate geometry-aware DXA DICOM augmentations after annotation is complete.

This module has no import-time side effects. Run `python -m augmentation.generate --help`
from dxa_project/ to inspect options; do not run against the dataset prematurely.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter, defaultdict

import numpy as np

from .core import (Transform, hip_position_ok, hip_roi_ok, spine_axis_angle,
                   spine_position_ok, transform_geometry, transformed_spacing,
                   warp_image)


REGIONS = ("SPINE", "LEG_LEFT", "LEG_RIGHT")
GROUPS = ("positive", "negative_position", "negative_axis_or_roi")
TARGET_PER_GROUP = {"positive": 500, "negative_position": 250,
                    "negative_axis_or_roi": 250}


def _region(label: str, side: str) -> str | None:
    if label == "SPINE":
        return "SPINE"
    if label == "LEG" and side in ("LEFT", "RIGHT"):
        return f"LEG_{side}"
    return None


def _read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_sources(workspace: Path, manifest_path: Path):
    manifest = {row["relative_path"].replace("\\", "/"): row
                for row in _read_rows(manifest_path)}
    entries = {}
    quarantined = set()
    problems = Counter()
    for root in sorted(workspace.glob("Размеченные*")):
        registry = root / "labels.csv"
        if not registry.is_file():
            continue
        for row in _read_rows(registry):
            rel = row.get("relative_path", "").replace("\\", "/")
            if rel in quarantined:
                continue
            region = _region(row.get("label", ""), row.get("side", ""))
            geometry_name = row.get("geometry_path", "")
            if not rel or not region or not geometry_name or rel not in manifest:
                continue
            geometry_path = root / geometry_name
            if not geometry_path.is_file():
                problems["missing_geometry_sidecar"] += 1
                continue
            item = manifest[rel]
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            key = "spine" if region == "SPINE" else "hip"
            if not geometry.get("complete", {}).get(key):
                problems["incomplete_geometry"] += 1
                continue
            from .core import validate_geometry
            try:
                geometry = validate_geometry(geometry, int(item["columns"]), int(item["rows"]))
            except (ValueError, KeyError, TypeError):
                problems["invalid_geometry"] += 1
                continue
            if rel in entries:
                if entries[rel]["geometry"] != geometry or entries[rel]["region"] != region:
                    problems["conflicting_complete_annotations"] += 1
                    entries.pop(rel, None)
                    quarantined.add(rel)
                continue
            entries[rel] = {"source": item, "geometry": geometry, "region": region}
    return list(entries.values()), problems


def _dicom_spacing(ds):
    for key in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        raw = getattr(ds, key, None)
        if raw is not None:
            try:
                values = tuple(float(v) for v in raw)
                if len(values) == 2 and all(math.isfinite(v) and v > 0 for v in values):
                    return values, key
            except (TypeError, ValueError):
                pass
    return None, "missing"


def _roi_proxy_spacing(geometry: dict, side: str):
    """Weak fallback from a *correctly drawn* reference ROI, with 10% slack."""
    box = geometry["hip"]["roi_box"]
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w, h = geometry["image_width"], geometry["image_height"]
    lateral_px = w - 1 - x2 if side == "LEFT" else x1
    if min(y1, h - 1 - y2, lateral_px) <= 0:
        return None
    return (1.1 * max(30 / y1, 30 / (h - 1 - y2)),
            1.1 * 20 / lateral_px)


def _spine_variant(geometry, group, rng, base_angle):
    w, h = geometry["image_width"], geometry["image_height"]
    lines = sorted(geometry["spine"]["disc_lines"],
                   key=lambda l: sum(p[1] for p in l["points"]) / 2)
    mids = [np.mean(l["points"], axis=0) for l in lines]
    gap = float(np.median(np.diff([m[1] for m in mids])))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    if group == "negative_axis_or_roi":
        target = rng.choice((rng.uniform(-20, -6), rng.uniform(-4, 4),
                             rng.uniform(6, 20)))
        angle = base_angle - target
        scale = max(1.02, abs(math.cos(math.radians(angle)))
                    + abs(math.sin(math.radians(angle))) + 0.025)
        scale += rng.uniform(0, 0.06)
        return Transform(w, h, scale, angle), f"axis_target_{target:.2f}"
    if group == "positive":
        target = rng.uniform(-4, 4)
        angle = base_angle - target
        scale = max(1.01, abs(math.cos(math.radians(angle)))
                    + abs(math.sin(math.radians(angle))) + 0.025)
        scale += rng.uniform(0, 0.05)
        desired_y = rng.uniform(0.35, 0.65) * gap * scale
        src_y = float(mids[0][1] + (cy - desired_y) /
                      (scale * math.cos(math.radians(angle))))
        return Transform(w, h, scale, angle, cx, src_y), f"half_Th12_axis_{target:.2f}"
    variant = rng.choice(("crop_Th12", "crop_Th12_and_next",
                          "crop_left_crest", "crop_right_crest", "crop_both_crests_bottom"))
    scale = rng.uniform(1.18, 1.65)
    src_x, src_y = cx, cy
    if variant == "crop_Th12":
        src_y = float(mids[0][1] + (cy - rng.uniform(-0.3, 0.15) * gap * scale) / scale)
    elif variant == "crop_Th12_and_next" and len(mids) >= 2:
        src_y = float(mids[1][1] + (cy - rng.uniform(-0.25, 0.2) * gap * scale) / scale)
    elif variant in ("crop_left_crest", "crop_right_crest"):
        key = "image_left" if variant == "crop_left_crest" else "image_right"
        p = geometry["spine"]["iliac_crests"][key]
        if p is not None:
            desired = -rng.uniform(2, 20) if key == "image_left" else w - 1 + rng.uniform(2, 20)
            src_x = p[0] + (cx - desired) / scale
    else:
        crests = list(geometry["spine"]["iliac_crests"].values())
        if all(p is not None for p in crests):
            y_top = min(p[1] for p in crests)
            src_y = y_top + (cy - (h + rng.uniform(2, 20))) / scale
    return Transform(w, h, scale, 0, src_x, src_y), variant


def _hip_variant(geometry, group, rng, side):
    w, h = geometry["image_width"], geometry["image_height"]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    box = geometry["hip"]["roi_box"]
    points = list(geometry["hip"]["landmarks"].values())
    if group == "positive":
        return Transform(w, h, rng.uniform(1.01, 1.06), 0,
                         cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)), "hip_safe_zoom"
    if group == "negative_position":
        p = rng.choice(points)
        edge = rng.choice(("top", "bottom", "left", "right"))
        target = {"top": (p[0], -rng.uniform(2, 15)),
                  "bottom": (p[0], h + rng.uniform(2, 15)),
                  "left": (-rng.uniform(2, 15), p[1]),
                  "right": (w + rng.uniform(2, 15), p[1])}[edge]
        scale = rng.uniform(1.18, 1.8)
        return Transform(w, h, scale, 0,
                         p[0] + (cx - target[0]) / scale,
                         p[1] + (cy - target[1]) / scale), f"landmark_{edge}"
    x1, y1, x2, y2 = box
    edge = rng.choice(("top", "bottom", "lateral"))
    anchor = {"top": (cx, y1), "bottom": (cx, y2),
              "lateral": (x2 if side == "LEFT" else x1, cy)}[edge]
    scale = rng.uniform(1.12, 1.55)
    if edge == "top":
        target = (cx, rng.uniform(1, 20))
    elif edge == "bottom":
        target = (cx, h - 1 - rng.uniform(1, 20))
    else:
        target = (w - 1 - rng.uniform(1, 15) if side == "LEFT" else rng.uniform(1, 15), cy)
    return Transform(w, h, scale, 0,
                     anchor[0] + (cx - target[0]) / scale,
                     anchor[1] + (cy - target[1]) / scale), f"roi_{edge}"


def _check_group(region, group, labels):
    if region == "SPINE":
        position, other = labels["spine_position"], labels["spine_axis"]
    else:
        position, other = labels["hip_position"], labels["hip_roi"]
    if group == "positive":
        return position == 0 and other == 0
    if group == "negative_position":
        return position == 1
    return other == 1


def _labels(region, geometry, info, image, spacing, source_row):
    if region == "SPINE":
        angle = spine_axis_angle(image, geometry)
        position = int(not spine_position_ok(geometry))
        axis = None if angle is None else int(abs(angle) > 5)
        source_artifact = source_row.get("spine_artifact", "")
        objects = geometry["spine"]["foreign_objects"]
        artifact = 1 if objects else (0 if source_artifact in ("0", "0.0") else None)
        return {"spine_position": position, "spine_axis": axis,
                "spine_axis_angle_deg": angle, "spine_artifact": artifact}
    side = region.removeprefix("LEG_")
    roi = hip_roi_ok(geometry, info["roi_fully_visible"], side, spacing)
    return {"hip_position": int(not hip_position_ok(geometry)),
            "hip_roi": None if roi is None else int(not roi),
            "hip_rotation": None}  # Rotation needs manual re-review if the tubercle was cropped.


def _read_dicom(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array
    if image.ndim != 2:
        raise ValueError("Only single-frame grayscale DICOM images are supported")
    return ds, image


def _write_dicom(ds, image, transform, destination: Path, uid_seed: str):
    import copy
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    out = copy.deepcopy(ds)
    out.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    out.PixelData = image.tobytes()
    out.Rows, out.Columns = image.shape
    out.SOPInstanceUID = generate_uid(entropy_srcs=[uid_seed, "instance"])
    out.SeriesInstanceUID = generate_uid(entropy_srcs=[uid_seed, "series"])
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.ImageType = ["DERIVED", "SECONDARY"]
    out.DerivationDescription = "Geometry-aware zoom/rotation for DXA quality research"
    spacing, _ = _dicom_spacing(ds)
    if spacing is not None:
        out.PixelSpacing = list(transformed_spacing(spacing, transform))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(destination), out, enforce_file_format=True)


def generate(workspace: Path, manifest_path: Path, output: Path,
             seed: int = 20260926, target_per_group=None, max_attempts_per_image=300,
             proxy_spacing=True):
    """Generate only from completed geometry; report every unmet quota."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    rng = random.Random(seed)
    sources, problems = _read_sources(workspace, manifest_path)
    by_region = defaultdict(list)
    for item in sources:
        by_region[item["region"]].append(item)
    targets = target_per_group or TARGET_PER_GROUP
    report = {"requested_per_region": dict(targets), "eligible_sources": {},
              "generated": {}, "skipped": dict(problems), "seed": seed}
    output_rows = []
    cache = {}
    for region in REGIONS:
        eligible = []
        for item in by_region[region]:
            source = item["source"]
            try:
                ds, image = _read_dicom(Path(source["path"]))
            except Exception:
                problems["dicom_read_error"] += 1
                continue
            if region == "SPINE":
                angle = spine_axis_angle(image, item["geometry"])
                if not spine_position_ok(item["geometry"]) or angle is None or abs(angle) > 5:
                    problems["source_not_valid_spine_positive"] += 1
                    continue
                spacing, basis = None, "not_applicable"
            else:
                if not hip_position_ok(item["geometry"]) or item["geometry"]["hip"]["roi_box"] is None:
                    problems["source_not_valid_hip_positive"] += 1
                    continue
                spacing, basis = _dicom_spacing(ds)
                if spacing is None and proxy_spacing:
                    spacing = _roi_proxy_spacing(item["geometry"], region.removeprefix("LEG_"))
                    basis = "roi_proxy" if spacing else "missing"
                if spacing is None or hip_roi_ok(item["geometry"], True,
                                                 region.removeprefix("LEG_"), spacing) is not True:
                    problems["source_roi_not_calibrated_or_valid"] += 1
                    continue
                angle = None
            item = {**item, "spacing": spacing, "spacing_basis": basis, "base_angle": angle}
            eligible.append(item)
            cache[source["relative_path"]] = (ds, image)
        report["eligible_sources"][region] = len(eligible)
        counts = Counter()
        if not eligible:
            report["generated"][region] = dict(counts)
            continue
        for group in GROUPS:
            requested = int(targets[group])
            attempts = 0
            limit = max_attempts_per_image * max(requested, 1)
            while counts[group] < requested and attempts < limit:
                attempts += 1
                item = rng.choice(eligible)
                row, geometry = item["source"], item["geometry"]
                ds, image = cache[row["relative_path"]]
                if region == "SPINE":
                    transform, variant = _spine_variant(geometry, group, rng, item["base_angle"])
                else:
                    transform, variant = _hip_variant(geometry, group, rng, region.removeprefix("LEG_"))
                if not transform.covers_output():
                    problems["transform_exceeds_source"] += 1
                    continue
                try:
                    moved_geometry, info = transform_geometry(geometry, transform)
                    moved_image = warp_image(image, transform)
                except (ValueError, IndexError):
                    problems["invalid_transform_or_geometry"] += 1
                    continue
                spacing = transformed_spacing(item["spacing"], transform)
                labels = _labels(region, moved_geometry, info, moved_image, spacing, row)
                if not _check_group(region, group, labels):
                    problems["candidate_does_not_match_target"] += 1
                    continue
                digest = hashlib.sha256(f"{seed}|{region}|{group}|{counts[group]}|"
                                        f"{row['relative_path']}|{attempts}".encode()).hexdigest()[:24]
                image_rel = f"images/{region}/{digest}.dcm"
                geometry_rel = f"geometry/{digest}.json"
                output.mkdir(parents=True, exist_ok=True)
                _write_dicom(ds, moved_image, transform, output / image_rel, digest)
                (output / "geometry").mkdir(exist_ok=True)
                (output / geometry_rel).write_text(
                    json.dumps(moved_geometry, ensure_ascii=False, indent=2), encoding="utf-8")
                output_rows.append({"image_path": image_rel, "geometry_path": geometry_rel,
                                    "source_relative_path": row["relative_path"],
                                    "source_study_uid": row["study_uid"], "region": region,
                                    "generation_group": group, "variant": variant,
                                    "scale": round(transform.scale, 6),
                                    "rotation_deg": round(transform.angle_deg, 6),
                                    "source_center_x": round(transform.source_center[0], 4),
                                    "source_center_y": round(transform.source_center[1], 4),
                                    "spacing_basis": item["spacing_basis"],
                                    "dropped_annotations": json.dumps(info["dropped"]),
                                    **labels})
                counts[group] += 1
        report["generated"][region] = {group: counts[group] for group in GROUPS}
    report["skipped"] = dict(problems)
    report["total_generated"] = len(output_rows)
    report["shortfall"] = {region: {group: int(targets[group]) - report["generated"][region].get(group, 0)
                                   for group in GROUPS} for region in REGIONS}
    report["actual_labels"] = {}
    for region in REGIONS:
        rows = [row for row in output_rows if row["region"] == region]
        tasks = ("spine_position", "spine_axis") if region == "SPINE" else ("hip_position", "hip_roi")
        report["actual_labels"][region] = {
            task: {"negative": sum(row[task] == 1 for row in rows),
                   "positive": sum(row[task] == 0 for row in rows),
                   "unknown": sum(row[task] is None for row in rows)}
            for task in tasks}
    output.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "geometry_path", "source_relative_path", "source_study_uid",
              "region", "generation_group", "variant", "scale", "rotation_deg",
              "source_center_x", "source_center_y", "spacing_basis", "dropped_annotations",
              "spine_position", "spine_axis", "spine_axis_angle_deg", "spine_artifact",
              "hip_position", "hip_roi", "hip_rotation"]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    (output / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--workspace", type=Path, default=project.parent)
    parser.add_argument("--manifest", type=Path, default=project / "outputs" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=project / "outputs" / "augmented_dataset")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--no-roi-proxy", action="store_true")
    args = parser.parse_args()
    report = generate(args.workspace, args.manifest, args.output, args.seed,
                      proxy_spacing=not args.no_roi_proxy)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
