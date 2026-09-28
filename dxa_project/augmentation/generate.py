"""Generate geometry-aware DXA DICOM augmentations from completed annotations.

Run ``python -m dxa_project.augmentation.generate --help`` from the
workspace root to inspect options.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import random
from collections import Counter, defaultdict
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .core import (Transform, hip_position_ok, hip_roi_ok, hip_roi_margins_mm,
                   lesser_trochanter_between_area, prepare_geometry,
                   physical_axis_angle, spine_axis_angle,
                   spine_position_ok, transform_geometry,
                   transformed_spacing, warp_image)
from .vertebral_axes import AxisConfig, analyze_spine, placement_from_axes


REGIONS = ("SPINE", "LEG_LEFT", "LEG_RIGHT")
GROUPS = ("positive", "negative_position", "negative_axis_or_roi")
TARGET_PER_GROUP = {"positive": 2500, "negative_position": 1250,
                    "negative_axis_or_roi": 1250}
SCANNER_NOMINAL_SPACING_MM = (1.05, 0.6)  # (row/Y, column/X), supplied by user


def _region(label: str, side: str) -> str | None:
    if label == "SPINE":
        return "SPINE"
    if label == "LEG" and side in ("LEFT", "RIGHT"):
        return f"LEG_{side}"
    return None


def _read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_sources(workspace: Path, manifest_path: Path,
                  annotations_root: Path | None = None):
    manifest = {row["relative_path"].replace("\\", "/"): row
                for row in _read_rows(manifest_path)}
    entries = {}
    quarantined = set()
    problems = Counter()
    # Other Размеченные* folders are old copies and must never silently
    # override or conflict with the authoritative finished annotation folder.
    for root in [annotations_root or workspace / "Размеченные"]:
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
                geometry = prepare_geometry(geometry, region)
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
            item["artifact_annotation_complete"] = bool(
                geometry["spine"]["foreign_objects"])
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


def _source_spacing(ds, allow_nominal=True):
    spacing, basis = _dicom_spacing(ds)
    if spacing is None and allow_nominal:
        return SCANNER_NOMINAL_SPACING_MM, "scanner_nominal_user_supplied"
    return spacing, basis


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


def _spine_variant(geometry, group, rng, base_angle, spacing_mm=SCANNER_NOMINAL_SPACING_MM):
    w, h = geometry["image_width"], geometry["image_height"]
    lines = sorted(geometry["spine"]["disc_lines"],
                   key=lambda l: sum(p[1] for p in l["points"]) / 2)
    mids = [np.mean(l["points"], axis=0) for l in lines]
    gap = float(np.median(np.diff([m[1] for m in mids])))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    reflect = rng.random() < 0.5
    normalized_angle = -base_angle if reflect else base_angle
    row_mm, col_mm = spacing_mm

    def coverage_scale(angle_deg):
        a = math.radians(angle_deg)
        return max(abs(math.cos(a)) + abs(math.sin(a)) * row_mm / col_mm,
                   abs(math.cos(a)) + abs(math.sin(a)) * col_mm / row_mm) + .025
    if group == "negative_axis_or_roi":
        target = rng.choice((rng.uniform(-10, -6), rng.uniform(6, 10)))
        angle = normalized_angle - target
        scale = max(1.02, coverage_scale(angle))
        scale += rng.uniform(0, 0.06)
        return Transform(w, h, scale, angle, reflect_x=reflect,
                         physical_spacing_mm=spacing_mm), f"axis_target_{target:.2f}"
    if group == "positive":
        target = rng.uniform(-4, 4)
        angle = normalized_angle - target
        scale = max(1.01, coverage_scale(angle))
        scale += rng.uniform(0, 0.05)
        desired_y = rng.uniform(0.35, 0.65) * gap * scale
        src_y = float(mids[0][1] + (cy - desired_y) /
                      (scale * math.cos(math.radians(angle))))
        return Transform(w, h, scale, angle, cx, src_y,
                         reflect_x=reflect, physical_spacing_mm=spacing_mm), f"half_Th12_axis_{target:.2f}"
    variant = rng.choice(("crop_Th12", "crop_Th12_and_next",
                          "crop_left_crest", "crop_right_crest", "crop_both_crests_bottom"))
    scale = rng.uniform(1.18, 1.65)
    axis_target = rng.uniform(-4, 4)
    angle = normalized_angle - axis_target
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
    return Transform(w, h, scale, angle, src_x, src_y,
                     reflect_x=reflect, physical_spacing_mm=spacing_mm), variant


def _positive_roi_center_intervals(geometry, side, scale, spacing_mm, slack_mm=0.0):
    """Safe source-center ranges from ROI-to-frame distances in millimeters."""
    w, h = geometry["image_width"], geometry["image_height"]
    x1, y1, x2, y2 = geometry["hip"]["roi_box"]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    row_mm, col_mm = spacing_mm
    vertical_px = (30 + slack_mm) * scale / row_mm
    lateral_px = (20 + slack_mm) * scale / col_mm
    x_low, x_high = cx / scale, w - 1 - cx / scale
    y_low, y_high = cy / scale, h - 1 - cy / scale
    y_low = max(y_low, y2 - (h - 1 - vertical_px - cy) / scale)
    y_high = min(y_high, y1 - (vertical_px - cy) / scale)
    if side == "LEFT":
        x_low = max(x_low, x2 - (w - 1 - lateral_px - cx) / scale)
    else:
        x_high = min(x_high, x1 - (lateral_px - cx) / scale)
    return (x_low, x_high), (y_low, y_high)


def _hip_variant(geometry, group, rng, side, spacing_mm=None):
    w, h = geometry["image_width"], geometry["image_height"]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    box = geometry["hip"]["roi_box"]
    points = list(geometry["hip"]["landmarks"].values())
    spacing_mm = spacing_mm or SCANNER_NOMINAL_SPACING_MM
    if group == "positive":
        scale = rng.uniform(1.005, 1.08)
        x_range, y_range = _positive_roi_center_intervals(
            geometry, side, scale, spacing_mm, slack_mm=0.5)
        if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
            # The caller rejects this candidate and samples another scale.
            return Transform(w, h, scale, 0, -w, -h), "roi_positive_no_slack"
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        return Transform(w, h, scale, 0, x, y), "roi_positive_distances"
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
    scale = rng.uniform(1.02, 1.30)
    x_range, y_range = _positive_roi_center_intervals(
        geometry, side, scale, spacing_mm, slack_mm=0.5)
    row_mm, col_mm = spacing_mm
    if edge in ("top", "bottom") and x_range[0] > x_range[1]:
        return Transform(w, h, scale, 0, -w, -h), "roi_no_safe_lateral_shift"
    if edge == "lateral" and y_range[0] > y_range[1]:
        return Transform(w, h, scale, 0, -w, -h), "roi_no_safe_vertical_shift"
    if edge == "top":
        target_mm = rng.uniform(22, 29)
        y = y1 + (cy - target_mm * scale / row_mm) / scale
        x = rng.uniform(*x_range)
    elif edge == "bottom":
        target_mm = rng.uniform(22, 29)
        target_y = h - 1 - target_mm * scale / row_mm
        y = y2 + (cy - target_y) / scale
        x = rng.uniform(*x_range)
    else:
        target_mm = rng.uniform(13, 19)
        target_x = (w - 1 - target_mm * scale / col_mm if side == "LEFT"
                    else target_mm * scale / col_mm)
        x = (x2 if side == "LEFT" else x1) + (cx - target_x) / scale
        y = rng.uniform(*y_range)
    return Transform(w, h, scale, 0, x, y), f"roi_{edge}_{target_mm:.1f}mm"


def _check_group(region, group, labels):
    if region == "SPINE":
        position, other = labels["spine_position"], labels["spine_axis"]
    else:
        position, other = labels["hip_position"], labels["hip_roi"]
    if group == "positive":
        if region == "SPINE":
            angle = labels["spine_axis_angle_deg"]
            return position == 0 and angle is not None and -4 <= angle <= 4
        return position == 0 and other == 0
    if group == "negative_position":
        if region == "SPINE":
            angle = labels["spine_axis_angle_deg"]
            return position == 1 and angle is not None and -4 <= angle <= 4
        return position == 1
    if region == "SPINE":
        angle = labels["spine_axis_angle_deg"]
        return angle is not None and (
            -10 <= angle <= -6 or 6 <= angle <= 10)
    return position == 0 and other == 1


def _labels(region, geometry, info, image, spacing, source_row):
    if region == "SPINE":
        analysis = analyze_spine(image,geometry,spacing or SCANNER_NOMINAL_SPACING_MM,
                                 polarity=source_row.get("axis_polarity","bright"))
        angle = analysis["global_angle_deg"]
        if any(not a["valid"] or a["review_reasons"] for a in analysis["axes"]):
            angle = None
        placement = placement_from_axes(geometry,analysis)
        position = None if placement is None else int(not placement)
        axis = None if angle is None else int(abs(angle) > 5)
        source_artifact = source_row.get("spine_artifact", "")
        objects = geometry["spine"]["foreign_objects"]
        artifact = 1 if objects else (0 if source_artifact in ("0", "0.0") or
                                     source_row.get("artifact_annotation_complete", False) else None)
        return {"spine_position": position, "spine_axis": axis,
                "spine_axis_angle_deg": angle, "spine_artifact": artifact}
    side = region.removeprefix("LEG_")
    roi = hip_roi_ok(geometry, info["roi_fully_visible"], side, spacing)
    margins = (hip_roi_margins_mm(geometry, side, spacing)
               if spacing is not None and geometry["hip"]["roi_box"] is not None else None)
    area, crossings = lesser_trochanter_between_area(geometry)
    box = geometry["hip"]["roi_box"]
    roi_area = (box[2] - box[0]) * (box[3] - box[1]) if box else 0
    source_rotation = source_row.get(f"{side.lower()}_hip_rotation", "")
    rotation = (0 if source_rotation in ("0", "0.0") else
                1 if source_rotation in ("1", "1.0") else None)
    partial = bool(geometry["hip"].get("lesser_trochanter_partial", False))
    return {"hip_position": int(not hip_position_ok(geometry)),
            "hip_roi": None if roi is None else int(not roi),
            "hip_roi_top_cm": margins["top"] / 10 if margins else None,
            "hip_roi_bottom_cm": margins["bottom"] / 10 if margins else None,
            "hip_roi_lateral_cm": margins["lateral"] / 10 if margins else None,
            "trochanter_between_area_px2": area,
            "trochanter_curve_crossings": crossings,
            "trochanter_area_fraction_roi": area / roi_area if roi_area > 0 else None,
            "trochanter_partial": int(partial),
            "hip_rotation": rotation if area > 0 and not partial else 1}


def _read_dicom(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array
    if image.ndim != 2:
        raise ValueError("Only single-frame grayscale DICOM images are supported")
    return ds, image


def _write_dicom(ds, image, transform, destination: Path, uid_seed: str,
                 geometry: dict | None = None, labels: dict | None = None,
                 region: str | None = None):
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
    if geometry is not None:
        block = out.private_block(0x0011, "DXA_MANUAL_LABELER", create=True)
        out.add_new(block.get_tag(0x09), "UT",
                    json.dumps(geometry, ensure_ascii=True, separators=(",", ":")))
        out.add_new(block.get_tag(0x0A), "UT",
                    json.dumps(labels or {}, ensure_ascii=True, separators=(",", ":")))
        if region:
            out.add_new(block.get_tag(0x01), "CS",
                        "SPINE" if region == "SPINE" else "LEG")
            out.add_new(block.get_tag(0x05), "CS",
                        region.removeprefix("LEG_") if region.startswith("LEG_") else "")
    spacing, _ = _dicom_spacing(ds)
    if spacing is not None:
        out.PixelSpacing = list(transformed_spacing(spacing, transform))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(destination), out, enforce_file_format=True)


def _spine_candidate(item, image, group, seed):
    rng=random.Random(seed)
    transform,variant=_spine_variant(item['geometry'],group,rng,item['base_angle'],item['spacing'])
    if not transform.covers_output():
        return 'transform_exceeds_source',None
    try:
        geometry,info=transform_geometry(item['geometry'],transform,region='SPINE')
        if group=='positive' and (len(geometry['spine']['disc_lines']) not in range(4,8) or
                any(p is None for p in geometry['spine']['iliac_crests'].values())):
            return 'candidate_positive_lost_required_geometry',None
        moved=warp_image(image,transform)
        spacing=transformed_spacing(item['spacing'],transform)
        labels=_labels('SPINE',geometry,info,moved,spacing,item['source'])
    except (ValueError,IndexError):
        return 'invalid_transform_or_geometry',None
    if not _check_group('SPINE',group,labels):
        return 'candidate_does_not_match_target',None
    return None,(transform,variant,geometry,info,moved,spacing,labels)


def generate(workspace: Path, manifest_path: Path, output: Path,
             seed: int = 20260926, target_per_group=None, max_attempts_per_image=300,
             allow_nominal_spacing=True, annotations_root: Path | None = None,
             workers: int = 1,resume: bool = False):
    """Generate only from completed geometry; report every unmet quota."""
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output}")
    rng = random.Random(seed)
    sources, problems = _read_sources(workspace, manifest_path, annotations_root)
    by_region = defaultdict(list)
    for item in sources:
        by_region[item["region"]].append(item)
    targets = target_per_group or TARGET_PER_GROUP
    report = {"axis_method": "joint_continuous_contour_l1",
              "axis_max_deviation_deg": AxisConfig().max_deviation_deg,
              "requested_per_region": dict(targets), "eligible_sources": {},
              "generated": {}, "skipped": dict(problems), "seed": seed}
    output_rows = _read_rows(output/'partial_manifest.csv') if resume else []
    if resume:
        for row in output_rows:
            for task in ('spine_position','spine_axis','spine_artifact','hip_position','hip_roi','hip_rotation'):
                if task in row:
                    row[task]=int(float(row[task])) if row[task] not in ('',None) else None
        report['resumed_images']=len(output_rows)
    started=time.perf_counter()
    cache = {}
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    for region in REGIONS:
        counts=Counter(row['generation_group'] for row in output_rows if row['region']==region)
        if all(counts[group]>=int(targets[group]) for group in GROUPS):
            report['generated'][region]={group:counts[group] for group in GROUPS}
            report['eligible_sources'][region]='completed_before_resume'
            continue
        eligible = []
        for item in by_region[region]:
            source = item["source"]
            try:
                ds, image = _read_dicom(Path(source["path"]))
            except Exception:
                problems["dicom_read_error"] += 1
                continue
            if region == "SPINE":
                spacing, basis = _source_spacing(ds, allow_nominal_spacing)
                if spacing is None:
                    problems["source_spine_spacing_missing"] += 1
                    continue
                source["axis_polarity"] = "dark" if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1" else "bright"
                analysis = analyze_spine(image,item["geometry"],spacing,polarity=source["axis_polarity"])
                angle = analysis["global_angle_deg"]
                if analysis["review_required"] or placement_from_axes(item["geometry"],analysis) is not True or angle is None or abs(angle)>5:
                    problems["source_not_valid_spine_positive"] += 1
                    continue
            else:
                if not hip_position_ok(item["geometry"]) or item["geometry"]["hip"]["roi_box"] is None:
                    problems["source_not_valid_hip_positive"] += 1
                    continue
                spacing, basis = _source_spacing(ds, allow_nominal_spacing)
                if spacing is None or hip_roi_ok(item["geometry"], True,
                                                 region.removeprefix("LEG_"), spacing) is not True:
                    problems["source_roi_not_calibrated_or_valid"] += 1
                    continue
                angle = None
            item = {**item, "spacing": spacing, "spacing_basis": basis, "base_angle": angle}
            eligible.append(item)
            cache[source["relative_path"]] = (ds, image)
        report["eligible_sources"][region] = len(eligible)
        print(json.dumps({'stage':'eligible_sources','region':region,'count':len(eligible)},ensure_ascii=False),flush=True)
        if not eligible:
            report["generated"][region] = dict(counts)
            continue
        for group in GROUPS:
            pending=deque()
            requested = int(targets[group])
            attempts = 0
            limit = max_attempts_per_image * max(requested, 1)
            while counts[group] < requested and attempts < limit:
                attempts += 1
                if region=='SPINE' and pool is not None:
                    if not pending:
                        for _ in range(workers*3):
                            candidate_item=rng.choice(eligible)
                            candidate_image=cache[candidate_item['source']['relative_path']][1]
                            pending.append((candidate_item,pool.submit(_spine_candidate,candidate_item,candidate_image,group,rng.getrandbits(64))))
                    item,future=pending.popleft()
                    error,candidate=future.result()
                    if error:
                        problems[error]+=1
                        continue
                else:
                    item = rng.choice(eligible)
                row, geometry = item["source"], item["geometry"]
                ds, image = cache[row["relative_path"]]
                if region=='SPINE' and pool is not None:
                    transform,variant,moved_geometry,info,moved_image,spacing,labels=candidate
                elif region == "SPINE":
                    transform, variant = _spine_variant(geometry, group, rng,
                                                        item["base_angle"], item["spacing"])
                else:
                    transform, variant = _hip_variant(geometry, group, rng,
                                                      region.removeprefix("LEG_"), item["spacing"])
                if not (region=='SPINE' and pool is not None) and not transform.covers_output():
                    problems["transform_exceeds_source"] += 1
                    continue
                if not (region=='SPINE' and pool is not None):
                    try:
                        moved_geometry, info = transform_geometry(geometry, transform, region=region)
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
                _write_dicom(ds, moved_image, transform, output / image_rel, digest,
                             moved_geometry, labels, region)
                (output / "geometry").mkdir(exist_ok=True)
                (output / geometry_rel).write_text(
                    json.dumps(moved_geometry, ensure_ascii=False, indent=2), encoding="utf-8")
                output_rows.append({"image_path": image_rel, "geometry_path": geometry_rel,
                                    "source_relative_path": row["relative_path"],
                                    "source_study_uid": row["study_uid"], "region": region,
                                    "generation_group": group, "variant": variant,
                                    "scale": round(transform.scale, 6),
                                    "rotation_deg": round(transform.angle_deg, 6),
                                    "reflect_x": int(transform.reflect_x),
                                    "source_center_x": round(transform.source_center[0], 4),
                                    "source_center_y": round(transform.source_center[1], 4),
                                    "spacing_basis": item["spacing_basis"],
                                    "row_spacing_mm": spacing[0] if spacing else None,
                                    "col_spacing_mm": spacing[1] if spacing else None,
                                    "dropped_annotations": json.dumps(info["dropped"]),
                                    **labels})
                counts[group] += 1
                if counts[group] % 50 == 0:
                    with (output/'partial_manifest.csv').open('w',encoding='utf-8-sig',newline='') as checkpoint:
                        checkpoint_fields=list(dict.fromkeys(key for checkpoint_row in output_rows for key in checkpoint_row))
                        checkpoint_writer=csv.DictWriter(checkpoint,fieldnames=checkpoint_fields)
                        checkpoint_writer.writeheader();checkpoint_writer.writerows(output_rows)
                    progress={'region':region,'group':group,'generated':counts[group],
                              'requested':requested,'attempts':attempts,
                              'total_generated':len(output_rows),'seconds':time.perf_counter()-started}
                    (output/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                    print(json.dumps(progress),flush=True)
        report["generated"][region] = {group: counts[group] for group in GROUPS}
    if pool is not None:
        pool.shutdown(wait=True)
    report["skipped"] = dict(problems)
    report["total_generated"] = len(output_rows)
    report['seconds']=time.perf_counter()-started
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
              "reflect_x",
              "source_center_x", "source_center_y", "spacing_basis",
              "row_spacing_mm", "col_spacing_mm", "dropped_annotations",
              "spine_position", "spine_axis", "spine_axis_angle_deg", "spine_artifact",
              "hip_position", "hip_roi", "hip_rotation", "hip_roi_top_cm",
              "hip_roi_bottom_cm", "hip_roi_lateral_cm",
              "trochanter_between_area_px2", "trochanter_curve_crossings",
              "trochanter_area_fraction_roi", "trochanter_partial"]
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
    parser.add_argument("--annotations-root", type=Path,
                        default=project.parent / "Размеченные")
    parser.add_argument("--output", type=Path, default=project / "outputs" / "augmented_15000")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument('--workers',type=int,default=1,help='Parallel CPU candidates for spine')
    parser.add_argument('--resume',action='store_true',help='Continue from partial_manifest.csv')
    parser.add_argument("--require-dicom-spacing", action="store_true",
                        help="Skip hips without measured DICOM pixel spacing")
    args = parser.parse_args()
    report = generate(args.workspace, args.manifest, args.output, args.seed,
                      allow_nominal_spacing=not args.require_dicom_spacing,
                      annotations_root=args.annotations_root,workers=args.workers,resume=args.resume)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
