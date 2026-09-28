"""Build 30 source DXA examples with five reviewed zooms and 2x6 contact sheets.

Run from the workspace root with ``python -m dxa_project.augmentation.pilot_30``.
Each sheet has unmarked images above the same images with annotations, and a
target caption under each column. The script does not edit source annotations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .core import (hip_position_ok, hip_roi_ok, physical_axis_angle,
                   spine_axis_angle, spine_position_ok, transform_geometry,
                   transformed_spacing, warp_image)
from .generate import (_check_group, _source_spacing, _hip_variant, _labels,
                       _read_dicom, _read_rows, _read_sources,
                       _spine_variant, _write_dicom)
from .vertebral_axes import analyze_spine, placement_from_axes


REGIONS = ("SPINE", "LEG_LEFT", "LEG_RIGHT")
GROUPS = ("positive", "negative_position", "negative_position",
          "negative_axis_or_roi", "negative_axis_or_roi")
TARGETS = ("spine_position", "spine_axis", "spine_artifact",
           "hip_position", "hip_roi", "hip_rotation")


def _reference_value(value):
    if value in ("0", "0.0"):
        return 0
    if value in ("1", "1.0"):
        return 1
    return None


def _font(size: int):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/seguisym.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _display_image(ds, pixels):
    values = pixels.astype(np.float32)
    low, high = np.percentile(values, (1, 99.5))
    image = np.clip((values - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = 255 - image
    return Image.fromarray(image).convert("RGB")


def _overlay(base: Image.Image, geometry: dict, region: str):
    out = base.copy()
    draw = ImageDraw.Draw(out)
    if region == "SPINE":
        for line in geometry["spine"]["disc_lines"]:
            draw.line([tuple(p) for p in line["points"]], fill="#35fa74", width=2)
        for name, point in geometry["spine"]["iliac_crests"].items():
            if point is None:
                continue
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ffcf44", outline="black")
            draw.text((x + 5, y - 13), "L" if name == "image_left" else "R",
                      fill="#ffcf44", font=_font(12), stroke_width=1, stroke_fill="black")
        for obj in geometry["spine"]["foreign_objects"]:
            draw.rectangle(obj["bbox"], outline="#ff4545", width=3)
    else:
        box = geometry["hip"]["roi_box"]
        if box is not None:
            draw.rectangle(box, outline="#cd68ff", width=3)
        for digit, key in enumerate(("greater_trochanter", "femoral_neck", "ischial_bone"), 1):
            point = geometry["hip"]["landmarks"][key]
            if point is None:
                continue
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#13d7f5", outline="black")
            draw.text((x + 6, y - 14), str(digit), fill="#13d7f5",
                      font=_font(14), stroke_width=1, stroke_fill="black")
        pixels = geometry["hip"].get("lesser_trochanter_pixels") or []
        if pixels:
            draw.point([tuple(point) for point in pixels], fill="#ff3030")
        else:
            for stroke in geometry["hip"]["lesser_trochanter_traces"]["trochanter"]:
                _dashed_polyline(draw, stroke["points"], "#ff3030")
    return out


def _dashed_polyline(draw, points, color, on=4.0, off=3.0):
    phase, period = 0.0, on + off
    for first, second in zip(points, points[1:]):
        a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        cursor = 0.0
        while cursor < length - 1e-9:
            visible = phase < on
            step = min(length - cursor, (on if visible else period) - phase)
            if visible:
                start_xy = a + (b - a) * (cursor / length)
                end_xy = a + (b - a) * ((cursor + step) / length)
                draw.line([tuple(start_xy), tuple(end_xy)], fill=color, width=2)
            cursor += step
            phase = (phase + step) % period


def _target_text(row, region):
    def value(name):
        current = row.get(name)
        return "?" if current is None else str(current)
    if region == "SPINE":
        first = f"Укладка {value('spine_position')} · Ось {value('spine_axis')}"
        second = f"Артефакт {value('spine_artifact')}"
        angle = row.get("spine_axis_angle_deg")
        third = f"Угол {angle:.1f}°" if angle is not None else "Угол ?"
    else:
        first = f"Точки {value('hip_position')} · ROI {value('hip_roi')}"
        second = f"Ротация {value('hip_rotation')} · Часть {value('trochanter_partial')}"
        third = f"Вертел {row.get('trochanter_between_area_px2', '?')} px²"
        top, bottom, lateral = (row.get(f"hip_roi_{key}_cm") for key in
                                ("top", "bottom", "lateral"))
        fourth = (f"Зазоры В/Н/Б: {top:.1f}/{bottom:.1f}/{lateral:.1f} см"
                  if None not in (top, bottom, lateral) else "Зазоры ROI: ?")
        return (first, second, third, fourth)
    return (first, second, third)


def _sheet(entries, region, destination):
    cell_width, image_size = 332, 306
    width, height = 6 * cell_width, 826
    sheet = Image.new("RGB", (width, height), "#121820")
    draw = ImageDraw.Draw(sheet)
    normal_font, small_font = _font(17), _font(15)
    for col, entry in enumerate(entries):
        label = "Исходное" if col == 0 else f"Аугментация {col}"
        draw.text((col * cell_width + 12, 8), label, fill="#ffffff", font=normal_font)
        base = _display_image(entry["ds"], entry["image"])
        marked = _overlay(base, entry["geometry"], region)
        for row_number, picture in enumerate((base, marked)):
            picture = ImageOps.contain(picture, (image_size, image_size))
            x = col * cell_width + (cell_width - picture.width) // 2
            y = 36 + row_number * 329 + (image_size - picture.height) // 2
            sheet.paste(picture, (x, y))
        for line_number, text in enumerate(_target_text(entry["labels"], region)):
            draw.text((col * cell_width + 8, 698 + line_number * 21), text,
                      fill="#f4f7fb" if line_number == 0 else "#bcd0e2", font=small_font)
        if col:
            draw.text((col * cell_width + 8, 792), entry["variant"][:37],
                      fill="#aebdcc", font=_font(13))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)


def _source_candidate(item, region):
    row, geometry = item["source"], item["geometry"]
    ds, image = _read_dicom(Path(row["path"]))
    if region == "SPINE":
        spacing, basis = _source_spacing(ds)
        row["axis_polarity"] = "dark" if str(getattr(ds,"PhotometricInterpretation",""))=="MONOCHROME1" else "bright"
        analysis = analyze_spine(image,geometry,spacing,polarity=row["axis_polarity"])
        angle = analysis["global_angle_deg"]
        if (analysis["review_required"] or placement_from_axes(geometry,analysis) is not True or angle is None or abs(angle) > 5 or
                any(_reference_value(row.get(key)) != 0
                    for key in ("spine_position", "spine_axis"))):
            return None
        ancillary = _reference_value(row.get("spine_artifact"))
        if ancillary == 1 and not geometry["spine"]["foreign_objects"]:
            return None
    else:
        side = region.removeprefix("LEG_")
        spacing, basis = _source_spacing(ds)
        key = f"{side.lower()}_hip_roi"
        if (not hip_position_ok(geometry) or
                hip_roi_ok(geometry, True, side, spacing) is not True or
                _reference_value(row.get(key)) != 0):
            return None
        angle = None
        ancillary = _reference_value(row.get(f"{side.lower()}_hip_rotation"))
    return {**item, "ds": ds, "image": image, "spacing": spacing,
            "spacing_basis": basis, "base_angle": angle, "ancillary": ancillary}


def _source_labels(item):
    region = item["region"]
    row, geometry, image = item["source"], item["geometry"], item["image"]
    labels = _labels(region, geometry, {"roi_fully_visible": True},
                     image, item["spacing"], row)
    if region == "SPINE":
        labels["spine_artifact"] = _reference_value(row.get("spine_artifact"))
    return labels


def _make_variants(item, rng, attempts_per_slot=250):
    region = item["region"]
    row, geometry = item["source"], item["geometry"]
    ds, image = item["ds"], item["image"]
    entries = [{"ds": ds, "image": image, "geometry": geometry,
                "labels": _source_labels(item), "variant": "source", "transform": None,
                "group": "source"}]
    used_variants = set()
    for group in GROUPS:
        chosen = None
        for _ in range(attempts_per_slot):
            if region == "SPINE":
                transform, variant = _spine_variant(geometry, group, rng,
                                                    item["base_angle"], item["spacing"])
            else:
                transform, variant = _hip_variant(geometry, group, rng,
                                                  region.removeprefix("LEG_"),
                                                  item["spacing"])
            if variant in used_variants or not transform.covers_output():
                continue
            try:
                moved_geometry, info = transform_geometry(geometry, transform, region=region)
                moved_image = warp_image(image, transform)
            except (ValueError, IndexError):
                continue
            spacing = transformed_spacing(item["spacing"], transform)
            labels = _labels(region, moved_geometry, info, moved_image, spacing, row)
            if not _check_group(region, group, labels):
                continue
            chosen = {"ds": ds, "image": moved_image, "geometry": moved_geometry,
                      "labels": labels, "variant": variant, "transform": transform,
                      "group": group, "spacing_basis": item["spacing_basis"]}
            break
        if chosen is None:
            return None
        used_variants.add(chosen["variant"])
        entries.append(chosen)
    return entries


def run(workspace: Path, output: Path, seed=20260927, per_region=10):
    workspace, output = workspace.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    manifest = workspace / "dxa_project" / "outputs" / "manifest.csv"
    annotation_root = workspace / "Размеченные"
    source_rows, problems = _read_sources(workspace, manifest, annotation_root)
    ordinal = {row["relative_path"].replace("\\", "/"): index
               for index, row in enumerate(_read_rows(annotation_root / "labels.csv"), 1)}
    rng = random.Random(seed)
    by_region = defaultdict(list)
    for item in source_rows:
        by_region[item["region"]].append(item)
    all_rows = []
    boards = []
    selected = Counter()
    failures = Counter(problems)
    for region in REGIONS:
        suitable = []
        for item in by_region[region]:
            try:
                candidate = _source_candidate(item, region)
            except (ValueError, KeyError, OSError):
                failures["source_read_or_geometry_error"] += 1
                continue
            if candidate is not None:
                suitable.append(candidate)
        positives = [item for item in suitable if item["ancillary"] == 1]
        others = [item for item in suitable if item["ancillary"] != 1]
        rng.shuffle(positives)
        rng.shuffle(others)
        # Include real artifact/rotation violations without letting them
        # dominate the small ten-source visual pilot.
        queue = positives[:2] + others + positives[2:]
        if region == "LEG_RIGHT":
            # Keep the user-reported contour case 358 in this fixed visual pilot.
            featured = [item for item in queue
                        if ordinal[item["source"]["relative_path"].replace("\\", "/")] == 358]
            featured_paths = {item["source"]["relative_path"] for item in featured}
            queue = featured + [item for item in queue
                                if item["source"]["relative_path"] not in featured_paths]
        for item in queue:
            if selected[region] >= per_region:
                break
            entries = _make_variants(item, rng)
            if entries is None:
                failures[f"candidate_five_variants_failed_{region}"] += 1
                continue
            selected[region] += 1
            source_num = ordinal[item["source"]["relative_path"].replace("\\", "/")]
            sample_id = f"{region.lower()}_{selected[region]:02d}_row{source_num:03d}"
            for number, entry in enumerate(entries):
                image_rel = None if number == 0 else f"images/{region}/{sample_id}_{number}.dcm"
                geo_rel = f"geometry/{sample_id}_{number}.json"
                sample_spacing = (transformed_spacing(item["spacing"], entry["transform"])
                                  if entry["transform"] else item["spacing"])
                if image_rel:
                    _write_dicom(item["ds"], entry["image"], entry["transform"],
                                 output / image_rel, f"pilot30-{seed}-{sample_id}-{number}",
                                 entry["geometry"], entry["labels"], region)
                destination = output / geo_rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(entry["geometry"], ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
                all_rows.append({"sample_id": sample_id, "source_row": source_num,
                                 "region": region, "column": number,
                                 "group": entry["group"], "variant": entry["variant"],
                                 "reflect_x": int(bool(entry["transform"] and
                                                       entry["transform"].reflect_x)),
                                 "image_path": image_rel or item["source"]["path"],
                                 "geometry_path": geo_rel,
                                 "source_relative_path": item["source"]["relative_path"],
                                 "source_study_uid": item["source"]["study_uid"],
                                 "spacing_basis": item["spacing_basis"],
                                 "row_spacing_mm": sample_spacing[0] if sample_spacing else None,
                                 "col_spacing_mm": sample_spacing[1] if sample_spacing else None,
                                 **entry["labels"]})
            board_rel = f"boards/{sample_id}.png"
            _sheet(entries, region, output / board_rel)
            boards.append({"source_row": source_num, "region": region,
                           "sample_id": sample_id, "board_path": board_rel})
        if selected[region] != per_region:
            raise RuntimeError(f"Only {selected[region]}/{per_region} complete examples for {region}")
    output.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "source_row", "region", "column", "group", "variant",
               "reflect_x",
               "image_path", "geometry_path", "source_relative_path", "source_study_uid",
               "spacing_basis", "row_spacing_mm", "col_spacing_mm",
               *TARGETS, "spine_axis_angle_deg",
               "hip_roi_top_cm", "hip_roi_bottom_cm", "hip_roi_lateral_cm",
               "trochanter_between_area_px2", "trochanter_curve_crossings",
               "trochanter_area_fraction_roi", "trochanter_partial"]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    coverage = {target: {"negative": sum(row.get(target) == 1 for row in all_rows),
                         "positive": sum(row.get(target) == 0 for row in all_rows),
                         "unknown": sum(row.get(target) is None for row in all_rows)}
                for target in TARGETS}
    report = {"seed": seed, "source_count": len(boards),
              "augmented_count": len(all_rows) - len(boards),
              "total_columns": len(all_rows), "by_region": dict(selected),
              "target_coverage": coverage, "failures": dict(failures),
              "boards": boards,
              "target_convention": "1=нарушение, 0=норма, ?=неизвестно",
              "roi_spacing_note": "1.05 mm/Y and 0.6 mm/X are user-supplied scanner nominal values; no spacing tags in these DICOMs"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2), encoding="utf-8")
    sections = []
    for region in REGIONS:
        cards = "\n".join(
            f'<article><h3>Исходный снимок №{board["source_row"]}</h3>'
            f'<a href="{html.escape(board["board_path"])}">'
            f'<img loading="lazy" src="{html.escape(board["board_path"])}" '
            f'alt="{html.escape(board["sample_id"])}"></a></article>'
            for board in boards if board["region"] == region)
        sections.append(f"<section><h2>{region}</h2>{cards}</section>")
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<title>DXA — 30 исходных снимков и 150 аугментаций</title>'
            '<style>body{font:16px system-ui;background:#101820;color:#f4f7fb;'
            'margin:24px}a{color:#87c9ff}article{margin:26px 0 46px}img{width:100%;'
            'max-width:1992px;height:auto}h1,h2{margin-top:40px}</style>'
            '<h1>30 исходных снимков и 150 аугментаций</h1>'
            '<p>Для каждого исходного снимка: сверху 6 изображений без разметки, '
            'снизу те же 6 изображений с разметкой. '
            '1 = нарушение, 0 = норма, ? = метка неизвестна. '
            'Метки стоят под соответствующим столбцом. '
            'Для ROI использовано 1,05 мм/Y и 0,6 мм/X: это номинальный масштаб сканера, '
            'его нет в DICOM. Зазоры указаны в сантиметрах. '
            '<a href="../roi_mismatch_review/index.html">Разбор 24 расхождений ROI</a>.</p>'
            + "\n".join(sections) + '</html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path,
                        default=workspace / "dxa_project" / "outputs" / "pilot_30")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--per-region", type=int, default=10)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
        result = run(args.workspace, args.output, args.seed, args.per_region)
    print(json.dumps({key: value for key, value in result.items() if key != "boards"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
