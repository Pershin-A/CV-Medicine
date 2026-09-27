"""Render all ROI disagreements between Excel and nominal pixel geometry."""
from __future__ import annotations

import argparse
import csv
import html
import json
import warnings
from pathlib import Path

from PIL import Image, ImageDraw

from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.generate import _read_dicom
from dxa_project.augmentation.pilot_30 import _display_image


def _dashed(draw, points, color="#41dce5", width=2, segment=9):
    (x1, y1), (x2, y2) = points
    length = abs(x2 - x1) + abs(y2 - y1)
    for offset in range(0, int(length), segment * 2):
        start, stop = offset / length, min(offset + segment, length) / length
        draw.line((x1 + (x2 - x1) * start, y1 + (y2 - y1) * start,
                   x1 + (x2 - x1) * stop, y1 + (y2 - y1) * stop),
                  fill=color, width=width)


def render(workspace: Path, audit_path: Path, output: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cases = audit["roi_reference_comparison"]["mismatches"]
    with (workspace / "Размеченные" / "labels.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        labels = {row["relative_path"].replace("\\", "/"): row
                  for row in csv.DictReader(stream)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    cards = []
    for case in cases:
        rel = case["relative_path"].replace("\\", "/")
        label = labels[rel]
        side = label["side"].upper()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
            ds, pixels = _read_dicom(workspace / "Исследования" / rel)
        raw = json.loads((workspace / "Размеченные" /
                          label["geometry_path"]).read_text(encoding="utf-8"))
        geometry = prepare_geometry(raw, "LEG_" + side)
        box = geometry["hip"]["roi_box"]
        w, h = geometry["image_width"], geometry["image_height"]
        factor = 2
        image = _display_image(ds, pixels).resize(
            (w * factor, h * factor), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(v * factor for v in box), outline="#cf65ff", width=3)
        top_y, bottom_y = 30 / 1.05, (h - 1) - 30 / 1.05
        lateral_x = ((w - 1) - 20 / 0.6 if side == "LEFT" else 20 / 0.6)
        _dashed(draw, ((0, top_y * factor), (w * factor, top_y * factor)))
        _dashed(draw, ((0, bottom_y * factor), (w * factor, bottom_y * factor)))
        _dashed(draw, ((lateral_x * factor, 0), (lateral_x * factor, h * factor)))
        for point in geometry["hip"]["landmarks"].values():
            if point is not None:
                x, y = point[0] * factor, point[1] * factor
                draw.ellipse((x - 4, y - 4, x + 4, y + 4),
                             fill="#37dcff", outline="black")
        filename = f"row{case['label_row']:03d}.png"
        image.save(output / "images" / filename, optimize=True)
        margins = case["margins_cm"]
        distances = "".join(
            f"<td class='{('fail' if value < threshold else 'pass')}'>"
            f"{value:.2f} / {threshold:.0f}</td>"
            for value, threshold in zip(
                (margins["top"], margins["bottom"], margins["lateral"]),
                (3, 3, 2)))
        cards.append(
            f"<article><h2>№{case['label_row']} · {'левое' if side == 'LEFT' else 'правое'} бедро</h2>"
            f"<img src='images/{filename}' alt='Снимок №{case['label_row']} с ROI и порогами'>"
            f"<p>Табличная метка: <b>{case['reference_violation']}</b> · "
            f"по номинальному масштабу: <b>{case['nominal_violation']}</b></p>"
            f"<table><tr><th>Верх, см</th><th>Низ, см</th><th>Бок, см</th></tr>"
            f"<tr>{distances}</tr></table>"
            f"<small>{html.escape(rel)}</small></article>"
        )
    count_good_to_bad = sum(c["reference_violation"] == 0 for c in cases)
    page = (
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>ROI — 24 расхождения</title>"
        "<style>body{background:#101820;color:#f4f7fb;font:16px system-ui;"
        "margin:24px;max-width:1400px}main{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(480px,1fr));gap:20px}article{background:#1b2834;"
        "padding:18px;border-radius:12px}img{display:block;max-width:100%;"
        "height:auto;margin:auto}h1,h2{margin:0 0 12px}p{line-height:1.5}"
        "table{border-collapse:collapse;width:100%}td,th{padding:7px;border:"
        "1px solid #536577;text-align:center}.fail{color:#ff7373}.pass{color:"
        "#8bedad}small{display:block;color:#a6b9ca;overflow-wrap:anywhere;"
        "margin-top:12px}</style>"
        "<h1>Почему расходятся метки ROI</h1>"
        "<p>Подложка взята непосредственно из папки «Исследования». "
        "Фиолетовая рамка — размеченный ROI после продления произвольной "
        "стороны до края снимка. Голубой пунктир — пороги: отступ 3 см сверху "
        "и снизу, 2 см с боковой стороны. Бирюзовые точки — ориентиры бедра. "
        "В таблице каждой карточки показано «измеренный зазор / требуемый». "
        "Красным выделен зазор ниже порога. Метка 1 означает нарушение.</p>"
        f"<p>Сравнено {audit['roi_reference_comparison']['checked']} снимков: "
        f"{len(cases)} расхождения. В {count_good_to_bad} случаях таблица даёт 0, "
        f"а геометрия по номинальному масштабу — 1; в {len(cases)-count_good_to_bad} "
        "случаях наоборот. Масштаб 1,05 мм/Y и 0,6 мм/X указан для сканера, "
        "но физический размер пикселя в DICOM отсутствует. Поэтому это "
        "разбор расхождений, а не доказательство ошибки в таблице. "
        "Дополнительная сверка показала совпадение пикселей, UID и размеров "
        "исходной и размеченной копий на всех 499 снимках; каждой строке "
        "таблицы соответствует правильный исходный путь.</p>"
        "<main>" + "".join(cards) + "</main></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return {"cases": len(cases), "table_ok_nominal_bad": count_good_to_bad,
            "table_bad_nominal_ok": len(cases) - count_good_to_bad}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--audit", type=Path,
                        default=workspace / "dxa_project" / "outputs" / "spacing_audit.json")
    parser.add_argument("--output", type=Path,
                        default=workspace / "dxa_project" / "outputs" /
                        "roi_mismatch_review")
    args = parser.parse_args()
    print(json.dumps(render(args.workspace.resolve(), args.audit.resolve(),
                            args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
