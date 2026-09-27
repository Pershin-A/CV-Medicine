"""Render every original DICOM beside its normalized visual annotation."""
from __future__ import annotations

import argparse
import csv
import html
import json
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.generate import _read_dicom
from dxa_project.augmentation.pilot_30 import _display_image, _font, _overlay


def _region(row: dict) -> str:
    label = row.get("label", "").upper()
    if label == "SPINE":
        return "SPINE"
    if label == "LEG" and row.get("side", "").upper() in ("LEFT", "RIGHT"):
        return "LEG_" + row["side"].upper()
    return "UNKNOWN"


def render(workspace: Path, output: Path) -> dict:
    labels_path = workspace / "Размеченные" / "labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    cards, counts = [], Counter()
    for number, row in enumerate(labels, 1):
        rel = row["relative_path"].replace("\\", "/")
        region = _region(row)
        image_path = workspace / "Исследования" / rel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
            ds, pixels = _read_dicom(image_path)
        base = _display_image(ds, pixels)
        geometry_path = workspace / "Размеченные" / row.get("geometry_path", "")
        geometry = None
        if geometry_path.is_file():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if geometry is not None and region != "UNKNOWN":
            geometry = prepare_geometry(geometry, region)
            marked = _overlay(base, geometry, region)
        else:
            marked = base.copy()
        trochanter = "not_applicable"
        if region.startswith("LEG_"):
            traces = (geometry or {}).get("hip", {}).get(
                "lesser_trochanter_traces", {})
            if (geometry or {}).get("hip", {}).get("lesser_trochanter_pixels"):
                trochanter = "filled"
            elif traces.get("trochanter"):
                trochanter = "dashed"
            else:
                trochanter = "unmarked"
        counts[region] += 1
        counts[f"trochanter_{trochanter}"] += 1
        board = Image.new("RGB", (640, 350), "#121820")
        draw = ImageDraw.Draw(board)
        for column, picture in enumerate((base, marked)):
            picture = ImageOps.contain(picture, (310, 310))
            x = column * 320 + (320 - picture.width) // 2
            y = 32 + (310 - picture.height) // 2
            board.paste(picture, (x, y))
        draw.text((9, 5), "Исходный снимок", fill="white", font=_font(17))
        draw.text((329, 5), "С разметкой", fill="white", font=_font(17))
        filename = f"row{number:03d}.jpg"
        board.save(output / "images" / filename, quality=90, optimize=True)
        description = {
            "filled": "малый вертел: красная заливка",
            "dashed": "малый вертел: красный пунктир",
            "unmarked": "малый вертел: контур не отмечен",
            "not_applicable": "",
        }[trochanter]
        cards.append(
            f"<article data-number='{number}' data-region='{region}' data-trochanter='{trochanter}'>"
            f"<h2>№{number} · {region}</h2>"
            f"<img loading='lazy' src='images/{filename}' "
            f"alt='Исходный снимок №{number} и его разметка'>"
            f"<p>{description}</p><small>{html.escape(rel)}</small></article>"
        )
    page = (
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>Все исходные DXA с разметкой</title>"
        "<style>body{background:#101820;color:#f4f7fb;font:16px system-ui;"
        "margin:24px}header{position:sticky;top:0;background:#101820;padding:"
        "0 0 12px;z-index:1}main{display:grid;grid-template-columns:repeat("
        "auto-fit,minmax(640px,1fr));gap:18px}article{background:#1b2834;"
        "padding:14px;border-radius:12px}h1,h2{margin:0 0 12px}img{max-width:"
        "100%;height:auto}p{margin:10px 0}small{color:#a6b9ca;overflow-wrap:"
        "anywhere}select{padding:7px;margin:0 12px 0 5px;font-size:16px}"
        "article[hidden]{display:none}</style>"
        "<header><h1>Все исходные DXA с разметкой</h1>"
        f"<p>Всего {len(labels)} снимков. Позвоночник: {counts['SPINE']}; "
        f"левое бедро: {counts['LEG_LEFT']}; правое бедро: {counts['LEG_RIGHT']}. "
        "На бедре красным закрашены пиксели малого вертела; при нулевой "
        "площади его видимый контур показан красным пунктиром.</p>"
        "<label>Область<select id='region'><option value='all'>Все</option>"
        "<option value='SPINE'>Позвоночник</option>"
        "<option value='LEG_LEFT'>Левое бедро</option>"
        "<option value='LEG_RIGHT'>Правое бедро</option>"
        "<option value='UNKNOWN'>Не определено</option></select></label>"
        "<label>Малый вертел<select id='trochanter'>"
        "<option value='all'>Все</option>"
        "<option value='filled'>Красная заливка</option>"
        "<option value='dashed'>Красный пунктир</option>"
        "<option value='unmarked'>Нет контура</option></select></label>"
        "<label>Номера через запятую <input id='numbers' "
        "placeholder='24, 77, 96' style='padding:7px;font-size:16px'></label>"
        "</header><main>" + "".join(cards) + "</main>"
        "<script>function filter(){const r=document.getElementById('region').value;"
        "const t=document.getElementById('trochanter').value;"
        "const raw=document.getElementById('numbers').value;"
        "const nums=new Set(raw.match(/\\d+/g)||[]);"
        "for(const card of document.querySelectorAll('article'))"
        "card.hidden=(r!=='all'&&card.dataset.region!==r)||"
        "(t!=='all'&&card.dataset.trochanter!==t)||"
        "(nums.size>0&&!nums.has(card.dataset.number))}"
        "document.getElementById('region').onchange=filter;"
        "document.getElementById('trochanter').onchange=filter;"
        "document.getElementById('numbers').oninput=filter;</script></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return {"total": len(labels), "counts": dict(counts),
            "html": str(output / "index.html")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path,
                        default=workspace / "dxa_project" / "outputs" /
                        "original_annotations")
    args = parser.parse_args()
    print(json.dumps(render(args.workspace.resolve(), args.output.resolve()),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
