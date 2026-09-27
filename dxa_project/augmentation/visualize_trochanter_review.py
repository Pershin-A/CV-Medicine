"""Build a visual review gallery for every annotated hip contour pair."""
from __future__ import annotations

import argparse
import csv
import html
import json
import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.generate import _read_dicom
from dxa_project.augmentation.pilot_30 import _display_image, _font, _overlay


def render(workspace: Path, output: Path) -> dict:
    with (workspace / "Размеченные" / "labels.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    with (workspace / "dxa_project" / "outputs" / "manifest.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        references = {row["relative_path"].replace("\\", "/"): row
                      for row in csv.DictReader(stream)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    cards = []
    counts = {"zero": 0, "positive": 0}
    for number, row in enumerate(labels, 1):
        if row.get("label", "").upper() != "LEG":
            continue
        side = row.get("side", "").upper()
        if side not in ("LEFT", "RIGHT"):
            continue
        geometry_path = workspace / "Размеченные" / row.get("geometry_path", "")
        if not geometry_path.is_file():
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        traces = geometry["hip"]["lesser_trochanter_traces"]
        if not all(traces.get(name) for name in ("trochanter", "adjacent_bone")):
            continue
        geometry = prepare_geometry(geometry, "LEG_" + side)
        rel = row["relative_path"].replace("\\", "/")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
            ds, pixels = _read_dicom(workspace / "Размеченные" / rel)
        base = _display_image(ds, pixels)
        marked = _overlay(base, geometry, "LEG_" + side)
        area = len(geometry["hip"]["lesser_trochanter_pixels"])
        status = "positive" if area else "zero"
        counts[status] += 1
        thumbnail = Image.new("RGB", (520, 288), "#121820")
        draw = ImageDraw.Draw(thumbnail)
        for col, picture in enumerate((base, marked)):
            picture = ImageOps.contain(picture, (250, 250))
            x = col * 260 + (260 - picture.width) // 2
            y = 28 + (250 - picture.height) // 2
            thumbnail.paste(picture, (x, y))
        draw.text((8, 5), "Исходное", fill="white", font=_font(15))
        draw.text((268, 5), "Расчёт", fill="white", font=_font(15))
        filename = f"row{number:03d}.jpg"
        thumbnail.save(output / "images" / filename, quality=88, optimize=True)
        ref = references.get(rel, {})
        rotation = ref.get(f"{side.lower()}_hip_rotation", "")
        cards.append({
            "number": number, "status": status,
            "html": (f"<article data-status='{status}'><h2>№{number} · "
                     f"{'левое' if side == 'LEFT' else 'правое'} бедро</h2>"
                     f"<img loading='lazy' src='images/{filename}' alt='Снимок №{number}'>"
                     f"<p>Площадь {area} px² · ротация в таблице "
                     f"{html.escape(str(rotation) or '?')}</p></article>"),
        })
    cards.sort(key=lambda item: (item["status"] != "zero", item["number"]))
    page = (
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>Проверка малого вертела</title>"
        "<style>body{background:#101820;color:#f4f7fb;font:16px system-ui;"
        "margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(520px,1fr));gap:16px}article{background:#1b2834;padding:12px;"
        "border-radius:10px}h2{margin:0 0 10px}img{max-width:100%;height:auto}"
        "p{margin:10px 0 0}select{padding:8px;font-size:16px}</style>"
        "<h1>Проверка выделения малого вертела</h1>"
        f"<p>Всего {sum(counts.values())} пар контуров; красная маска: "
        f"{counts['positive']}, только пунктир: {counts['zero']}. "
        "Справа показан расчёт после подготовки исходной разметки. "
        "Нулевая площадь означает отсутствие подходящего участка расширения "
        "между контурами; это решение алгоритма, которое можно проверить визуально.</p>"
        "<label>Показать: <select id='filter'><option value='all'>Все</option>"
        "<option value='zero'>Только нулевая площадь</option>"
        "<option value='positive'>Только маска</option></select></label>"
        "<main>" + "".join(item["html"] for item in cards) + "</main>"
        "<script>document.getElementById('filter').onchange=e=>{"
        "for(const card of document.querySelectorAll('article'))"
        "card.hidden=e.target.value!=='all'&&card.dataset.status!==e.target.value;"
        "}</script></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--output", type=Path,
                        default=workspace / "dxa_project" / "outputs" /
                        "trochanter_review")
    args = parser.parse_args()
    print(json.dumps(render(args.workspace.resolve(), args.output.resolve()),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
