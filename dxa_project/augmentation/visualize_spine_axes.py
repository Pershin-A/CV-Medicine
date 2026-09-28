"""Read-only gallery of 50 spine originals, annotations and L1 fitted axes."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import warnings

import numpy as np
import pydicom
from PIL import ImageDraw

from .core import prepare_geometry
from .generate import _read_rows, _source_spacing
from .pilot_30 import _display_image, _overlay
from .vertebral_axes import analyze_spine, placement_from_axes


def draw_axes(base, report, detailed=False):
    image=base.copy()
    draw=ImageDraw.Draw(image)
    for axis in report['axes']+[report['upper_fragment']]:
        if not axis:
            continue
        if detailed and axis.get('roi_polygon'):
            corners=axis['roi_polygon']
            draw.line([tuple(p) for p in corners+[corners[0]]],fill='#61baff',width=1)
            for section in axis.get('symmetric_sections',axis['sections']):
                x,y=section['point']
                draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill='#ff9e32')
        if axis['valid']:
            draw.line([tuple(p) for p in axis['axis_points']],fill='#ffec4c',width=2)
    if report.get('joint_fit',{}).get('success'):
        for x,y in report['joint_fit']['joint_points']:
            draw.ellipse((x-2.5,y-2.5,x+2.5,y+2.5),fill='#f066dd')
    return image


def run(root,output,count=50):
    if (output/'report.json').exists():
        raise FileExistsError('Use a new output folder to preserve previous diagnostic results')
    output.mkdir(parents=True,exist_ok=True)
    for folder in ('originals','axes','diagnostics','geometry_analysis'):
        (output/folder).mkdir(exist_ok=True)
    labels=_read_rows(root/'Размеченные/labels.csv')
    spines=[(n,row) for n,row in enumerate(labels,1) if row['label']=='SPINE']
    indices=np.linspace(0,len(spines)-1,min(count,len(spines))).round().astype(int)
    selected=[spines[i] for i in indices]
    cards=[]
    metric_rows=[]
    summaries=[]
    for n,row in selected:
        ds=pydicom.dcmread(root/'Исследования'/row['relative_path'],force=True)
        pixels=ds.pixel_array
        geometry=prepare_geometry(json.loads((root/'Размеченные'/row['geometry_path']).read_text(encoding='utf-8')),'SPINE')
        spacing,basis=_source_spacing(ds)
        report=analyze_spine(pixels,geometry,spacing,
                             polarity='dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright')
        report['scan']=n
        report['spacing_basis']=basis
        report['spine_position_ok']=placement_from_axes(geometry,report)
        base=_display_image(ds,pixels)
        annotated=_overlay(base,geometry,'SPINE')
        base.save(output/'originals'/f'{n}.png')
        draw_axes(annotated,report).save(output/'axes'/f'{n}.png')
        draw_axes(annotated,report,True).save(output/'diagnostics'/f'{n}.png')
        (output/'geometry_analysis'/f'{n}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        angle=report['global_angle_deg']
        angle_text='не определён' if angle is None else f'{angle:.2f}°'
        ratio=report['top_ratio']
        ratio_text='?' if ratio is None else f'{ratio:.2f}'
        reasons=sorted({reason for a in report['axes']+[report['upper_fragment']] if a for reason in a['review_reasons']})
        axes_summary=[]
        for i,axis in enumerate(report['axes']+[report['upper_fragment']]):
            if not axis:
                continue
            record={'scan':n,'vertebra':'Th12_fragment' if axis['kind']=='upper_fragment' else i+1,
                    'valid':axis['valid'],'angle_deg':axis.get('angle_deg'),
                    'shift_mm':axis.get('shift_mm'),'error_mm':axis.get('mean_absolute_error_mm'),
                    'sections':axis.get('valid_sections',0),
                    'review_reasons':'|'.join(axis['review_reasons'])}
            metric_rows.append(record)
            if axis['valid']:
                axes_summary.append(f"{record['vertebra']}: {record['angle_deg']:.1f}°, E={record['error_mm']:.2f} мм, N={record['sections']}")
            else:
                axes_summary.append(f"{record['vertebra']}: не определена ({record['review_reasons']})")
        valid=sum(a['valid'] for a in report['axes'])
        summaries.append({'scan':n,'full_axes':len(report['axes']),'valid_axes':valid,
                          'global_angle_deg':angle,'top_ratio':ratio,
                          'review_required':report['review_required'],'reasons':reasons})
        details=html.escape('\n'.join(axes_summary))
        cards.append(f'<article data-review="{int(report["review_required"])}"><h2>№{n}</h2>'
                     f'<img class="original" src="originals/{n}.png" loading="lazy">'
                     f'<img class="annotated" data-simple="axes/{n}.png" data-detail="diagnostics/{n}.png" src="diagnostics/{n}.png" loading="lazy">'
                     f'<p>Общая ось: {angle_text}<br>Оси: {valid}/{len(report["axes"])} · Th12/шаг: {ratio_text}</p>'
                     f'<p>{"Нужен просмотр" if report["review_required"] else "Автоматические проверки пройдены"}</p>'
                     f'<details><summary>Метрики позвонков</summary><pre>{details}</pre></details></article>')
    result={'selected_images':len(selected),'full_axes':sum(s['full_axes'] for s in summaries),
            'valid_full_axes':sum(s['valid_axes'] for s in summaries),
            'images_requiring_review':sum(s['review_required'] for s in summaries),
            'selection':'50 evenly spaced scans in the authoritative spine registry',
            'augmentation_generated':False,'images':summaries}
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'axes_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]))
        writer.writeheader();writer.writerows(metric_rows)
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>50 позвоночников: оси</title>
<style>body{background:#111a23;color:#eef5fb;font:15px system-ui;margin:20px}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:15px}article{min-width:0}h2{font-size:18px}img{display:block;width:100%;height:350px;object-fit:contain;background:#080d12;margin:6px 0}p{font-size:13px}pre{white-space:pre-wrap;font-size:11px}header{position:sticky;top:0;background:#111a23;padding:12px;z-index:1}a{color:#91caff}</style>
<header><h1>50 исходных позвоночников: контуры и оси по геометрическим серединам</h1><p>В каждом столбце сверху оригинал, снизу тот же снимок с разметкой и осями. Шесть пар в строке. Зелёные линии — разметка; жёлтые — совместные непрерывные оси; розовые точки — общие узлы; синий — контур позвонка; оранжевые точки — середины между боковыми границами. Углы и ошибки рассчитаны в миллиметрах при номинале 1,05 мм/Y, 0,6 мм/X. Предупреждения означают необходимость просмотра, а не ошибочную ручную разметку.</p>
<label><input id="detail" type="checkbox" checked>Показать контуры и середины</label> <label><input id="review" type="checkbox">Только с предупреждениями</label> <a href="axes_metrics.csv">Таблица метрик</a></header><main class="grid">'''+''.join(cards)+'''</main><script>
document.getElementById('detail').onchange=e=>document.querySelectorAll('.annotated').forEach(img=>img.src=e.target.checked?img.dataset.detail:img.dataset.simple);
document.getElementById('review').onchange=e=>document.querySelectorAll('article').forEach(card=>card.style.display=e.target.checked&&card.dataset.review==='0'?'none':'');
</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/spine_axes_50')
    p.add_argument('--count',type=int,default=50)
    args=p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        run(args.workspace.resolve(),args.output.resolve(),args.count)
