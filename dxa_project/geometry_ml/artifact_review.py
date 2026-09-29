"""Prepare the eight annotation discrepancies for human review, without edits."""
from pathlib import Path
import json,csv,html
import numpy as np
from PIL import Image
from .data import load_records,read_dicom_image
from dxa_project.augmentation.pilot_30 import _overlay

def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'dxa_project/outputs/improvements_v2/artifact_review';out.mkdir(parents=True,exist_ok=True)
    with (root/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:reference={r['relative_path']:r for r in csv.DictReader(f)}
    with (root/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:
        numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(csv.DictReader(f),1)}
    rows=[];parts=['<h1>Расхождения рамок и бинарной метки артефакта</h1><p>Это очередь проверки, данные не изменены. Рамка не обязательно означает именно нарушение, указанное в таблице: сначала подтвердить смысл разметки.</p>']
    for r in load_records(root):
        if r.region!='SPINE':continue
        g=json.loads(r.geometry_path.read_text(encoding='utf-8'));value=reference[r.relative_path].get('spine_artifact','')
        if value in ('',None) or int(float(value))==int(bool(g['spine']['foreign_objects'])):continue
        i=len(rows)+1;name=f'case_{i}.png'
        base=Image.fromarray((read_dicom_image(r.source_path)*255).astype(np.uint8)).convert('RGB')
        _overlay(base,g,r.region).save(out/name)
        rows.append({'case':i,'image_number':numbers[r.relative_path],'relative_path':r.relative_path,'author_label':int(float(value)),
                     'has_visual_boxes':int(bool(g['spine']['foreign_objects'])),'review_decision':''})
        parts.append(f'<article><h2>Снимок №{numbers[r.relative_path]}</h2><img src="{name}"><p>Таблица: {value}; рамки: {len(g["spine"]["foreign_objects"])}.</p><details><summary>Путь</summary>{html.escape(r.relative_path)}</details></article>')
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','image_number','relative_path','author_label','has_visual_boxes','review_decision']);writer.writeheader();writer.writerows(rows)
    if not rows:parts.append('<p>Расхождений нет. Подтвержденные исправления от 29 сентября учтены.</p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;background:#152130;color:white;padding:25px}article{display:inline-block;width:45%;vertical-align:top;padding:15px}img{max-width:100%;height:500px;object-fit:contain}</style>'+''.join(parts),encoding='utf-8')
    print(json.dumps({'review_cases':len(rows),'output':str(out)}))

if __name__=='__main__':main()
