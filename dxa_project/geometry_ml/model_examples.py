"""Separate successes/errors for every deployed module and both point variants."""
from pathlib import Path
from copy import deepcopy
import csv,json,html,hashlib
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from .data import load_records,read_dicom_image,LANDMARKS,CRESTS
from .predict import predict_file,_load_model
from .landmarks import predict_points
from .train import _box_iou
from .build_team_report import filtered,REGIONS
from dxa_project.augmentation.core import prepare_geometry,hip_roi_ok
from dxa_project.augmentation.pilot_30 import _overlay

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def point_score(gt,pred,keys,spacing,tolerance):
    distances=[];missing=extra=0
    for key in keys:
        a,b=gt.get(key),pred.get(key)
        if a is None and b is not None:extra+=1
        if a is not None and b is None:missing+=1
        if a is not None and b is not None:distances.append(float(np.hypot((a[0]-b[0])*spacing[1],(a[1]-b[1])*spacing[0])))
    worst=max(distances,default=0);passed=missing==extra==0 and worst<=tolerance
    text=f'Средняя ошибка: {np.mean(distances):.1f} мм; максимум: {worst:.1f} мм.' if distances else 'Нет пар точек для измерения расстояния.'
    return passed,worst+100*(missing+extra),text+f' Пропущено видимых: {missing}; лишних: {extra}.',bool(distances)

def score(key,gt,pred,record):
    if key in ('points','coordinate','crests'):
        spine=key=='crests';branch='spine' if spine else 'hip';field='iliac_crests' if spine else 'landmarks'
        return point_score(gt[branch][field],pred[branch][field],CRESTS if spine else LANDMARKS,record.spacing_mm,5 if spine else 10)
    if key=='roi':
        a,b=gt['hip']['roi_box'],pred['hip']['roi_box'];iou=_box_iou(a,b) if a is not None and b is not None else 0
        return iou>=.85,1-iou,f'IoU ROI = {iou:.3f}.',True
    if key=='mask':
        a={tuple(p) for p in gt['hip']['lesser_trochanter_pixels']};b={tuple(p) for p in pred['hip']['lesser_trochanter_pixels']}
        dice=2*len(a&b)/(len(a)+len(b)) if a or b else 1
        return dice>=.7,1-dice,f'Dice = {dice:.3f}; площадь эталона / модели: {len(a)} / {len(b)} пикселей.',bool(a)
    if key=='artifact':
        a=[o['bbox'] for o in gt['spine']['foreign_objects']];b=[o['bbox'] for o in pred['spine']['foreign_objects']]
        matches=0
        if a and b:
            ious=np.asarray([[_box_iou(x,y) for y in b] for x in a])
            costs=np.where(ious>=.5,-(min(len(a),len(b))+1)-ious,0)
            i,j=linear_sum_assignment(costs)
            matches=sum(ious[x,y]>=.5 for x,y in zip(i,j))
        missing=len(a)-matches;extra=len(b)-matches
        return missing==extra==0,missing+extra,f'Рамок эталон / модель: {len(a)} / {len(b)}; совпало при IoU ≥ 0,5: {matches}; пропуски: {missing}; лишние: {extra}.',bool(a)
    a=gt['spine']['disc_lines'];b=pred['spine']['disc_lines'];distances=[]
    if a and b:
        xs=np.asarray([.25,.5,.75])*(gt['image_width']-1)
        def positions(line):
            p,q=line['points'];return p[1]+(xs-p[0])*(q[1]-p[1])/max(q[0]-p[0],1e-6)
        costs=np.asarray([[np.mean(abs(positions(x)-positions(y)))*record.spacing_mm[0] for y in b] for x in a])
        i,j=linear_sum_assignment(costs);distances=costs[i,j].tolist()
    error=float(np.mean(distances)) if distances else 100
    mismatch=abs(len(a)-len(b));passed=not mismatch and error<=5
    return passed,error+50*mismatch,f'Линий эталон / модель: {len(a)} / {len(b)}; среднее вертикальное отклонение сопоставленных линий: {error:.1f} мм.',True

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2';public=root/'dxa_project/team_demo'
    assets=public/'model_example_assets';assets.mkdir(exist_ok=True)
    labelrows=rows(root/'Размеченные/labels.csv');numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(labelrows,1)}
    sorted_files=sorted((p.relative_to(root/'Исследования').as_posix() for p in (root/'Исследования').rglob('*') if p.is_file() and p.suffix.lower()=='.dcm'),key=str.casefold)
    ui_numbers={r:i for i,r in enumerate(sorted_files,1)}
    records={r.relative_path:r for r in load_records(root)};reference={r['relative_path']:r for r in rows(root/'dxa_project/outputs/manifest.csv')}
    saved=read(out/'pipeline/evaluation/predictions.json');predictions={r['relative_path']:read(out/f'pipeline/evaluation/prediction_{i:03}.json') for i,r in enumerate(saved,1)}
    cache={};device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def example(rel):
        if rel not in cache:
            r=records[rel];image=read_dicom_image(r.source_path);gt=prepare_geometry(read(r.geometry_path),r.region)
            p=predictions[rel]
            if p['region']!=r.region:p=predict_file(r.source_path,out/'pipeline',force_region=r.region)
            cache[rel]=(image,gt,p)
        return cache[rel]
    alternate={}
    coord_model,coord_size=_load_model('hip_points',out/'coordinate_pipeline',device)
    sections=[('router','Определение области тела (router)','Верная область тела.'),
              ('lines','Модель позвоночника: межпозвоночные линии (spine lines)','Количество совпало; среднее вертикальное отклонение ≤ 5 мм.'),
              ('crests','Модель позвоночника: подвздошные точки (iliac crests)','Нет пропусков/лишних точек; максимальная ошибка ≤ 5 мм.'),
              ('artifact','Детектор артефактов (artifact detector)','Все рамки сопоставлены при IoU ≥ 0,5; нет пропусков и лишних.'),
              ('points','Отдельная сеть точек: shared256 / карты точек (heatmap)','Нет пропусков/лишних точек; максимальная ошибка ≤ 10 мм.'),
              ('coordinate','Отдельная сеть точек: coordinate256 / координатный loss','Нет пропусков/лишних точек; максимальная ошибка ≤ 10 мм.'),
              ('roi','Модель бедра: область интереса (hip ROI)','IoU ≥ 0,85.'),
              ('mask','Модель бедра: малый вертел (segmentation)','Dice ≥ 0,70; среди успехов приоритет непустым эталонным маскам.')]
    parts=['<h1>Успехи и ошибки каждого модуля DXA</h1><p><a href="improvements.html">Итоговые метрики</a> · <a href="#review">Номера для ручной проверки</a></p>',
           '<p>Текущие веса improvements_v2, прежние 100 контрольных оригиналов. По возможности показаны пять успехов и пять ошибок на модуль, с разными пиксельными массивами. Если успехов меньше, показывается фактическое число и лучшие, но ещё несовершенные результаты. Это отбор иллюстраций, не дополнительная оценка качества.</p>',
           '<p>Номера совпадают с галереей всех исходных снимков: строки labels.csv, начиная с 1. Номер в приложении без фильтра отдельно указан, если отличается. Для геометрических модулей задана верная область тела, чтобы ошибка маршрутизатора не подменяла ошибку локализации. Допуски ниже служат отбору примеров, не являются клиническими критериями.</p>',
           '<nav>'+ ' · '.join(f'<a href="#{k}">{html.escape(t)}</a>' for k,t,_ in sections)+'</nav>']
    summary={}
    for key,title,criterion in sections:
        candidates=[]
        for row in saved:
            rel=row['relative_path'];r=records[rel]
            if key!='router' and (r.region=='SPINE')!=(key in ('lines','crests','artifact')):continue
            image,gt,p=example(rel)
            if key=='router':
                actual=predictions[rel];passed=actual['region']==r.region;severity=1-max(actual['router_probabilities'].values())
                text=f'Эталон: {REGIONS[r.region]}; модель: {REGIONS[actual["region"]]}.';target=True
            else:
                geometry=p['geometry']
                if key=='coordinate':
                    if rel not in alternate:
                        points,_=predict_points(coord_model,image,coord_size,r.region.removeprefix('LEG_'))
                        alternate[rel]=deepcopy(gt);alternate[rel]['hip']['landmarks']=points
                    geometry=alternate[rel]
                passed,severity,text,target=score(key,gt,geometry,r)
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            candidates.append(dict(rel=rel,passed=bool(passed),severity=float(severity),text=text,target=target,digest=digest))
        def unique(items):
            selected=[];seen=set()
            for c in items:
                if c['digest'] not in seen:seen.add(c['digest']);selected.append(c)
                if len(selected)==5:break
            return selected
        good=unique(sorted((c for c in candidates if c['passed']),key=lambda c:(not c['target'],c['severity'])))
        bad=unique(sorted((c for c in candidates if not c['passed']),key=lambda c:-c['severity']))
        groups=[('Успехи',good),('Ошибки',bad)]
        if not good:groups.insert(0,('Лучшие из имеющихся — критерий успеха не достигнут',unique(sorted(candidates,key=lambda c:c['severity']))))
        parts.append(f'<section id="{key}"><h2>{title}</h2><p>Критерий отбора успеха: {criterion}</p>');summary[key]=[]
        for label,selected in groups:
            parts.append(f'<h3>{label}: {len(selected)} примеров</h3>')
            if not selected:parts.append('<p>Таких случаев в контрольной части нет.</p>')
            for index,c in enumerate(selected,1):
                rel=c['rel'];r=records[rel];image,gt,p=example(rel);base=Image.fromarray((image*255).astype('uint8')).convert('RGB')
                n=numbers[rel];label_number=f'№{n}'+(f' · номер в приложении без фильтра: {ui_numbers[rel]}' if ui_numbers[rel]!=n else '')
                parts.append(f'<article><h4>{label_number} · {REGIONS[r.region]} · '+('Успех' if c['passed'] else 'Ошибка')+'</h4><div class="grid">')
                pictures=[('original',base,'Исходный снимок')]
                if key!='router':
                    geometry=alternate[rel] if key=='coordinate' else p['geometry'];overlaykey='points' if key=='coordinate' else key
                    pictures.extend([('truth',_overlay(base.copy(),filtered(gt,overlaykey),r.region),'Эталонная разметка'),
                                     ('prediction',_overlay(base.copy(),filtered(geometry,overlaykey),r.region),'Предсказание модели')])
                for suffix,picture,caption in pictures:
                    name=f'{key}_{n}_{suffix}.png';picture.thumbnail((420,480));picture.save(assets/name)
                    parts.append(f'<figure><img loading="lazy" src="model_example_assets/{name}"><figcaption>{caption}</figcaption></figure>')
                parts.append('</div><p>'+html.escape(c['text'])+'</p></article>')
                summary[key].append({'image_number':n,'unfiltered_labeler_number':ui_numbers[rel],'category':label,'success':c['passed'],'measurement':c['text']})
        parts.append('</section>')
    review=[]
    for item in rows(out/'artifact_review/review.csv'):
        rel=item['relative_path'];r=records[rel];gt=read(r.geometry_path);boxes=len(gt['spine']['foreign_objects'])
        review.append({'image_number':numbers[rel],'unfiltered_labeler_number':ui_numbers[rel],'reason':'artifact_label_vs_visual_boxes',
                       'table_label':int(item['author_label']),'visual_box_count':boxes})
        base=Image.fromarray((read_dicom_image(r.source_path)*255).astype('uint8')).convert('RGB')
        picture=_overlay(base,filtered(gt,'artifact'),r.region);picture.thumbnail((420,480));picture.save(assets/f'review_{numbers[rel]}.png')
    roi_review=[]
    for rel,r in records.items():
        if r.region=='SPINE':continue
        key=('left_' if r.region=='LEG_LEFT' else 'right_')+'hip_roi';value=reference[rel].get(key,'')
        if value in ('',None):continue
        gt=prepare_geometry(read(r.geometry_path),r.region)
        if gt['hip']['roi_box'] is None:continue
        expected=int(not hip_roi_ok(gt,True,r.region.removeprefix('LEG_'),r.spacing_mm))
        if expected!=int(float(value)):roi_review.append({'image_number':numbers[rel],'table_label':int(float(value)),'geometry_label':expected})
    parts.append('<h2 id="review">Ручная проверка разметки: конкретные номера</h2><p><b>'+', '.join('№'+str(r['image_number']) for r in review)+'</b></p><p>Это расхождения эталонов, а не список ошибок модели. Флаг 1 означает нарушение. Наличие рамки само по себе ещё не доказывает ошибку таблицы; уточните смысл объекта. Исходная разметка не изменена.</p>')
    for item in review:
        n=item['image_number'];parts.append(f'<article><h3>№{n}</h3><img src="model_example_assets/review_{n}.png"><p>Табличный флаг артефакта: {item["table_label"]}; рамок в визуальной разметке: {item["visual_box_count"]}.</p></article>')
    parts.append('<h3>Перепроверка ROI по номинальному масштабу</h3>')
    parts.append('<p>Расхождений не осталось.</p>' if not roi_review else '<p>Дополнительные кандидаты: '+', '.join(f'№{x["image_number"]} (таблица {x["table_label"]}, геометрия {x["geometry_label"]})' for x in roi_review)+'. Это проверка при номинальном масштабе; она не доказывает ошибку метки.</p>')
    (public/'model_examples.json').write_text(json.dumps({'modules':summary,'manual_review':review,'roi_nominal_review':roi_review},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    document='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DXA — успехи и ошибки моделей</title><style>body{font:16px system-ui;line-height:1.5;max-width:1450px;margin:24px auto;padding:20px;background:#111a23;color:#ecf3fa}a{color:#96ceff}article{background:#1b2938;padding:18px;margin:18px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}figure{margin:0}img{max-width:100%;max-height:480px}section{margin-top:50px}nav{line-height:2}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>'+''.join(parts)+'</html>'
    (public/'model_examples.html').write_text(document,encoding='utf-8')
    for filename in ('improvements.html','index.html'):
        path=public/filename;text=path.read_text(encoding='utf-8')
        if 'href="model_examples.html"' not in text:text=text.replace('</h1>','</h1><p><a href="model_examples.html">Успехи и ошибки каждого модуля; номера для ручной проверки</a></p>',1)
        path.write_text(text,encoding='utf-8')
    print(json.dumps({'page':str(public/'model_examples.html'),'manual_review_numbers':[r['image_number'] for r in review],
                      'roi_nominal_review':roi_review,'sections':{k:len(v) for k,v in summary.items()}},ensure_ascii=False))

if __name__=='__main__':main()
