"""Saved-model component comparison and annotation-oracle spine-axis audit."""
import csv
import json
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image, ImageDraw
from .data import load_records
from .model_examples import score
from .compare_reviewed import OUT, ROOT, read
from .report_reviewed import table, n
from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.vertebral_axes import analyze_spine

def alternative_angles(analysis):
    """Diagnostic alternatives in physical space; not deployed predictions."""
    xy=np.asarray(analysis['spacing_mm_row_col'][::-1])
    axes=analysis['axes'];top=analysis.get('upper_fragment')
    including_top=None
    if axes and top and top['valid'] and axes[-1]['valid']:
        delta=(np.asarray(axes[-1]['axis_points'][1])-top['axis_points'][0])*xy
        including_top=float(np.degrees(np.arctan2(delta[0],delta[1])))
    points=analysis.get('joint_fit',{}).get('joint_points',[])
    robust=None
    if len(points)>=3:
        p=np.asarray(points)*xy; weights=np.ones(len(p))
        for _ in range(8):
            center=np.average(p,axis=0,weights=weights);z=p-center
            values,vectors=np.linalg.eigh((z*weights[:,None]).T@z)
            direction=vectors[:,-1]
            if direction[1]<0:direction=-direction
            residual=np.abs(z@np.array([direction[1],-direction[0]]))
            weights=np.minimum(1.,2./np.maximum(residual,1e-9))
        robust=float(np.degrees(np.arctan2(direction[0],direction[1])))
    return {'manual_including_top_angle_deg':including_top,'manual_robust_angle_deg':robust}

def run():
    records={r.relative_path:r for r in load_records(ROOT)}
    with (ROOT/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:
        numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(csv.DictReader(f),1)}
    runs={}; geometries={}; oracle={}
    for kind,folder in [('light',OUT/'light'),('heavy',OUT/'heavy_stratified')]:
        rows=read(folder/'evaluation/predictions.json')
        runs[kind]={row['relative_path']:(row,read(folder/f'evaluation/prediction_{i:03}.json')) for i,row in enumerate(rows,1)}
    common=set(runs['light']) & set(runs['heavy'])
    summary={}; detail={}; axes=[]
    for kind,cache in runs.items():
        component_rows={k:[] for k in ('lines','crests','artifact','points','roi','mask')}
        for rel,(row,pred) in cache.items():
            r=records[rel]
            if rel not in geometries:geometries[rel]=prepare_geometry(read(r.geometry_path),r.region)
            gt=geometries[rel]
            routed=row['predicted_region']==r.region
            if routed:
                for key in (('lines','crests','artifact') if r.region=='SPINE' else ('points','roi','mask')):
                    passed,severity,text,has_target=score(key,gt,pred['geometry'],r)
                    component_rows[key].append({'number':numbers[rel],'relative_path':rel,'study':r.study,'passed':bool(passed),'severity':float(severity),'text':text,'has_target':bool(has_target),'common':rel in common})
            if r.region=='SPINE':
                if rel not in oracle:
                    ds=pydicom.dcmread(str(r.source_path),force=True)
                    raw=np.squeeze(ds.pixel_array)
                    basis=pred.get('spacing_basis')
                    spacing=pred.get('vertebral_axes',{}).get('spacing_mm_row_col',r.spacing_mm)
                    a=analyze_spine(raw,gt,spacing,polarity='dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright')
                    oracle[rel]={'analysis':a,'spacing_basis':basis}
                a=oracle[rel]['analysis'];angle=a['global_angle_deg'];model_angle=pred.get('spine_axis_angle_deg') if routed else None
                axes.append({'model':kind,'number':numbers[rel],'relative_path':rel,'study':r.study,'common':rel in common,'author_flag':row['truth']['spine_axis'],'model_flag':row['predicted'].get('spine_axis') if routed else None,'manual_lines_angle_deg':angle,'manual_lines_flag':int(abs(angle)>5) if angle is not None else None,'model_angle_deg':model_angle,'angle_error_deg':abs(model_angle-angle) if model_angle is not None and angle is not None else None,'gt_line_count':len(gt['spine']['disc_lines']),'model_line_count':len(pred['geometry']['spine']['disc_lines']) if routed else None,'manual_review_required':a['review_required'],'model_review_required':pred.get('vertebral_axes',{}).get('review_required'),'pixel_angle_deg':a['global_angle_pixel_deg'],'spacing_basis':oracle[rel]['spacing_basis'],**alternative_angles(a)})
        detail[kind]=component_rows
        summary[kind]={key:{'evaluated':len(rows),'passed':sum(r['passed'] for r in rows),'studies':len(set(r['study'] for r in rows)),'mean_severity':float(np.mean([r['severity'] for r in rows])) if rows else None,'paired_evaluated':sum(r['common'] for r in rows),'paired_passed':sum(r['common'] and r['passed'] for r in rows)} for key,rows in component_rows.items()}
        print(json.dumps({'component_audit':kind,'spine_oracle_files':len(oracle)}),flush=True)
    axis_summary={}
    for kind in runs:
        rows=[a for a in axes if a['model']==kind]
        known=[a for a in rows if a['manual_lines_flag'] is not None]
        positive=[a for a in rows if a['author_flag']==1]
        axis_summary[kind]={'n':len(rows),'manual_axis_defined':len(known),'manual_flag_author_disagreements':sum(a['manual_lines_flag']!=a['author_flag'] for a in known),'mean_angle_error_deg':float(np.mean([a['angle_error_deg'] for a in rows if a['angle_error_deg'] is not None])),'positives':positive,'manual_axes_review_required':sum(a['manual_review_required'] for a in rows),'model_axes_review_required':sum(bool(a['model_review_required']) for a in rows)}
    result={'scope':'Saved actual-route original test predictions, no training. Component pass thresholds are engineering diagnostics, not clinical thresholds. Oracle means same heuristic with manual lines, not independent true axes. Own tests differ; shared subset is small.', 'component_summary':summary,'component_details':detail,'axis_summary':axis_summary,'axis_details':axes}
    (OUT/'component_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    labels={'lines':'Линии разделения (spine lines)','crests':'Подвздошные точки (iliac crests)','artifact':'Рамки артефактов (artifact boxes)','points':'Три точки бедра (hip landmarks)','roi':'Область интереса (ROI)','mask':'Малый вертел (segmentation)'}
    criteria={'lines':'Количество совпадает, среднее вертикальное отклонение ≤5 мм','crests':'Нет пропусков/лишних, все точки ≤5 мм','artifact':'Все рамки сопоставлены при IoU≥0,5, нет лишних','points':'Нет пропусков/лишних, все точки ≤10 мм','roi':'IoU≥0,85','mask':'Dice≥0,70; обе пустые маски считаются совпадением'}
    parts=['<h1>Сравнение частей пайплайна и диагностика оси</h1><p><a href="model_comparison_20260929.html">Общее сравнение</a></p><p>Использованы сохранённые предсказания на исходных тестовых снимках. Веса, таргеты и пороги не менялись. Оценка геометрии доступна только при правильном определении области; ошибки определения области не скрываются: малая версия 98/100, большая 96/99.</p><h2>Отдельные компоненты: доля примеров, прошедших критерий</h2><p>Это строгие инженерные критерии для диагностики. Таблица не заменяет непрерывные метрики и не доказывает превосходство: собственные тесты разные. Числа — прошедшие / оценённые.</p>',table(['Компонент','Критерий','Малая: свой тест','Большая: свой тест','Малая: общие','Большая: общие'],[[labels[key],criteria[key]]+[f"{summary[kind][key]['passed']}/{summary[kind][key]['evaluated']}" for kind in runs]+[f"{summary[kind][key]['paired_passed']}/{summary[kind][key]['paired_evaluated']}" for kind in runs] for key in labels])]
    parts += ['<h3>Отдельно непустые эталоны</h3><p>Пустые эталоны могут создавать ложное впечатление высокого качества. Здесь исключены снимки без рамок/пикселей в ручной разметке; критерии прежние. Для артефактов требуется найти все рамки без лишних: это строже обнаружения хотя бы одного артефакта.</p>',table(['Компонент','Малая: прошедшие / непустые','Большая: прошедшие / непустые'],[[labels[key]]+[f"{sum(r['passed'] and r['has_target'] for r in detail[kind][key])}/{sum(r['has_target'] for r in detail[kind][key])}" for kind in runs] for key in ('artifact','mask')])]
    parts+=['<h2>Проверка оси с ручными линиями</h2><p>Для каждого тестового позвоночника выполнен тот же алгоритм построения контуров и осей, но вместо линий модели поданы ручные линии. Это позволяет отделить ошибки предсказания линий от проблем геометрического алгоритма и бинарного эталона. Ручные линии не являются независимой разметкой осей: расхождение требует проверки, а не автоматического исправления таблицы.</p>',table(['Модель','Снимков позвоночника','Ось по ручным линиям определена','Расхождений с таблицей','Средняя ошибка угла модели к ручным линиям, °','Оси модели с флагом проверки'],[[kind,s['n'],s['manual_axis_defined'],s['manual_flag_author_disagreements'],n(s['mean_angle_error_deg']),s['model_axes_review_required']] for kind,s in axis_summary.items()])]
    for kind,s in axis_summary.items():
        parts += [f'<h3>{kind}: все табличные нарушения оси</h3>',table(['№','Таблица','Ручные: физический угол, °','Ручные: пиксельный угол, °','Модель: физический угол, °','Линий ручных / модели','Ручной алгоритм требует проверки'],[[a['number'],a['author_flag'],n(a['manual_lines_angle_deg']),n(a['pixel_angle_deg']),n(a['model_angle_deg']),f"{a['gt_line_count']}/{a['model_line_count']}",a['manual_review_required']] for a in s['positives']])]
    mismatches=sorted({a['number'] for a in axes if a['manual_lines_flag'] is not None and a['manual_lines_flag']!=a['author_flag']})
    parts += ['<h2>Номера для ручной проверки оси</h2><p>'+', '.join(map(str,mismatches))+'</p><p>Нумерация — строки Размеченные/labels.csv, как в галерее оригиналов. Проверяется не только бинарная метка: также контуры, крайние позвонки и физический масштаб.</p>', '<h2>Причины и план проверки</h2><ol><li><b>Локализация линий и их декодирование.</b> Общая карта тонких линий даёт слабый Dice; декодер ограничен семью линиями, локальной полосой ±7 пикселей и жёсткими порогами. Проверить альтернативное декодирование на валидации, затем сравнить семь упорядоченных линий с вероятностями или карты концов линий.</li><li><b>Контуры и крайние точки.</b> Рёбра, таз и неполные крайние позвонки влияют на эвристические границы; ошибка общего угла особенно чувствительна к концам. Сравнить текущую линию между крайними осями с устойчивой аппроксимацией всех межпозвоночных узлов в физических координатах. Не менять определение целевого угла без согласования.</li><li><b>Несовпадение геометрии и табличного эталона.</b> Проверить перечисленные снимки вручную. Даже ручные линии не гарантируют правильную ось при ошибке контура; таблицу автоматически не менять.</li><li><b>Синтетический сдвиг распределения.</b> Аугментационные углы известны и отделены от порога. На оригиналах встречаются пограничные углы, рёбра, иной масштаб и наклон. Проверить распределения углов и обучение с большим весом оригиналов; добавить контролируемые примеры вокруг 5° только с явно заданной меткой.</li><li><b>Неопределённость.</b> Инференс выдаёт бинарную метку оси даже при review_required. Измерить ошибки отдельно в надёжной группе и группе проверки; предусмотреть неопределённый ответ при плохой геометрии. Это увеличивает честность вывода, но само по себе не исправляет чувствительность.</li></ol><p>Приоритет: проверка ручных линий и бинарных меток → улучшение декодера → сравнение способов построения общей оси → сравнение архитектур на едином разбиении. Порог нарушения остаётся 5°, его нельзя подбирать под тест.</p>']
    parts += ['<h2>Дополнительная проверка определения оси</h2><p>В текущем коде общий отрезок начинается на первой линии разделения: ось верхнего неполного фрагмента в него не входит. Проверены две альтернативы на ручных линиях: включение верхнего фрагмента и устойчивая прямая через все общие узлы осей (ортогональная аппроксимация с Huber-весами, масштаб 2 мм). Это исследовательские варианты определения, не новые рабочие предсказания.</p>',table(['№','Текущий угол, °','С верхним фрагментом, °','Устойчивая прямая, °'],[[a['number'],n(a['manual_lines_angle_deg']),n(a['manual_including_top_angle_deg']),n(a['manual_robust_angle_deg'])] for a in axes if a['author_flag']==1]), '<p>Пять табличных нарушений — только три исследования и три уникальных пиксельных массива. №264, 267 и 270 имеют одинаковые пиксели, но немного разные ручные линии. В исходных DICOM этих пяти снимков не найдено PixelSpacing, ImagerPixelSpacing или NominalScannedPixelSpacing: физические углы опираются на предоставленный номинальный масштаб 1,05 мм по Y и 0,6 мм по X. Нельзя заключать, что именно такой масштаб использовали организаторы.</p>']
    parts += ['<h2>Варианты улучшения локализации</h2><p>При нынешней разметке можно сравнить семь упорядоченных линий с наличием и параметрами «высота в центре + наклон», либо карты двух концов каждой линии с координатной потерей и ограничением порядка. Концы нужно определять одинаково на фиксированных вертикальных сечениях, поскольку ручные линии продлены до краёв. Использовать координатную ошибку в физических единицах, BCE для наличия и штраф пересечения соседних линий. Дифференцируемый переход от карт точек к координатам описан в <a href="https://arxiv.org/abs/1801.07372">DSNT</a>; это основание для эксперимента, а не доказательство улучшения на DXA.</p><p>Альтернатива после дополнительной разметки — сегментация тел позвонков с построением осей по контурам. Сочетание сегментации и оценки осевой линии используется в <a href="https://arxiv.org/abs/2403.12115">исследовании автоматического измерения угла Кобба</a>, но угол Кобба и общий наклон к вертикали — разные цели; результаты исследования нельзя напрямую переносить на эту задачу.</p>']
    # Draw all tabular positive axis cases, making both geometry stages visible.
    assets=ROOT/'dxa_project/team_demo/component_axis_assets';assets.mkdir(exist_ok=True)
    parts+=['<h2>Все исходные нарушения оси: ручные линии и модель</h2>']
    for a in [x for x in axes if x['author_flag']==1]:
        rel=a['relative_path'];r=records[rel];ds=pydicom.dcmread(str(r.source_path),force=True)
        raw=np.squeeze(ds.pixel_array).astype(float);lo,hi=np.percentile(raw,[.5,99.5]);image=np.clip((raw-lo)/max(hi-lo,1),0,1)
        if ds.PhotometricInterpretation=='MONOCHROME1':image=1-image
        base=Image.fromarray((image*255).astype('uint8')).convert('RGB'); row,pred=runs[a['model']][rel]
        parts.append(f"<h3>№{a['number']} · {a['model']}</h3><div class='grid'>")
        for tag,geo,analysis in [('manual',geometries[rel],oracle[rel]['analysis']),('model',pred['geometry'],pred.get('vertebral_axes',{}))]:
            pic=base.copy();draw=ImageDraw.Draw(pic)
            for line in geo['spine']['disc_lines']:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
            for axis in analysis.get('axes',[]):
                if axis and axis.get('axis_points'):draw.line([tuple(p) for p in axis['axis_points']],fill='cyan',width=2)
            if analysis.get('global_axis_points'):draw.line([tuple(p) for p in analysis['global_axis_points']],fill='red',width=3)
            filename=f"{a['model']}_{a['number']}_{tag}.png";pic.thumbnail((520,650));pic.save(assets/filename)
            parts.append(f"<figure><img src='component_axis_assets/{filename}'><figcaption>{'Ручные линии' if tag=='manual' else 'Линии модели'}; зелёный — линии, голубой — оси позвонков, красный — общая ось; угол {n(analysis.get('global_angle_deg'))}°</figcaption></figure>")
        parts.append('</div>')
    page=ROOT/'dxa_project/team_demo/component_diagnostics_20260929.html'
    page.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика модулей DXA</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px;background:#101923;color:#e8eff5}p,li{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526477;padding:8px;text-align:left}a{color:#9cf}.grid{display:flex;gap:20px}img{max-width:100%}figure{margin:0;flex:1}</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print(json.dumps({'page':str(page),'summary':summary,'axis_summary':axis_summary},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
