"""Publish aggregate experiment results and five paired landmark examples."""
from pathlib import Path
import json,html
import numpy as np
from PIL import Image,ImageDraw
from .data import load_records,read_dicom_image,LANDMARKS

LABELS={'SPINE':'Позвоночник','LEG_LEFT':'Левое бедро','LEG_RIGHT':'Правое бедро',
        'spine_position':'Укладка позвоночника','spine_axis':'Наклон позвоночника',
        'spine_artifact':'Артефакты','hip_position':'Позиционирование бедра',
        'hip_roi':'Отступы ROI','hip_rotation':'Ротация бедра'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def number(v):return '—' if v is None else f'{v:.3f}'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def metric_table(old,new):
    oldmap={(x['region'],x['task']):x for x in old['metrics']};rows=[]
    for x in new['metrics']:
        previous=oldmap[(x['region'],x['task'])];a=previous.get('metrics') or {};b=x.get('metrics') or {}
        interval=x.get('ci95',{}).get('f1');ci='—' if interval is None else '–'.join(number(v) for v in interval)
        rows.append([LABELS[x['region']],LABELS.get(x['task'],x['task']),number(a.get('f1')),number(b.get('f1')),ci,
                     number(a.get('roc_auc')),number(b.get('roc_auc')),number(b.get('sensitivity')),number(b.get('specificity')),
                     number(previous['coverage']),number(x['coverage']),x['n']])
    return table(['Область','Проверка','F1 раньше','F1 сейчас','95% CI F1','AUC раньше','AUC сейчас',
                  'Чувствительность','Специфичность','Покрытие раньше','Покрытие сейчас','N'],rows)

def point_examples(root,out,destination):
    oldrows=read(out/'decoder_legacy_rows.json');selection=read(out/'point_selection.json')['chosen']
    newrows=read(out/'trained'/selection['variant']/'test_rows.json')
    oldmap={r['relative_path']:r for r in oldrows};records={r.relative_path:r for r in load_records(root)}
    oldeval=root/'dxa_project/outputs/geometry_ml_augmented_5epochs/evaluation'
    neweval=out/'pipeline/evaluation'
    oldpred={r['relative_path']:read(oldeval/f'prediction_{i:03}.json') for i,r in enumerate(read(oldeval/'predictions.json'),1)}
    newpred={r['relative_path']:read(neweval/f'prediction_{i:03}.json') for i,r in enumerate(read(neweval/'predictions.json'),1)}
    # Quantiles of the change, including deteriorations: avoid picking only successes.
    pairs=sorted((r for r in newrows if r['relative_path'] in oldmap and r['errors_mm'] and oldmap[r['relative_path']]['errors_mm']
                  and oldpred[r['relative_path']]['region']==records[r['relative_path']].region
                  and newpred[r['relative_path']]['region']==records[r['relative_path']].region),
                 key=lambda r:np.mean(r['errors_mm'])-np.mean(oldmap[r['relative_path']]['errors_mm']))
    examples=[];assets=destination/'improvement_assets';assets.mkdir(exist_ok=True)
    for i,index in enumerate(np.linspace(0,len(pairs)-1,5).astype(int),1):
        r=pairs[index];rel=r['relative_path'];record=records[rel];truth=read(record.geometry_path)
        image=Image.fromarray((read_dicom_image(record.source_path)*255).astype('uint8')).convert('RGB')
        blocks=[]
        for points,color in ((truth['hip']['landmarks'],'cyan'),(oldpred[rel]['geometry']['hip']['landmarks'],'orange'),
                             (newpred[rel]['geometry']['hip']['landmarks'],'lime')):
            panel=image.copy();draw=ImageDraw.Draw(panel)
            radius=max(3,min(panel.size)//75)
            for n,key in enumerate(LANDMARKS,1):
                p=points.get(key)
                if p is not None:
                    x,y=p;draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(2,radius//2));draw.text((x+radius,y),str(n),fill=color)
            panel.thumbnail((320,420));blocks.append(panel)
        height=max(p.height for p in blocks);canvas=Image.new('RGB',(960,height),(15,20,28))
        for k,p in enumerate(blocks):canvas.paste(p,(k*320+(320-p.width)//2,0))
        name=f'points_{i}.png';canvas.save(assets/name)
        before=float(np.mean(oldmap[rel]['errors_mm']));after=float(np.mean(r['errors_mm']))
        examples.append({'example':i,'old_error_mm':before,'new_error_mm':after,'image':'improvement_assets/'+name})
    return examples

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    baseline=root/'dxa_project/outputs/geometry_ml_augmented_5epochs';dest=root/'dxa_project/team_demo';dest.mkdir(exist_ok=True)
    old=read(baseline/'evaluation/report.json');new=read(out/'pipeline/evaluation/report.json')
    synthetic=read(out/'pipeline/synthetic_evaluation/report.json');oldsynthetic=read(baseline/'synthetic_evaluation/report.json')
    localization=read(out/'localization_comparison.json')
    protocol=read(out/'protocol.json');pointtest=read(out/'test_comparison.json');decoders=read(out/'decoder_comparison.json')
    training={name:read(out/'modules'/name/'report.json')['tasks'][0] for name in ('spine','hip','artifact')}
    selected=read(out/'point_selection.json')['chosen'];chosen=selected['variant'];examples=point_examples(root,out,dest)
    alternative_path=out/'coordinate_pipeline/synthetic_evaluation/report.json'
    alternative=read(alternative_path) if alternative_path.exists() else None
    # Public JSON deliberately contains aggregates only, no source paths/study IDs.
    public={'partition_counts':protocol['summary'],'original_evaluation':new,'synthetic_evaluation':synthetic,'localization':localization,
            'point_comparison':pointtest,'decoders':decoders,'selected_points':{'variant':chosen,'epoch':selected['epoch']},
            'training':{k:{f:v[f] for f in ('epochs','max_epochs','best_epoch','seconds')} for k,v in training.items()},'examples':examples,
            'coordinate_variant_synthetic':alternative}
    assert not new['failed_files'] and not synthetic['failures']
    (dest/'improvements_metrics.json').write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    splitrows=[[k,v['images'],v['studies'],v['spine_images'],v['artifact_positive'],number(v['artifact_fraction'])] for k,v in protocol['summary'].items()]
    pointrows=[['Старые веса / старый декодер',number(decoders['legacy']['mean_error_mm']),number(decoders['legacy']['pck_10mm'])],
               ['Старые веса / максимум внутри кадра',number(decoders['masked_logit_argmax']['mean_error_mm']),number(decoders['masked_logit_argmax']['pck_10mm'])]]
    pointrows += [[k+' / эпоха '+str(v['epoch']),number(v['mean_error_mm']),number(v['pck_10mm'])] for k,v in pointtest.items()]
    trainingrows=[[LABELS.get(k,k),v['epochs'],v['best_epoch'],f"{v['seconds']/60:.1f}"] for k,v in training.items()]
    for variant in ('shared256','coordinate256'):
        v=read(out/'trained'/variant/'report.json');trainingrows.append([variant,v['actual_epochs'],'отбор best/last на внутренней validation',f"{v['seconds']/60:.1f}"])
    overall=new['overall_any_violation']['metrics'];previous=old['overall_any_violation']['metrics']
    body='<h1>DXA: проверка улучшений</h1><p><a href="index.html">Первоначальный отчёт</a> · <a href="improvements_metrics.json">Полные агрегированные метрики</a></p>'
    if (dest/'model_examples.html').exists():body+='<p><a href="model_examples.html">Успехи и ошибки каждого модуля; номера для ручной проверки разметки</a></p>'
    body+='<h2>Что изменено</h2><ul><li>Разделение по исследованиям и близкие доли артефактов; одинаковый вес исходных снимков независимо от количества аугментаций.</li><li>Отдельная модель трёх точек бедра; сравнение heatmap и дополнительного координатного loss.</li><li>Меньший LR для энкодера, cosine scheduler, до 20 эпох, ранняя остановка и сохранение лучшей модели.</li><li>Максимумы карт точек ограничены кадром; поиск по логитам исключает ложные равные максимумы из-за насыщения sigmoid.</li><li>Расстояния между точками служат проверкой правдоподобия. Дополнительный режим может воздержаться от ответа, но не объявляет нарушение только по анатомической вариативности.</li></ul>'
    body+='<h2>Разделение данных</h2>'+table(['Часть','Снимки','Исследования','Позвоночник','Артефакты','Доля'],splitrows)
    body+='<p>Исследования и точные пиксельные дубли между частями не пересекаются. Train и внутренняя validation используются для обучения и выбора. Прежние 100 контрольных снимков сохранены для сравнения; их уже изучали, поэтому это не новый независимый test. В этой контрольной выборке доля артефактов остаётся высокой. Маршрутизатор (router) взят из предыдущего запуска; остальные три модуля обучены заново.</p>'
    body+='<h2>Точки бедра (hip landmarks)</h2>'+table(['Вариант','Средняя ошибка, мм','Доля точек в пределах 10 мм'],pointrows)
    body+=f'<p>В рабочий пайплайн выбран {html.escape(chosen)}, эпоха {selected["epoch"]}, только по внутренней validation. Из каждой модели дополнительно сравнивались сохранённые best и last с рабочим декодером. На 65 исходных контрольных снимках бедра все три точки видны: чувствительность и ROC AUC позиционирования на них не определены. Миллиметры рассчитаны при номинальном масштабе 1,05 × 0,6 мм; это ограничение физической точности.</p>'
    body+='<h2>Полный пайплайн: исходные контрольные снимки</h2>'+metric_table(old,new)
    body+=table(['Показатель','Раньше','Сейчас'],[['F1: любое нарушение',number(previous['f1']),number(overall['f1'])],
                  ['ROC AUC: любое нарушение',number(previous['roc_auc']),number(overall['roc_auc'])],
                  ['Покрытие: любое нарушение',number(old['overall_any_violation_coverage']),number(new['overall_any_violation_coverage'])],
                  ['Успешно обработанные файлы',old['processed_files'],new['processed_files']],
                  ['Macro-F1 маршрутизации',number(old['router_macro_f1']),number(new['router_macro_f1'])]])
    body+='<p>F1 отражает баланс точности обнаружения нарушений и чувствительности; больше — лучше. ROC AUC оценивает ранжирование по непрерывному score, а не точность бинарного ответа на выбранном пороге. Чувствительность — доля обнаруженных нарушений, специфичность — доля качественных снимков без ложной тревоги. Покрытие — доля случаев, в которых правило смогло вернуть определённый ответ; пропуски исключены из F1 и не считаются правильными. Сравнивайте F1 вместе с покрытием. 95% CI получены bootstrap по исходным исследованиям; малое число исследований даёт широкие интервалы. Балансированная точность, PR AUC и матрицы ошибок находятся в JSON.</p>'
    body+='<h2>Отдельная проверка на аугментациях контрольных источников</h2>'+metric_table(oldsynthetic,synthetic)
    body+='<p>135 синтетических изображений, без пересечения с обучающими исследованиями. Здесь появляются пропавшие точки и нарушенные отступы. Это проверка заданных преобразований, её нельзя выдавать за диагностическое качество на реальных нарушениях.</p>'
    if alternative is not None:
        rows=[]
        for variant,report in ((chosen,synthetic),('coordinate256',alternative)):
            for metric in report['metrics']:
                if metric['task']!='hip_position':continue
                m=metric.get('metrics') or {}
                rows.append([variant,LABELS[metric['region']],number(m.get('f1')),number(m.get('sensitivity')),
                             number(m.get('specificity')),number(m.get('roc_auc')),number(metric['coverage'])])
        body+='<h3>Два варианта точек в одном полном пайплайне</h3>'+table(['Вариант','Бедро','F1 позиции','Чувствительность','Специфичность','ROC AUC','Покрытие'],rows)
        body+='<p>Все остальные веса, источники и преобразования одинаковы. Это диагностическое сравнение после внутреннего отбора; выбранная основная модель не меняется по результатам контрольной выборки.</p>'
    location_names={'mean_roi_iou':'Средний IoU ROI','trochanter_micro_dice':'Dice маски малого вертела',
                    'trochanter_micro_iou':'IoU маски малого вертела','mean_crest_error_mm':'Ошибка подвздошных точек, мм',
                    'box_recall_iou50':'Recall рамок артефактов, IoU ≥ 0,5','box_precision_iou50':'Precision рамок артефактов, IoU ≥ 0,5'}
    body+='<h2>Локализация после полного инференса</h2>'+table(['Метрика','Раньше','Сейчас'],
         [[label,number(localization['baseline'][key]),number(localization['improved'][key])] for key,label in location_names.items()])
    body+='<p>Только исходные снимки, направленные в верную ветку. Dice и IoU ближе к 1 означают лучшее совпадение масок/прямоугольников; ошибка точек меньше — лучше. Mask Dice/IoU здесь посчитаны по сумме пикселей на исходном разрешении, без искусственной оценки 1 для пустых масок. Эталон маски вертела получен из ручных контуров, поэтому ошибки построения эталона ограничивают интерпретацию. F1 наличия артефакта может быть высоким даже при плохой локализации его рамки.</p>'
    body+='<h2>Время обучения</h2>'+table(['Модуль','Эпох выполнено','Выбранная эпоха','Минуты'],trainingrows)
    body+=f'<p>GPU: RTX 5060 Ti. Средний полный инференс сейчас {new["seconds_per_file_mean"]:.3f} с на файл после прогрева. Старое измерение частично включало холодный запуск: напрямую сравнивать ускорение нельзя.</p>'
    body+='<h2>Пять примеров: разметка → старые точки → новые точки</h2><p>Слева ручная разметка (голубой), в центре прежняя модель (оранжевый), справа новая (зелёный). Примеры выбраны по квантилям изменения ошибки, включая ухудшения.</p>'
    for e in examples:body+=f'<figure><img src="{e["image"]}" loading="lazy"><figcaption>Пример {e["example"]}: ошибка {e["old_error_mm"]:.1f} → {e["new_error_mm"]:.1f} мм.</figcaption></figure>'
    body+='<h2>Выводы по выполненному запуску</h2>'
    body+=f'<ul><li>Отдельная сеть точек дала практический выигрыш: средняя ошибка {decoders["legacy"]["mean_error_mm"]:.1f} → {pointtest[chosen]["mean_error_mm"]:.1f} мм. Доля точек в пределах 10 мм: {decoders["legacy"]["pck_10mm"]:.1%} → {pointtest[chosen]["pck_10mm"]:.1%}.</li>'
    body+=f'<li>Общий F1 вырос {previous["f1"]:.3f} → {overall["f1"]:.3f} при том же покрытии {new["overall_any_violation_coverage"]:.0%}. Специфичность выросла {previous["specificity"]:.3f} → {overall["specificity"]:.3f}, но чувствительность снизилась {previous["sensitivity"]:.3f} → {overall["sensitivity"]:.3f}; ROC AUC также снизился {previous["roc_auc"]:.3f} → {overall["roc_auc"]:.3f}. Ложных тревог меньше, однако пропусков нарушений больше.</li>'
    body+=f'<li>Уменьшение LR и больше эпох не улучшили всё: IoU ROI снизился {localization["baseline"]["mean_roi_iou"]:.3f} → {localization["improved"]["mean_roi_iou"]:.3f}; ошибка подвздошных точек выросла {localization["baseline"]["mean_crest_error_mm"]:.1f} → {localization["improved"]["mean_crest_error_mm"]:.1f} мм. Для этих модулей следующий эксперимент должен менять один фактор и отдельно проверять дообучение прежних весов.</li>'
    body+='<li>Стоит сохранить отдельную модель точек и исправление декодера; весь новый комплект весов пока нельзя считать безусловной заменой прежнего. Укладка/ось позвоночника и ротация бедра требуют следующего цикла улучшения. Наличие артефактов и их рамки оценивайте отдельно.</li></ul>'
    if alternative is not None:
        current_f1={m['region']:m['metrics']['f1'] for m in synthetic['metrics'] if m['task']=='hip_position'}
        alternate_f1={m['region']:m['metrics']['f1'] for m in alternative['metrics'] if m['task']=='hip_position'}
        body+=f'<p>Координатный loss оказался лучше для бинарного позиционирования на контрольных аугментациях: F1 слева {current_f1["LEG_LEFT"]:.3f} → {alternate_f1["LEG_LEFT"]:.3f}, справа {current_f1["LEG_RIGHT"]:.3f} → {alternate_f1["LEG_RIGHT"]:.3f}, хотя точность координат/PCK у него хуже. Это показывает различие целей локализации и итоговой классификации. Следующий отбор по F1 следует провести на аугментациях внутренней validation, прежде чем менять основной вариант.</p>'
    body+='<h2>Что ещё проверить</h2><ul><li>Восемь расхождений визуальных отметок артефактов и таблицы требуют ручного решения; исходная разметка автоматически не изменялась.</li><li>Шесть вариантов архитектур/размеров/ROI прошли короткий forward/backward. Полностью обучены только shared256 и coordinate256; по остальным нельзя делать вывод о превосходстве.</li><li>Одновременно изменены split, sampler, LR и длительность: этот запуск проверяет комбинацию улучшений, а не доказывает вклад каждого изменения. Для строгого сравнения нужны отдельные абляции.</li><li>Следующий независимый набор должен содержать реальные пропуски структур и нарушения укладки; аугментации этого не заменяют.</li></ul>'
    document='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DXA — улучшения</title><style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#202b3c;background:#f5f7fa}table{border-collapse:collapse;font-size:14px;margin:15px 0;background:white}td,th{padding:9px;border:1px solid #cfd6e0;text-align:left}h2{margin-top:32px}img{max-width:100%}figure{background:white;padding:14px;margin:16px 0}p,li{line-height:1.5}</style>'+body+'</html>'
    (dest/'improvements.html').write_text(document,encoding='utf-8')
    index=dest/'index.html'
    if index.exists():
        original=index.read_text(encoding='utf-8')
        if 'href="improvements.html"' not in original:
            original=original.replace('</h1>','</h1><p><a href="improvements.html">Новые эксперименты и сравнение улучшений</a></p>',1)
            index.write_text(original,encoding='utf-8')
    (out/'REPORT.md').write_text('# Проверка улучшений DXA\n\nПолный отчёт: `dxa_project/team_demo/improvements.html`.\n\n'
          +f'Выбраны точки: {chosen}, эпоха {selected["epoch"]}. F1 любого нарушения: {previous["f1"]:.3f} → {overall["f1"]:.3f}; '
          +f'ROC AUC: {previous["roc_auc"]:.3f} → {overall["roc_auc"]:.3f}.\n',encoding='utf-8')
    print(json.dumps({'report':str(dest/'improvements.html'),'selected_points':chosen,'overall':overall},ensure_ascii=False))

if __name__=='__main__':main()
