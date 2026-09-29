"""Human-readable final module selection, split audit and held-out metrics."""
import json,html
from .final_assembly import ROOT,OUT,save


def build():
    bundle=OUT/'bundle';report=json.loads((bundle/'evaluation/report.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));data=json.loads((OUT/'data_summary.json').read_text(encoding='utf-8'))
    hip=json.loads((OUT/'hip_selection.json').read_text(encoding='utf-8'));points=json.loads((OUT/'point_selection.json').read_text(encoding='utf-8'))
    scoliosis=json.loads((bundle/'scoliosis_report.json').read_text(encoding='utf-8'));sbest=json.loads((bundle/'scoliosis_calibration.json').read_text(encoding='utf-8'))
    smoke=json.loads((OUT/'api_smoke_report.json').read_text(encoding='utf-8'))
    artifact=json.loads((OUT/'artifact_selection.json').read_text(encoding='utf-8'))
    spine=json.loads((OUT/'spine_selection.json').read_text(encoding='utf-8'))
    geometry=json.loads((bundle/'evaluation/spine_geometry_test.json').read_text(encoding='utf-8'))
    spatial=json.loads((bundle/'evaluation/spatial_test_metrics.json').read_text(encoding='utf-8'))
    crests=json.loads((OUT/'crest_selection.json').read_text(encoding='utf-8'))
    def n(x):return '—' if x is None else f'{x:.3f}'
    def interval(c):return f'[{n(c[0])}; {n(c[1])}]' if c else '—'
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    split=[]
    tasks=sorted(protocol['summary']['train']['labels'])
    for task in tasks:
        row=[task]
        for part in ('train','validation','test'):
            s=protocol['summary'][part]['labels'][task];total=s['positive']+s['negative']
            row.append(f'{s["positive"]}/{total} ({100*s["positive"]/max(1,total):.1f}%)')
        split.append(row)
    metrics=[];intervals=[]
    for r in report['metrics']:
        m=r.get('metrics') or {};metrics.append([r['region'],r['task'],r['labeled'],n(r['coverage']),n(m.get('f1')),n(m.get('sensitivity')),n(m.get('specificity')),n(m.get('balanced_accuracy')),n(m.get('roc_auc')),n(m.get('pr_auc_average_precision'))])
        for key in ('f1','roc_auc','pr_auc_average_precision','sensitivity','specificity','balanced_accuracy'):
            ci=r.get('ci95',{}).get(key);intervals.append([r['region'],r['task'],key,n(m.get(key)),f'[{n(ci[0])}; {n(ci[1])}]' if ci else '—'])
    modules=[['Область','ResNet18','Validation accuracy; historical alternatives both 1.0'],
             ['Позвоночник','ResNet50, all_peaks '+spine['selected']+', strict terminals','0.7 original angular MAE +0.3 synthetic angular MAE + divider F1; undefined angle penalty'],
             ['Подвздошные точки','Independent point-only ResNet50' if crests['separate_module_selected'] else 'Joint spine checkpoint','Maximum original validation PCK10; presence balanced accuracy and distance as tie terms'],
             ['ROI бедра',hip['roi'],'Maximum validation ROI IoU among retrained light/heavy'],
             ['Малый вертел',hip['mask'],'Maximum validation mask Dice among retrained light/heavy'],
             ['Три точки',points['selected']+' shared256 full frame','Maximum validation PCK10; misses count as failures'],
             ['Артефакты',artifact['selected']+' FasterRCNN','Maximum box F1 IoU50 on common original validation; image confidence threshold validation only'],
             ['Сколиоз','ResNet18 classifier','Mean positive F1 and balanced accuracy; threshold from original validation only']]
    lines=['# Итоговый пайплайн DXA — 29.09.2026','',
        'Собрана версия из отдельных модулей, обученных заново на общем разбиении. Финальные файлы: `outputs/final_20260929/bundle`. Предыдущие веса не заменены; существующий активный сервис не переключён. Новая версия проверена через изолированный API.', '',
        f'Главный результат на исходном test: MAE угла с вертикалью {n(geometry["angles"]["mae_deg"])}° при покрытии {n(geometry["angles"]["coverage"])}; macro-F1 нарушений {n(report["quality_macro_f1"])}. Выбор лучших модулей по validation не обеспечил высокое качество всех итоговых правил.', '',
        'Критические проблемы этой версии: ротация бедра отмечается как нарушение у всех правильно маршрутизированных test-снимков; табличные положительные метки наклона позвоночника пропущены; оценка положения позвоночника даёт много ложных срабатываний. Подробные матрицы ошибок, неопределённые ответы и следующие шаги приведены ниже. Итоговые веса и пороги после просмотра test не менялись.', '',
        '## Модули и критерии выбора','',table(['Модуль','Выбранный вариант','Метрика'],modules),'',
        'Лучшие сохранённые эпохи hip-кандидатов выбирались по среднему ROI IoU и Dice. Затем ROI и маска выбраны отдельно среди этих эпох по validation в исходных координатах DICOM, чтобы сравнить разные разрешения сетей на одном эталоне. Это выбор среди проверенных кандидатов, не исчерпывающий поиск всех архитектур. Для детектора артефактов оба семейства также заново обучены и сравнены на одной validation, поскольку прежние исторические сравнения имели разные части данных.', '',
        f'Артефакты: box F1 light={n(artifact["box_f1_iou50_validation"]["light"])}, heavy={n(artifact["box_f1_iou50_validation"]["heavy"])}; выбран {artifact["selected"]}. Порог присутствия артефакта {artifact["image_flag_threshold"]:.6f}, выбирался по среднему F1 и balanced accuracy на исходной validation. Меняет решение и отображаемые рамки, не влияет на обучение или выбор по test.', '',
        f'Позвоночник: проверены c=2 и c=0,5 на новом общем разбиении, критерии {n(spine["validation"]["c2"]["score"])} и {n(spine["validation"]["c05"]["score"])}. Выбран {spine["selected"]}. Прямая угловая голова исключена по прежнему отрицательному результату; здесь вся контурная цепочка оценивается по реальному углу.', '',
        f'Подвздошные ориентиры: PCK10 совместного углового checkpoint {n(crests["joint_validation"]["crest_pck_10mm"])}, отдельного модуля {n(crests["candidate_validation"]["crest_pck_10mm"])}. Отдельный модуль выбран: {crests["separate_module_selected"]}. Это устраняет конфликт выбора разделителей по углу и точек по локализации; новая сеть обучена с нуля от стандартного ImageNet только на loss ориентиров, без штрафов разделителей.', '',
        f'Точек: validation PCK10 light={n(points["validation"]["light"]["pck_10mm"])}, heavy={n(points["validation"]["heavy"]["pck_10mm"])}. ROI IoU: light={n(hip["validation"]["light"]["mean_roi_iou"])}, heavy={n(hip["validation"]["heavy"]["mean_roi_iou"])}. Dice малого вертела: light={n(hip["validation"]["light"]["pixel_dice"])}, heavy={n(hip["validation"]["heavy"]["pixel_dice"])}.', '',
        '## Разбиение по нарушениям','',
        f'{protocol["summary"]["train"]["images"]} train / {protocol["summary"]["validation"]["images"]} validation / {protocol["summary"]["test"]["images"]} test исходных снимков. Обучение: {data["training_images"]} снимков, из них {data["training_augmented"]} аугментаций. Все 15 000 аугментаций распределяются по исходному исследованию; в обучение входят только train-источники.', '',
        'Балансируются оба класса каждой метки и области снимков. Исследования и группы идентичных пикселей неделимы. Пересечение исследований и пиксельных дубликатов между частями равно нулю. Целевые доли 64%/16%/20%; равные доли нарушений важнее равного абсолютного числа в частях разных размеров.', '',
        table(['Нарушение','Train: положительные','Validation','Test'],split),'',
        'Редкие классы, которые невозможно представить во всех трёх частях без утечки: '+', '.join(protocol['unstratifiable_rare_classes'])+'. Это дополнительные страты разбиения, новые детекторы перелома/люмбализации здесь не обучались.', '',
        '## Детектор сколиоза','',
        'Цель — локальная метка `spine_issue=SCOLIOSIS`, всего 27 исходных положительных снимков. Сеть отличает эту метку от общего наклона оси; угол Кобба не вычисляется и диагноз по клиническому порогу не заявляется. Прочие явно размеченные NONE/LUMBARIZATION — отрицательный класс сколиоза.', '',
        'Аугментации наследуют метку только при сохранении всех исходных разделителей и минимум четырёх видимых линиях. После обрезки метка остаётся приближённой; примеры с потерянной геометрией исключены из этой задачи. Sampling балансирует классы и родителей, чтобы многочисленные варианты одного источника не задавали класс.', '',
        f'Validation: выбранная эпоха {sbest["epoch"]}, порог {sbest["threshold"]:.6f}, F1={n(sbest["validation"]["f1"])}, sensitivity={n(sbest["validation"]["sensitivity"])}, specificity={n(sbest["validation"]["specificity"])}. Validation и test малы: всего {protocol["summary"]["validation"]["labels"]["spine_scoliosis"]["positive"]} и {protocol["summary"]["test"]["labels"]["spine_scoliosis"]["positive"]} положительных случаев. Порог и эпоха не выбирались по test.', '',
        '## Test на исходных снимках','',table(['Область','Нарушение','Размечено','Покрытие','F1','Sensitivity','Specificity','Balanced accuracy','ROC AUC','PR AUC/AP'],metrics),'',
        f'Router macro-F1={n(report["router_macro_f1"])}; macro-F1 нарушений={n(report["quality_macro_f1"])}; все {report["processed_files"]}/{report["validation_originals"]} файлов обработаны. У неопределённых правил показано покрытие; они не считаются верными ответами. 95% интервалы по исследованиям находятся в `bundle/evaluation/report.json`.', '',
        f'Целевая геометрическая метрика на test: MAE угла с вертикалью {n(geometry["angles"]["mae_deg"])}°, покрытие {n(geometry["angles"]["coverage"])}; MAE со штрафом за неопределённый угол {n(geometry["angles"]["failure_penalized_mae_deg"])}°. Разделители: precision ≤5 мм {n(geometry["dividers"]["precision_5mm"])}, recall ≤5 мм {n(geometry["dividers"]["recall_5mm"])}. Детали и доверительные интервалы — `spine_geometry_test.json`.', '',
        f'95% CI угловой MAE: {interval(geometry["angles"].get("ci95",{}).get("mae_deg"))}°; MAE со штрафом: {interval(geometry["angles"].get("ci95",{}).get("failure_penalized_mae_deg"))}°. Разделители: precision {interval(geometry["dividers_ci95"].get("precision_5mm"))}, recall {interval(geometry["dividers_ci95"].get("recall_5mm"))}.', '',
        'Вся исходная коллекция уже изучалась в прежних экспериментах. Это внутренний test, не новый внешний набор. Для нового разбиения использованы только стандартные ImageNet/COCO веса, без прежних локально обученных моделей. Масштаб без PixelSpacing остаётся номинальным 1.05/0.6 мм. Численный угол для выбора позвоночника использует геометрический псевдоэталон по ручным линиям; табличный флаг >5° проверяется отдельно.', '',
        '## API и логи','',
        '`spine_scoliosis` добавлен в `quality_flags`, `quality_scores`, JSON и CSV. Детали порога — `scoliosis`. Дополнительные веса маски и сколиоза сохраняются при регистрации и пакетном предсказании. Старые пакеты из пяти моделей остаются совместимыми.', '',
        'Во время предсказания worker выводит «Фотография 1/N: обработка», затем «готово» или «ошибка». События сохраняются в `jobs/<id>/prediction.jsonl`. Читать с пагинацией: `GET /v1/jobs/<id>/logs?offset=0&limit=100`; прогресс — `GET /v1/jobs/<id>`. Повреждённый файл не прерывает обработку остальных.', '',
        f'API smoke: {smoke["images"]} файла, все три области, {smoke["logs"]["total"]} пофайловых событий; выгрузки CSV/JSON/ZIP успешны. Тестовый сервис изолирован от действующего.', '',
        'Свежая полная регрессия после добавления модулей и API: **99 passed**, три предупреждения библиотек о deprecated API. Включает проверку того, что loss подвздошных ориентиров не обучает канал разделителей, а недекодированная точка учитывается как пропуск. Worker откладывает GPU-предсказание при активном полном обучении; CPU-задания продолжаются. Новая версия регистрируется вместе с протоколом и калибровкой.', '',
        '## Запуск','',
        '```powershell',r'$env:PYTHONPATH="$PWD\analysis_20260929\runtime;$PWD"',r'.venv\Scripts\python.exe -m dxa_project.geometry_ml.final_assembly --phase train',r'.venv\Scripts\python.exe -m dxa_project.geometry_ml.final_assembly --phase evaluate','```','',
        'Повторный запуск использует завершённые этапы этой версии. Фаза evaluate включает завершение обучения кандидатов подвздошных ориентиров и тяжёлого детектора, затем калибровку на validation и итоговый test. При новых данных/условиях нужен новый каталог и новое обучение. Протокол, fingerprint, выбранные эпохи, метрики и версии находятся рядом с bundle. API-регистрация: `POST /v1/model/versions`, checkpoints=bundle, protocol=outputs/final_20260929/protocol.json; epochs — будущий бюджет partial_fit пяти основных модулей, activate — настройка активной версии.', '',
        'Следующие приоритеты: проверить ошибочные сколиозы вручную и увеличить число независимых положительных исследований; разметить истинную общую ось; проверить редкие нарушения и калибровку ротации на внешнем наборе. Дополнительные scoliosis/hip_mask/spine_crests при partial_fit пока сохраняются без дообучения; обновлять их следует полным версионным обучением.', '',
    ]
    pipeline=['## Как работает модель','',
              'Вход — один однокадровый серый DICOM DXA. Интенсивность нормализуется, router выбирает позвоночник, левое или правое бедро. Правое бедро зеркально приводится к общей ориентации на входе сети; выход возвращается в исходные координаты. Для пространственных моделей используется letterbox без искажения пропорций, для router — resize.', '',
              'Позвоночник: сеть строит разделители и подвздошные точки; детектор находит инородные объекты; по разделителям и изображению восстанавливаются локальные и общая оси со строгими крайними перпендикулярами. Отдельная сеть классифицирует метку сколиоза. Правила формируют флаги положения и наклона общей оси >5°. Бедро: отдельные выбранные модули находят ROI, три ориентира и маску малого вертела; правила проверяют положение, ROI и ротацию по площади маски с порогом, выбранным на validation.', '',
              'Выход — область, флаги и непрерывные оценки, координатная геометрия и артефакты PNG/JSON. Таргеты позвоночника: spine_position, spine_axis, spine_artifact, spine_scoliosis; обеих ног: hip_position, hip_roi, hip_rotation. **1 означает нарушение, 0 — отсутствие нарушения, null — неопределённый ответ.** Сколиоз не заменяется общим наклоном и не является вычислением угла Кобба.', '']
    lines=lines[:4]+pipeline+lines[4:]
    overall=report['overall_any_violation'];om=overall.get('metrics') or {}
    lines+=['## Общая оценка и интервалы','',
            f'Любое нарушение в исследуемом снимке: покрытие {n(report["overall_any_violation_coverage"])}, N={overall["n"]}. Набор учитываемых таргетов зависит от области. Общий вывод считается только при определённых ответах и метках всех её таргетов.', '',
            table(['Метрика','Значение','95% CI'],[[k,n(om.get(k)),f'[{n(overall["ci95"][k][0])}; {n(overall["ci95"][k][1])}]' if overall.get('ci95',{}).get(k) else '—'] for k in ('f1','roc_auc','pr_auc_average_precision','sensitivity','specificity','balanced_accuracy')]), '',
            'Интервалы для каждого таргета. PR AUC представлен average precision (AP). Bootstrap группирует снимки исходного исследования; 300 повторов, seed=42. При отсутствии обоих классов ROC AUC/PR AUC не определены. Количество пригодных bootstrap-повторов сохранено в JSON.', '',
            table(['Область','Таргет','Метрика','Значение','95% CI'],intervals),'',
            '## Локализация, сегментация и производительность','']
    macro=spatial['macro'];any_region=[]
    study_overall=spatial['any_violation_studies'];sm=study_overall.get('metrics') or {}
    lines+=['### Итог на уровне исследования','',
            f'«Есть хотя бы одно нарушение» среди всех снимков исследования: N={study_overall["n"]}/{study_overall["total_studies"]}, покрытие {n(study_overall["coverage"])}. Исследование положительное при любом определённом нарушении; отрицательное только когда все применимые флаги всех его снимков известны и равны нулю; иначе ответ неопределён.', '',
            table(['Метрика','Значение','95% CI'],[[k,n(sm.get(k)),interval(study_overall.get('ci95',{}).get(k))] for k in ('f1','roc_auc','pr_auc_average_precision','sensitivity','specificity','balanced_accuracy')]),'']
    for region,item in spatial['any_violation_by_region'].items():
        for key in ('f1','roc_auc','pr_auc_average_precision','sensitivity','specificity','balanced_accuracy'):
            value=(item.get('metrics') or {}).get(key);ci=item.get('ci95',{}).get(key)
            any_region.append([region,key,n(value),f'[{n(ci[0])}; {n(ci[1])}]' if ci else '—',n(item['coverage'])])
    lines+=['Macro-F1 с интервалами: quality_macro_f1 усредняет пары «область × таргет»; target_macro_f1 усредняет семь типов нарушений, объединяя обе ноги для одинаковых hip-таргетов; router_macro_f1 относится к трём областям. Неопределённые ответы учитываются через отдельное покрытие.', '',table(['Метрика','Значение','95% CI'],[[k,n(v),f'[{n(macro["ci95"][k][0])}; {n(macro["ci95"][k][1])}]' if macro['ci95'].get(k) else '—'] for k,v in macro['metrics'].items()]),'',
            'Любое нарушение отдельно по области:', '',table(['Область','Метрика','Значение','95% CI','Покрытие'],any_region),'']
    rows=[]
    for region in ('SPINE','LEG_LEFT','LEG_RIGHT'):
        item=spatial[region]
        for key,value in item['metrics'].items():
            if value is None:continue
            ci=item['ci95'].get(key);rows.append([region,key,n(value),f'[{n(ci[0])}; {n(ci[1])}]' if ci else '—'])
    lines+=[table(['Область','Метрика','Значение','95% CI'],rows),'',
            'Для SPINE расстояния/PCK относятся к подвздошным ориентирам, для ног — к трём точкам бедра. Dice и mask_iou — среднее по снимкам в исходном разрешении; пустая эталонная и пустая предсказанная маски дают 1. PCK включает пропуски как ошибки; средняя дистанция рассчитана по найденным точкам и сопровождается покрытием. box_* — рамки артефактов при IoU≥0,5 и выбранном validation-пороге confidence. Масштаб в мм номинальный.', '',
            f'Успешные файлы: {report["processed_files"]}/{report["validation_originals"]} ({100*report["file_processing_success_fraction"]:.1f}%). Время на файл: среднее {n(report["seconds_per_file_mean"])} с, p95 {n(report["seconds_per_file_p95"])} с. На исследование: среднее {n(spatial["timing"]["mean_seconds_per_study"])} с, p95 {n(spatial["timing"]["p95_seconds_per_study"])} с; включает все его test-снимки.', '',
            'Замер с прогретыми весами, последовательное предсказание на GPU: чтение DICOM, forward и геометрическая постобработка. Загрузка весов и HTTP-передача исключены; это не время пакетного API-задания.', '']
    from .final_assembly import prepare
    _,augmented,groups,labels,protocol=prepare()
    from .evaluate import TASKS
    region_split=[]
    for region,tasks in TASKS.items():
        for task in tasks+(('spine_scoliosis',) if region=='SPINE' else ()):
            row=[region,task]
            for part in ('train','validation','test'):
                values=[labels[r.relative_path].get(task) for r in groups[part] if r.region==region and not r.source_id]
                row.append(f'{values.count(1)}/{values.count(0)+values.count(1)}')
            region_split.append(row)
    lines+=['## Баланс по области и таргету','',table(['Область','Таргет','Train: нарушения/размечено','Validation','Test'],region_split),'',
            'Оптимизация разбиения балансировала каждый тип нарушения в целом и размер анатомических областей. В отдельных сочетаниях «область × таргет» один класс отсутствует: например, ROI левой ноги в test и положение правой ноги в test. Там sensitivity/ROC AUC нельзя надёжно оценить; синтетические примеры ниже восполняют только демонстрацию, не метрики. Для следующего независимого набора нужно дополнительно балансировать именно эти сочетания.', '']
    from .final_examples import build as examples
    lines+=examples(groups['test'],bundle/'evaluation',augmented,labels,protocol)
    from .final_api_reference import build as api_reference
    lines+=api_reference()
    actions={'spine_axis':'Проверить соответствие табличного флага геометрическому углу >5°; разметить независимую общую ось в градусах.',
             'spine_position':'Отдельно улучшать подвздошные ориентиры и крайние разделители; выбирать их также по локализации и F1 итогового правила.',
             'spine_artifact':'Добавить разные виды инородных объектов и похожие отрицательные случаи; проверить рамки, confidence и IoU.',
             'spine_scoliosis':'Увеличить число независимых положительных исследований и проверить локальную разметку ошибочных случаев.',
             'hip_position':'Добрать реальные нарушения положения и обрезанные ориентиры; проверить правило видимости трёх точек.',
             'hip_roi':'Добрать реальные нарушения отступов ROI отдельно для обеих сторон; улучшить границы ROI и проверить масштаб.',
             'hip_rotation':'Проверить маски малого вертела и достоверность PixelSpacing; перекалибровать физическую площадь на новых validation-исследованиях.'}
    problems=[]
    for item in report['metrics']:
        m=item.get('metrics') or {};signals=[]
        if m.get('f1') is not None and m['f1']<.5:signals.append('F1='+n(m['f1']))
        if m.get('sensitivity') is not None and m['sensitivity']<.5:signals.append('sensitivity='+n(m['sensitivity']))
        if item['coverage'] is not None and item['coverage']<.95:signals.append('coverage='+n(item['coverage']))
        if m.get('roc_auc') is None:signals.append('ROC AUC не определён: не хватает обоих классов')
        if signals:problems.append([item['region'],item['task'],'; '.join(signals),actions[item['task']]])
    lines+=['## Итог и оставшиеся проблемы','',
            f'По validation выбраны: ROI — {hip["roi"]}, маска — {hip["mask"]}, точки — {points["selected"]}, разделители/оси — {spine["selected"]}, артефакты — {artifact["selected"]}. Сколиоз обучен отдельно. Выбор выполнялся до test; показатели test не использованы для перестановки модулей или порогов.', '',
            table(['Область','Подзадача','Сигнал проблемы на test','Следующий шаг'],problems) if problems else 'Для всех определённых задач F1 и sensitivity ≥0,5, coverage ≥0,95; ограничения размера и независимости набора сохраняются.', '',
            'Повышение этих метрик следует проверять на новом validation/test-протоколе и внешних исследованиях. Приведённые иллюстрации объясняют поведение, а не заменяют оценку всего набора.', '']
    text='\n'.join(lines);(ROOT/'dxa_project/FINAL_MODEL_20260929.md').write_text(text,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code']).replace('src="team_demo/','src="')
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    page=ROOT/'dxa_project/team_demo/final_model_20260929.html';page.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Итоговый DXA</title><style>body{max-width:1400px;margin:32px auto;font:16px system-ui;padding:20px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #ccd}pre{overflow:auto}img{max-width:100%;height:auto}h2{margin-top:50px}</style>'+body+'</html>',encoding='utf-8')
