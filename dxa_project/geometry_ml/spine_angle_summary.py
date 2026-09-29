"""Build the final per-step research summary, figures and review notebook."""
import json,math,html
from pathlib import Path
import numpy as np
from .spine_angle_study import ROOT,DEST,save


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((DEST/'report.json').read_text(encoding='utf-8'))
    results={k:json.loads((DEST/(k+'_result.json')).read_text(encoding='utf-8')) for k in report['results']}
    winner=report['winner_validation']
    best_geometry=min(results,key=lambda k:results[k]['validation']['geometry_angle']['failure_penalized_mae_deg'])
    def line_f1(v):
        a=v['recall_5mm'];b=v['precision_5mm'];return 2*a*b/max(a+b,1e-9)
    best_lines=max(results,key=lambda k:line_f1(results[k]['validation']['lines']))
    from .data import load_records,load_augmented_records
    originals=load_records(ROOT);all_records=originals+load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',originals)
    by={r.relative_path:r for r in all_records}
    gaps={}
    for name,v in results.items():
        counts={'close_c2':0,'close_c05':0,'missing_top_5mm':0,'missing_bottom_5mm':0,'no_lines':0}
        for rel,lines in v['test_lines'].items():
            r=by[rel];g=json.loads(r.geometry_path.read_text(encoding='utf-8'));h=g['image_height']-1;w=g['image_width']-1
            def centers(ls):
                values=[]
                for line in ls:
                    a,b=sorted(line['points'],key=lambda p:p[0]);values.append((a[1]+(w/2-a[0])*(b[1]-a[1])/max(b[0]-a[0],1e-6))/h)
                return np.asarray(sorted(values))
            gt=centers(g['spine']['disc_lines']);ys=centers(lines);ref=np.diff(gt).mean();d=np.diff(ys)
            if not len(ys):counts['no_lines']+=1;counts['missing_top_5mm']+=1;counts['missing_bottom_5mm']+=1;continue
            for c,key in [(2,'close_c2'),(.5,'close_c05')]:counts[key]+=int(bool(np.any(d<1/(1/ref+c))))
            counts['missing_top_5mm']+=int((ys[0]-gt[0])*h*r.spacing_mm[0]>5)
            counts['missing_bottom_5mm']+=int((gt[-1]-ys[-1])*h*r.spacing_mm[0]>5)
        gaps[name]=counts
    base=results['control_c2'];chosen=results[winner]
    a={r['relative_path']:r for r in base['rows']};b={r['relative_path']:r for r in (chosen['head_rows'] if chosen['config']['angle'] else chosen['rows'])}
    pairs=[]
    for path,r in a.items():
        if r['reference_angle'] is None:continue
        def error(x):return abs(x['angle']-x['reference_angle']) if x['angle'] is not None else 15.
        pairs.append({'study':r['study'],'difference_deg':error(b[path])-error(r)})
    ids=sorted({r['study'] for r in pairs});rng=np.random.default_rng(42);draw=[]
    for _ in range(1000):
        sample=[r['difference_deg'] for i in rng.integers(0,len(ids),len(ids)) for r in pairs if r['study']==ids[i]]
        draw.append(float(np.mean(sample)))
    paired={'winner_minus_control_failure_penalized_mae_deg':float(np.mean([r['difference_deg'] for r in pairs])),
            'ci95':np.percentile(draw,[2.5,97.5]).tolist(),'test_studies':len(ids),'method':'paired original-study bootstrap; undefined angle=15 degrees, previously inspected diagnostic test'}
    head_frame_failures={k:sum(r.get('axis_points') is None for r in v['head_rows']) for k,v in results.items() if v['config']['angle']}
    summary={'best_angle_validation':winner,'best_geometry_angle_validation':best_geometry,'best_lines_validation':best_lines,'standardized_gap_audit':gaps,'paired_angle_difference':paired,'head_axis_outside_frame':head_frame_failures}
    save(DEST/'step_summary.json',summary)
    assets=ROOT/'dxa_project/team_demo/spine_angle_assets';assets.mkdir(exist_ok=True)
    colors=['#4353b4','#17a689','#e39432','#a052b8','#e0567a']
    fig,ax=plt.subplots(figsize=(9,5))
    for color,(name,v) in zip(colors,results.items()):
        ax.plot([r['epoch'] for r in v['history']],[r['validation']['angle']['failure_penalized_mae_deg'] for r in v['history']],marker='o',label=name,color=color)
    ax.set(xlabel='Epoch',ylabel='Validation angular MAE with abstention penalty, deg',title='Model selection uses validation angle');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(assets/'validation_angle.png',dpi=180);plt.close(fig)
    rows=chosen['head_rows'] if chosen['config']['angle'] else chosen['rows']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for synthetic,color,label in [(False,'#168a7e','Original scans'),(True,'#9d54b8','Augmentations')]:
        usable=[r for r in rows if r['angle'] is not None and r['reference_angle'] is not None and r['relative_path'].startswith('aug/')==synthetic]
        axes[0].scatter([r['reference_angle'] for r in usable],[r['angle'] for r in usable],s=24,alpha=.7,label=label,color=color)
    axes[0].plot([-15,15],[-15,15],'k--',alpha=.5);axes[0].set(xlabel='Manual-geometry reference, deg',ylabel='Predicted global angle, deg',title=winner);axes[0].legend();axes[0].grid(alpha=.2)
    names=list(results);errors=[]
    for name in names:
        rs=results[name]['head_rows'] if results[name]['config']['angle'] else results[name]['rows']
        errors.append([abs(r['angle']-r['reference_angle']) for r in rs if r['angle'] is not None and r['reference_angle'] is not None])
    axes[1].boxplot(errors,tick_labels=names,showfliers=True);axes[1].set(ylabel='Absolute angular error, deg',title='Defined angles only; coverage in report');axes[1].tick_params(axis='x',rotation=25);axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(assets/'test_angle.png',dpi=180);plt.close(fig)
    def n(x):return '—' if x is None else f'{x:.3f}'
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    rows=[[k,v['best_epoch'],n(v['validation']['angle']['failure_penalized_mae_deg']),n(v['test_angles']['mae_deg']),n(v['test_angles']['coverage']),n(v['original_angles']['mae_deg']),n(v['synthetic_angles']['mae_deg'])] for k,v in results.items()]
    lines=[
        '# Исследование целевого угла позвоночника — 29.09.2026', '',
        f'По внутренней validation лучший метод угла: **{winner}**. Лучший контурный угол: **{best_geometry}**. Лучшее разделение линий по validation F1 в пределах 5 мм: **{best_lines}**. Это разные подзадачи, и победители не обязаны совпадать.', '',
        f'Пять контролируемых запусков по четыре эпохи, {report["seconds"]/60:.1f} минуты. 489 train, 147 validation, 154 test; 34 test-оригинала и 120 test-аугментаций из 20 исследований. Тот же состав, начальные тяжёлые веса и порядок примеров, что в предыдущем исследовании пространственного декодера. Энкодер заморожен; обучаются декодер и, где указано, общая ось. Рабочие веса, API и метки аугментаций не заменены.', '',
        'Эталон численного угла — новая строгая контурная геометрия по ручным разделителям. Это **псевдоэталон**, не независимое измерение истинной оси специалистом. На оригиналах табличное нарушение >5° оценено отдельно. Использованный test уже рассматривался ранее и остаётся диагностическим. Выбор эпох и вариантов только по validation.', '',
        '## Целевая метрика', '',table(['Вариант','Эпоха','Validation MAE с пропусками','Test MAE, °','Покрытие','Оригиналы MAE','Аугментации MAE'],rows),'',
        'При неопределённом угле критерий выбора добавляет 15° на такой снимок, чтобы улучшение MAE не достигалось отказом от сложных случаев. Контурные методы и обученная общая ось сравниваются как разные явно названные способы решения задачи.', '',
        f'У обученной головы угол направления определён даже при оси вне рамки; отдельно проверено отсутствие отрезка в рамке: {head_frame_failures}. Лучшая модель разделителей выбирается среди сохранённых по угловому критерию эпох, а не среди всех возможных эпох по отдельному критерию линий.', '',
        f'Парная разность критерия test (победитель − control): {paired["winner_minus_control_failure_penalized_mae_deg"]:.3f}°, 95% интервал [{paired["ci95"][0]:.3f}; {paired["ci95"][1]:.3f}]. Bootstrap по исследованиям, 1000 повторов; он не учитывает неопределённость переобучения.', '',
        '## Промежуточные итоги по правкам', '',
        '1. **Пропуски и пустые зоны.** Введены относительные квадратичные штрафы больших интервалов и отсутствующего покрытия сверху/снизу. Они действуют на все найденные пики, а не только на кандидатов около эталонных линий. Крайнее покрытие привязано к ручной разметке; отсутствующая анатомия не должна дорисовываться. Не принуждаем расстояния к абсолютной равномерности.', '',
        '2. **Перестроение разделителей.** Реализованы биссектрисы через общий узел соседних осей в физических координатах. При фиксированных осях угол не меняется по построению. Отдельно измерено последующее повторное построение контуров: возможны смена кластеров, отказы и пересечение линий. Это диагностическая абляция, не выбранный по test новый постпроцессор.', '',
        '3. **Угловая потеря.** Добавлена отдельная общая ось: сеть предсказывает две X-координаты на верхней и нижней рамке. Угол вычисляется atan2 в мм, loss содержит квадрат ошибки в градусах, делённый на 5°, и координатную потерю. Чем больше ошибка, тем больше штраф. Выбор эпох перенесён на угловой MAE, а не на количество линий.', '',
        '4. **Кластеры близких линий.** Старый штраф был мал в масштабе loss и не охватывал лишние пики. Новый штраф нормирован на минимальный интервал и имеет вес 8. Координаты кандидатов считаются локальным soft-argmax; их дискретное обнаружение не дифференцируется, но координаты и уверенность получают градиенты. В таблице ниже кластеры всех вариантов измерены по одинаковому c=2 — сравнивать их с разными порогами было бы некорректно.', '',
        '5. **Крайние позвонки.** Направление верхнего/нижнего фрагмента строго равно физическому перпендикуляру к крайнему разделителю. Боковые границы начинаются в контактных точках контура соседнего полного позвонка и продолжаются параллельно этому перпендикуляру до рамки. Ткани ниже/выше не подбирают новое направление и не тянут ось к подвздошной кости. Общий узел сохраняется. Это геометрическая экстраполяция, а не доказанный контур тела крайнего позвонка.', '',
        '6. **c=0,5.** Проверен отдельно от c=2 в паре all_peaks и паре angle. В train строгий минимум конфликтует с 214/489 эталонных геометрий против 2/489 для c=2; среди исходников — 67 против 2. Это приближение интервалами в центре изображения, не полный контурный зазор. Негативы и естественно разные размеры позвонков сохраняются; жёстко навязывать минимум всем снимкам нельзя.', '',
        table(['Вариант','Кластеры c=2 /154','Кластеры c=0,5 /154','Пропущен верх >5 мм','Пропущен низ >5 мм','Recall линий','Precision линий'],[[k,gaps[k]['close_c2'],gaps[k]['close_c05'],gaps[k]['missing_top_5mm'],gaps[k]['missing_bottom_5mm'],n(v['test']['lines']['recall_5mm']),n(v['test']['lines']['precision_5mm'])] for k,v in results.items()]),'',
        'Штрафы всех пиков, больших пробелов и крайних пропусков изменены совместно: эта серия оценивает их пакет, а не доказывает отдельный эффект каждого коэффициента. c и наличие общей оси имеют отдельные сравнения.', '',
        f'Пакет all_peaks_c2 против control: кластеры по единому c=2 {gaps["control_c2"]["close_c2"]} → {gaps["all_peaks_c2"]["close_c2"]}; пропуски сверху {gaps["control_c2"]["missing_top_5mm"]} → {gaps["all_peaks_c2"]["missing_top_5mm"]}, снизу {gaps["control_c2"]["missing_bottom_5mm"]} → {gaps["all_peaks_c2"]["missing_bottom_5mm"]}. Recall {n(base["test"]["lines"]["recall_5mm"])} → {n(results["all_peaks_c2"]["test"]["lines"]["recall_5mm"])}; precision {n(base["test"]["lines"]["precision_5mm"])} → {n(results["all_peaks_c2"]["test"]["lines"]["precision_5mm"])}. Потеря одной линии не обязана образовать интервал выше широкого порога holes, поэтому одновременно учитываем recall и крайние пропуски.', '',
        f'Усиление c=2 → 0,5 без угловой головы: validation-критерий {n(results["all_peaks_c2"]["validation"]["angle"]["failure_penalized_mae_deg"])} → {n(results["all_peaks_c05"]["validation"]["angle"]["failure_penalized_mae_deg"])}; test MAE {n(results["all_peaks_c2"]["test_angles"]["mae_deg"])} → {n(results["all_peaks_c05"]["test_angles"]["mae_deg"])}. Итог зависит от части данных; это не устойчивое доказательство пользы строгого минимума.', '',
        f'Обученная общая ось: test MAE angle_c2={n(results["angle_c2"]["test_angles"]["mae_deg"])}°, angle_c05={n(results["angle_c05"]["test_angles"]["mae_deg"])}°. Контурные углы тех же моделей: {n(results["angle_c2"]["geometry_angles"]["mae_deg"])}° и {n(results["angle_c05"]["geometry_angles"]["mae_deg"])}°. Угловая голова и контурная цепочка — отдельные результаты; дополнительная цель может улучшить разделители, но сама голова требует проверки обобщения.', '',
        '## Контурные оси и перестроение линий', '',
        table(['Источник линий','Прежние крайние MAE','Strict MAE','Strict coverage','Биссектрисы + refit MAE','Coverage','Отказы'],[[k,n(v['weighted_terminals']['mae_deg']),n(v['strict_terminals']['mae_deg']),n(v['strict_terminals']['coverage']),n(v['strict_plus_bisectors_refit']['mae_deg']),n(v['strict_plus_bisectors_refit']['coverage']),v['bisector_rejections']] for k,v in report['geometry_comparisons'].items()]),'',
        'Все сравнения используют один численный эталон. Он построен методом strict, поэтому преимущество strict относительно него само по себе не является доказательством превосходства над старым методом по истинной клинической оси.', '',
        '## Итог по исходной клинической цели', '',
        table(['Вариант','Оригиналы F1 >5°','Sensitivity','Specificity','AUC','Синтетика F1 >5°'],[[k,*[n((v['original_angles'].get('binary_existing_labels') or {}).get(m)) for m in ['f1','sensitivity','specificity','roc_auc']],n((v['synthetic_angles'].get('binary_existing_labels') or {}).get('f1'))] for k,v in results.items()]),'',
        'На 34 test-оригиналах четыре табличных положительных случая: ни один вариант не обнаружил их при пороге 5°. Табличный флаг, анатомическая ось и глобальный наклон не полностью совпадают: среди всех 166 исходных позвоночников 17 табличных нарушений, но только восемь строгих ручных геометрических осей превышают 5°. Если новое обучение точнее воспроизводит геометрию, это не обязано улучшить F1 по этой таблице. Нужны независимые два конца общей оси и угол на исходных снимках, особенно на расхождениях около 5°.', '',
        '## Что делать дальше', '',
        f'- Для численного угла сохранять **{winner} как исследовательского кандидата по validation** и контроль c=2: парный test-интервал включает ноль, подтверждённого улучшения общего угла нет. До независимой угловой разметки не менять рабочий API.',
        f'- Для разделительных линий отправная точка — **{best_lines} по validation**, а не автоматически модель с лучшим угловым head. Проверять одновременно recall, precision, края и зазоры.',
        '- Угловая голова может обходить ошибки линий. Следующее контролируемое обучение должно согласовать её с контурной цепочкой, сохранив независимый эталон общей оси; сначала проверить расхождения, а не увеличивать все штрафы.',
        '- Минимальный интервал сделать локальным/зависящим от размера тела позвонка и подтверждённого дефекта. Усиление c=0,5 применять только после проверки допустимости на эталоне.',
        '- Для пропусков и лишних линий проверить упорядоченную геометрическую голову с присутствием/неполной видимостью, мягким отталкиванием и отдельной разметкой крайних фрагментов; фиксированный лимит декодера сейчас семь линий.',
        '- Биссектрисы оставить безопасным геометрическим вариантом с контролем пересечений и без обязательного повторного fit, пока он не подтверждён на validation с независимыми осями.',
        '- После независимой угловой разметки сравнить partial/full encoder training и больший бюджет на одном протоколе. Нынешние четыре эпохи — проверка подхода, не финальное обучение.', '',
        '## Итог всей последовательности исследований', '',
        '- **Регрессия упорядоченных линий на замороженных признаках:** число линий улучшилось, но полнота локализации ≤5 мм упала до 10–17% против 65,4% прежней пространственной карты. Такой вариант не подходит как основной локализатор. Следующий шаг — обучение пространственных признаков вместе с геометрией, а не усиление штрафа числа.',
        '- **Пространственный декодер и квадратичные штрафы:** в предыдущей серии исходное изображение с новыми штрафами подняло test recall с 66,6% до 82,4% и снизило ошибку сопоставленных линий с 8,52 до 4,23 мм, но породило лишние линии на 77,9% снимков. Это лучшая отправная точка локализации той серии; нынешняя проверка всех пиков адресует именно этот недостаток.',
        '- **Отсечение яркости до верхних 10–20% пикселей:** ухудшило полноту до 64,7–70,2% и точность до 45,1–49,6%. Продолжать с исходной яркостью. Мягкая маска/дополнительный канал остаются гипотезами, пока не проверенными вариантами.',
        '- **Совместные внутренние оси:** общие узлы сохраняют непрерывность. Крайние контуры были уязвимы к соседним костям; строгая нормаль решает направление конструктивно, но требует отдельной проверки правильности крайних контактных точек.',
        f'- **Нынешняя угловая серия:** победитель численной цели — {winner}; разделителей — {best_lines}. Сравнение исторических серий не является чистой абляцией: различались головы, train-бюджет, критерий выбора и определение крайних осей. Чистые сравнения — внутри каждой серии.', '',
        '## Воспроизводимость и файлы', '',
        '`geometry_ml/spine_angle_study.py`, `spine_angle_geometry.py`, `spine_angle_finalize.py`, `spine_angle_summary.py`; `outputs/spine_angle_study_20260929/` хранит протокол, версии, хеши, эталоны, веса, истории, подробные прогнозы, CI и сравнения. HTML: `team_demo/spine_angle_study_20260929.html`. Ноутбук: `DXA_Spine_Angle_20260929.ipynb` в корне. Численные проверки и регрессии decoder/metrics: 9 passed; полный набор: **91 passed**, три предупреждения.', '',
        'Использовать Python с совместимыми torch/torchvision и доступным pydicom. В данном сеансе pydicom 3.0.1 подключён из локального wheel через `analysis_20260929/runtime`, основной venv не переустанавливался. Масштаб исходников номинальный Y=1,05 / X=0,6 мм; этот источник неопределённости остаётся.', '',
        '```powershell', r'$env:PYTHONPATH="$PWD\analysis_20260929\runtime;$PWD"', r'.venv\Scripts\python.exe -m dxa_project.geometry_ml.spine_angle_study --epochs 4', r'.venv\Scripts\python.exe -m dxa_project.geometry_ml.spine_angle_summary', '```', '',
        'Повторный запуск исследования в этой же папке использует уже сохранённые завершённые варианты. Для нового эксперимента с иными условиями нужна отдельная версия и каталог; текущие результаты нельзя выдавать за новое обучение.',
    ]
    text='\n'.join(lines);(ROOT/'dxa_project/SPINE_ANGLE_RESULTS_20260929.md').write_text(text,encoding='utf-8')
    page=ROOT/'dxa_project/team_demo/spine_angle_study_20260929.html';content=page.read_text(encoding='utf-8')
    extra='<section id="step-summary"><h2>Промежуточные итоги по всем шагам</h2><p>По validation: угол — <b>'+winner+'</b>; контурный угол — <b>'+best_geometry+'</b>; линии — <b>'+best_lines+'</b>.</p><p>Парная разность углового критерия с control: '+n(paired['winner_minus_control_failure_penalized_mae_deg'])+'°, 95% ДИ '+str(paired['ci95'])+'. Независимые оси специалистов пока не размечены.</p><p><a href="../SPINE_ANGLE_RESULTS_20260929.md">Полный разбор каждого изменения и следующих шагов</a></p><img style="max-width:950px;width:100%" src="spine_angle_assets/validation_angle.png"><img style="max-width:1200px;width:100%" src="spine_angle_assets/test_angle.png"></section>'
    if 'id="step-summary"' not in content:content=content.replace('<h2>Одни и те же примеры</h2>',extra+'<h2>Одни и те же примеры</h2>')
    page.write_text(content,encoding='utf-8')
    def cell(kind,source,outputs=None):
        r={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':r.update(execution_count=None,outputs=outputs or [])
        return r
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        cell('markdown','# DXA: целевой угол позвоночника\n\nПять контролируемых вариантов. Геометрический эталон по ручным линиям; отдельная оценка табличного >5°. Повторное обучение выключено.'),
        cell('code','from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path.cwd()\nif not (ROOT / "dxa_project").exists():\n    raise RuntimeError("Откройте ноутбук из корня Хакатон")\nreport = json.loads((ROOT / "dxa_project/outputs/spine_angle_study_20260929/report.json").read_text(encoding="utf-8"))\nreport["winner_validation"]'),
        cell('code','rows = [{"variant": k, "epoch": v["best_epoch"], "test_mae_deg": v["test_angles"]["mae_deg"], "coverage": v["test_angles"]["coverage"], "original_mae_deg": v["original_angles"]["mae_deg"], "synthetic_mae_deg": v["synthetic_angles"]["mae_deg"]} for k,v in report["results"].items()]\ndisplay(pd.DataFrame(rows))'),
        cell('markdown',text),
        cell('code','from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "dxa_project/team_demo/spine_angle_assets/validation_angle.png")))\ndisplay(Image(filename=str(ROOT / "dxa_project/team_demo/spine_angle_assets/test_angle.png")))'),
        cell('code','RUN_TRAINING = False\nif RUN_TRAINING:\n    from dxa_project.geometry_ml.spine_angle_study import run\n    run(epochs=4)\n# Существующие завершённые варианты используются повторно; новые условия требуют новой версии.'),
    ]}
    for i,c in enumerate(nb['cells']):c['id']=f'angle-study-{i}'
    save(ROOT/'DXA_Spine_Angle_20260929.ipynb',nb)
    print(json.dumps({'winner':winner,'best_geometry':best_geometry,'best_lines':best_lines,'paired':paired,'gap_audit':gaps},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
