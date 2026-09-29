"""Compare saved runs without training or changing calibrated thresholds."""
import csv
import json
from pathlib import Path
from .evaluate import with_ci, binary_metrics
from .report_reviewed import LABELS, table, n

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dxa_project/outputs/retrained_20260929'
DEST = ROOT / 'dxa_project/team_demo/model_comparison_20260929.html'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def run():
    folders = {'light': OUT/'light', 'heavy': OUT/'heavy_stratified'}
    reports = {k: read(p/'evaluation/report.json') for k,p in folders.items()}
    predictions = {k: {r['relative_path']: r for r in read(p/'evaluation/predictions.json')} for k,p in folders.items()}
    common = sorted(set(predictions['light']) & set(predictions['heavy']))
    paired = {}
    for kind in folders:
        with (folders[kind]/'evaluation/quality_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for field in ('truth','prediction'):
                r[field] = int(r[field]) if r[field] not in ('','None') else None
            r['score'] = float(r['score']) if r['score'] not in ('','None') else None
        paired[kind] = rows
    shared = []
    keys = sorted({(r['region'],r['task']) for r in paired['light']})
    for region,task in keys:
        sets = {k: {r['relative_path']:r for r in rows if r['relative_path'] in common and r['region']==region and r['task']==task and r['truth'] is not None} for k,rows in paired.items()}
        labeled = set(sets['light']) & set(sets['heavy'])
        assert all(sets['light'][p]['truth']==sets['heavy'][p]['truth'] for p in labeled)
        usable = sorted(p for p in labeled if all(sets[k][p]['prediction'] is not None for k in sets))
        shared.append({'region':region,'task':task,'labeled':len(labeled),'paired_usable':len(usable), **{k:with_ci([sets[k][p] for p in usable]) for k in sets}})
    calibrations = {}
    for k,p in folders.items():
        c = read(p/'evaluation/calibration.json')
        a = c['examples']; t = c['rotation_threshold_px2']
        def rows(threshold):
            return [{'truth':y,'prediction':int(area<threshold),'score':-area} for area,y in a]
        candidates = sorted(set([0.,1.] + [float(area)+.5 for area,y in a]))
        balanced = max(candidates,key=lambda x:binary_metrics(rows(x))['balanced_accuracy'])
        calibrations[k] = {'threshold_px2':t,'n':len(a),'violation_count':sum(y for area,y in a),'max_area_px2':max(area for area,y in a),'selected_validation_metrics':binary_metrics(rows(t)), 'diagnostic_balanced_accuracy_threshold_px2':balanced,'diagnostic_validation_metrics':binary_metrics(rows(balanced)), 'note':'Diagnostic only: thresholds and predictions were not changed; no test-based calibration.'}
    module_metrics = {}
    for kind in folders:
        modules = {}
        for task in ('spine','hip','artifact'):
            path = folders[kind]/(task+'_report.json') if kind=='heavy' else OUT/('spine_encoder_lr_2e4' if task=='spine' else 'light_base')/(task+'_report.json')
            d=read(path)
            modules[task]=next(x['validation_metrics'] for x in d['history'] if x['epoch']==d['best_epoch'])
        if kind=='light':
            modules['points']=read(OUT/'selection.json')['selected']['metrics']['original']
        else:
            d=read(folders[kind]/'points_report.json')
            modules['points']=min(d['history'],key=lambda x:x['validation_loss'])['metrics']
        module_metrics[kind]=modules
    result = {'status':read(OUT/'status.json'),'own_test_reports':reports,'module_inner_validation_metrics':module_metrics,'common_test_files':len(common),'common_test_studies':len({predictions['light'][p]['study'] for p in common}),'common_test_paths':common,'paired_tasks':shared,'rotation_calibration':calibrations,'warning':'Own tests differ. Common subset has only four independent studies; no reliable architecture ranking.'}
    (OUT/'model_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    parts = ['<h1>DXA: малая и большая модели</h1><p>Большой прогон завершён 29.09.2026 в 06:43, код завершения 0. Все 99 исходных и 135 синтетических тестовых файлов обработаны без ошибок.</p>',
        '<h2>Вывод</h2><p>Увеличение модели не дало убедительного улучшения полного решения. Главная проблема большой версии — порог ротации: нарушение предсказывается для всех 62 оценённых снимков бедра, включая 49 качественных. Малую версию также нельзя считать готовой: она пропускает нарушения оси и часть нарушений ротации.</p>',
        '<h2>Условия сравнения</h2><p>Малая версия: 100 тестовых снимков, 19 исследований. Большая: 99 снимков, 20 исследований; новое разбиение по исследованиям со стратификацией по артефактам. В тесте позвоночника с артефактами соответственно 19/35 и 8/34 снимков. Различия ниже отражают одновременно архитектуру, состав данных и калибровку. Аугментации исходных исследований не переходят между обучением и контролем.</p>']
    if (DEST.parent/'component_diagnostics_20260929.html').exists():
        parts.insert(1,'<p><a href="component_diagnostics_20260929.html" style="color:#9cf">Сравнение отдельных модулей и диагностика оси по ручным линиям</a></p>')
    names = {'light':'Малая','heavy':'Большая'}
    overallrows=[]
    for k,r in reports.items():
        m=r['overall_any_violation']['metrics']; ci=r['overall_any_violation']['ci95']['f1']
        overallrows.append([names[k],n(r['router_macro_f1']),n(m['f1']),f'{ci[0]:.3f}–{ci[1]:.3f}',n(m['roc_auc']),n(m['balanced_accuracy']),n(m['sensitivity']),n(m['specificity']),n(r['overall_any_violation_coverage'])])
    parts += ['<h2>Полный пайплайн: собственные тестовые выборки</h2>',table(['Модель','Область: macro-F1','Нарушение: F1','95% интервал F1','ROC AUC','Сбаланс. точность','Чувствительность','Специфичность','Покрытие'],overallrows), '<p>1 означает нарушение. Покрытие — доля файлов с полным определённым ответом; отказы исключены из F1. Интервалы получены повторной выборкой исследований. Общий ROC AUC использует максимум оценок разных правил, а не обученную и откалиброванную вероятность; его нельзя интерпретировать как медицинскую вероятность.</p>']
    rows=[]
    for region,task in keys:
        ms={k:next(x for x in r['metrics'] if x['region']==region and x['task']==task) for k,r in reports.items()}
        rows.append([LABELS[region],LABELS[task]]+[n((ms[k].get('metrics') or {}).get(field)) for field in ('f1','roc_auc') for k in folders]+[f"{ms[k]['n']}/{ms[k]['labeled']}" for k in folders])
    parts += ['<h2>По каждому нарушению: собственные тесты</h2>',table(['Область','Проверка','F1 малая','F1 большая','AUC малая','AUC большая','Ответы малая','Ответы большая'],rows), '<p>В исходном тесте позиционирования бедра нет нарушений: чувствительность и AUC проверить невозможно. F1=0 при ложных тревогах и отсутствие F1 без тревог не означают сравнение способности обнаруживать нарушения.</p>',
        f"<h2>Одни и те же тестовые снимки: {len(common)}, исследований: {result['common_test_studies']}</h2><p>Оба прогона исключали их из обучения. Таблица использует одинаковые снимки с определёнными ответами обеих моделей. При четырёх исследованиях это диагностическая проверка, а не доказательство превосходства.</p>",table(['Область','Проверка','Парных ответов / размечено','F1 малая','F1 большая','AUC малая','AUC большая'],[[LABELS[x['region']],LABELS[x['task']],f"{x['paired_usable']}/{x['labeled']}"]+[n((x[k].get('metrics') or {}).get(field)) for field in ('f1','roc_auc') for k in folders] for x in shared])]
    parts += ['<h2>Почему сломалась ротация</h2><p>Порог выбирался по максимуму F1 только на внутренней валидации. Для большой версии выбран 442,5 пикселя², при максимальной предсказанной площади 442: все валидационные бедра стали нарушениями. Это допустимый максимум F1 при таком распределении, но непригодная рабочая точка. На тесте специфичность ротации равна нулю. У малой версии порог 0,5: нарушениями становятся только нулевые маски. Нужны более надёжные маски и выбор порога с контролем специфичности на валидации; менять порог по тестовым результатам нельзя.</p>',table(['Модель','Порог px²','Нарушений / валидация','Специфичность валидации','Порог по balanced accuracy (только диагностика)'],[[names[k],c['threshold_px2'],f"{c['violation_count']}/{c['n']}",n(c['selected_validation_metrics']['specificity']),c['diagnostic_balanced_accuracy_threshold_px2']] for k,c in calibrations.items()])]
    parts += ['<p>Проблема глубже выбора порога: AUC по отрицательной площади на внутренней валидации большой модели — 0,178, то есть площадь ранжирует нарушения в противоположную сторону. Даже диагностический порог 12,5 по сбалансированной точности обнаруживает лишь 4 из 24 нарушений. Нужно проверить маски, физический масштаб площади и связь площади с меткой; одной перестановки порога недостаточно.</p>', '<h2>Отдельные модули: внутренняя валидация лучших весов</h2><p>Это выборочные оценки на разных валидационных исследованиях, использованные при выборе весов, а не независимый тест. Ошибки в мм используют номинальный масштаб исходников. В модуле бедра встроенные точки заменены отдельной моделью точек; поэтому их ошибку сюда не включаем.</p>',table(['Модуль / метрика','Малая','Большая'],[[label]+[n(module_metrics[k][module].get(metric)) for k in folders] for label,module,metric in [('Подвздошные точки: ошибка, мм ↓','spine','mean_point_error_nominal_mm'),('Маска линий позвоночника: Dice ↑','spine','pixel_dice'),('ROI бедра: IoU ↑','hip','mean_roi_iou'),('Малый вертел: Dice ↑','hip','pixel_dice'),('Точки бедра: ошибка, мм ↓','points','mean_error_mm'),('Точки бедра: доля в 10 мм ↑','points','pck_10mm'),('Рамки артефактов: полнота при IoU≥0,5 ↑','artifact','box_recall_iou50')]])]
    for k,p in folders.items():
        syn=read(p/'synthetic_evaluation/report.json')
        parts += [f'<h2>{names[k]}: синтетический тест, {syn["selected_images"]} снимков</h2>',table(['Область','Проверка','F1','AUC','Покрытие'],[[LABELS[x['region']],LABELS[x['task']],n((x.get('metrics') or {}).get('f1')),n((x.get('metrics') or {}).get('roc_auc')),n(x['coverage'])] for x in syn['metrics']])]
    parts += ['<h2>Следующий шаг</h2><ol><li>Оставить оба набора весов, не заменять малый пайплайн большим целиком.</li><li>Пересмотреть выбор порога ротации на валидации: ограничить ложные тревоги, проверить пересчёт площади в мм² и качество масок.</li><li>Проверить пропуски нарушений оси на исходниках: ошибки линий/оси, масштаб и соответствие бинарной разметки; хороший синтетический тест не заменяет исходный.</li><li>Сравнить обе архитектуры при одном зафиксированном разбиении по исследованиям и артефактам. После использования этого теста для анализа для окончательной оценки нужны новые исследования или внешний контроль.</li></ol><p>Скорость: малая 0,135 с, большая 0,089 с в среднем, без первоначальной загрузки весов. Замеры сделаны в разные моменты, поэтому вывод, что большая быстрее, не обоснован: нужен повторный замер на одинаковых файлах и условиях.</p>']
    DEST.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение DXA</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#101923;color:#e8eff5}p,li{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526477;padding:8px;text-align:left}h2{margin-top:36px}</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print(json.dumps({'report':str(DEST),'common_files':len(common),'common_studies':result['common_test_studies'],'paired_tasks':[{k:v for k,v in x.items() if k not in ('light','heavy')}|{k:x[k].get('metrics') for k in folders} for x in shared]},ensure_ascii=False))

if __name__=='__main__':
    run()
