"""Current training progress and final Russian team report, no raw study identifiers."""
from pathlib import Path
import json,csv,html,time,argparse
from .retrain_reviewed import ROOT,OUT,AUG

DEST=ROOT/'dxa_project/team_demo'
LABELS={'SPINE':'Позвоночник (spine)','LEG_LEFT':'Левое бедро (left hip)','LEG_RIGHT':'Правое бедро (right hip)',
        'spine_position':'Укладка позвоночника','spine_axis':'Наклон позвоночника','spine_artifact':'Артефакты (artifact)',
        'hip_position':'Позиционирование бедра','hip_roi':'Отступы области интереса (ROI)','hip_rotation':'Ротация бедра'}
MODULES={'router':'Определение области (router)','spine':'Позвоночник (spine)','hip':'ROI и маска бедра (hip)',
         'artifact':'Артефакты (artifact)','hip_points':'Точки бедра (hip landmarks)'}
VARIANT_LABELS={'shared256':'Общая сеть, 256 (shared256)','coordinate256':'Координатная потеря, 256 (coordinate256)',
                'shared384':'Общая сеть, 384 (shared384)','shared512':'Общая сеть, 512 (shared512)',
                'roi384':'ROI с контекстом, 384 (roi384)','independent256':'Три отдельные сети, 256 (independent256)'}
STAGES={'waiting_for_new_augmentation':'Генерация новой аугментации','augmentation_audit':'Проверка всех новых изображений',
        'light_training':'Обучение модулей простой модели','light_complete':'Малая модель завершена; большая ожидает запуска','point_screening':'Сравнение шести вариантов модели точек',
        'point_finalist_training':'Полное обучение двух финалистов','pipeline_inference':'Проверка полного пайплайна',
        'heavy_real_timing':'Измерение времени большой модели','heavy_training':'Обучение большой модели','heavy_transition':'Переход к большой модели: ориентир 8 часов, максимум 12','complete':'Работа завершена',
        'spine_learning_rate_comparison':'Сравнение скоростей обучения позвоночника','failed':'Ошибка выполнения'}
def read(p):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    except (json.JSONDecodeError,OSError):return None  # A writer may be finishing a report.
def n(v):return '—' if v is None else f'{v:.3f}'
def table(head,rows):return '<table><tr>'+''.join(f'<th>{html.escape(str(s))}</th>' for s in head)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(s))}</td>' for s in row)+'</tr>' for row in rows)+'</table>'

def render():
    DEST.mkdir(exist_ok=True);status=read(OUT/'status.json') or {};parts=['<h1>DXA: повторное обучение после исправления разметки</h1>',
        '<p>Версия данных: 29 сентября 2026. Старые аугментации и веса сохранены отдельно.</p>',
        f'<h2>Состояние: {html.escape(STAGES.get(status.get("stage"),status.get("stage","Подготовка")))}</h2>',
            '<p>'+html.escape(str(status.get('updated','')))+' · '+html.escape(MODULES.get(status.get('task'),VARIANT_LABELS.get(status.get('variant'),str(status.get('architecture','')))))+'</p>']
    logs=list(OUT.glob('training*.log'))+list(OUT.glob('heavy_12h.log'))+list(OUT.glob('heavy_unattended.log'));log=max(logs,key=lambda p:p.stat().st_mtime) if logs else OUT/'training.log'
    if log.exists():
        last=next((line for line in reversed(log.read_text(encoding='utf-8',errors='replace').splitlines()) if line.startswith('{') and '"epoch"' in line),None)
        if last:
            try:
                progress=json.loads(last);key=progress.get('task',progress.get('variant'));parts+=[f'<p>Последняя завершённая эпоха: {progress.get("epoch")}; {html.escape(MODULES.get(key,VARIANT_LABELS.get(key,str(key))))}.</p>']
            except ValueError:pass
    corrections=read(OUT/'corrections.json')
    if corrections:parts+=['<h2>Разметка</h2><p>Новые рамки на №52, 55, 58 подтверждены. Флаг артефакта на №102, 121, 122, 287, 288 установлен в 1 в таблицах и DICOM. Пиксели исходников не менялись; сохранены резервные копии. После исправления расхождений рамок и флагов нет.</p>']
    if status.get('stage')=='failed':parts+=['<p>'+html.escape(str(status.get('error')))+'</p>']
    generation=read(AUG/'generation_report.json');progress=read(AUG/'progress.json')
    if generation:parts+=['<h2>Новые аугментации</h2><p>Сформировано 15 000 изображений: по 5 000 на каждую область. По 2 500 примеров соблюдения двух правил генерации (таргеты 0), 1 250 нарушений укладки/позиционирования и 1 250 нарушений оси/ROI (флаг нарушения 1). Метки артефактов и ротации учитываются отдельно.</p>']
    elif progress:parts+=[f'<p>Генерация: {progress["total_generated"]}/15000 изображений; область {LABELS[progress["region"]]}.</p>']
    audit=read(AUG/'audit_report.json')
    if audit:parts+=[f'<p>Проверены все {audit["files"]} файлов, координаты и метаданные. Найдено проблем: {len(audit["problems"])}. Полностью пересчитаны правила бедра; правила оси позвоночника дополнительно пересчитаны на {audit["recomputed_examples"]} выбранных примерах.</p>']
    protocol=read(OUT/'protocol.json')
    training_data=read(OUT/'training_data.json')
    if training_data:parts+=[f'<p>Обучение использует {training_data["train_images"]} изображений: 318 исходных и {training_data["train_images"]-318} аугментированных. Остальные исходные исследования и их аугментации исключены из обучения.</p>']
    if protocol:
        parts+=['<h2>Как разделены данные</h2>',table(['Часть','Исходные изображения','Исследования','С артефактами / позвоночники'],
            [[{'train':'Обучение','validation':'Выбор модели','test':'Контрольный набор'}[key],r['images'],r['studies'],f"{r['artifact_positive']} / {r['spine_images']}"] for key,r in protocol['summary'].items()]),
            '<p>Аугментации одного исходного исследования находятся в той же части. Повторы пикселей между частями запрещены. Контрольные 100 снимков уже рассматривались ранее: это сравнительный ориентир, не новый независимый тест.</p>']
    screening=read(OUT/'screening.json')
    if screening:
        parts+=['<h2>Сравнение моделей точек бедра</h2><p>Каждый вариант обучается восемь полных эпох с одинаковыми скоростями обучения. Координаты оцениваются в миллиметрах. Для варианта ROI на проверке используется предсказанная рамка; если точки не найдены, повторяется поиск на полном снимке.</p>',
            '<p>Оценка выбора = 55% доли точек ≤10 мм на исходных снимках + 25% этой доли на аугментированных проверочных снимках + 20% сбалансированной точности позиционирования на них. Все проверочные исследования исключены из обучения.</p>',
            table(['Вариант','Лучшая эпоха','Оценка ↑','Точки ≤10 мм: оригиналы ↑','Точки ≤10 мм: аугментации ↑','Позиционирование: F1 ↑','Время, мин'],
                  [[VARIANT_LABELS[c['variant']],c['epoch'],n(c['score']),n(c['metrics']['original']['pck_10mm']),n(c['metrics']['synthetic']['pck_10mm']),n(c['metrics']['synthetic']['position']['metrics']['f1']),n(c['seconds']/60)] for c in screening['candidates']])]
    selection=read(OUT/'selection.json')
    spine=read(OUT/'spine_selection.json')
    if spine:
        parts+=['<h2>Позвоночник: сравнение скоростей обучения</h2>',table(['Encoder LR','Head LR','Эпоха','Функция потерь на валидации ↓'],
            [[c['encoder_lr'],c['head_lr'],c['best_epoch'],n(c['validation_loss'])] for c in spine['candidates']]),
            f'<p>Выбран encoder LR {spine["selected"]["encoder_lr"]}. Данные, архитектура и функции потерь одинаковы; контрольный набор не использовался для выбора.</p>']
    if selection:
        s=selection['selected'];parts+=[f'<h2>Выбранная простая модель</h2><p>Модель точек: <b>{s["variant"]}</b>, сохранена эпоха {s["epoch"]}. Два финалиста прошли до 24 эпох с остановкой при отсутствии улучшения. Контрольные снимки при выборе не использовались.</p>']
    heavy_protocol=read(OUT/'heavy_stratified/protocol.json')
    if heavy_protocol:
        parts+=['<h2>Большая модель: новое разбиение по исследованиям и артефактам</h2>',
                table(['Выборка','Исследований','Снимков позвоночника','Артефакты','Доля артефактов'],[[p,s['studies'],s['spine_images'],s['artifact_positive'],n(s['artifact_fraction'])] for p,s in heavy_protocol['summary'].items()]),
                '<p>Все снимки исследования и аугментации остаются в одной выборке. Контрольные выборки малой и большой моделей различаются, поэтому их итоговые числа нельзя сравнивать напрямую. Конфигурация точек выбрана в прежнем эксперименте; новый контроль не является внешним независимым тестом.</p>']
    for folder,title in ((OUT/'light','Простой пайплайн'),(OUT/'heavy_stratified','Большой пайплайн')):
        report=read(folder/'evaluation/report.json')
        if not report:continue
        parts+=[f'<h2>{title}: итоговая проверка</h2>',f'<p>Обработано {report["processed_files"]}/{report["validation_originals"]} файлов. Macro-F1 классификатора области (router): {n(report["router_macro_f1"])}. Среднее время после прогрева: {n(report["seconds_per_file_mean"])} с.</p>']
        overall=report['overall_any_violation'];m=overall.get('metrics') or {};ci=overall.get('ci95',{}).get('f1')
        parts+=[f'<p>Обнаружение любого нарушения: F1 {n(m.get("f1"))}, ROC AUC {n(m.get("roc_auc"))}; 95% интервал F1: {"—" if ci is None else "–".join(n(x) for x in ci)}. Покрытие: {n(report["overall_any_violation_coverage"])}.</p>',
            table(['Область','Проверка','N','Чувствительность ↑','Специфичность ↑','F1 ↑','ROC AUC ↑','Покрытие ↑'],
                [[LABELS[r['region']],LABELS[r['task']],r['n'],*[n((r.get('metrics') or {}).get(k)) for k in ('sensitivity','specificity','f1','roc_auc')],n(r['coverage'])] for r in report['metrics']])]
        synthetic=read(folder/'synthetic_evaluation/report.json')
        if synthetic:
            parts+=[f'<h3>{title}: отдельная проверка синтетических нарушений</h3><p>{synthetic["selected_images"]} аугментаций из контрольных исследований. Они исключены из обучения и выбора модели. Эти результаты показывают устойчивость к искусственным изменениям, а не заменяют проверку оригиналов.</p>',
                table(['Область','Проверка','N','F1 ↑','ROC AUC ↑','Покрытие ↑'],[[LABELS[r['region']],LABELS[r['task']],r['n'],n((r.get('metrics') or {}).get('f1')),n((r.get('metrics') or {}).get('roc_auc')),n(r['coverage'])] for r in synthetic['metrics']])]
    if (DEST/'model_comparison_20260929.html').exists():parts+=['<p><a href="model_comparison_20260929.html" style="color:#75caff">Сравнение малой и большой моделей: результаты, проблемы и общие тестовые снимки</a></p>']
    if (DEST/'retraining_examples_20260929.html').exists():parts+=['<p><a href="retraining_examples_20260929.html" style="color:#75caff">Примеры успехов и ошибок отдельно для каждого модуля</a></p>']
    parts+=['<h2>Как читать метрики</h2><ul><li>Чувствительность: какая доля нарушений обнаружена. Низкое значение означает пропущенные проблемы.</li><li>Специфичность: какая доля хороших снимков правильно принята. Низкое значение означает много ложных тревог.</li><li>F1: баланс обнаруженных нарушений и ложных тревог; 1 — идеально. Метрика зависит от доли нарушений.</li><li>ROC AUC: качество ранжирования; 0,5 соответствует случайному ранжированию, 1 — идеальному. При наличии только одного класса AUC не определена.</li><li>Покрытие: доля снимков, для которых система дала определённый ответ. Неопределённые ответы исключаются из F1, поэтому F1 нужно читать вместе с покрытием.</li><li>Точки ≤10 мм: доля видимых точек, найденных не дальше 10 мм. Пропущенные точки считаются неуспехом.</li><li>95% интервалы считаются по исходным исследованиям. 15 000 аугментаций не означают 15 000 независимых пациентов.</li></ul><p>Риски: мало независимых артефактов; частично пропавшие подвздошные кости на аугментациях ещё нуждаются в доразметке; физический масштаб исходников основан на номинальном размере пикселя, где в DICOM нет измеренного масштаба.</p>']
    timing=read(OUT/'heavy_timing.json')
    if timing:
        parts+=['<h2>Большая модель: измерение времени</h2>',f'<p>Оценка полного цикла: {timing["estimate_hours"]:.2f} ч. Ориентир — 8 часов, верхняя граница — 12 часов. Включены 30% запаса и 10 минут для итоговой проверки. '+('Полное обучение разрешено.' if timing['allowed_full_training'] else 'Полное обучение не запущено: оценка превышает 12 часов.')+'</p>',
            table(['Модуль','Размер входа','Пакет изображений (batch)','Эпохи','Оценка обучения, мин'],[[MODULES[r['task']],r['input_size'],r['batch_size'],r['planned_epochs'],n(r['estimated_seconds']/60)] for r in timing['tasks']]),
            '<p>Измерялись обучение и проверка на новых реальных DICOM; большая модель использует ResNet50, детектор — Faster R-CNN ResNet50 FPN v2 с внутренним размером 800. Ранняя остановка не предполагается при расчёте лимита.</p>']
    parts+=['<h2>Воспроизводимость</h2><p>Ноутбуки DXA_*_20260929.ipynb содержат результаты и команды повторного запуска. Веса и подробные отчёты: dxa_project/outputs/retrained_20260929. Данные: dxa_project/outputs/augmented_15000_20260929.</p>']
    css='body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 25px;background:#101923;color:#e8eff5}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;border:1px solid #3a4a5a;text-align:left}th{background:#233549}p,li{line-height:1.55}h2{margin-top:35px}'
    refresh='<meta http-equiv="refresh" content="30">' if status.get('stage') not in ('complete','failed') else ''
    (DEST/'retraining_20260929.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'+refresh+'<title>DXA — повторное обучение</title><style>'+css+'</style>'+''.join(parts)+'</html>',encoding='utf-8')

def finish_notebooks():
    from .retrain_reviewed import notebooks
    import nbformat
    from nbclient import NotebookClient
    notebooks()
    for path in ROOT.glob('DXA_*_20260929.ipynb'):
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,path)

def main(watch=False):
    while True:
        render();status=read(OUT/'status.json') or {}
        if status.get('stage')=='complete':
            finish_notebooks();render();break
        if status.get('stage')=='failed':break
        if not watch:break
        time.sleep(10)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
