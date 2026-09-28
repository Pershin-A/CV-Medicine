"""Collect immutable experiment provenance and a readable report."""
from pathlib import Path
import csv, hashlib, json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mistune


def main():
    root=Path(__file__).resolve().parents[2]
    project=root/'dxa_project'; run=project/'outputs/geometry_ml_augmented_5epochs'
    aug=project/'outputs/augmented_15000_final'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    training=read(run/'report.json'); evaluation=read(run/'evaluation/report.json')
    synthetic=read(run/'synthetic_evaluation/report.json'); audit=read(aug/'audit_report.json')
    with (aug/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    counts={}
    for region,targets in {'SPINE':['spine_position','spine_axis','spine_artifact'],
                          'LEG_LEFT':['hip_position','hip_roi','hip_rotation'],
                          'LEG_RIGHT':['hip_position','hip_roi','hip_rotation']}.items():
        counts[region]={key:dict(Counter(r.get(key,'') or 'unknown' for r in rows if r['region']==region)) for key in targets}
    (run/'actual_target_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    provenance={}
    for name,path in {'authoritative_labels':root/'Размеченные/labels.csv',
                      'corrected_reference_manifest':project/'outputs/manifest.csv',
                      'augmentation_manifest':aug/'manifest.csv'}.items():
        provenance[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    provenance['generated_geometry_sha256']={r['geometry_path']:hashlib.sha256((aug/r['geometry_path']).read_bytes()).hexdigest() for r in rows}
    provenance['original_geometry_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'Размеченные/geometry').glob('*.json')}
    provenance['checkpoints_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.pt')}
    provenance['protocol']={'fold':0,'seed':42,'epochs':5,'size':256,'batch_size':8,'manual_overlay':None}
    (run/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,4,figsize=(15,3.3))
    for ax,t in zip(axes,training['tasks']):
        h=t['history']; ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label='train')
        if t['task']!='artifact': ax.plot([x['epoch'] for x in h],[x['validation_loss_or_detection_count'] for x in h],label='validation')
        ax.set_title(t['task']); ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(run/'learning_curves.png',dpi=150); plt.close(fig)
    label={'SPINE':'Позвоночник','LEG_LEFT':'Левое бедро','LEG_RIGHT':'Правое бедро',
           'spine_position':'Укладка','spine_axis':'Ось','spine_artifact':'Артефакты',
           'hip_position':'Позиционирование','hip_roi':'ROI','hip_rotation':'Ротация'}
    def formatted(m,key):
        value=(m.get('metrics') or {}).get(key)
        if value is None:return '—'
        ci=m.get('ci95',{}).get(key)
        return f'{value:.3f}'+(f' [{ci[0]:.3f}; {ci[1]:.3f}]' if ci else '')
    def table(results):
        text=['| Область | Проверка | N | Покрытие | Чувствительность | Специфичность | F1 [95% ДИ] | ROC AUC [95% ДИ] |',
              '|---|---|---:|---:|---:|---:|---|---|']
        for m in results:
            v=m.get('metrics') or {}; number=lambda k:'—' if v.get(k) is None else f'{v[k]:.3f}'
            text.append(f"|{label[m['region']]}|{label[m['task']]}|{m['n']}|{m['coverage']:.1%}|{number('sensitivity')}|{number('specificity')}|{formatted(m,'f1')}|{formatted(m,'roc_auc')}|")
        return text
    lines=['# Эксперимент DXA: 15 000 аугментаций и обучение','',
           '## 1. Оси и контуры','',
           '[50 обновлённых примеров](../spine_axes_50_terminal_final/index.html). Крайние оси смешаны с соседними по физическим высотам; внутренние общие точки сохранены. Верхний контур перестроен независимо, симметрично относительно новой оси. Учитывается анизотропия пикселя и ограничение локального наклона ±12°. Из 225 полных осей все проходят геометрические проверки; 16 из 50 снимков имеют предупреждения для визуального просмотра. Это не оценка точности относительно эталонных контуров.',
           '', '## 2. Генерация и аудит','',
           'Готовый набор: `outputs/augmented_15000_final`. Ровно 15 000 DICOM и 15 000 JSON: по 5000 на область, внутри каждой 2500 позитивной стратегии и по 1250 двух негативных стратегий. Недобора нет. Из-за одновременного изменения нескольких условий реальные частоты таргетов отличаются от квот стратегий: см. [частоты](actual_target_counts.json). Для 25 аугментаций укладка позвоночника неопределённа, для 111 изображений правого бедра не определена переносимая метка ротации; они не записаны как норма. Геометрические таргеты этих изображений используются независимо от неизвестного бинарного флага.',
           '', 'Угол сначала компенсируется в физических координатах; рабочая конфигурация набора: позитивы [-4;4], негативы [-10;-6] ∪ [6;10]. Эти негативы лежат внутри ранее оговорённых диапазонов до ±15°. Используется отражение позвоночника с обменом L/R. ROI левого бедра продлён влево, правого — вправо. Маски и частичные контуры малого вертела сохраняются при обрезании; все зависящие от преобразования флаги пересчитываются.',
           '', 'Проверены читаемость, конечность пикселей, размеры, допустимость геометрии, согласованность CSV/JSON/private DICOM и уникальность SOP UID у всех 15 000 файлов. Все условия бедра пересчитаны; дорогие яркостные условия позвоночника повторно проверены в стратифицированной выборке из 90 примеров. Ошибок не найдено. Это автоматическая проверка; подвздошные кости после crop/поворота ещё требуют ручного просмотра. [Полный аудит](../augmented_15000_final/audit_report.json).',
           '', 'Генерация была восстановлена после ошибки записи промежуточного CSV. Итоговый набор полный; первые 5000 файлов сохранены, незарегистрированные промежуточные файлы вынесены в backups/incomplete_generation_checkpoint. Поле seconds в generation_report относится только к восстановленному этапу: 196,7 с для бёдер; этап позвоночника занимал около 657 с. Старый незавершённый outputs/augmented_15000 не используется.',
           '', '## 3. Обучение','',
           'GPU: RTX 5060 Ti, CUDA, torch 2.12.1+cu130. Четыре компонента, по 5 эпох, 256×256, batch 8. Маршрутизатор — замороженный ResNet18; две пространственные модели — ResNet18 с U-Net декодером; артефакты — Faster R-CNN MobileNetV3 FPN. Правое бедро отражается для общей модели бедра. Потери: Dice+BCE, heatmaps/координаты/видимость, регрессия ROI, стандартные потери детектора.',
           '', 'Обучение: 399 оригиналов и 11 446 производных из 81 исследования. Валидация: 100 оригиналов из других 19 исследований. Study overlap = 0. Из 499 оригиналов 263 уникальных пиксельных массива; ни одна группа точных дублей не пересекает train/validation. Метрики ниже считаются на всех 100 файлах, поэтому повторные сканы имеют повторный вес; bootstrap группируется по исследованию. [Аудит дублей](pixel_duplicate_audit.json).',
           '', '| Компонент | Train / validation | Время 5 эпох, с | Геометрические метрики последней эпохи |','|---|---:|---:|---|']
    for t in training['tasks']:
        m=t['history'][-1]['validation_metrics']
        if t['task']=='router':detail=f"Accuracy {m.get('accuracy',m.get('router_accuracy',0)):.3f}"
        elif t['task']=='spine':detail=f"Dice линий {m['pixel_dice']:.3f}; IoU {m['pixel_iou']:.3f}; ошибка точек {m['mean_point_error_nominal_mm']:.1f} мм"
        elif t['task']=='hip':detail=f"Dice вертела {m['pixel_dice']:.3f}; IoU {m['pixel_iou']:.3f}; ROI IoU {m['mean_roi_iou']:.3f}; точки {m['mean_point_error_nominal_mm']:.1f} мм"
        else:detail=f"Box recall IoU≥0.5 {m['box_recall_iou50']:.3f}; precision {m['box_precision_iou50']:.3f}"
        lines.append(f"|{t['task']}|{t['train_images']} / {t['validation_images']}|{t['seconds']:.1f}|{detail}|")
    lines+=['', 'Показана последняя, заранее заданная пятая эпоха; лучший чекпойнт по test не выбирался. Сумма успешных этапов обучения около 10 минут, без учёта чтения набора и неудачного первого запуска детектора. Детектор повторно обучен после исправления NaN при пустых предложениях на негативном кадре; RPN в таком случае продолжает получать supervision. [Кривые обучения](learning_curves.png).',
            '', '## 4. Инференс и метрики на исходных данных','',
            f"Успешно обработано {evaluation['processed_files']}/{evaluation['validation_originals']} файлов. Macro-F1 маршрутизации {evaluation['router_macro_f1']:.3f}. Среднее время {evaluation['seconds_per_file_mean']:.3f} с; p95 {evaluation['seconds_per_file_p95']:.3f} с. Измерено на прогретых моделях с чтением DICOM и геометрией; загрузка весов исключена.",
            '', 'Класс 1 — нарушение. Неопределённые ответы исключены из метрик конкретной задачи и явно показаны через покрытие; они не засчитываются как правильные. N — число полученных определённых ответов. 95% интервалы — percentile bootstrap по исходным исследованиям, 300 повторов. Для класса, отсутствующего в выборке, ROC AUC не определён.', '']+table(evaluation['metrics'])
    overall=evaluation['overall_any_violation']
    lines+=['', f"Для любого нарушения на {overall['n']} полностью определённых файлах (покрытие {evaluation['overall_any_violation_coverage']:.0%}): F1 {formatted(overall,'f1')}, ROC AUC {formatted(overall,'roc_auc')}. Специфичность {overall['metrics']['specificity']:.3f}: много ложных тревог. Среднее F1 положительного класса по девяти парам область/задача {evaluation['quality_macro_f1']:.3f}.",
            '', '[Полные метрики](evaluation/report.json): balanced accuracy, PR AUC (average precision), матрицы ошибок, интервалы и число допустимых bootstrap-повторов. Порог ротации подобран только по 80 train-оригиналам; инференс автоматически читает evaluation/calibration.json. Остальные пороги фиксированы. Ранжирующие оценки правил не являются откалиброванными клиническими вероятностями.',
            '', '## 5. Отдельная синтетическая проверка','',
            f"135 изображений: по 15 для каждой стратегии каждой области, из {synthetic['eligible_heldout_augmentations']} производных только validation-источников. Ни одно не использовалось в обучении. Метрики здесь проверяют устойчивость к заданным преобразованиям, отдельно от исходной выборки; 200 bootstrap-повторов.",'']+table(synthetic['metrics'])
    lines+=['', '## 6. Выводы и ограничения','',
            '- ROI и маска вертела уже работают существенно лучше точек. F1=1 для ROI на оригиналах основан на небольшой выборке и широких интервалах; это не доказательство идеальной модели.',
            '- Позиционирование бедра требует доработки heatmap-декодирования и порогов на отдельной внутренней validation-части: на качественных оригиналах почти всегда возникает ложное нарушение. Высокий AUC на синтетике не исправляет плохой рабочий порог.',
            '- Линии позвоночника имеют низкий Dice. Ось распознаётся на синтетических поворотах, но реальное нарушение оси не выявлено; требуется больше исходных негативов и анализ отличий автора/геометрии. Порог не менялся по результатам этой test-части.',
            '- Артефакты: хороший F1 наличия на оригиналах, но слабый recall рамок и резкое ухудшение на преобразованных снимках. Нельзя подменять оценку локализации бинарной метрикой.',
            '- Ротация правого бедра слабая. Нулевая/малая предсказанная площадь и исходная анатомическая ротация ещё требуют проверки соответствия. Номинальный размер пикселя не подтверждает масштаб ранее увеличенных сканов.',
            '- Автоматические контуры, остаточная полнота разметки костей, малый размер test и повторные пиксельные массивы ограничивают выводы. Это рабочий эксперимент, а не готовая клиническая система.',
            '', '## 7. Тяжёлый вариант и последующая доразметка','',
            '[Код и серверный запуск](../../geometry_ml/HEAVY_PIPELINE.md): ResNet50 + U-Net для обеих пространственных моделей; Faster R-CNN ResNet50 FPN v2 для артефактов. Переключатель --architecture heavy. На GPU проверены forward/backward с конечными значениями; полноценно этот вариант пока не обучался. [Проверка](../heavy_model_verification.json).',
            '', 'Для доразметки использовать отдельную папку с labels.csv и --augmented-annotations-root. Конфигурация labeler: labeler/.env.augmented.example. При сравнении сохранить fold 0, seed 42, архитектуру, число эпох и неизменные тестовые исследования. Текущие JSON и чекпойнты зафиксированы хешами в [provenance.json](provenance.json). Число аугментаций не равно числу независимых исследований.',
            '', '[Ноутбук](../../DXA_augmented_pipeline.ipynb) показывает отчёты и кривые и содержит команды нового запуска. [Исполненный ноутбук](experiment_executed.ipynb) содержит готовые выводы. 38 автоматических тестов прошли; полный генератор и обучение проверены на реальных данных.']
    lines+=['', '## Иллюстрации','', 'Кривые:', '', '![Кривые обучения](learning_curves.png)',
            '', 'Бедро: изображение / истинная маска / предсказанная маска (один пример):', '', '![Малый вертел](hip_preview.png)',
            '', 'Позвоночник: изображение / истинные линии / предсказанная маска линий:', '', '![Линии](spine_preview.png)']
    markdown='\n'.join(lines)
    (run/'REPORT.md').write_text(markdown,encoding='utf-8')
    body=mistune.create_markdown(plugins=['table'])(markdown)
    (run/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>DXA: результаты</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:35px auto;padding:20px;background:#111a23;color:#e9f0f7;line-height:1.5}'
        'a{color:#91caff}table{border-collapse:collapse;font-size:14px;width:100%;display:block;overflow:auto}'
        'td,th{padding:8px;border:1px solid #405368}img{max-width:100%}h2{margin-top:35px}code{color:#ffda8a}</style>'+body+'</html>',encoding='utf-8')
    print(run/'REPORT.md')


if __name__=='__main__':main()
