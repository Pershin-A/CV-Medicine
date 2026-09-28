"""Write the reproducible DXA geometry training notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build(destination: Path):
    notebook = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# DXA: изображение → геометрия → контроль качества

Используются **только исходные DICOM и завершённая ручная разметка** из `Размеченные/`. Синтетические изображения в этом запуске не используются. Все фолды разделяются по исследованию. Короткий прогон проверяет код, а не качество моделей; для реальной оценки обучите несколько эпох и сравните фолды.\n\nМаршрутизатор видит исходные стороны без отражения. В общей ветви бедра правые снимки отражаются до левой ориентации. Физический размер пикселя в DICOM отсутствует: для ROI и угла позвоночника по умолчанию используются заданные 1,05 мм/Y и 0,6 мм/X. Порог площади вертела пока предварительный, его нужно калибровать только на обучающих исследованиях после исправления меток."""),
        nbf.v4.new_code_cell("""from pathlib import Path
import json, subprocess, sys
import pandas as pd
import torch
from IPython.display import display
from PIL import Image

ROOT = Path.cwd().resolve()
if not (ROOT / 'Размеченные').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'Размеченные' / 'labels.csv').is_file()
print('Проект:', ROOT)
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))"""),
        nbf.v4.new_markdown_cell("## 1. Проверка разметки до обучения"),
        nbf.v4.new_code_cell("""sys.path.insert(0, str(ROOT))
from dxa_project.audit_geometry_targets import audit
from dxa_project.geometry_ml.data import DxaDataset, load_records
from dxa_project.geometry_ml.train import split_records

print(json.dumps(audit(ROOT), ensure_ascii=False, indent=2))
records = load_records(ROOT)
train_records, valid_records = split_records(records, fold=0)
assert not ({r.study for r in train_records} & {r.study for r in valid_records})
print('Снимков:', len(records), 'обучение:', len(train_records),
      'проверка:', len(valid_records))

for region in ('SPINE', 'LEG_LEFT', 'LEG_RIGHT'):
    record = next(r for r in records if r.region == region)
    image, target = DxaDataset([record], size=256)[0]
    print(region, 'линии:', int(target['line'].sum()),
          'точки:', int(target['hip_present'].sum()),
          'ROI:', bool(target['roi_present']),
          'пиксели вертела:', int(target['trochanter'].sum()),
          'рамки:', len(target['artifact_boxes']))
right = next(r for r in records if r.region == 'LEG_RIGHT')
assert DxaDataset([right], size=224, router_mode=True)[0][1]['flipped'] is False
assert DxaDataset([right], size=256)[0][1]['flipped'] is True"""),
        nbf.v4.new_markdown_cell("## 2. Короткий проверочный запуск всех моделей"),
        nbf.v4.new_code_cell("""SMOKE_DIR = ROOT / 'dxa_project' / 'outputs' / 'geometry_ml_notebook_smoke'
command = [sys.executable, '-m', 'dxa_project.geometry_ml.train',
           '--task', 'all', '--smoke', '--size', '128',
           '--output', str(SMOKE_DIR)]
print(' '.join(command))
subprocess.run(command, cwd=ROOT, check=True)
smoke_report = json.loads((SMOKE_DIR / 'report.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{**{k: v for k, v in task.items() if k not in ('history','preview')},
                       **task['history'][-1]['validation_metrics']}
                      for task in smoke_report['tasks']]))"""),
        nbf.v4.new_markdown_cell("""## 3. Полное обучение по одному фолду

Поставьте `RUN_FULL_TRAINING = True`, чтобы обучить все четыре ветви на всех подходящих оригинальных снимках. Когда полный набор аугментаций будет готов, укажите его папку в `AUGMENTED_ROOT`: синтетические снимки будут добавлены только к train, а проверка останется на исходных DICOM других исследований. Начните с 5 эпох, затем увеличьте число по графикам обучения и проверки. Повторите для пяти фолдов. На данном компьютере доступна RTX 5060 Ti. Не используйте итоговые метрики короткого прогона как оценку качества."""),
        nbf.v4.new_code_cell("""RUN_FULL_TRAINING = False
EPOCHS = 5
FOLD = 0
AUGMENTED_ROOT = None  # папка с результатом generate.py после полной генерации
TRAIN_DIR = ROOT / 'dxa_project' / 'outputs' / f'geometry_ml_fold_{FOLD}'
if RUN_FULL_TRAINING:
    command = [sys.executable, '-m', 'dxa_project.geometry_ml.train',
               '--task', 'all', '--epochs', str(EPOCHS), '--fold', str(FOLD),
               '--size', '256', '--batch-size', '4', '--output', str(TRAIN_DIR)]
    if AUGMENTED_ROOT is not None:
        command += ['--augmented-root', str(AUGMENTED_ROOT)]
    subprocess.run(command, cwd=ROOT, check=True)
CHECKPOINT_DIR = TRAIN_DIR if (TRAIN_DIR / 'report.json').exists() else SMOKE_DIR
report = json.loads((CHECKPOINT_DIR / 'report.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{**{k: v for k, v in task.items() if k not in ('history','preview')},
                       **task['history'][-1]['validation_metrics']}
                      for task in report['tasks']]))"""),
        nbf.v4.new_markdown_cell("## 4. Визуальная проверка и сквозной инференс"),
        nbf.v4.new_code_cell("""for name in ('spine', 'hip', 'artifact'):
    path = CHECKPOINT_DIR / f'{name}_preview.png'
    if path.is_file():
        print(name, '— исходный снимок | разметка | прогноз')
        display(Image.open(path))

from dxa_project.geometry_ml.predict import predict_file
for region in ('SPINE', 'LEG_LEFT', 'LEG_RIGHT'):
    record = next(r for r in valid_records if r.region == region)
    result = predict_file(record.source_path, CHECKPOINT_DIR, force_region=region)
    print(region, record.relative_path, result['quality_flags'])
    if region == 'SPINE':
        print('линий:', len(result['geometry']['spine']['disc_lines']),
              'угол:', result['spine_axis_angle_deg'])
    else:
        print('ROI, мм:', result['roi_margins_mm'],
              'площадь вертела, пикселей²:', result['lesser_trochanter_area_px2'])"""),
        nbf.v4.new_markdown_cell("""## Что считать успешной проверкой

- Нет пересечения исследований между обучением и проверкой.
- Для всех ветвей конечные значения потерь и градиентов; модель выдаёт ожидаемые размеры тензоров.
- На отдельной выборке улучшаются *геометрические* метрики: Dice для линий/маски, ошибка точек, IoU ROI, recall рамок. Одна эпоха этого не доказывает.
- Итоговые флаги сверяются с исправленными целевыми метками только после настройки геометрии и порогов на обучающих исследованиях.
- Сквозной инференс нужно проверять и с принудительно правильной ветвью, и с предсказанной маршрутизатором."""),
    ]
    notebook.cells = cells
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    destination.write_text(nbf.writes(notebook), encoding="utf-8")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "DXA_geometry_pipeline.ipynb")
