"""Create the reproducible augmentation experiment and report notebook."""
from pathlib import Path
import nbformat as nbf

def build():
    nb=nbf.v4.new_notebook()
    markdown=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    nb.cells=[markdown('''# DXA: обучение на аугментированных данных

15 000 производных изображений, четыре компонента пайплайна, фиксированный fold 0.
Валидация — только исходные исследования вне обучения. Текущий запуск: 5 эпох,
размер 256, batch 8, CUDA. Выполнение обучающей ячейки повторно запускает обучение;
для новой разметки задайте отдельный RUN и MANUAL_OVERLAY.
'''),code('''from pathlib import Path
import json, sys, subprocess
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if ROOT.name == 'dxa_project': ROOT = ROOT.parent
AUG = ROOT / 'dxa_project/outputs/augmented_15000_final'
RUN = ROOT / 'dxa_project/outputs/geometry_ml_augmented_5epochs'
ARCHITECTURE = 'light' # 'heavy' для серверного варианта
MANUAL_OVERLAY = None # папка с labels.csv после доразметки
def cli(*args):
    return subprocess.run([sys.executable, '-u', '-m', *map(str,args)], cwd=ROOT, check=True)
'''),markdown('## Отчёт генерации и проверка'),code('''generation = json.loads((AUG/'generation_report.json').read_text(encoding='utf-8'))
display(pd.DataFrame(generation['generated']).T)
audit = json.loads((AUG/'audit_report.json').read_text(encoding='utf-8'))
assert not audit['problems'], audit['problems'][:10]
print('Проверено файлов:', audit['files'], 'повторно вычислено таргетов:', audit['recomputed_examples'])
'''),markdown('''## Обучение (запускать намеренно)

Для сравнения после доразметки используйте отдельную выходную папку и тот же fold.
Сохранённые исходные JSON генерации не заменяются ручной версией.
'''),code('''# Раскомментируйте для нового запуска:
# args = ['dxa_project.geometry_ml.train', '--augmented-root', AUG,
#         '--output', RUN, '--architecture', ARCHITECTURE, '--epochs', '5',
#         '--size', '256', '--batch-size', '8', '--loader-workers', '2', '--fold', '0']
# if MANUAL_OVERLAY is not None: args += ['--augmented-annotations-root', MANUAL_OVERLAY]
# cli(*args)
'''),markdown('## Кривые и геометрические метрики'),code('''training = json.loads((RUN/'report.json').read_text(encoding='utf-8'))
rows=[]
fig, axs = plt.subplots(1,4,figsize=(16,3))
for ax, task in zip(axs, training['tasks']):
    history = pd.DataFrame(task['history'])
    ax.plot(history['epoch'], history['train_loss'], label='train')
    if task['task'] != 'artifact': ax.plot(history['epoch'], history['validation_loss_or_detection_count'], label='validation')
    ax.set_title(task['task']); ax.legend()
    rows.append({'task':task['task'],'train':task['train_images'],'validation':task['validation_images'],
                 'seconds':task['seconds'],**task['history'][-1]['validation_metrics']})
plt.tight_layout()
display(pd.DataFrame(rows))
'''),markdown('## Сквозные метрики и 95% интервалы'),code('''# Для нового запуска оценки:
# cli('dxa_project.geometry_ml.evaluate','--checkpoints',RUN,'--output',RUN/'evaluation','--fold','0')
evaluation=json.loads((RUN/'evaluation/report.json').read_text(encoding='utf-8'))
rows=[]
for result in evaluation['metrics']:
    row={k:result.get(k) for k in ['region','task','n','coverage']}
    row.update(result.get('metrics') or {})
    for name, interval in (result.get('ci95') or {}).items(): row[name+'_ci95']=interval
    rows.append(row)
display(pd.DataFrame(rows))
print('Файлы обработаны:',evaluation['file_processing_success_fraction'])
print('Среднее время:',evaluation['seconds_per_file_mean'])
print('Macro-F1 маршрутизации:',evaluation['router_macro_f1'])
print('Общее нарушение:',evaluation['overall_any_violation'])
'''),markdown('''## Ограничения

Это короткий внутренний эксперимент, не клиническая валидация. Размер пикселя
номинальный. Неопределённые ответы не засчитываются как правильные: смотрите
покрытие каждой задачи рядом с F1/ROC AUC. Для гипотезы об улучшении после
доразметки необходимо повторить тот же протокол и сохранить тестовые исследования.
Серверный вариант описан в `geometry_ml/HEAVY_PIPELINE.md`.
''')]
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    out=Path(__file__).resolve().parents[1]/'DXA_augmented_pipeline.ipynb'
    out.write_text(nbf.writes(nb),encoding='utf-8');print(out)

if __name__=='__main__':build()
