"""Generate the separate, executable notebook for improvement variants."""
from pathlib import Path
import nbformat as nbf

def main():
    m=nbf.v4.new_markdown_cell;c=nbf.v4.new_code_cell
    nb=nbf.v4.new_notebook(cells=[m('''# DXA: проверка вариантов улучшения

Этот ноутбук отделяет проверку исполняемости от оценки качества. Варианты:
декодеры координат, отдельная общая / три независимые модели точек,
heatmap / координатный loss, 256/384/512, полный кадр / ROI с контекстом,
постоянный / cosine / plateau learning rate.
Длительное обучение выключено по умолчанию. Предыдущий fold 0 уже просмотрен;
это сравнительный benchmark, а не новый независимый test.
'''),c('''from pathlib import Path
import json,sys,subprocess
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd()
if ROOT.name=='dxa_project': ROOT=ROOT.parent
OUT=ROOT/'dxa_project/outputs/improvements_v2'
from dxa_project.geometry_ml.experiments import VARIANTS
def cli(*args):
    return subprocess.run([sys.executable,'-u','-m',*map(str,args)],cwd=ROOT,check=True)
'''),m('## Разделение для выбора настроек'),c('''protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
display(pd.DataFrame(protocol['summary']).T)
assert protocol['group_overlap']==0
assert protocol['cross_partition_exact_duplicates']==0
print(protocol['outer_scope'])
'''),m('## Координаты на текущих весах: пять декодеров'),c('''# cli('dxa_project.geometry_ml.experiments','--phase','decode','--output',OUT)
results=json.loads((OUT/'decoder_comparison.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'decoder':k,'error_mm':v['mean_error_mm'],'PCK_10mm':v['pck_10mm'],
                     'specificity':v['position']['metrics']['specificity']} for k,v in results.items()]))
print('На исходных 65 бедрах нет истинных нарушений позиционирования: sensitivity/AUC здесь не определены.')
'''),m('## Варианты архитектур, loss и разрешения'),c('''display(pd.DataFrame(VARIANTS).T)
# Каждый вариант был проверен на реальных данных: два обучающих батча,
# обратное распространение, validation, сохранение чекпойнта.
smoke=[]
for name in VARIANTS:
    p=OUT/'smoke'/name/'report.json'
    if p.exists():
        r=json.loads(p.read_text(encoding='utf-8'))
        smoke.append({'variant':name,'smoke':r['smoke'],'seconds':r['seconds'],'epochs':r['actual_epochs']})
display(pd.DataFrame(smoke))
assert len(smoke)==len(VARIANTS), 'Сначала выполните следующую проверочную ячейку'
integration=OUT/'point_integration_smoke.json'
if integration.exists():
    values=json.loads(integration.read_text(encoding='utf-8'))
    display(pd.DataFrame(values))
    assert len(values)==4 and all(x['passed'] for x in values)
    print('Полный инференс crop/independent проверен на CPU для обеих сторон; это проверка интеграции на smoke-весах, не качества.')
'''),c('''RERUN_SMOKE=False
if RERUN_SMOKE:
    cli('dxa_project.geometry_ml.experiments','--phase','smoke',
        '--variants',','.join(VARIANTS),'--output',OUT)
'''),m('''## Проверка forward/backward на разных scheduler

Здесь один синтетический шаг оптимизатора проверяет расписания; это не оценка модели.
'''),c('''import torch
for name in ['constant','cosine','plateau']:
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=2e-4)
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,5) if name=='cosine'
               else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=1) if name=='plateau' else None)
    loss=parameter.square().sum();loss.backward();optimizer.step()
    if name=='cosine':scheduler.step()
    if name=='plateau':scheduler.step(float(loss.detach()))
    assert torch.isfinite(parameter).all()
    print(name,optimizer.param_groups[0]['lr'])
'''),m('''## Длительное обучение: запускать варианты по одному

Для честного сравнения меняйте один фактор. Новые LR-варианты сохраняйте в
отдельный OUTPUT, чтобы не заменять предыдущие веса. Test не влияет на выбор.
'''),c('''RUN_LONG=False
VARIANT='shared256'
ENCODER_LR=2e-5
HEAD_LR=2e-4
SCHEDULER='cosine' # none / cosine / plateau
EPOCHS=20
WARMUP_EPOCHS=0 # например 2, только для cosine
OUTPUT=OUT
if RUN_LONG:
    cli('dxa_project.geometry_ml.experiments','--phase','train','--variants',VARIANT,
        '--epochs',EPOCHS,'--encoder-lr',ENCODER_LR,'--head-lr',HEAD_LR,
        '--scheduler',SCHEDULER,'--warmup-epochs',WARMUP_EPOCHS,'--output',OUTPUT)
'''),m('''## Остальные модули полного пайплайна

Выполняйте этот блок после создания protocol.json. Он обучает позвоночник,
ROI/маску бедра и детектор артефактов отдельно; готовые точки подключаются
на инференсе. Маршрутизатор текущего эксперимента сохранён из прежнего запуска.
'''),c('''RETRAIN_MODULES=False
if RETRAIN_MODULES:
    for task in ['spine','hip','artifact']:
        cli('dxa_project.geometry_ml.train','--task',task,'--epochs',20,'--size',256,'--batch-size',8,
            '--encoder-lr',1e-5 if task=='artifact' else 2e-5,
            '--head-lr',3e-5 if task=='artifact' else 1e-4,'--scheduler','cosine','--patience',5,
            '--loader-workers',2,'--balance-sources','--selection-protocol',OUT/'protocol.json',
            '--augmented-root',ROOT/'dxa_project/outputs/augmented_15000_final',
            '--output',OUT/'modules'/task)
'''),m('## Готовые кривые и результаты длительного запуска'),c('''trained=[]
for folder in sorted((OUT/'trained').glob('*')):
    p=folder/'report.json'
    if not p.exists():continue
    report=json.loads(p.read_text(encoding='utf-8'));h=pd.DataFrame(report['history'])
    plt.plot(h['epoch'],[x['mean_error_mm'] for x in h['metrics']],label=folder.name)
    trained.append({'variant':folder.name,'epochs':report['actual_epochs'],'seconds':report['seconds'],
                    'best_inner_error_mm':report['best_inner_validation_error_mm']})
if trained:
    plt.xlabel('Эпоха');plt.ylabel('Ошибка точек, мм');plt.legend();plt.grid();plt.show()
display(pd.DataFrame(trained))
test=OUT/'test_comparison.json'
if test.exists():
    r=json.loads(test.read_text(encoding='utf-8'))
    display(pd.DataFrame([{'variant':k,'error_mm':v['mean_error_mm'],'PCK_10mm':v['pck_10mm'],
                          'specificity':v['position']['metrics']['specificity'],'crop_scope':v['crop_scope']} for k,v in r.items()]))
synthetic=[]
for variant,folder in [('shared256',OUT/'pipeline'),('coordinate256',OUT/'coordinate_pipeline')]:
    p=folder/'synthetic_evaluation/report.json'
    if p.exists():
        for metric in json.loads(p.read_text(encoding='utf-8'))['metrics']:
            if metric['task']=='hip_position':
                values=metric.get('metrics') or {}
                synthetic.append({'variant':variant,'region':metric['region'],'F1':values.get('f1'),
                                  'sensitivity':values.get('sensitivity'),'specificity':values.get('specificity'),
                                  'AUC':values.get('roc_auc'),'coverage':metric['coverage']})
display(pd.DataFrame(synthetic))
'''),m('## Полный пайплайн: результаты на тех же контрольных источниках'),c('''evaluation=OUT/'pipeline/evaluation/report.json'
if evaluation.exists():
    current=json.loads(evaluation.read_text(encoding='utf-8'))
    previous=json.loads((ROOT/'dxa_project/outputs/geometry_ml_augmented_5epochs/evaluation/report.json').read_text(encoding='utf-8'))
    before={(x['region'],x['task']):x for x in previous['metrics']}
    rows=[]
    for x in current['metrics']:
        old=before[(x['region'],x['task'])]
        rows.append({'region':x['region'],'task':x['task'],'old_F1':(old.get('metrics') or {}).get('f1'),
                     'new_F1':(x.get('metrics') or {}).get('f1'),'new_AUC':(x.get('metrics') or {}).get('roc_auc'),
                     'coverage':x['coverage'],'F1_CI95':x.get('ci95',{}).get('f1')})
    display(pd.DataFrame(rows))
    print('Пропуски не считаются правильными. F1 сравнивайте с coverage. Score правил не калиброван как клиническая вероятность.')
'''),m('## Геометрия: проверка правдоподобия вместо замены видимости'),c('''from dxa_project.geometry_ml.landmark_geometry import distance_features,plausibility
from dxa_project.geometry_ml.data import LANDMARKS
points=dict(zip(LANDMARKS,[[20.,20.],[40.,25.],[60.,60.]]))
features=distance_features(points,[1.05,.6])
assert features is not None and features['distance_12_mm']>0
display(pd.DataFrame([features]))
print(plausibility(features))
print(plausibility(None))
sidecar=OUT/'pipeline/landmark_geometry_bounds.json'
if sidecar.exists():
    bounds=json.loads(sidecar.read_text(encoding='utf-8'))
    print('Границы только из train:',bounds)
    print('Режим framing: только видимость и отступы; framing_and_geometry: дополнительное воздержание при неверной геометрии.')
comparison=OUT/'geometry_rule_comparison.json'
if comparison.exists():
    values=json.loads(comparison.read_text(encoding='utf-8'))
    display(pd.DataFrame([{'rule':key,'coverage':value['coverage'],'abstentions':value['abstentions'],
                          'specificity':(value.get('metrics') or {}).get('specificity')} for key,value in values.items()]))
'''),m('''## ROI и ограничения

Вариант roi384 обучается на истинной ROI с контекстом и при автономной оценке
является oracle-экспериментом. В полном инференсе используется предсказанная ROI;
при пропусках точек предусмотрен возврат к полному кадру. Нельзя выдавать oracle
результат за качество полного пайплайна.
Технический crop не меняет разметку ROI и её физические отступы.
Расстояния между точками выводятся как признаки правдоподобия, а не заменяют видимость.
Два батча проверяют только исполняемость, не позволяют выбрать лучшую архитектуру.
''')])
    nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'variants','exec')
    p=Path(__file__).resolve().parents[1]/'DXA_improvement_variants.ipynb'
    nbf.write(nb,p);print(p)

if __name__=='__main__':main()
