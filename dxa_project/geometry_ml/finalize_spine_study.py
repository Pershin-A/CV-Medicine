"""Add existing-label checks, study CIs, readable source IDs and conclusions."""
import csv,json
from pathlib import Path
import numpy as np
from .spine_penalty_study import DEST,ROOT
from .evaluate import with_ci
from .data import load_records,load_augmented_records
from .report_reviewed import table,n

def run():
    path=DEST/'report.json';report=json.loads(path.read_text(encoding='utf-8'))
    with (ROOT/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        flags={r['relative_path']:r.get('spine_axis','') for r in csv.DictReader(f)}
    with (ROOT/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:
        numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(csv.DictReader(f),1)}
    with (ROOT/'dxa_project/outputs/augmented_15000_20260929/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        flags.update({'aug/'+r['image_path']:r.get('spine_axis','') for r in csv.DictReader(f)})
    originals=load_records(ROOT)
    records={r.relative_path:r for r in originals+load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',originals)}
    tables=[]
    for name,result in report['axes'].items():
        rows=result['details']
        for r in rows:
            value=flags.get(r['relative_path'],'');r['truth_flag']=int(float(value)) if value not in ('','None',None) else None
            record=records[r['relative_path']];r['source_number']=numbers.get(record.source_id or record.relative_path)
            r['augmented']=r['relative_path'].startswith('aug/')
        for field,angle_key in [('binary_vs_existing_labels','angle'),('previous_axis_vs_existing_labels','previous_angle')]:
            result[field]=with_ci([{'study':r['study'],'truth':r['truth_flag'],'prediction':int(abs(r[angle_key])>5),'score':abs(r[angle_key])} for r in rows if r['truth_flag'] is not None and r[angle_key] is not None],200)
        result['by_scope']={}
        for augmented,label in [(False,'originals'),(True,'augmentations')]:
            result['by_scope'][label]={field:with_ci([{'study':r['study'],'truth':r['truth_flag'],'prediction':int(abs(r[key])>5),'score':abs(r[key])} for r in rows if r['augmented']==augmented and r['truth_flag'] is not None and r[key] is not None],200) for field,key in [('new','angle'),('old','previous_angle')]}
        old=result['previous_axis_vs_existing_labels'];new=result['binary_vs_existing_labels']
        tables.append([name,old['n'],n((old.get('metrics') or {}).get('f1')),n((old.get('metrics') or {}).get('roc_auc')),new['n'],n((new.get('metrics') or {}).get('f1')),n((new.get('metrics') or {}).get('roc_auc'))])
    # Evaluate actual physical axis lengths, rather than training surrogates.
    manual={r['relative_path']:r for r in report['axes']['manual']['details']}
    prior_rows=[]
    for name,v in report['axes'].items():
        complete=[];violations=[]
        for r in v['details']:
            reference=manual[r['relative_path']];lengths=[a for a in reference['axis_lengths_mm'] if a is not None]
            record=records[r['relative_path']];geometry=json.loads(record.geometry_path.read_text(encoding='utf-8'))
            height=(geometry['image_height']-1)*record.spacing_mm[0]
            if not lengths or not r['axis_lengths_mm'] or any(a is None for a in r['axis_lengths_mm']) or r['top_length_mm'] is None or r['bottom_length_mm'] is None:continue
            minimum=height/(height/np.mean(lengths)+2)
            ok=min(r['axis_lengths_mm'])>=minimum and r['top_length_mm']<=height/6 and r['bottom_length_mm']<=height/6
            complete.append(ok);violations.append({'relative_path':r['relative_path'],'minimum_mm':float(minimum),'all_priors_passed':bool(ok)})
        v['physical_prior_check_c2_h6']={'evaluated':len(complete),'passed':int(sum(complete)),'details':violations}
        prior_rows.append([name,len(complete),sum(complete)])
    rng=np.random.default_rng(42)
    for name,result in report['results'].items():
        rows=result['details'];studies=sorted({r['study'] for r in rows});groups={s:[r for r in rows if r['study']==s] for s in studies}
        draws=[]
        for _ in range(300):
            sample=[r for i in rng.integers(0,len(studies),len(studies)) for r in groups[studies[i]]]
            draws.append([np.mean([r['under']>0 for r in sample]),np.mean([r['recall_5mm'] for r in sample])])
        intervals=np.percentile(draws,[2.5,97.5],axis=0)
        result['test_ci95_by_study']={'under_fraction':intervals[:,0].tolist(),'recall_5mm':intervals[:,1].tolist()}
    report['conclusion']='Count penalties reduce shortage but do not solve localization; none of the new frozen-feature heads is ready for production. Adaptive c=2 is softer than c=1. Validate full spatial training and decoder changes next.'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page=ROOT/'dxa_project/team_demo/spine_penalty_study_20260929.html';text=page.read_text(encoding='utf-8')
    extra='<h2>Проверка против существующих меток и физических длин</h2><p>Интервалы оценены по исходным исследованиям; аугментации не считаются независимыми пациентами. Метки аугментаций сформированы старым определением оси. Сравнение с новым правилом требует пересчёта геометрических меток, не подгонки порога.</p>'+table(['Вариант','Старая ось: N','F1','AUC','Новая ось: N','F1','AUC'],tables)+table(['Вариант','Физические длины проверены','Все приоры c=2 и H/6 соблюдены'],prior_rows)
    extra+='<p>Новый отрезок от границы до границы меняет определение угла. Не использовать старые бинарные аугментационные метки как безусловный эталон новой геометрии. Ограничения на длину мягкие: статистика показывает, что они не гарантируют соблюдения всех условий.</p>'
    rows=[]
    for k,v in report['results'].items():
        c=v['test_ci95_by_study'];rows.append([k,n(v['test']['under_fraction']),f"{c['under_fraction'][0]:.3f}–{c['under_fraction'][1]:.3f}",n(v['test']['recall_5mm']),f"{c['recall_5mm'][0]:.3f}–{c['recall_5mm'][1]:.3f}"])
    extra+='<h2>95% интервалы по исследованиям</h2>'+table(['Вариант','Недобор','95% интервал','Полнота ≤5 мм','95% интервал'],rows)
    extra+='<h2>Вывод исследования</h2><p>Приоры идеальной укладки не следует жёстко навязывать негативным примерам: локализатор должен находить существующие структуры, даже если укладка нарушена. Предел H/6 конфликтует с верхними высотами 232/873 и нижними 299/873 обучающих эталонов. Внутренний флаг построения контура не доказывает его правильность.</p><p>Штрафы уменьшают недобор, но не решают локализацию. Новые регрессионные головы на замороженных признаках часто ошибаются на десятки миллиметров. В общий пайплайн их не переносим. Следующий этап — применить дополнительные потери при обучении пространственных карт/канонических точек с разморозкой декодера, сохранить координатную точность и отдельно сравнить декодирование. Требовать одновременно достаточное число линий, их правильное положение и надёжные контуры.</p><p><a href="../../DXA_Spine_Penalties_20260929.ipynb">Отдельный ноутбук</a> · <a href="../HYBRID_PIPELINE_PLAN_20260929.md">План гибрида</a></p>'
    begin=text.find('<h2>Проверка против существующих меток и физических длин</h2>')
    end=text.find('<h2>Данные и конфликт')
    if begin>=0 and end>begin:text=text[:begin]+text[end:]
    text=text.replace('<h2>Данные и конфликт',extra+'<h2>Данные и конфликт',1)
    # Label demonstration sources explicitly; IDs correspond across all three rows.
    names=['manual','previous_heavy_map',report['winner_validation']]
    for name in names:
        selections=report['axes'][name]['details'][:5]+report['axes'][name]['details'][34:39]
        for r in selections:
            basename=r['relative_path'].split('/')[-1]
            token=f'{name} · {basename} ·'
            label=f"{name} · исходный №{r['source_number']} · "+('аугментация '+basename if r['augmented'] else 'оригинал')+' ·'
            text=text.replace(token,label,1)
    page.write_text(text,encoding='utf-8')
    import nbformat
    from nbclient import NotebookClient
    notebook=ROOT/'DXA_Spine_Penalties_20260929.ipynb';nb=nbformat.read(notebook,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code' and 'display(HTML' in cell.source:
            cell.source="display(HTML((ROOT/'dxa_project/team_demo/spine_penalty_study_20260929.html').read_text(encoding='utf-8').replace('src=\"spine_penalty_assets/', 'src=\"dxa_project/team_demo/spine_penalty_assets/')))"
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute();nbformat.write(nb,notebook)
    print(json.dumps({'winner':report['winner_validation'],'minutes':report['seconds']/60,'prior':report['prior_audit'],'results':{k:v['test'] for k,v in report['results'].items()},'axes':{k:{kk:vv for kk,vv in v.items() if kk in ['n','defined','mean_angle_error_manual_deg','reliable_terminal_fits']} for k,v in report['axes'].items()}},ensure_ascii=False))

if __name__=='__main__':run()
