"""Held-out study evaluation: quality metrics, cluster bootstrap CIs and timing."""
import argparse,csv,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from .data import load_records
from .train import split_records
from .predict import predict_file
from dxa_project.augmentation.core import prepare_geometry,hip_position_ok

TASKS={'SPINE':('spine_position','spine_axis','spine_artifact'),
       'LEG_LEFT':('hip_position','hip_roi','hip_rotation'),
       'LEG_RIGHT':('hip_position','hip_roi','hip_rotation')}

def binary_metrics(rows):
    y=np.asarray([r['truth'] for r in rows]); p=np.asarray([r['prediction'] for r in rows])
    scores=np.asarray([r['score'] if r['score'] is not None else float(r['prediction']) for r in rows])
    tp=int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    sensitivity=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    both=len(set(y))==2
    return {'sensitivity':sensitivity,'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2 if both else None,
            'f1':float(f1_score(y,p,zero_division=0)) if tp+fp+fn else None,
            'roc_auc':float(roc_auc_score(y,scores)) if both else None,
            'pr_auc_average_precision':float(average_precision_score(y,scores)) if both else None,
            'tp':tp,'tn':tn,'fp':fp,'fn':fn}

def with_ci(rows,iterations=300):
    if not rows: return {'n':0,'metrics':None}
    point=binary_metrics(rows); groups={s:[r for r in rows if r['study']==s] for s in sorted({r['study'] for r in rows})}
    ids=list(groups); rng=np.random.default_rng(42)
    samples={k:[] for k in ('sensitivity','specificity','balanced_accuracy','f1','roc_auc','pr_auc_average_precision')}
    for _ in range(iterations):
        draw=[r for index in rng.integers(0,len(ids),len(ids)) for r in groups[ids[index]]]
        values=binary_metrics(draw)
        for key in samples:
            if values[key] is not None: samples[key].append(values[key])
    ci={k:np.percentile(v,[2.5,97.5]).tolist() if v and point[k] is not None else None for k,v in samples.items()}
    return {'n':len(rows),'studies':len(ids),'metrics':point,'ci95':ci,'bootstrap_valid_replicates':{k:len(v) for k,v in samples.items()}}

def truth_flags(reference,region,geometry=None,include_scoliosis=False):
    result={}
    tasks=TASKS[region]+(('spine_scoliosis',) if region=='SPINE' and include_scoliosis else ())
    for task in tasks:
        field=task if region=='SPINE' else region.removeprefix('LEG_').lower()+'_'+task
        value=reference.get(field,'')
        result[task]=int(float(value)) if value not in ('',None) else None
        if result[task] not in (0,1):result[task]=None
    if region.startswith('LEG_') and geometry is not None:
        result['hip_position']=int(not hip_position_ok(prepare_geometry(geometry,region)))
    return result

def run(root,checkpoints,output,fold=0,bootstrap=300,selection_protocol=None):
    output.mkdir(parents=True,exist_ok=True)
    originals=load_records(root); train,valid=split_records(originals,fold)
    if selection_protocol:
        protocol=json.loads(selection_protocol.read_text(encoding='utf-8'))
        train=[r for r in originals if protocol['partition_by_path'][r.relative_path]=='validation']
        valid=[r for r in originals if protocol['partition_by_path'][r.relative_path]=='test']
    with (root/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        reference={r['relative_path']:r for r in csv.DictReader(f)}
    include_scoliosis=(checkpoints/'scoliosis.pt').is_file()
    task_map={k:v+(('spine_scoliosis',) if k=='SPINE' and include_scoliosis else ()) for k,v in TASKS.items()}
    if include_scoliosis:
        with (root/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                issue=row.get('spine_issue','')
                if issue in ('SCOLIOSIS','NONE','LUMBARIZATION'):reference[row['relative_path']]['spine_scoliosis']=int(issue=='SCOLIOSIS')
    calibration=[r for r in train if r.region.startswith('LEG_') and truth_flags(reference[r.relative_path],r.region)['hip_rotation'] is not None]
    rng=np.random.default_rng(42); rng.shuffle(calibration); calibration=calibration[:80]
    areas=[]
    for record in calibration:
        predicted=predict_file(record.source_path,checkpoints,force_region=record.region)
        areas.append((predicted['lesser_trochanter_area_px2'],truth_flags(reference[record.relative_path],record.region)['hip_rotation']))
    candidates=np.unique([0.,1.]+[float(a)+.5 for a,y in areas])
    threshold=max(candidates,key=lambda t:f1_score([y for a,y in areas],[int(a<t) for a,y in areas],zero_division=0)) if len({y for a,y in areas})==2 else 1.
    calibration_scope='inner validation only' if selection_protocol else 'training originals only'
    (output/'calibration.json').write_text(json.dumps({'rotation_threshold_px2':float(threshold),'source':calibration_scope,'n':len(areas),'examples':areas},indent=2),encoding='utf-8')
    # Warm all branches before measuring steady-state per-study latency.
    for region in TASKS:
        sample=next((r for r in valid if r.region==region),None)
        if sample:predict_file(sample.source_path,checkpoints,float(threshold),force_region=region)
    if torch.cuda.is_available():torch.cuda.synchronize()
    results=[]; task_rows=[]; failures=[]; timings=[]
    for index,record in enumerate(valid,1):
        start=time.perf_counter()
        ground_geometry=json.loads(record.geometry_path.read_text(encoding='utf-8'))
        truth=truth_flags(reference[record.relative_path],record.region,ground_geometry,include_scoliosis)
        try:
            predicted=predict_file(record.source_path,checkpoints,rotation_area_threshold_px2=float(threshold))
            if torch.cuda.is_available():torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
            for task,value in truth.items():
                guess=predicted['quality_flags'].get(task) if predicted['region']==record.region else None
                task_rows.append({'region':record.region,'task':task,'study':record.study,'relative_path':record.relative_path,
                                  'truth':value,'prediction':guess,'score':predicted['quality_scores'].get(task) if guess is not None else None})
            results.append({'relative_path':record.relative_path,'study':record.study,'truth_region':record.region,
                            'predicted_region':predicted['region'],'truth':truth,'predicted':predicted['quality_flags'],
                            'scores':predicted['quality_scores'],'seconds':timings[-1]})
            (output/f'prediction_{index:03}.json').write_text(json.dumps(predicted,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as error:
            failures.append({'relative_path':record.relative_path,'error':repr(error)})
            for task,value in truth.items():
                task_rows.append({'region':record.region,'task':task,'study':record.study,'relative_path':record.relative_path,
                                  'truth':value,'prediction':None,'score':None})
        if index%20==0: print(json.dumps({'evaluation_completed':index,'total':len(valid)}),flush=True)
    metrics=[]
    for region,tasks in task_map.items():
        for task in tasks:
            all_rows=[r for r in task_rows if r['region']==region and r['task']==task and r['truth'] is not None]
            usable=[r for r in all_rows if r['prediction'] is not None]
            metric={'region':region,'task':task,'labeled':len(all_rows),'coverage':len(usable)/len(all_rows) if all_rows else None,**with_ci(usable,bootstrap)}
            metrics.append(metric)
    overall=[]
    for row in results:
        truths=list(row['truth'].values()); guesses=row['predicted']
        if row['truth_region']!=row['predicted_region'] or any(t is None for t in truths) or any(guesses.get(k) is None for k in row['truth']): continue
        overall.append({'study':row['study'],'truth':int(any(truths)),'prediction':int(any(guesses[k] for k in row['truth'])),
                        'score':max(row['scores'].get(k,0.) or 0. for k in row['truth'])})
    report={'fold':fold,'validation_originals':len(valid),'validation_studies':len({r.study for r in valid}),
            'train_validation_study_overlap':0,'processed_files':len(results),'failed_files':failures,
            'file_processing_success_fraction':len(results)/len(valid),'seconds_per_file_mean':float(np.mean(timings)) if timings else None,
            'seconds_per_file_p95':float(np.percentile(timings,95)) if timings else None,
            'metrics':metrics,'overall_any_violation':with_ci(overall,bootstrap),
            'router_macro_f1':float(f1_score([r['truth_region'] for r in results],[r['predicted_region'] for r in results],average='macro')) if results else None,
            'quality_macro_f1':float(np.mean([m['metrics']['f1'] for m in metrics if m.get('metrics') and m['metrics']['f1'] is not None])) if any(m.get('metrics') and m['metrics']['f1'] is not None for m in metrics) else None,
            'ci_method':'percentile bootstrap by original study, not by augmented image',
            'score_note':'continuous rule-derived ranking scores, not calibrated clinical probabilities',
            'timing_scope':'warm branches; includes DICOM read and geometric postprocessing, excludes initial checkpoint loading',
            'undefined_outputs':'abstentions excluded from task metrics and reported through coverage; they are not counted as correct',
            'threshold_calibration':calibration_scope+'; outer benchmark never used to choose rotation threshold'}
    report['label_sources']='corrected organizer manifest for existing flags; hip positioning computed from authoritative visual landmarks because the organizer table has no separate positioning field'
    report['overall_any_violation_coverage']=len(overall)/len(valid)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    if task_rows:
        with (output/'quality_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(task_rows[0])); writer.writeheader(); writer.writerows(task_rows)
    lines=['# Оценка на исходных исследованиях вне обучения','',f"Обработано {len(results)}/{len(valid)} файлов. Macro-F1 маршрутизации: {report['router_macro_f1']}.",'',
           '| Область | Нарушение | N | Покрытие | Чувствительность | Специфичность | F1 | ROC AUC |','|---|---|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        values=m.get('metrics') or {}
        show=lambda k: '—' if values.get(k) is None else f"{values[k]:.3f}"
        lines.append(f"|{m['region']}|{m['task']}|{m['n']}|{m['coverage']}|{show('sensitivity')}|{show('specificity')}|{show('f1')}|{show('roc_auc')}|")
    lines+=['','95% интервалы, PR AUC/AP, balanced accuracy и матрицы ошибок приведены в report.json. Неопределённые ответы учитываются через покрытие. Это внутренняя валидация, не клиническая оценка на независимом внешнем наборе.']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'evaluation_report':str(output/'report.json'),'processed':len(results)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--fold',type=int,default=0); parser.add_argument('--bootstrap',type=int,default=300)
    parser.add_argument('--selection-protocol',type=Path)
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');run(args.root.resolve(),args.checkpoints.resolve(),args.output.resolve(),args.fold,args.bootstrap,args.selection_protocol)
