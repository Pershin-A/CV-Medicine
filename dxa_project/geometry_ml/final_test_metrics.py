"""Spatial metrics in original coordinates for the final held-out predictions."""
import json,csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from .final_assembly import save


def box_iou(a,b):
    if a is None or b is None:return 0.
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda c:max(0,c[2]-c[0])*max(0,c[3]-c[1])
    intersection=x*y;return intersection/max(1e-9,area(a)+area(b)-intersection)


def aggregate(rows):
    result={}
    for field in ('dice','mask_iou','roi_iou'):
        values=[r[field] for r in rows if r.get(field) is not None]
        result[field]=float(np.mean(values)) if values else None
    visible=sum(r.get('visible',0) for r in rows);found=sum(r.get('found',0) for r in rows)
    distances=[d for r in rows for d in r.get('distances_mm',[])]
    result.update(point_error_mm=float(np.mean(distances)) if distances else None,
                  point_coverage=found/visible if visible else None,
                  pck_5mm=sum(r.get('pck5',0) for r in rows)/visible if visible else None,
                  pck_10mm=sum(r.get('pck10',0) for r in rows)/visible if visible else None)
    tp=sum(r.get('box_tp',0) for r in rows);fp=sum(r.get('box_fp',0) for r in rows);fn=sum(r.get('box_fn',0) for r in rows)
    result.update(box_precision_iou50=tp/(tp+fp) if tp+fp else None,
                  box_recall_iou50=tp/(tp+fn) if tp+fn else None,
                  box_f1_iou50=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return result


def bootstrap(rows):
    result=aggregate(rows);groups={s:[r for r in rows if r['study']==s] for s in sorted({r['study'] for r in rows})}
    ids=list(groups);rng=np.random.default_rng(42);samples={k:[] for k in result}
    for _ in range(300):
        m=aggregate([r for i in rng.integers(0,len(ids),len(ids)) for r in groups[ids[i]]])
        for k,v in m.items():
            if v is not None:samples[k].append(v)
    return {'images':len(rows),'studies':len(ids),'metrics':result,
            'ci95':{k:np.percentile(v,[2.5,97.5]).tolist() if v and result[k] is not None else None for k,v in samples.items()}}


def run(records,output):
    from .data import LANDMARKS,CRESTS
    from dxa_project.augmentation.core import prepare_geometry
    rows=[]
    for i,r in enumerate(records,1):
        gt=prepare_geometry(json.loads(r.geometry_path.read_text(encoding='utf-8')),r.region);path=output/f'prediction_{i:03}.json'
        pred=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        geometry=pred.get('geometry') if pred.get('region')==r.region else None
        row={'relative_path':r.relative_path,'study':r.study,'region':r.region,'visible':0,'found':0,'distances_mm':[],'pck5':0,'pck10':0}
        part='spine' if r.region=='SPINE' else 'hip';field='iliac_crests' if part=='spine' else 'landmarks'
        for name in CRESTS if part=='spine' else LANDMARKS:
            point=gt[part][field].get(name);guess=geometry[part][field].get(name) if geometry else None
            if point is None:continue
            row['visible']+=1
            if guess is None:continue
            delta=np.asarray(point)-np.asarray(guess);distance=float(np.hypot(delta[0]*r.spacing_mm[1],delta[1]*r.spacing_mm[0]))
            row['found']+=1;row['distances_mm'].append(distance);row['pck5']+=distance<=5;row['pck10']+=distance<=10
        if part=='hip':
            def pixels(g):return {(int(round(x)),int(round(y))) for x,y in g['hip'].get('lesser_trochanter_pixels',[])}
            target=pixels(gt);guess=pixels(geometry) if geometry else set();intersection=len(target&guess);union=len(target|guess)
            row['dice']=2*intersection/(len(target)+len(guess)) if target or guess else 1.
            row['mask_iou']=intersection/union if union else 1.
            row['empty_reference_mask']=not bool(target)
            row['roi_iou']=box_iou(gt['hip']['roi_box'],geometry['hip']['roi_box'] if geometry else None) if gt['hip']['roi_box'] else None
        else:
            target=[b['bbox'] for b in gt['spine'].get('foreign_objects',[])];guess=[b['bbox'] for b in geometry['spine'].get('foreign_objects',[])] if geometry else []
            overlaps=np.asarray([[box_iou(a,b) for b in guess] for a in target])
            hits=0
            if target and guess:
                a,b=linear_sum_assignment(-overlaps);hits=int((overlaps[a,b]>=.5).sum())
            row.update(box_tp=hits,box_fp=len(guess)-hits,box_fn=len(target)-hits)
        rows.append(row)
    report={region:bootstrap([r for r in rows if r['region']==region]) for region in ('SPINE','LEG_LEFT','LEG_RIGHT')}
    timing=json.loads((output/'predictions.json').read_text(encoding='utf-8'));studies={}
    for r in timing:studies[r['study']]=studies.get(r['study'],0.)+r['seconds']
    report['timing']={'processed_studies':len(studies),'mean_seconds_per_study':float(np.mean(list(studies.values()))),
                      'p95_seconds_per_study':float(np.percentile(list(studies.values()),95)),
                      'scope':'sum of sequential warm per-file prediction times within each study; read + forward + geometry; not API batch latency'}
    from .evaluate import binary_metrics,with_ci
    from sklearn.metrics import f1_score
    with (output/'quality_predictions.csv').open(encoding='utf-8-sig',newline='') as f:all_quality=list(csv.DictReader(f))
    quality=all_quality
    quality=[{**r,'truth':int(r['truth']),'prediction':int(r['prediction']),'score':float(r['score']) if r['score'] else None} for r in quality if r['truth'] and r['prediction']]
    def macro(ids):
        selected=[r for study in ids for r in quality if r['study']==study];tasks={}
        for r in selected:tasks.setdefault((r['region'],r['task']),[]).append(r)
        values=[binary_metrics(v)['f1'] for v in tasks.values()];values=[v for v in values if v is not None]
        types={}
        for r in selected:types.setdefault(r['task'],[]).append(r)
        type_values=[binary_metrics(v)['f1'] for v in types.values()];type_values=[v for v in type_values if v is not None]
        routed=[r for study in ids for r in timing if r['study']==study]
        return {'quality_macro_f1':float(np.mean(values)) if values else None,
                'target_macro_f1':float(np.mean(type_values)) if type_values else None,
                'router_macro_f1':float(f1_score([r['truth_region'] for r in routed],[r['predicted_region'] for r in routed],labels=['SPINE','LEG_LEFT','LEG_RIGHT'],average='macro',zero_division=0))}
    ids=sorted(studies);rng=np.random.default_rng(42);samples={k:[] for k in ('quality_macro_f1','target_macro_f1','router_macro_f1')}
    for _ in range(300):
        for k,v in macro([ids[i] for i in rng.integers(0,len(ids),len(ids))]).items():
            if v is not None:samples[k].append(v)
    report['macro']={'metrics':macro(ids),'ci95':{k:np.percentile(v,[2.5,97.5]).tolist() if v else None for k,v in samples.items()},
                     'note':'Quality macro averages defined positive-class F1 across region/target pairs; abstentions excluded, coverage reported separately.'}
    primary=output/'report.json'
    if primary.exists():
        baseline=json.loads(primary.read_text(encoding='utf-8'))
        for key,value in report['macro']['metrics'].items():
            if value is not None and baseline.get(key) is not None and not np.isclose(value,baseline[key]):raise AssertionError(f'Macro metric audit mismatch: {key}')
    report['any_violation_by_region']={}
    for region in ('SPINE','LEG_LEFT','LEG_RIGHT'):
        all_rows=[r for r in timing if r['truth_region']==region];usable=[]
        for r in all_rows:
            if not r['truth'] or r['predicted_region']!=region or any(v is None for v in r['truth'].values()) or any(r['predicted'].get(k) is None for k in r['truth']):continue
            usable.append({'study':r['study'],'truth':int(any(r['truth'].values())),
                           'prediction':int(any(r['predicted'][k] for k in r['truth'])),
                           'score':max(r['scores'].get(k,0.) or 0. for k in r['truth'])})
        report['any_violation_by_region'][region]={'coverage':len(usable)/len(all_rows),**with_ci(usable,300)}
    study_ids=sorted({r['study'] for r in all_quality});study_rows=[]
    def any_flag(values):
        if 1 in values:return 1
        return 0 if values and all(v==0 for v in values) else None
    for study in study_ids:
        items=[r for r in all_quality if r['study']==study]
        truth=any_flag([int(r['truth']) if r['truth'] else None for r in items])
        guess=any_flag([int(r['prediction']) if r['prediction'] else None for r in items])
        if truth is None or guess is None:continue
        study_rows.append({'study':study,'truth':truth,'prediction':guess,'score':max([float(r['score']) for r in items if r['score']] or [0.])})
    report['any_violation_studies']={'coverage':len(study_rows)/max(1,len(study_ids)),'total_studies':len(study_ids),**with_ci(study_rows,300),
                                    'definition':'Any known positive flag makes study positive; study is negative only when every applicable flag is known zero. Otherwise abstain.'}
    report['notes']=['Distances use nominal pixel spacing; original coordinates.',
                     'PCK includes missing points as failures; distance averages only found points, coverage is separate.',
                     'Mask Dice/IoU are image means; empty/empty masks score 1; region mismatch predicts empty geometry.',
                     'Box metrics use final calibrated confidence threshold and IoU >= .5; CIs resample original studies.']
    save(output/'spatial_test_rows.json',rows);save(output/'spatial_test_metrics.json',report);return report
