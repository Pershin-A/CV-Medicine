"""Candidate pipeline checks and rotation calibration using validation studies only."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from .store import PROJECT,write_json
from .inference import InferenceEngine
from dxa_project.geometry_ml.predict import _load_model
from dxa_project.geometry_ml.evaluate import truth_flags,TASKS,with_ci

def evaluate(folder,records,device,calibrate=False):
    _load_model.cache_clear();torch.cuda.empty_cache()
    engine=InferenceEngine(folder,str(device));values=[]
    with (PROJECT/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        reference={r['relative_path']:r for r in csv.DictReader(f)}
    from dxa_project.geometry_ml.final_protocol import labels_for_originals
    local=labels_for_originals(PROJECT,records) if (folder/'scoliosis.pt').exists() else {}
    for offset in range(0,len(records),8):
        batch=records[offset:offset+8]
        predicted=engine.predict([r.source_path for r in batch])
        for r,p in zip(batch,predicted):
            truth=truth_flags(reference[r.relative_path],r.region,json.loads(r.geometry_path.read_text(encoding='utf-8')))
            if r.region=='SPINE' and local:truth['spine_scoliosis']=local[r.relative_path].get('spine_scoliosis')
            values.append({'record':r,'truth':truth,'region':p['region'],'predictions':p['quality_flags'],'scores':p['quality_scores'],
                           'area':p.get('lesser_trochanter_area_mm2')})
    if calibrate:
        areas=[(v['area'],v['truth']['hip_rotation']) for v in values if v['record'].region!='SPINE' and v['region']==v['record'].region and v['truth'].get('hip_rotation') is not None]
        if len({y for a,y in areas})==2:
            candidates=np.unique([0.,.63,*[a+.001 for a,y in areas]])
            threshold=float(max(candidates,key=lambda t:f1_score([y for a,y in areas],[int(a<t) for a,y in areas],zero_division=0)))
            write_json(folder/'evaluation/calibration.json',{'rotation_threshold_mm2':threshold,'rotation_threshold_px2':threshold/(1.05*.6),'source':'service inner validation only','n':len(areas)})
            for v in values:
                if v['area'] is not None:
                    v['predictions']['hip_rotation']=int(v['area']<threshold)
                    v['scores']['hip_rotation']=float(1/(1+np.exp(np.clip((v['area']-threshold)/max(.63,threshold*.2),-60,60))))
    metrics=[]
    for region,tasks in TASKS.items():
        if region=='SPINE' and local:tasks=tasks+('spine_scoliosis',)
        for task in tasks:
            rows=[];labeled=0
            for v in values:
                if v['record'].region!=region or v['truth'][task] is None:continue
                labeled+=1
                if v['region']!=region or v['predictions'].get(task) is None:continue
                rows.append(dict(study=v['record'].study,truth=v['truth'][task],prediction=v['predictions'][task],score=v['scores'].get(task)))
            metrics.append(dict(region=region,task=task,labeled=labeled,coverage=len(rows)/labeled if labeled else None,**with_ci(rows,100)))
    defined=[m['metrics']['f1'] for m in metrics if m.get('metrics') and m['metrics']['f1'] is not None]
    known=sum(m['labeled'] for m in metrics);processed=sum(m['n'] for m in metrics)
    report=dict(images=len(records),studies=len({r.study for r in records}),metrics=metrics,
                macro_f1=float(np.mean(defined)) if defined else None,coverage=processed/known if known else 0,
                scope='inner validation, not untouched test; study bootstrap 100 replicates',processed_files=len(values))
    _load_model.cache_clear();torch.cuda.empty_cache();return report

def compare(base,folder,records,device):
    before=evaluate(Path(base['path']),records,device)
    after=evaluate(folder,records,device,True)
    passed=(after['macro_f1'] is not None and before['macro_f1'] is not None and
            after['macro_f1']>=before['macro_f1']-.02 and after['coverage']>=before['coverage']-.02)
    report=dict(before=before,after=after,passed=passed,allowed_macro_f1_drop=.02,allowed_coverage_drop=.02)
    write_json(folder/'pipeline_validation.json',report);return report
