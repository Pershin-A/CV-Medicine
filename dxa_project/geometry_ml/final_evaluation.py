"""Evaluate the assembled bundle and verify actual batch API prediction."""
import json,time,hashlib,csv,copy
from pathlib import Path
import numpy as np
import torch
from .final_assembly import ROOT,OUT,AUG,save,event,prepare
from .evaluate import run as evaluate,with_ci,truth_flags
from .predict import predict_file,_load_model


def run():
    bundle=OUT/'bundle';required=('router','spine','hip','hip_mask','hip_points','artifact','scoliosis')
    if not all((bundle/(k+'.pt')).exists() for k in required):raise RuntimeError('Training is incomplete')
    originals,augmented,groups,labels,protocol=prepare();torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    # Midpoints preserve selected decisions without sitting exactly on a float score.
    from .scoliosis import choose_threshold
    validation_rows=json.loads((bundle/'scoliosis_validation_rows.json').read_text(encoding='utf-8'))
    threshold,metrics,score=choose_threshold(validation_rows)
    ckpt=torch.load(bundle/'scoliosis.pt',map_location='cpu',weights_only=False);ckpt['threshold']=threshold;torch.save(ckpt,bundle/'scoliosis.pt')
    save(bundle/'scoliosis_calibration.json',{'threshold':threshold,'validation':metrics,'epoch':ckpt['epoch'],'selection_score':score,'source':'selected epoch original validation; decision-stable midpoint thresholds'})
    torch.hub.set_dir(str(ROOT/'dxa_project/outputs/torch_hub'))
    from .final_crests import train_and_select as select_crests
    select_crests(groups,bundle,torch.device('cuda'))
    if (bundle/'spine_crests.pt').is_file():required+=('spine_crests',)
    from .final_hip_selection import select as select_hip
    select_hip(groups,bundle,torch.device('cuda'))
    from .final_artifact import select_and_calibrate
    select_and_calibrate(groups,bundle,torch.device('cuda'))
    event('evaluation',scope='test originals; never model selection')
    evaluate(ROOT,bundle,bundle/'evaluation',bootstrap=300,selection_protocol=OUT/'protocol.json')
    calibration=json.loads((bundle/'evaluation/calibration.json').read_text(encoding='utf-8'))
    calibration['rotation_threshold_mm2']=calibration['rotation_threshold_px2']*1.05*.6
    save(bundle/'evaluation/calibration.json',calibration)
    report=json.loads((bundle/'evaluation/report.json').read_text(encoding='utf-8'))
    geometry_test=evaluate_spine_geometry(groups['test'],bundle/'evaluation',labels)
    from .final_test_metrics import run as spatial_audit
    spatial_test=spatial_audit(groups['test'],bundle/'evaluation')
    manifest={'version':'final_20260929','modules':{k:{'file':k+'.pt','sha256':hashlib.sha256((bundle/(k+'.pt')).read_bytes()).hexdigest()} for k in required},
              'protocol':str(OUT/'protocol.json'),'label_sources':'organizer correction manifest + local SCOLIOSIS labels; inherited only in sufficiently visible rigid augmentation',
              'hip_selection':json.loads((OUT/'hip_selection.json').read_text(encoding='utf-8')),
              'point_selection':json.loads((OUT/'point_selection.json').read_text(encoding='utf-8')),
              'artifact_selection':json.loads((OUT/'artifact_selection.json').read_text(encoding='utf-8')),
              'spine_selection':json.loads((OUT/'spine_selection.json').read_text(encoding='utf-8')),
              'crest_selection':json.loads((OUT/'crest_selection.json').read_text(encoding='utf-8')),
              'initialization':'generic ImageNet/COCO; no prior local trained weights','test':report,'spine_geometry_test':geometry_test,'spatial_test':spatial_test,'activation':'Versioned bundle ready for registration; existing active version is preserved'}
    save(bundle/'model_manifest.json',manifest)
    event('api_smoke')
    smoke_api(bundle,groups['test'])
    from .final_report import build
    build();event('complete',bundle=str(bundle),report=str(ROOT/'dxa_project/FINAL_MODEL_20260929.md'))


def evaluate_spine_geometry(records,output,labels):
    """Audit the selected module on test; reuse predictions, never select on test."""
    import pydicom
    from concurrent.futures import ProcessPoolExecutor
    from .final_spine import strict_angle
    from .spine_angle_study import angle_summary
    from .spine_penalty_study import measure,summarize
    selected=[(i,r) for i,r in enumerate(records,1) if r.region=='SPINE']
    tasks=[]
    for _,r in selected:
        ds=pydicom.dcmread(r.source_path)
        tasks.append((np.squeeze(ds.pixel_array),json.loads(r.geometry_path.read_text(encoding='utf-8')),r.spacing_mm,
                      'dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright'))
    with ProcessPoolExecutor(max_workers=6) as pool:references=list(pool.map(strict_angle,tasks,chunksize=2))
    rows=[];lines=[]
    for (i,r),reference in zip(selected,references):
        path=output/f'prediction_{i:03}.json'
        pred=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        correct=pred.get('region')=='SPINE'
        rows.append({'relative_path':r.relative_path,'study':r.study,'reference_angle':reference,
                     'angle':pred.get('spine_axis_angle_deg') if correct else None,
                     'flag':labels[r.relative_path].get('spine_axis')})
        lines.append(measure(pred['geometry']['spine']['disc_lines'] if correct else [],r))
    summary={'angles':angle_summary(rows,bootstrap=300),'dividers':summarize(lines),
             'reference':'strict geometry from manual dividers; pseudo-reference, not independently annotated axis',
             'selection':'test audit only; no threshold or checkpoint selection'}
    by_study={s:[r for r in lines if r['study']==s] for s in sorted({r['study'] for r in lines})}
    ids=list(by_study);rng=np.random.default_rng(42);samples={k:[] for k in summary['dividers'] if k not in ('images','studies')}
    for _ in range(300):
        metrics=summarize([r for i in rng.integers(0,len(ids),len(ids)) for r in by_study[ids[i]]])
        for k in samples:
            if metrics[k] is not None:samples[k].append(metrics[k])
    summary['dividers_ci95']={k:np.percentile(v,[2.5,97.5]).tolist() if v else None for k,v in samples.items()}
    save(output/'spine_geometry_test.json',summary);save(output/'spine_geometry_rows.json',rows)
    return summary


def smoke_api(bundle,records):
    from fastapi.testclient import TestClient
    from dxa_project.service.app import create_app
    from dxa_project.service.training import register
    from dxa_project.service.worker import Worker
    # Isolated smoke store does not switch the user's active service model.
    app=create_app(OUT/'api_smoke');store=app.state.store
    model=register(store,{'checkpoints':str(bundle),'protocol':str(OUT/'protocol.json'),
                         'epochs':{k:1 for k in ('router','spine','hip','artifact','hip_points')},'activate':True})
    tested=json.loads((bundle/'evaluation/predictions.json').read_text(encoding='utf-8'))
    routed={r['relative_path'] for r in tested if r['truth_region']==r['predicted_region']}
    chosen=[next(r for r in records if r.region==region and r.relative_path in routed) for region in ('SPINE','LEG_LEFT','LEG_RIGHT')]
    paths=[r.source_path for r in chosen];job=store.create_job('predict',{'paths':[str(p) for p in paths],'model_version':model['version']},None)
    Worker(store,'cuda').run(store.job(job));result=store.job(job)
    if result['status']!='completed':raise RuntimeError(f'API inference failed: {result}')
    client=TestClient(app);logs=client.get(f'/v1/jobs/{job}/logs').json()
    if len(logs['events'])!=6:raise RuntimeError('Missing per-image prediction events')
    predictions=[store.get('result',rid) for rid in result['result']['result_ids']]
    if 'spine_scoliosis' not in predictions[0]['quality_flags']:raise RuntimeError('API dropped scoliosis module')
    if (bundle/'spine_crests.pt').is_file() and 'spine_crests' not in result['result']['batch_sizes']:raise RuntimeError('API dropped independent iliac landmark module')
    exports={}
    for fmt in ('csv','json','zip'):
        response=client.get(f'/v1/jobs/{job}/export?format={fmt}');response.raise_for_status();exports[fmt]=len(response.content)
    save(OUT/'api_smoke_report.json',{'status':result['status'],'job_id':job,'images':len(paths),'logs':logs,'export_bytes':exports,
                                   'batch_sizes':result['result']['batch_sizes'],'regions':[r['region'] for r in predictions]})
