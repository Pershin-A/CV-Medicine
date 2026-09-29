"""Separate synthetic robustness check using held-out source studies only."""
import argparse,csv,json,warnings,time
from pathlib import Path
import numpy as np
from .data import load_records
from .train import split_records
from .predict import predict_file
from .evaluate import TASKS,with_ci

def run(root,augmented,checkpoints,output,per_group=15,selection_protocol=None):
    output.mkdir(parents=True,exist_ok=True)
    originals=load_records(root); train,valid=split_records(originals,0)
    if selection_protocol is not None:
        from .protocol import make_protocol
        protocol=make_protocol(root,originals,train,valid,selection_protocol)
        mapping=protocol['partition_by_path']
        train=[r for r in originals if mapping[r.relative_path]!='test']
        valid=[r for r in originals if mapping[r.relative_path]=='test']
    valid_sources={r.relative_path:r.study for r in valid}
    with (augmented/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    eligible=[r for r in all_rows if r['source_relative_path'] in valid_sources]
    rng=np.random.default_rng(42); selected=[]
    for region in TASKS:
        for group in ('positive','negative_position','negative_axis_or_roi'):
            rows=[r for r in eligible if r['region']==region and r['generation_group']==group]
            if rows:selected.extend(rows[i] for i in rng.choice(len(rows),min(per_group,len(rows)),replace=False))
    calibration=json.loads((checkpoints/'evaluation/calibration.json').read_text())
    area_threshold_mm2=calibration['rotation_threshold_px2']*1.05*.6
    tasks=[]; failures=[]; times=[]
    for row in selected:
        study=valid_sources[row['source_relative_path']]
        assert study not in {r.study for r in train}
        spacing=float(row['row_spacing_mm']),float(row['col_spacing_mm'])
        start=time.perf_counter()
        try:
            prediction=predict_file(augmented/row['image_path'],checkpoints,
                                   rotation_area_threshold_px2=area_threshold_mm2/(spacing[0]*spacing[1]),spacing_override_mm=spacing)
            times.append(time.perf_counter()-start)
            for task in TASKS[row['region']]:
                truth=row.get(task,'')
                tasks.append({'region':row['region'],'task':task,'study':study,'image':row['image_path'],
                              'truth':int(float(truth)) if truth!='' else None,
                              'prediction':prediction['quality_flags'].get(task) if prediction['region']==row['region'] else None,
                              'score':prediction['quality_scores'].get(task) if prediction['region']==row['region'] else None})
        except Exception as error:failures.append({'image':row['image_path'],'error':repr(error)})
    metrics=[]
    for region,conditions in TASKS.items():
        for task in conditions:
            rows=[r for r in tasks if r['region']==region and r['task']==task and r['truth'] is not None]
            usable=[r for r in rows if r['prediction'] is not None]
            metrics.append({'region':region,'task':task,'coverage':len(usable)/len(rows) if rows else None,**with_ci(usable,200)})
    report={'selected_images':len(selected),'eligible_heldout_augmentations':len(eligible),
            'train_study_overlap':0,'failures':failures,'metrics':metrics,'mean_seconds':float(np.mean(times)),
            'scope':'synthetic robustness only; reported separately from original validation; generated labels inherit remaining annotation limitations',
            'spacing':'known transformed nominal spacing from augmentation manifest; rotation threshold converted through mm²'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'synthetic_evaluation':str(output/'report.json'),'n':len(selected)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--augmented-root',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection-protocol',type=Path)
    args=p.parse_args()
    with warnings.catch_warnings():warnings.simplefilter('ignore');run(args.root,args.augmented_root,args.checkpoints,args.output,selection_protocol=args.selection_protocol)
