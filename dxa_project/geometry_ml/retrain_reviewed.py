"""Versioned retraining after reviewed annotations; every stage is resumable.

Run: python -m dxa_project.geometry_ml.retrain_reviewed
The runner waits for the new generation report, audits data, screens six point
variants, trains two finalists, and enforces the user's twelve-hour heavy-model gate.
"""
from pathlib import Path
import csv,json,time,hashlib,subprocess,sys,shutil,math,random,gc,warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_records,load_augmented_records,collate,DxaDataset,read_dicom_image
from .train import train_task,split_records,_model,_loss,_subset
from .protocol import make_protocol,source_sampler
from .experiments import VARIANTS,train_variant,point_stats,summarize
from .landmarks import PointDataset,PointsNet
from .predict import _load_model,prepare_input,_original_point
from .evaluate import binary_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dxa_project/outputs/retrained_20260929'
AUG=ROOT/'dxa_project/outputs/augmented_15000_20260929'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def event(stage,**details):
    save(OUT/'status.json',{'stage':stage,'updated':time.strftime('%Y-%m-%d %H:%M:%S'),**details})
    print(json.dumps({'stage':stage,**details},ensure_ascii=False),flush=True)

def fingerprint(records):
    files=[ROOT/'разметка.xlsx',ROOT/'Размеченные/labels.csv',ROOT/'dxa_project/outputs/manifest.csv']
    files += [p for r in records for p in (r.source_path,r.geometry_path)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def clear_gpu():
    _load_model.cache_clear();gc.collect();torch.cuda.empty_cache()

def setup():
    OUT.mkdir(parents=True,exist_ok=True);torch.hub.set_dir(str(ROOT/'dxa_project/outputs/torch_hub'))
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this scheduled training run')
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    return torch.device('cuda')

def protocol(records):
    path=OUT/'protocol.json'
    if not path.exists():
        # Keep identical source groups to the earlier comparison, refresh counts.
        p=json.loads((ROOT/'dxa_project/outputs/improvements_v2/protocol.json').read_text(encoding='utf-8'))
        for part in ('train','validation','test'):
            rows=[r for r in records if p['partition_by_path'][r.relative_path]==part];sp=[r for r in rows if r.region=='SPINE']
            n=sum(bool(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects']) for r in sp)
            p['summary'][part]={'images':len(rows),'studies':len({r.study for r in rows}),'spine_images':len(sp),'artifact_positive':n,'artifact_fraction':n/len(sp)}
        p['label_revision']='2026-09-29';save(path,p)
    a,b=split_records(records,0)
    return make_protocol(ROOT,records,a,b,path)

def validation_augments(aug,parts):
    # Both ROI and point visibility examples, with independent source groups.
    with (AUG/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:table={f"aug/{r['image_path']}":r for r in csv.DictReader(f)}
    buckets={};rng=random.Random(42)
    for r in aug:
        if r.region=='SPINE' or parts[r.source_id]!='validation':continue
        key=(r.region,table[r.relative_path]['generation_group']);buckets.setdefault(key,[]).append(r)
    result=[]
    for key,pool in sorted(buckets.items()):rng.shuffle(pool);result+=pool[:20]
    return result

def proposed_boxes(records,folder,device):
    model,size=_load_model('hip',folder,device);result={}
    model.eval()
    with torch.no_grad():
        for r in records:
            image=read_dicom_image(r.source_path);h,w=image.shape;side=r.region.removeprefix('LEG_')
            x,m=prepare_input(image,size,side=='RIGHT');out=model(x[None].to(device))
            y1,y2,lateral=(out['roi'][0].cpu().numpy()*size).tolist()
            top=_original_point(0,min(y1,y2),m)[1];bottom=_original_point(0,max(y1,y2),m)[1];xlat=_original_point(lateral,0,m)[0]
            result[r.relative_path]=[0.,top,xlat,bottom] if side=='LEFT' else [xlat,top,float(w-1),bottom]
    return result

def deployed_points(model,config,records,boxes,device):
    dataset=PointDataset(records,config['size'],config['crop'],boxes)
    loader=DataLoader(dataset,batch_size=2 if config['size']>=512 or config['mode']=='independent' else 8,num_workers=2,collate_fn=collate,persistent_workers=True)
    rows=[];model.eval()
    with torch.no_grad():
        for images,targets in loader:
            outputs=model(images.to(device));batchrows=point_stats(outputs,targets,'masked_logit_argmax')
            if config['crop']=='roi':
                # Exactly the production fallback, never use a true validation ROI.
                for i,row in enumerate(batchrows):
                    if sum(row[k] for k in ('tp_visible','fp_visible'))<3:
                        full=PointDataset([targets[i]['record']],config['size'],'full');x,t=collate([full[0]])
                        batchrows[i]=point_stats(model(x.to(device)),t,'masked_logit_argmax')[0]
            rows.extend(batchrows)
    return rows

class Validation:
    def __init__(self,originals,synthetic,boxes):self.originals=originals;self.synthetic=synthetic;self.boxes=boxes
    def __call__(self,model,config,device):
        rows=deployed_points(model,config,self.originals+self.synthetic,self.boxes,device)
        n=len(self.originals);original=summarize(rows[:n]);synthetic=summarize(rows[n:])
        bal=synthetic['position']['metrics']['balanced_accuracy']
        value=.55*original['pck_10mm']+.25*synthetic['pck_10mm']+.20*(bal if bal is not None else .5)
        metrics={**original,'original':original,'synthetic':synthetic,'selection_score':value,'selection_loss':1-value,
                 'roi_validation':'predicted ROI + deployed full-image fallback'}
        return metrics,rows,1-value

def fit_modules(train,valid,folder,device,architecture='light'):
    folder.mkdir(parents=True,exist_ok=True)
    for task in ('router','spine','hip','artifact'):
        report=folder/f'{task}_report.json'
        if report.exists():continue
        clear_gpu();event(f'{architecture}_training',task=task)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        # Detector keeps its real internal resize (320 light / 800 heavy).
        size=256 if task=='router' or architecture=='light' else 800 if task=='artifact' else 384
        batch=8 if architecture=='light' or task=='router' else 2
        result=train_task(task,train,valid,folder,device,12 if task=='router' else 24,size,batch,None,True,
                          architecture,2,1e-5 if task=='artifact' else 2e-5,
                          3e-5 if task=='artifact' else 1e-3 if task=='router' else 2e-4,'cosine',0 if task=='artifact' else 6,True)
        save(report,result)

def select_spine_variant(train,valid,base,device):
    path=OUT/'spine_selection.json'
    if path.exists():return
    alternative=OUT/'spine_encoder_lr_2e4';alternative.mkdir(exist_ok=True)
    report_path=alternative/'spine_report.json'
    if not report_path.exists():
        clear_gpu();event('spine_learning_rate_comparison')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        report=train_task('spine',train,valid,alternative,device,24,256,8,None,True,'light',2,2e-4,2e-4,'cosine',6,True)
        save(report_path,report)
    reports=[(base,json.loads((base/'spine_report.json').read_text(encoding='utf-8')),2e-5),
             (alternative,json.loads(report_path.read_text(encoding='utf-8')),2e-4)]
    candidates=[]
    for folder,r,lr in reports:
        best=min(r['history'],key=lambda h:h['validation_loss_or_detection_count'])
        candidates.append({'encoder_lr':lr,'head_lr':2e-4,'best_epoch':best['epoch'],
                           'validation_loss':best['validation_loss_or_detection_count'],'validation_metrics':best['validation_metrics'],
                           'checkpoint':str(folder/'spine.pt'),'seconds':r['seconds']})
    chosen=min(candidates,key=lambda c:c['validation_loss'])
    if chosen['encoder_lr']!=2e-5:
        shutil.copy2(base/'spine.pt',base/'spine_encoder_lr_2e5.pt');shutil.copy2(chosen['checkpoint'],base/'spine.pt')
    save(path,{'candidates':candidates,'selected':chosen,'criterion':'minimum inner validation loss, same architecture and losses; outer test not used'})
    clear_gpu()

def screen_and_train(train,originals,synthetic,base,device):
    hiptrain=[r for r in train if r.region!='SPINE'];allvalid=originals+synthetic
    clear_gpu();boxes=proposed_boxes(allvalid,base,device);clear_gpu()
    validation=Validation(originals,synthetic,boxes)
    candidates=[]
    # Same number of full epochs and identical optimization for all six options.
    for name,conf in VARIANTS.items():
        folder=OUT/'screening'/name
        if not (folder/'report.json').exists():
            clear_gpu();event('point_screening',variant=name)
            train_variant({**conf,'architecture':'light','batch_size':8},hiptrain,originals,folder,device,epochs=8,patience=0,validation_evaluator=validation)
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        best=max(report['history'],key=lambda h:h['metrics']['selection_score'])
        candidates.append({'variant':name,'score':best['metrics']['selection_score'],'epoch':best['epoch'],'metrics':best['metrics'],'seconds':report['seconds']})
        save(OUT/'screening.json',{'budget_epochs':8,'selection':'inner validation only; original PCK10 55%, synthetic PCK10 25%, synthetic balanced accuracy 20%','candidates':candidates})
    finalists=sorted(candidates,key=lambda c:c['score'],reverse=True)[:2]
    full=[]
    for candidate in finalists:
        name=candidate['variant'];folder=OUT/'finalists'/name
        if not (folder/'report.json').exists():
            clear_gpu();event('point_finalist_training',variant=name)
            train_variant({**VARIANTS[name],'architecture':'light','batch_size':8},hiptrain,originals,folder,device,epochs=24,patience=7,validation_evaluator=validation)
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'));best=max(report['history'],key=lambda h:h['metrics']['selection_score'])
        full.append({'variant':name,'score':best['metrics']['selection_score'],'epoch':best['epoch'],'metrics':best['metrics'],'checkpoint':str(folder/'hip_points.pt'),'seconds':report['seconds']})
    selected=max(full,key=lambda x:x['score']);save(OUT/'selection.json',{'screening':candidates,'finalists':full,'selected':selected,'test_used_for_selection':False})
    destination=OUT/'light';destination.mkdir(exist_ok=True)
    for name in ('router','spine','hip','artifact'):shutil.copy2(base/f'{name}.pt',destination/f'{name}.pt')
    shutil.copy2(selected['checkpoint'],destination/'hip_points.pt')
    from .landmark_geometry import fit_geometry_ranges
    save(destination/'landmark_geometry_bounds.json',fit_geometry_ranges([r for r in hiptrain if not r.source_id]))
    clear_gpu()
    return selected

def heavy_timing(train,valid,selected,device):
    path=OUT/'heavy_timing.json'
    limit_seconds=12*3600
    if path.exists():
        report=json.loads(path.read_text(encoding='utf-8'))
        report.update(limit_seconds=limit_seconds,target_seconds=8*3600,allowed_full_training=report['estimate_seconds']<=limit_seconds)
        save(path,report);return report
    event('heavy_real_timing');timings=[]
    for task in ('router','spine','hip','artifact','hip_points'):
        clear_gpu();torch.manual_seed(42);size=256 if task=='router' else 800 if task=='artifact' else 384
        batch=8 if task=='router' else 2
        records=_subset(train,'hip' if task=='hip_points' else task,False)
        validation=_subset(valid,'hip' if task=='hip_points' else task,False)
        if task=='hip_points':
            config={**VARIANTS[selected['variant']],'architecture':'heavy'};size=config['size']
            model=PointsNet(True,config['mode'],'heavy').to(device)
            ds=PointDataset(records,size,config['crop']);vs=PointDataset(validation,size,config['crop'])
        else:model=_model(task,True,'heavy').to(device);ds=DxaDataset(records,size,router_mode=task=='router');vs=DxaDataset(validation,size,router_mode=task=='router')
        loader=DataLoader(ds,batch_size=batch,sampler=source_sampler(records,artifact=task=='artifact'),num_workers=2,persistent_workers=True,collate_fn=collate)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
        from .landmarks import point_loss
        times=[];iterator=iter(loader)
        for i in range(min(15,len(loader))):
            torch.cuda.synchronize();start=time.perf_counter();images,targets=next(iterator);model.train();optimizer.zero_grad(set_to_none=True)
            loss=point_loss(model(images.to(device)),targets,config['loss']) if task=='hip_points' else _loss(task,model,images,targets,device)[0]
            if not torch.isfinite(loss):raise FloatingPointError(f'Heavy {task} nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            if i>=3:times.append(time.perf_counter()-start)
        model.eval();vt=[]
        with torch.no_grad():
            for i,(images,targets) in enumerate(DataLoader(vs,batch_size=batch,num_workers=2,collate_fn=collate)):
                if i>=5:break
                torch.cuda.synchronize();start=time.perf_counter()
                if task=='artifact':
                    from .models import detector_images
                    model(detector_images(images.to(device)))
                else:model(images.to(device))
                torch.cuda.synchronize();vt.append(time.perf_counter()-start)
        # 30% margin includes loader, checkpoints, model-selection decoding and inference.
        steps=math.ceil(len(records)/batch);vsteps=math.ceil(len(validation)/batch)
        epoch_seconds=steps*float(np.quantile(times,.75))+vsteps*float(np.quantile(vt,.75))
        epochs=12 if task=='router' else 24
        timings.append({'task':task,'input_size':size,'batch_size':batch,'train_images':len(records),'train_batches':steps,
                        'measured_training_batches':len(times),'train_seconds_p75':float(np.quantile(times,.75)),
                        'validation_seconds_p75':float(np.quantile(vt,.75)),'planned_epochs':epochs,'estimated_seconds':epoch_seconds*epochs})
        print(json.dumps(timings[-1]),flush=True)
        del model,optimizer,loader,ds,vs;clear_gpu()
    estimate=sum(t['estimated_seconds'] for t in timings)*1.30+600
    report={'device':torch.cuda.get_device_name(),'tasks':timings,'estimate_seconds':estimate,'estimate_hours':estimate/3600,
            'limit_seconds':limit_seconds,'target_seconds':8*3600,'allowed_full_training':estimate<=limit_seconds,'safety_margin':.30,'additional_inference_seconds':600,
            'scope':'real regenerated training data; full planned epochs without relying on early stopping; detector internal resize unchanged'}
    save(path,report);return report

def evaluate_pipeline(folder,selection_protocol=None):
    clear_gpu();event('pipeline_inference',architecture=folder.name)
    selection_protocol=selection_protocol or OUT/'protocol.json'
    if not (folder/'evaluation/report.json').exists():
        subprocess.run([sys.executable,'-m','dxa_project.geometry_ml.evaluate','--checkpoints',str(folder),'--output',str(folder/'evaluation'),
                        '--selection-protocol',str(selection_protocol),'--bootstrap','300'],cwd=ROOT,check=True)
    report=json.loads((folder/'evaluation/report.json').read_text(encoding='utf-8'))
    if report['failed_files']:raise RuntimeError(f'Pipeline inference failed on {len(report["failed_files"])} files')
    if not (folder/'synthetic_evaluation/report.json').exists():
        subprocess.run([sys.executable,'-m','dxa_project.geometry_ml.evaluate_augmented','--checkpoints',str(folder),'--output',str(folder/'synthetic_evaluation'),
                        '--augmented-root',str(AUG),'--selection-protocol',str(selection_protocol)],cwd=ROOT,check=True)
    report=json.loads((folder/'synthetic_evaluation/report.json').read_text(encoding='utf-8'))
    if report['failures']:raise RuntimeError(f'Synthetic inference failed on {len(report["failures"])} files')

def notebooks():
    import nbformat
    for name in VARIANTS:
        nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell(f'# DXA: {name}\nВосемь полных эпох сравнения на новой аугментации. Выбор по внутренней валидации; ROI берётся из предсказаний модели.\nПовторное обучение доступно через общий воспроизводимый скрипт.'),
            nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'dxa_project').exists())\n"+f"report = json.loads((root/'dxa_project/outputs/retrained_20260929/screening/{name}/report.json').read_text(encoding='utf-8'))\n"+"[(h['epoch'], h['metrics']['selection_score'], h['metrics']['original']['pck_10mm'], h['metrics']['synthetic']['position']['metrics']) for h in report['history']]"),
            nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\nh = report['history']\nplt.plot([r['epoch'] for r in h], [r['metrics']['selection_score'] for r in h]); plt.xlabel('Эпоха'); plt.ylabel('Оценка на внутренней валидации'); plt.grid()"),
            nbformat.v4.new_code_cell("# Измените флаг, чтобы повторить именно этот эксперимент.\nRUN_EXPERIMENT = False\nif RUN_EXPERIMENT:\n    import subprocess, sys\n"+f"    subprocess.run([sys.executable, '-m', 'dxa_project.geometry_ml.retrain_reviewed', '--variant', '{name}', '--epochs', '8'], cwd=root, check=True)")]
        nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}
        nbformat.write(nb,ROOT/f'DXA_{name}_20260929.ipynb')
    nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# Полный повторный запуск DXA после исправления разметки\nПоследовательность: новый набор → аудит → обучение модулей → шесть сравнений → два финалиста → инференс → измерение времени большой модели.\nБольшая модель запускается только при оценке полного времени ≤ 12 часов.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'dxa_project').exists())\nout = root/'dxa_project/outputs/retrained_20260929'\nfor name in ['corrections.json', 'screening.json', 'selection.json', 'heavy_timing.json', 'status.json']:\n    path = out/name\n    print(name, json.loads(path.read_text(encoding='utf-8')) if path.exists() else 'Ожидается')"),
        nbformat.v4.new_code_cell("RUN_TRAINING = False\nif RUN_TRAINING:\n    import subprocess, sys\n    subprocess.run([sys.executable, '-m', 'dxa_project.geometry_ml.retrain_reviewed'], cwd=root, check=True)")]
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}};nbformat.write(nb,ROOT/'DXA_retraining_20260929.ipynb')
    nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# Позвоночник: сравнение скоростей обучения encoder\n2e-5 и 2e-4 при одинаковой архитектуре, функциях потерь, данных и бюджете 24 эпох. Выбор по внутренней валидационной функции потерь.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'dxa_project').exists())\nreport = json.loads((root/'dxa_project/outputs/retrained_20260929/spine_selection.json').read_text(encoding='utf-8'))\nreport"),
        nbformat.v4.new_code_cell("RUN_EXPERIMENT = False\nif RUN_EXPERIMENT:\n    import subprocess, sys\n    subprocess.run([sys.executable, '-m', 'dxa_project.geometry_ml.train', '--task', 'spine', '--epochs', '24', '--batch-size', '8', '--loader-workers', '2', '--encoder-lr', '2e-4', '--head-lr', '2e-4', '--scheduler', 'cosine', '--patience', '6', '--balance-sources', '--selection-protocol', 'dxa_project/outputs/retrained_20260929/protocol.json', '--augmented-root', 'dxa_project/outputs/augmented_15000_20260929', '--output', 'dxa_project/outputs/retrained_20260929/notebook_runs/spine_lr'], cwd=root, check=True)")]
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}};nbformat.write(nb,ROOT/'DXA_spine_lr_20260929.ipynb')
    nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# Большая модель: реальное измерение времени\nResNet50 и Faster R-CNN ResNet50 FPN v2. Полный запуск разрешён только если консервативная оценка ≤ 12 часов. Таблица отражает реальные тренировочные пакеты новых данных, а не случайные тензоры.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'dxa_project').exists())\nreport = json.loads((root/'dxa_project/outputs/retrained_20260929/heavy_timing.json').read_text(encoding='utf-8'))\nreport"),
        nbformat.v4.new_code_cell("import pandas as pd\npd.DataFrame(report['tasks'])"),
        nbformat.v4.new_code_cell("print('Оценка, часов:', report['estimate_hours'])\nprint('Полное обучение разрешено:', report['allowed_full_training'])")]
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}};nbformat.write(nb,ROOT/'DXA_heavy_timing_20260929.ipynb')
    # Statistics can run in another notebook kernel; training must use CUDA venv.
    for path in ROOT.glob('DXA_*_20260929.ipynb'):
        notebook=nbformat.read(path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                cell.source=cell.source.replace('    subprocess.run([sys.executable,',
                    "    training_python = next((p for p in (root/'.venv/Scripts/python.exe', root/'.venv/bin/python') if p.is_file()), Path(sys.executable))\n    subprocess.run([str(training_python),")
        nbformat.write(notebook,path)

def main():
    warnings.filterwarnings('ignore',category=UserWarning,module='pydicom');device=setup();notebooks();records=load_records(ROOT)
    current=fingerprint(records);snapshot=OUT/'data_snapshot.json'
    if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8'))!=current:raise RuntimeError('Source data changed during this versioned run')
    save(snapshot,current);p=protocol(records);event('waiting_for_new_augmentation',protocol_summary=p['summary'])
    while not (AUG/'generation_report.json').exists():time.sleep(5)
    if fingerprint(records)!=current:raise RuntimeError('Sources changed during augmentation')
    audit_path=AUG/'audit_report.json'
    if not audit_path.exists():
        event('augmentation_audit');subprocess.run([sys.executable,'-m','dxa_project.augmentation.audit_generated',str(AUG),'--output',str(audit_path)],cwd=ROOT,check=True)
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['files']!=15000 or audit['problems']:raise RuntimeError(f'Augmentation audit failed: {audit["problems"][:3]}')
    aug=load_augmented_records(ROOT,AUG,records);parts=p['partition_by_path']
    train=[r for r in records if parts[r.relative_path]=='train']+[r for r in aug if parts[r.source_id]=='train']
    valid=[r for r in records if parts[r.relative_path]=='validation'];hipvalid=[r for r in valid if r.region!='SPINE']
    synthetic=validation_augments(aug,parts)
    save(OUT/'training_data.json',{'train_images':len(train),'validation_originals':len(valid),'point_synthetic_validation':len(synthetic),
                                  'augmented_directory':str(AUG),'group_overlap':len({r.study for r in train}&{r.study for r in valid}),
                                  'point_synthetic_paths':[r.relative_path for r in synthetic]})
    base=OUT/'light_base';fit_modules(train,valid,base,device);select_spine_variant(train,valid,base,device)
    selected=screen_and_train(train,hipvalid,synthetic,base,device);evaluate_pipeline(OUT/'light')
    if not (ROOT/'dxa_project/team_demo/retraining_examples_20260929.html').exists():
        subprocess.run([sys.executable,'-m','dxa_project.geometry_ml.reviewed_examples'],cwd=ROOT,check=True)
    from .heavy_continuation import run as run_heavy
    run_heavy()

def single_variant(name,epochs):
    device=setup();records=load_records(ROOT);p=protocol(records)
    if not (AUG/'audit_report.json').exists() or json.loads((AUG/'audit_report.json').read_text())['problems']:
        raise RuntimeError('Regenerated data must pass the audit first')
    if fingerprint(records)!=json.loads((OUT/'data_snapshot.json').read_text(encoding='utf-8')):
        raise RuntimeError('Sources have changed')
    parts=p['partition_by_path'];aug=load_augmented_records(ROOT,AUG,records)
    train=[r for r in records if parts[r.relative_path]=='train' and r.region!='SPINE']+[r for r in aug if parts[r.source_id]=='train' and r.region!='SPINE']
    valid=[r for r in records if parts[r.relative_path]=='validation' and r.region!='SPINE'];synthetic=validation_augments(aug,parts)
    boxes=proposed_boxes(valid+synthetic,OUT/'light_base',device);clear_gpu()
    train_variant({**VARIANTS[name],'architecture':'light','batch_size':8},train,valid,OUT/'notebook_runs'/name,device,epochs,patience=0,
                  validation_evaluator=Validation(valid,synthetic,boxes))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--variant',choices=tuple(VARIANTS));parser.add_argument('--epochs',type=int,default=8)
    args=parser.parse_args()
    try:
        if args.variant:single_variant(args.variant,args.epochs)
        else:main()
    except Exception as error:
        event('failed',error=repr(error));raise
