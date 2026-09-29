"""Warm start, replay and study-safe validation; candidates never replace active weights."""
import json
import random
import shutil
import uuid
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader
from .store import PROJECT,allowed_path,write_json
from .data_ops import image_hash
from dxa_project.geometry_ml.data import Record,load_records,collate,DxaDataset
from dxa_project.geometry_ml.protocol import source_sampler
from dxa_project.geometry_ml.train import train_task,_subset,_loss,_validation_stats,_aggregate_stats
from dxa_project.geometry_ml.predict import _load_model
from dxa_project.geometry_ml.landmarks import PointDataset,point_loss

TASKS=('router','spine','hip','artifact','hip_points')

def register(store,payload):
    folder=allowed_path(payload['checkpoints']);protocol=allowed_path(payload['protocol'])
    metadata_path=folder/'training_config.json'
    config=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    payload=dict(payload);payload['epochs']=payload.get('epochs') or config.get('epochs') or dict(router=12,spine=24,hip=24,artifact=24,hip_points=24)
    for task in TASKS:
        if not (folder/f'{task}.pt').is_file():raise ValueError(f'Missing checkpoint: {task}.pt')
    if not (folder/'evaluation/calibration.json').is_file():raise ValueError('Bundle requires evaluation/calibration.json')
    p=json.loads(protocol.read_text(encoding='utf-8'))
    try:records=load_records(PROJECT)
    except FileNotFoundError:records=[]
    parts=p['partition_by_path']
    if not parts or not set(parts.values())<={'train','validation','test'}:raise ValueError('Invalid training protocol')
    if records and set(parts)!={r.relative_path for r in records}:raise ValueError('Protocol must cover all current original images')
    groups={}
    for r in records:
        if groups.setdefault(r.study,parts[r.relative_path])!=parts[r.relative_path]:raise ValueError('Protocol study leakage')
    for task in TASKS:
        epochs=payload['epochs'].get(task)
        if not isinstance(epochs,int) or not 1<=epochs<=200:raise ValueError(f'Invalid epoch budget: {task}')
    for task,epochs in payload['epochs'].items():
        if task not in (*TASKS,'scoliosis','hip_mask','spine_crests') or type(epochs) is not int or not 1<=epochs<=200:
            raise ValueError(f'Invalid saved module/epoch budget: {task}')
    version=uuid.uuid4().hex;dest=store.root/'models'/version;dest.mkdir(parents=True)
    for name in [*(f'{t}.pt' for t in TASKS),'scoliosis.pt','hip_mask.pt','spine_crests.pt','landmark_geometry_bounds.json','training_config.json']:
        if (folder/name).exists():shutil.copy2(folder/name,dest/name)
    (dest/'evaluation').mkdir();shutil.copy2(folder/'evaluation/calibration.json',dest/'evaluation/calibration.json')
    shutil.copy2(protocol,dest/'protocol.json')
    item=dict(version=version,path=str(dest),epochs=payload['epochs'],status='imported',origin=str(folder),protocol=str(dest/'protocol.json'),
              training_data_available=bool(records))
    write_json(dest/'model_manifest.json',item);store.put('model',version,item)
    if payload['activate']:store.put('config','active_model',{'version':version})
    return item

def model(store,version=None):
    if version is None:version=store.get('config','active_model')['version']
    return store.get('model',version)

def validate_rows(store,base,ids=None):
    rows=[r for r in store.all('queue') if r['status']=='ready' and (ids is None or r['id'] in ids)]
    if not rows:raise ValueError('No ready training examples')
    if ids is not None and len(rows)!=len(set(ids)):raise ValueError('Some requested queue items are not ready')
    records=load_records(PROJECT);parts=json.loads(Path(base['protocol']).read_text(encoding='utf-8'))['partition_by_path']
    study_parts={r.study:parts[r.relative_path] for r in records};pixels={}
    for r in records:
        key=image_hash(pydicom.dcmread(r.source_path).pixel_array)
        old=pixels.setdefault(key,parts[r.relative_path])
        if old!=parts[r.relative_path]:raise ValueError('Base protocol contains cross-partition identical images')
    for row in rows:
        im=store.get('image',row['image_id'])
        if study_parts.get(row['study'],'train')!='train' or pixels.get(im['pixel_hash'],'train')!='train':
            raise ValueError(f"Held-out study/image cannot enter training: {row['image_id']}")
    return rows,records,parts

def select_new(rows,task):
    if task=='router':return rows
    return [r for r in rows if ('hip_points' in r['reviewed'] or 'hip' in r['reviewed']) and r['region']!='SPINE'] if task=='hip_points' else [
        r for r in rows if task in r['reviewed'] and (r['region']=='SPINE')==(task in ('spine','artifact'))]

def metric(model,task,records,size,device,config=None):
    ds=PointDataset(records,size,config['crop']) if task=='hip_points' else DxaDataset(records,size,router_mode=task=='router')
    model.eval();losses=[];stats=[]
    with torch.no_grad():
        for images,targets in DataLoader(ds,batch_size=2,collate_fn=collate):
            if task=='hip_points':losses.append(float(point_loss(model(images.to(device)),targets,config['loss'] or 'heatmap')))
            elif task=='artifact':
                from dxa_project.geometry_ml.models import detector_images
                stats.append(_validation_stats(task,model(detector_images(images.to(device))),targets,size))
            else:losses.append(float(_loss(task,model,images,targets,device)[0]))
    if task=='artifact':
        s=_aggregate_stats(stats,task);p=s['box_precision_iou50'];r=s['box_recall_iou50'];return -2*p*r/max(1e-9,p+r)
    return float(np.mean(losses))

def partial_fit(store,job,device):
    if job['payload'].get('human') is not None:
        from .staged_fit import run
        return run(store,job,device)
    payload=job['payload'];base=model(store,payload.get('base_model_version'))
    rows,originals,parts=validate_rows(store,base,payload.get('queue_ids'))
    version=job['id'];folder=store.root/'models'/version
    if folder.exists():raise ValueError('Interrupted training candidate exists; create a new job after reviewing it')
    shutil.copytree(base['path'],folder)
    # Persist source ownership BEFORE training, including newly seen studies.
    for row in rows:
        ownership=next((s for s in store.all('study') if s['id']==row['study']),None)
        if ownership and ownership['partition']!='train':raise ValueError('New study is reserved for validation/test')
        store.put('study',row['study'],{'id':row['study'],'partition':'train'})
        row['status']='reserved';row['training_job']=job['id'];store.put('queue',row['id'],row)
    write_json(folder/'queue_snapshot.json',rows);reports=[]
    try:
        for task in TASKS:
            new=select_new(rows,task)
            if not new:continue
            fresh=[Record('service/'+r['id'],Path(r['path']),Path(r['geometry_path']),r['study'],r['region'],tuple(r['spacing_mm']),r['source_id']) for r in new]
            old=_subset([r for r in originals if parts[r.relative_path]=='train'],'hip' if task=='hip_points' else task,False)
            rng=random.Random(42);replay=rng.sample(old,min(payload['replay_per_task'],len(old)))
            valid=_subset([r for r in originals if parts[r.relative_path]=='validation'],'hip' if task=='hip_points' else task,False)
            if not valid:raise ValueError(f'No held-out validation records for {task}')
            _load_model.cache_clear();torch.cuda.empty_cache()
            net,size=_load_model(task,Path(base['path']),device);config=getattr(net,'point_configuration',None)
            before=metric(net,task,valid,size,device,config)
            architecture=torch.load(Path(base['path'])/f'{task}.pt',map_location='cpu',weights_only=False).get('architecture','light')
            budget=base['epochs'][task]
            store.update(job['id'],result={'version':version,'task':task,'epochs':budget,'completed_modules':len(reports)})
            if task=='hip_points':
                loader=DataLoader(PointDataset(fresh+replay,size,config['crop']),batch_size=2,sampler=source_sampler(fresh+replay),collate_fn=collate)
                opt=torch.optim.AdamW(net.parameters(),lr=payload['learning_rate'],weight_decay=1e-4)
                sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budget,eta_min=1e-6);history=[];best=float('inf')
                checkpoint=torch.load(Path(base['path'])/'hip_points.pt',map_location='cpu',weights_only=False)
                for epoch in range(budget):
                    net.train();losses=[]
                    for images,targets in loader:
                        opt.zero_grad(set_to_none=True);loss=point_loss(net(images.to(device)),targets,config['loss'] or 'heatmap')
                        if not torch.isfinite(loss):raise FloatingPointError('Non-finite point loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                    score=metric(net,task,valid,size,device,config);sched.step()
                    history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=score))
                    if score<best:
                        best=score;checkpoint['state_dict']=net.state_dict();torch.save(checkpoint,folder/'hip_points.pt')
                    write_json(folder/'hip_points_history.json',history)
                    store.update(job['id'],result={'version':version,'task':task,'epoch':epoch+1,'epochs':budget})
                report=dict(task=task,history=history,epochs=budget)
            else:
                del net;_load_model.cache_clear();torch.cuda.empty_cache()
                report=train_task(task,fresh+replay,valid,folder,device,budget,size,2,None,False,architecture,0,
                                  payload['learning_rate'],payload['learning_rate'],'cosine',0,True,Path(base['path'])/f'{task}.pt')
            _load_model.cache_clear();net,size=_load_model(task,folder,device)
            after=metric(net,task,valid,size,device,getattr(net,'point_configuration',None))
            report.update(validation_before=before,validation_after=after,passed=after<=before+max(.01,abs(before)*.05))
            reports.append(report);write_json(folder/'partial_fit_report.json',reports)
        if not reports:raise ValueError('No completely reviewed modules available for training')
        del net;_load_model.cache_clear();torch.cuda.empty_cache()
        store.update(job['id'],result={'version':version,'stage':'pipeline_validation','completed_modules':len(reports)})
        from .validation import compare
        pipeline=compare(base,folder,[r for r in originals if parts[r.relative_path]=='validation'],device)
        item=dict(version=version,path=str(folder),epochs=base['epochs'],status='validated' if all(r['passed'] for r in reports) and pipeline['passed'] else 'candidate',
                  parent=base['version'],protocol=str(folder/'protocol.json'),report=str(folder/'partial_fit_report.json'),
                  validation_scope='module losses and complete pipeline F1/AUC/coverage on inner validation; no test used')
        write_json(folder/'model_manifest.json',item);store.put('model',version,item)
        for row in rows:row['status']='consumed' if item['status']=='validated' else 'ready';store.put('queue',row['id'],row)
        store.update(job['id'],'completed',{'model':item,'trained_modules':[r['task'] for r in reports]})
    except Exception:
        for row in rows:row['status']='ready';store.put('queue',row['id'],row)
        raise
    finally:_load_model.cache_clear();torch.cuda.empty_cache()
