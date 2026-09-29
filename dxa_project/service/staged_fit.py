"""Four ordered fine-tuning stages; independent human/model snapshots and augmentation."""
import copy,json,logging,math,shutil,time,uuid
from pathlib import Path
import numpy as np
import pydicom,torch
from torch.utils.data import DataLoader
from .store import PROJECT,allowed_path,write_json
from .schemas import TrainingSample
from .data_ops import import_image,annotate,image_hash,augment
from .training import model
from dxa_project.geometry_ml.data import Record,DxaDataset,collate,load_records
from dxa_project.geometry_ml.predict import _load_model
from dxa_project.geometry_ml.train import _loss,_subset,_validation_stats,_aggregate_stats
from dxa_project.geometry_ml.landmarks import PointDataset,point_loss

MODULES=('router','spine','spine_crests','scoliosis','hip','hip_mask','hip_points','artifact')
TARGETS={'SPINE':('spine_position','spine_axis','spine_artifact','spine_scoliosis'),
         'LEG_LEFT':('hip_position','hip_roi','hip_rotation'),'LEG_RIGHT':('hip_position','hip_roi','hip_rotation')}
logger=logging.getLogger('dxa.training')

def event(store,job_id,stage,**details):
    e=dict(time=time.time(),stage=stage,**details)
    message=f'Набор {stage}'
    if 'module' in e:message+=f', модуль {e["module"]}'
    if 'epoch' in e:message+=f': эпоха {e["epoch"]}/{e["epochs"]}'
    e['message']=message;logger.info(message)
    path=store.root/'jobs'/job_id/'training.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    store.update(job_id,result={'stage':stage,**details})

def embedded(path,prefer_manual=False):
    ds=pydicom.dcmread(path,stop_before_pixels=True)
    has_manual=False
    if prefer_manual:
        try:
            b=ds.private_block(0x0011,'DXA_MANUAL_LABELER',create=False)
            has_manual=b.get_tag(9) in ds
        except KeyError:pass
    try:
        if has_manual:raise KeyError('Prefer reviewed manual annotation')
        b=ds.private_block(0x0011,'DXA_MODEL_V1',create=False)
        p=json.loads(ds[b.get_tag(1)].value)['prediction']
        return dict(region=p['region'],geometry=json.loads(ds[b.get_tag(2)].value),targets=p['quality_flags'],
                    reviewed=['spine','artifact','spine_crests','scoliosis'] if p['region']=='SPINE' else ['hip','hip_mask','hip_points'])
    except KeyError:pass
    try:
        b=ds.private_block(0x0011,'DXA_MANUAL_LABELER',create=False)
        region=str(ds[b.get_tag(1)].value)
        if region=='LEG':region='LEG_'+str(ds[b.get_tag(5)].value)
        g=json.loads(ds[b.get_tag(9)].value);targets={}
        if b.get_tag(10) in ds:targets=json.loads(ds[b.get_tag(10)].value)
        if b.get_tag(8) in ds and region=='SPINE':
            issue=str(ds[b.get_tag(8)].value)
            if issue in ('NONE','SCOLIOSIS','LUMBARIZATION'):targets['spine_scoliosis']=int(issue=='SCOLIOSIS')
        return dict(region=region,geometry=g,targets={k:v for k,v in targets.items() if k in TARGETS.get(region,())},
                    reviewed=['spine','artifact','spine_crests'] if region=='SPINE' else ['hip','hip_mask','hip_points'])
    except KeyError:raise ValueError('Provide region/geometry annotation or supported private DICOM annotation')

def validate_datasets(store,base,payload):
    if payload.get('human') is None or payload.get('model') is None:raise ValueError('Both human and model datasets are required')
    if payload.get('queue_ids') is not None:raise ValueError('Do not mix queue_ids with staged datasets')
    try:originals=load_records(PROJECT)
    except FileNotFoundError as e:raise ValueError('partial_fit requires original training/validation data, labels and manifest; see API_GUIDE') from e
    parts=json.loads(Path(base['protocol']).read_text(encoding='utf-8'))['partition_by_path']
    # Pixels are checked as well as study ownership, including anonymous copied DICOMs.
    pixel_parts={image_hash(pydicom.dcmread(r.source_path).pixel_array):parts[r.relative_path] for r in originals}
    owned={r['id']:r['partition'] for r in store.all('study')};seen=set();out=copy.deepcopy(payload)
    for kind in ('human','model'):
        dataset=payload[kind];items=list(dataset['images'])
        if dataset.get('manifest_path'):
            items+=json.loads(allowed_path(dataset['manifest_path']).read_text(encoding='utf-8'))
        if not items:raise ValueError(f'{kind} dataset is empty')
        if len(items)>10000:raise ValueError('Dataset exceeds 10000 files')
        rows=[]
        for raw in items:
            s=TrainingSample.model_validate(raw).model_dump()
            if bool(s.get('image_id'))==bool(s.get('path')):raise ValueError('Specify exactly one of image_id or path')
            if s.get('image_id'):
                im=store.get('image',s['image_id'])
                if s.get('annotation_version') is None:raise ValueError('annotation_version required with image_id')
                a=store.get('annotation',f'{im["image_id"]}:{s["annotation_version"]}')
            else:
                path=allowed_path(s['path']);im=import_image(store,path)
                meta=embedded(path,prefer_manual=kind=='human') if s.get('geometry') is None and not s.get('geometry_path') else {}
                g=s.get('geometry') or (json.loads(allowed_path(s['geometry_path']).read_text(encoding='utf-8')) if s.get('geometry_path') else meta.get('geometry'))
                region=s.get('region') or meta.get('region')
                if region not in TARGETS:raise ValueError('Annotation region required')
                targets=s.get('targets') or meta.get('targets',{})
                # An explicitly supplied full geometry is the same training contract as labeler JSON.
                reviewed=s.get('reviewed') or meta.get('reviewed') or (['spine','artifact','spine_crests'] if region=='SPINE' else ['hip','hip_mask','hip_points'])
                if targets.get('spine_scoliosis') in (0,1) and 'scoliosis' not in reviewed:reviewed=reviewed+['scoliosis']
                a=annotate(store,im['image_id'],dict(region=region,geometry=g,reviewed=reviewed,targets=targets,
                    spacing_mm=s.get('spacing_mm'),expected_version=im['annotation_version'],mask_result_id=None))
            if pixel_parts.get(im['pixel_hash'],'train')!='train' or owned.get(im['study'],'train')!='train':
                raise ValueError('Validation/test image or study cannot enter partial_fit')
            if im['pixel_hash'] in seen:raise ValueError('Duplicate pixels within/across human and model datasets; keep one authoritative annotation')
            seen.add(im['pixel_hash'])
            section='spine' if a['region']=='SPINE' else 'hip'
            if section not in a['reviewed']:raise ValueError('Provide complete reviewed geometric annotation')
            rows.append(dict(id=uuid.uuid4().hex,image_id=im['image_id'],annotation_version=a['version'],
                path=im['path'],geometry_path=a['geometry_path'],region=a['region'],reviewed=a['reviewed'],
                labels=a['targets'],spacing_mm=a['spacing_mm'],study=im['study'],source_id=im['image_id'],pixel_hash=im['pixel_hash']))
        out[kind]={'rows':rows}
    for task in MODULES:
        if (Path(base['path'])/(task+'.pt')).exists() and task not in base['epochs']:
            raise ValueError(f'Missing saved original training epoch budget for {task}; register training_config.json')
    return out

def snapshot(store,job,kind):
    root=store.root/'jobs'/job['id']/'output';rows=[]
    for raw in job['payload'][kind]['rows']:
        r=dict(raw);folder=root/'originals'/kind/r['id'];folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['path'],folder/'image.dcm');shutil.copy2(r['geometry_path'],folder/'geometry.json')
        r.update(path=str(folder/'image.dcm'),geometry_path=str(folder/'geometry.json'));write_json(folder/'annotation.json',r)
        from .inference import render_overlay
        from dxa_project.geometry_ml.data import read_dicom_image
        render_overlay(read_dicom_image(Path(r['path'])),json.loads(Path(r['geometry_path']).read_text(encoding='utf-8')),
                       root/'annotated'/kind/r['id']/'overlay.png')
        store.put('study',r['study'],{'id':r['study'],'partition':'train'});rows.append(r)
    (root/'temporary/augmented'/kind).mkdir(parents=True,exist_ok=True)
    return rows

def augment_set(store,job,kind,rows):
    cfg=job['payload']['augmentation'];root=store.root/'jobs'/job['id']/'output/temporary/augmented'/kind
    augmented=[];reports=[]
    for i,row in enumerate(rows):
        supported=('spine_position','spine_axis') if row['region']=='SPINE' else ('hip_position','hip_roi')
        child=dict(id=job['id'],payload=dict(image_id=row['image_id'],annotation_version=row['annotation_version'],
            output_directory=str(root/row['id']),strict_targets=True,manage_job=False,
            config=dict(positive_count=cfg['n_pp'],negative_count_by_target={k:cfg['n_pn'] for k in supported},
                negative_source_count=cfg['n_nn'],seed=cfg['seed']+i,max_attempts_per_sample=cfg['max_attempts_per_sample'])))
        report=augment(store,child,False);reports.append(report)
        for p in sorted((root/row['id']).glob('*/metadata.json')):
            r=json.loads(p.read_text(encoding='utf-8'));r['labels']['spine_scoliosis']=row['labels'].get('spine_scoliosis') if row['region']=='SPINE' else None
            if row['labels'].get('spine_scoliosis')==1:
                g=json.loads(Path(r['geometry_path']).read_text());original=json.loads(Path(row['geometry_path']).read_text())
                if len(g['spine']['disc_lines'])!=len(original['spine']['disc_lines']):r['labels']['spine_scoliosis']=None
            write_json(p,r);augmented.append(r)
    write_json(root/'augmentation_report.json',reports)
    if any(any(k.startswith('Unmet quota') for k in r['rejected']) for r in reports):
        raise ValueError(f'Augmentation quota not met in {kind}; inspect {root}/augmentation_report.json')
    return augmented

def selected(rows,task):
    if task=='router':return rows
    key={'hip_mask':'hip','hip_points':'hip','spine_crests':'spine','scoliosis':'scoliosis'}.get(task,task)
    return [r for r in rows if (r['region']=='SPINE')==(task in ('spine','spine_crests','scoliosis','artifact'))
            and (key in r['reviewed'] or task in r['reviewed']) and (task!='scoliosis' or r['labels'].get('spine_scoliosis') in (0,1))]

def records(rows):return [Record('service/'+r['id'],Path(r['path']),Path(r['geometry_path']),r['study'],r['region'],tuple(r['spacing_mm']),r['source_id']) for r in rows]

def loss_for(task,net,x,ts,device,labels,spine_loss=None):
    if task=='hip_points':return point_loss(net(x.to(device)),ts,net.point_configuration['loss'] or 'heatmap')
    if task=='scoliosis':
        y=torch.tensor([labels[t['relative_path']]['spine_scoliosis'] for t in ts],device=device,dtype=torch.float32)
        return torch.nn.functional.binary_cross_entropy_with_logits(net(x.to(device)),y)
    if task=='hip_mask':
        from dxa_project.geometry_ml.models import dice_bce
        return dice_bce(net(x.to(device))['spatial'][:,3:],torch.stack([t['trochanter'] for t in ts]).to(device),5)
    if task=='spine_crests':
        from dxa_project.geometry_ml.final_crests import point_loss as crest_loss
        return crest_loss('spine',net,x,ts,device)[0]
    if task=='spine' and spine_loss is not None:return spine_loss(task,net,x,ts,device)[0]
    return _loss(task,net,x,ts,device)[0]

def validation_score(task,net,loader,device,labels,spine_loss=None):
    net.eval();values=[];stats=[]
    with torch.no_grad():
        for x,ts in loader:
            if task=='artifact':
                from dxa_project.geometry_ml.models import detector_images
                stats.append(_validation_stats(task,net(detector_images(x.to(device))),ts,0))
            else:values.append(float(loss_for(task,net,x,ts,device,labels,spine_loss)))
    if task=='artifact':
        m=_aggregate_stats(stats,task);p=m['box_precision_iou50'];r=m['box_recall_iou50'];return -2*p*r/max(1e-9,p+r)
    return float(np.mean(values))

def train_stage(store,job,stage,rows,folder,base,originals,parts,device):
    reports=[]
    # Fixed inner validation and original labels; never train on these records.
    from dxa_project.geometry_ml.final_protocol import labels_for_originals
    original_labels=labels_for_originals(PROJECT,originals)
    for task in MODULES:
        if not (folder/(task+'.pt')).exists():continue
        eligible=selected(rows,task)
        if not eligible:event(store,job['id'],stage,module=task,status='skipped_no_labeled_examples');continue
        train=records(eligible);valid=_subset([r for r in originals if parts[r.relative_path]=='validation'],'hip' if task in ('hip_points','hip_mask') else 'spine' if task in ('spine_crests','scoliosis') else task,False)
        if task=='scoliosis':valid=[r for r in valid if original_labels[r.relative_path].get('spine_scoliosis') in (0,1)]
        if not valid:raise ValueError(f'No inner validation for {task}')
        labels={**original_labels,**{'service/'+r['id']:r['labels'] for r in eligible}}
        _load_model.cache_clear();torch.cuda.empty_cache();net,size=_load_model(task,folder,device)
        config=getattr(net,'point_configuration',None)
        ds=lambda rs:PointDataset(rs,size,config['crop']) if task=='hip_points' else DxaDataset(rs,size,router_mode=task=='router')
        tl=DataLoader(ds(train),batch_size=2,shuffle=True,collate_fn=collate);vl=DataLoader(ds(valid),batch_size=2,collate_fn=collate)
        spine_loss=None
        if task=='spine':
            from dxa_project.geometry_ml.final_spine import SpineLoss
            spine_loss=SpineLoss(train+valid,c=2.)
        before=validation_score(task,net,vl,device,labels,spine_loss)
        checkpoint=torch.load(folder/(task+'.pt'),map_location='cpu',weights_only=False)
        epochs=base['epochs'][task];opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=job['payload']['learning_rate'],weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=min(1e-6,job['payload']['learning_rate']))
        best=math.inf;history=[]
        for epoch in range(1,epochs+1):
            event(store,job['id'],stage,module=task,epoch=epoch,epochs=epochs,status='started')
            net.train();losses=[]
            for x,ts in tl:
                opt.zero_grad(set_to_none=True);loss=loss_for(task,net,x,ts,device,labels,spine_loss)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            score=validation_score(task,net,vl,device,labels,spine_loss);scheduler.step()
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_score=score))
            if score<best:
                best=score;checkpoint['state_dict']={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};checkpoint['partial_fit_stage']=stage
                torch.save(checkpoint,folder/(task+'.pt'))
            write_json(folder/f'{stage}_{task}_history.json',history)
            event(store,job['id'],stage,module=task,epoch=epoch,epochs=epochs,status='completed',validation_score=score)
        reports.append(dict(stage=stage,module=task,epochs=epochs,examples=len(train),validation_before=before,validation_best=best,history=history))
        del net;_load_model.cache_clear();torch.cuda.empty_cache()
    return reports

def run(store,job,device):
    base=model(store,job['payload']['base_model_version']);folder=store.root/'models'/job['id']
    if folder.exists():raise ValueError('Interrupted candidate exists; submit a new request_id')
    shutil.copytree(base['path'],folder)
    originals=load_records(PROJECT);parts=json.loads(Path(base['protocol']).read_text())['partition_by_path'];reports=[]
    human=snapshot(store,job,'human');machine=snapshot(store,job,'model')
    for stage,rows in (('human_original',human),('model_original',machine)):
        event(store,job['id'],stage,examples=len(rows));reports+=train_stage(store,job,stage,rows,folder,base,originals,parts,device)
        write_json(folder/'partial_fit_report.json',reports)
    event(store,job['id'],'augmentation',dataset='human');ha=augment_set(store,job,'human',human)
    event(store,job['id'],'augmentation',dataset='model');ma=augment_set(store,job,'model',machine)
    for stage,rows in (('human_augmented',ha),('model_augmented',ma)):
        if rows:reports+=train_stage(store,job,stage,rows,folder,base,originals,parts,device)
        else:event(store,job['id'],stage,status='skipped_zero_augmentation_quota')
        write_json(folder/'partial_fit_report.json',reports)
    from .validation import compare
    event(store,job['id'],'validation');pipeline=compare(base,folder,[r for r in originals if parts[r.relative_path]=='validation'],device)
    item=dict(version=job['id'],path=str(folder),epochs=base['epochs'],parent=base['version'],protocol=str(folder/'protocol.json'),
              status='validated' if pipeline['passed'] else 'candidate',report=str(folder/'partial_fit_report.json'))
    write_json(folder/'model_manifest.json',item);store.put('model',job['id'],item)
    event(store,job['id'],'complete',version=job['id'])
    root=store.root/'jobs'/job['id']/'output'
    store.update(job['id'],'completed',{'model':item,'stages':['human_original','model_original','human_augmented','model_augmented'],
        'reports':reports,'output_directory':str(root),'temporary_directory':str(root/'temporary/augmented'),
        'activation':'explicit POST /v1/model/versions/{version}/activate; current active weights preserved'})
