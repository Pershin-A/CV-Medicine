"""Reproducible landmark ablations and dedicated model training."""
import argparse,json,time,random,csv,shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_records,load_augmented_records,collate,LANDMARKS,DxaDataset
from .train import split_records
from .protocol import make_protocol,source_sampler
from .landmarks import PointsNet,PointDataset,point_loss
from .decoding import decode_heatmap
from .predict import _load_model
from .evaluate import with_ci
from dxa_project.augmentation.core import hip_position_ok

VARIANTS={
    'shared256':dict(mode='shared',size=256,crop='full',loss='heatmap'),
    'coordinate256':dict(mode='shared',size=256,crop='full',loss='coordinate'),
    'shared384':dict(mode='shared',size=384,crop='full',loss='heatmap'),
    'shared512':dict(mode='shared',size=512,crop='full',loss='heatmap'),
    'roi384':dict(mode='shared',size=384,crop='roi',loss='heatmap'),
    'independent256':dict(mode='independent',size=256,crop='full',loss='heatmap'),
}

def point_stats(outputs,targets,decoder='masked_argmax',legacy=False):
    maps=torch.sigmoid(outputs['spatial']).detach().cpu().numpy()
    logits=outputs['spatial'].detach().cpu().numpy()
    if legacy:maps=maps[:,:3]
    presence=torch.sigmoid(outputs['presence']).detach().cpu().numpy()
    rows=[]
    for n,t in enumerate(targets):
        r=t['record'];g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
        pred={};errors=[];misses=0;tp=fp=fn=0;in_padding=0;points=[]
        for k,key in enumerate(LANDMARKS):
            p,peak=decode_heatmap(maps[n,k],t['meta'],decoder,raw_logits=logits[n,k])
            is_present=presence[n,k]>=.5 and peak>=.35 and p is not None
            gt=g['hip']['landmarks'][key];visible=gt is not None
            if is_present and visible:tp+=1
            if is_present and not visible:fp+=1
            if not is_present and visible:fn+=1
            pred[key]=p if is_present else None
            if visible:
                error=float(np.hypot((p[0]-gt[0])*r.spacing_mm[1],(p[1]-gt[1])*r.spacing_mm[0])) if p is not None else None
                if error is not None:errors.append(error)
                if not is_present:misses+=1
                points.append({'name':key,'error_mm':error,'visible_prediction':is_present})
            py,px=np.unravel_index(maps[n,k].argmax(),maps[n,k].shape)
            m=t['meta'];dw=m.get('resized_width',round(m['width']*m['scale']));dh=m.get('resized_height',round(m['height']*m['scale']))
            in_padding+=not(m['left']<=px<m['left']+dw and m['top']<=py<m['top']+dh)
        predicted=json.loads(json.dumps(g));predicted['hip']['landmarks']=pred
        margin=.025*min(g['image_width'],g['image_height'])
        distances=[min(p[0],p[1],g['image_width']-1-p[0],g['image_height']-1-p[1])/margin for p in pred.values() if p is not None]
        score=max(1-float(presence[n].min()),1/(1+np.exp(np.clip((min(distances)-1)*5,-50,50))) if len(distances)==3 else 1.)
        rows.append({'study':r.study,'relative_path':r.relative_path,'truth':int(not hip_position_ok(g)),
                     'prediction':int(not hip_position_ok(predicted)),'score':float(score),'errors_mm':errors,
                     'points':points,'misses':misses,'tp_visible':tp,'fp_visible':fp,'fn_visible':fn,'raw_padding_peaks':int(in_padding)})
    return rows

def summarize(rows):
    errors=[e for r in rows for e in r['errors_mm']]
    visible=[p for r in rows for p in r['points']]
    per_point={k:[p['error_mm'] for p in visible if p['name']==k and p['error_mm'] is not None] for k in LANDMARKS}
    groups=sorted({r['study'] for r in rows});rng=np.random.default_rng(42);bootstrap=[]
    grouped={g:[r for r in rows if r['study']==g] for g in groups}
    for _ in range(200):
        sample=[r for g in rng.choice(groups,len(groups),replace=True) for r in grouped[g]] if groups else []
        sample_errors=[e for r in sample for e in r['errors_mm']];sample_points=[p for r in sample for p in r['points']]
        if sample_errors and sample_points:
            bootstrap.append([np.mean(sample_errors),sum(p['visible_prediction'] and p['error_mm'] is not None and p['error_mm']<=10 for p in sample_points)/len(sample_points)])
    ci={key:[float(x) for x in np.quantile(np.asarray(bootstrap)[:,i],[.025,.975])] if bootstrap else None
        for i,key in enumerate(('mean_error_mm','pck_10mm'))}
    return {'images':len(rows),'mean_error_mm':float(np.mean(errors)) if errors else None,'localization_ci95':ci,
            'median_error_mm':float(np.median(errors)) if errors else None,
            'pck_5mm':sum(p['visible_prediction'] and p['error_mm'] is not None and p['error_mm']<=5 for p in visible)/len(visible) if visible else None,
            'pck_10mm':sum(p['visible_prediction'] and p['error_mm'] is not None and p['error_mm']<=10 for p in visible)/len(visible) if visible else None,
            'mean_error_by_point_mm':{k:float(np.mean(v)) if v else None for k,v in per_point.items()},
            'missing_visible_points':sum(r['misses'] for r in rows),'raw_padding_peaks':sum(r['raw_padding_peaks'] for r in rows),
            'position':with_ci(rows,200)}

def evaluate(model,loader,device,decoder,loss_kind='heatmap',legacy=False):
    model.eval();rows=[];losses=[]
    with torch.no_grad():
        for images,targets in loader:
            outputs=model(images.to(device))
            if not legacy:losses.append(float(point_loss(outputs,targets,loss_kind)))
            rows.extend(point_stats(outputs,targets,decoder,legacy))
    return summarize(rows),rows,float(np.mean(losses)) if losses else None

def train_variant(config,train,valid,out,device,epochs=20,max_batches=None,encoder_lr=2e-5,head_lr=2e-4,scheduler='cosine',patience=7,warmup_epochs=0,validation_evaluator=None,mixed_precision=False):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    out.mkdir(parents=True,exist_ok=True)
    model=PointsNet(True,config['mode'],config.get('architecture','light')).to(device)
    datasets=[PointDataset(rows,config['size'],config['crop']) for rows in (train,valid)]
    batch=config.get('batch_size',2 if config['mode']=='independent' or config['size']==512 or config.get('architecture')=='heavy' else 8)
    loaders=[DataLoader(datasets[0],batch_size=batch,sampler=source_sampler(train),num_workers=2,collate_fn=collate,persistent_workers=True,pin_memory=True),
             DataLoader(datasets[1],batch_size=batch,num_workers=2,collate_fn=collate,persistent_workers=True,pin_memory=True)]
    encoder=[];heads=[]
    for name,p in model.named_parameters():
        (encoder if any(x in name for x in ('.stem.','.layer1.','.layer2.','.layer3.','.layer4.')) else heads).append(p)
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':encoder_lr},{'params':heads,'lr':head_lr}],weight_decay=1e-4)
    schedule=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs),eta_min=1e-6) if scheduler=='cosine'
              else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,factor=.5) if scheduler=='plateau' else None)
    if warmup_epochs:
        if scheduler!='cosine' or warmup_epochs>=epochs:raise ValueError('Warmup requires cosine and warmup_epochs < epochs')
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:(e+1)/warmup_epochs if e<warmup_epochs
              else .01+.99*(1+np.cos(np.pi*(e-warmup_epochs)/max(1,epochs-warmup_epochs)))/2)
    best=float('inf');stale=0;history=[];start=time.perf_counter()
    use_amp=bool(mixed_precision and device.type=='cuda' and torch.cuda.is_bf16_supported())
    for epoch in range(epochs):
        model.train();losses=[]
        for step,(images,targets) in enumerate(loaders[0]):
            if max_batches is not None and step>=max_batches:break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_amp):
                outputs={k:v.float() for k,v in model(images.to(device)).items()};loss=point_loss(outputs,targets,config['loss'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite point loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        if max_batches is not None:
            # Smoke remains short and cannot be confused with quality assessment.
            samples=[datasets[1][i] for i in range(min(4,len(datasets[1])))]
            images,targets=collate(samples);model.eval()
            with torch.no_grad():outputs=model(images.to(device))
            metrics=summarize(point_stats(outputs,targets,'masked_logit_argmax'));vl=float(point_loss(outputs,targets,config['loss']))
        elif validation_evaluator is not None:metrics,_,vl=validation_evaluator(model,config,device)
        else:metrics,_,vl=evaluate(model,loaders[1],device,'masked_logit_argmax',config['loss'])
        score=metrics.get('selection_loss',metrics['mean_error_mm'])
        checkpoint={'task':'hip_points','state_dict':model.state_dict(),'size':config['size'],'mode':config['mode'],
            'architecture':config.get('architecture','light'),'crop':config['crop'],'loss':config['loss'],'decoder':'masked_logit_argmax','epoch':epoch+1,'training_batch_size':batch}
        torch.save(checkpoint,out/'hip_points_last.pt')
        if score<best:
            best=score;stale=0;torch.save(checkpoint,out/'hip_points.pt')
        else:stale+=1
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':vl,'metrics':metrics,'learning_rates':[g['lr'] for g in optimizer.param_groups]})
        (out/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'variant':out.name,'epoch':epoch+1,'error_mm':metrics['mean_error_mm'],'selection_value':score,'validation_loss':vl,'seconds':time.perf_counter()-start}),flush=True)
        if schedule:
            if scheduler=='plateau':schedule.step(score)
            else:schedule.step()
        if patience and stale>=patience:break
    report={'config':config,'train_images':len(train),'validation_images':len(valid),'max_epochs':epochs,
            'actual_epochs':len(history),'smoke':max_batches is not None,'encoder_lr':encoder_lr,'head_lr':head_lr,'scheduler':scheduler,'warmup_epochs':warmup_epochs,
            'seconds':time.perf_counter()-start,'best_selection_value':best,'mixed_precision':'bf16 forward / fp32 weights, loss, validation' if use_amp else 'fp32',
            'selection_kind':'composite deployed validation' if validation_evaluator is not None else 'mean_error_mm',
            'best_inner_validation_error_mm':min(history,key=lambda h:h['metrics'].get('selection_loss',h['metrics']['mean_error_mm']))['metrics']['mean_error_mm'],'history':history}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def run(args):
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.hub.set_dir(str(root/'dxa_project/outputs/torch_hub'))
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    originals=load_records(root);oldtrain,oldtest=split_records(originals,0)
    protocol=make_protocol(root,originals,oldtrain,oldtest,out/'protocol.json')
    parts={k:[r for r in originals if protocol['partition_by_path'][r.relative_path]==k and r.region!='SPINE'] for k in ('train','validation','test')}
    aug=load_augmented_records(root,args.augmented_root.resolve(),originals)
    source_parts=protocol['partition_by_path']
    parts['train'] += [r for r in aug if r.region!='SPINE' and source_parts[r.source_id]=='train']
    if args.phase=='decode':
        model,_=_load_model('hip',args.baseline.resolve(),device)
        records=[r for r in oldtest if r.region!='SPINE']
        loader=DataLoader(PointDataset(records,256),batch_size=8,num_workers=2,collate_fn=collate)
        reports={}
        for method in ('legacy','masked_argmax','local_softargmax','masked_logit_argmax','local_logit_softargmax'):
            metrics,rows,_=evaluate(model,loader,device,method,legacy=True);reports[method]=metrics
            (out/f'decoder_{method}_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'decoder_comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(reports));return
    if args.phase in ('smoke','train'):
        for variant in args.variants.split(','):
            config={**VARIANTS[variant],'architecture':args.architecture}
            report=train_variant(config,parts['train'],parts['validation'],out/('smoke' if args.phase=='smoke' else 'trained')/variant,
                                 device,1 if args.phase=='smoke' else args.epochs,2 if args.phase=='smoke' else None,
                                 args.encoder_lr,args.head_lr,args.scheduler,args.patience,args.warmup_epochs)
        return
    if args.phase=='test':
        reports={}
        for variant in args.variants.split(','):
            folder=out/'trained'/variant;checkpoint_path=folder/'hip_points.pt'
            selection_path=out/'point_selection.json'
            if selection_path.exists():
                candidates=[x for x in json.loads(selection_path.read_text(encoding='utf-8'))['candidates'] if x['variant']==variant]
                if candidates:checkpoint_path=Path(min(candidates,key=lambda x:x['metrics']['mean_error_mm'])['path'])
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            model=PointsNet(False,checkpoint['mode'],checkpoint['architecture']).to(device);model.load_state_dict(checkpoint['state_dict'])
            # Crop checkpoints evaluated here with true ROI are explicitly oracle only.
            loader=DataLoader(PointDataset(parts['test'],checkpoint['size'],checkpoint['crop']),batch_size=4,num_workers=2,collate_fn=collate)
            metrics,rows,_=evaluate(model,loader,device,'masked_logit_argmax',checkpoint['loss'])
            reports[variant]={**metrics,'checkpoint':checkpoint_path.name,'epoch':checkpoint['epoch'],
                              'crop_scope':'oracle true ROI' if checkpoint['crop']=='roi' else 'full image','outer_scope':protocol['outer_scope']}
            (folder/'test_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'test_comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(reports))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,default=Path('dxa_project/outputs/improvements_v2'))
    p.add_argument('--augmented-root',type=Path,default=Path('dxa_project/outputs/augmented_15000_final'))
    p.add_argument('--baseline',type=Path,default=Path('dxa_project/outputs/geometry_ml_augmented_5epochs'))
    p.add_argument('--phase',choices=('decode','smoke','train','test'),required=True)
    p.add_argument('--variants',default='shared256');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--encoder-lr',type=float,default=2e-5);p.add_argument('--head-lr',type=float,default=2e-4)
    p.add_argument('--scheduler',choices=('none','cosine','plateau'),default='cosine');p.add_argument('--patience',type=int,default=7)
    p.add_argument('--warmup-epochs',type=int,default=0)
    p.add_argument('--architecture',choices=('light','heavy'),default='light')
    run(p.parse_args())
