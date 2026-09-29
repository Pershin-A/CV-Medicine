"""Isolated controlled study of global spine angle, all peaks and strict terminals."""
import argparse,copy,csv,json,math,time,warnings
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import find_peaks
from .data import load_records,load_augmented_records,DxaDataset,collate,read_dicom_image
from .models import SpatialNet,dice_bce
from .predict import _spine_lines
from .spine_penalty_study import target,measure,summarize
from .spine_brightness_study import quadratic_priors
from .spine_angle_geometry import analyze_strict,strict_terminals,bisect_dividers,axis_from_frame_x
from .experimental_spine_axes import analyze_frame_axes
from .evaluate import binary_metrics

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'dxa_project/outputs/spine_angle_study_20260929'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


class AngleSpatial(nn.Module):
    def __init__(self,state):
        super().__init__();self.base=SpatialNet('SPINE',False,'resnet50');self.base.load_state_dict(state)
        self.base.layer4.register_forward_hook(self._hook)
        self.axis_head=nn.Sequential(nn.LayerNorm(2048*8+8),nn.Linear(2048*8+8,128),nn.ReLU(),nn.Linear(128,2))
    def _hook(self,module,inputs,output):self.encoded=output
    def forward(self,x):
        out=self.base(x)
        features=torch.cat([F.adaptive_avg_pool2d(self.encoded,(4,2)).flatten(1),F.adaptive_avg_pool2d(torch.sigmoid(out['spatial'][:,:1]),(4,2)).flatten(1)],1)
        out['frame_x']=.5+torch.tanh(self.axis_head(features))
        return out


def frame_angle_loss(frame_x,targets):
    eligible=[i for i,t in enumerate(targets) if t.get('reference_angle') is not None]
    zero=frame_x.sum()*0
    if not eligible:return zero,zero
    x=frame_x[eligible];t=[targets[i] for i in eligible]
    aspect=x.new_tensor([(a['width']-1)*a['spacing_mm'][1]/((a['height']-1)*a['spacing_mm'][0]) for a in t])
    angles=torch.atan2((x[:,1]-x[:,0])*aspect,torch.ones_like(aspect))*180/math.pi
    reference=x.new_tensor([a['reference_angle'] for a in t])
    error=torch.atan2(torch.sin((angles-reference)*math.pi/180),torch.cos((angles-reference)*math.pi/180))*180/math.pi
    endpoint=F.smooth_l1_loss(x,x.new_tensor([a['reference_frame_x'] for a in t]),beta=.05)
    return (error/5).square().mean(),endpoint


def all_peak_priors(logits,targets,c):
    """Candidate finding is discrete; local peak coordinates/confidence retain gradients.

    Every detected candidate (including extras) enters the penalties. Geometry
    terms are relative errors, unlike the tiny squared normalized gaps before.
    Missing-end constraints follow the labeled coverage, never an ideal fixed
    vertebra count. Widest GT gap prevents forcing equal spacing on real defects.
    """
    values={k:[] for k in ['close','holes','tails']}
    for i,t in enumerate(targets):
        gt=t['ordered_lines'].to(logits.device);left,top=t['pad_left'],t['pad_top']
        dw=round(t['width']*t['scale']);dh=round(t['height']*t['scale'])
        profile=torch.sigmoid(logits[i,0,top:top+dh,left+dw//4:left+3*dw//4]).mean(1)
        indices=find_peaks(profile.detach().cpu().numpy(),prominence=.03,distance=3)[0]
        grid=torch.linspace(0,1,dh,device=logits.device);ys=[];ps=[]
        for index in indices:
            lo=max(0,index-3);hi=min(dh,index+4);q=profile[lo:hi]
            mass=torch.softmax(q/.04,0);ys.append((mass*grid[lo:hi]).sum())
            ps.append((mass*q).sum())
        zero=profile.sum()*0
        if len(ys)<2:
            for key in values:values[key].append(zero)
            continue
        y=torch.stack(ys);p=torch.stack(ps);gaps=y[1:]-y[:-1];weight=p[:-1]*p[1:]
        reference=torch.diff(gt[:,0]);mean=reference.mean().clamp_min(.01)
        minimum=1/(1/mean+c);maximum=torch.maximum(1.5*mean,reference.max())
        values['close'].append((F.relu((minimum-gaps)/minimum).square()*weight).mean())
        values['holes'].append((F.relu((gaps-maximum)/mean).square()*weight).mean())
        values['tails'].append(F.relu((y[0]-gt[0,0])/mean).square()*p[0]+F.relu((gt[-1,0]-y[-1])/mean).square()*p[-1])
    return {k:torch.stack(v).mean() for k,v in values.items()}


def angle_summary(rows,bootstrap=200):
    if not rows:return {'n':0,'defined':0,'coverage':None,'mae_deg':None}
    usable=[r for r in rows if r['reference_angle'] is not None and r['angle'] is not None]
    reference=[r for r in rows if r['reference_angle'] is not None]
    errors=[abs(r['angle']-r['reference_angle']) for r in usable]
    score=sum(abs(r['angle']-r['reference_angle']) if r['angle'] is not None else 15. for r in reference)/max(1,len(reference))
    binary=[{'truth':r['flag'],'prediction':int(abs(r['angle'])>5),'score':abs(r['angle'])} for r in rows if r['angle'] is not None and r.get('flag') is not None]
    result={'n':len(rows),'reference_defined':len(reference),'defined':len(usable),'coverage':len(usable)/max(1,len(reference)),
            'mae_deg':float(np.mean(errors)) if errors else None,'median_error_deg':float(np.median(errors)) if errors else None,
            'rmse_deg':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'p95_error_deg':float(np.percentile(errors,95)) if errors else None,
            'within_1deg':float(np.mean(np.array(errors)<=1)) if errors else None,
            'failure_penalized_mae_deg':float(score),
            'binary_existing_labels':binary_metrics(binary) if binary else None,
            'originals':len([r for r in rows if not r['relative_path'].startswith('aug/')])}
    if bootstrap and reference:
        groups={s:[r for r in reference if r['study']==s] for s in sorted({r['study'] for r in reference})};ids=list(groups);rng=np.random.default_rng(42);draw=[]
        for _ in range(bootstrap):
            sample=[r for j in rng.integers(0,len(ids),len(ids)) for r in groups[ids[j]]]
            e=[abs(r['angle']-r['reference_angle']) for r in sample if r['angle'] is not None]
            draw.append([float(np.mean(e)) if e else 15.,sum(abs(r['angle']-r['reference_angle']) if r['angle'] is not None else 15. for r in sample)/len(sample)])
        result['ci95']={k:np.percentile(np.asarray(draw)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['mae_deg','failure_penalized_mae_deg'])}
    return result


def run(epochs=4):
    DEST.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('Controlled spatial study requires the available CUDA runtime')
    previous=json.loads((ROOT/'dxa_project/outputs/spine_brightness_study_20260929/protocol.json').read_text(encoding='utf-8'))
    originals=load_records(ROOT);records=originals+load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',originals)
    by={r.relative_path:r for r in records};groups={k:[by[p] for p in paths] for k,paths in previous['partitions'].items()}
    assert not any({r.study for r in groups[a]}&{r.study for r in groups[b]} for a,b in [('train','validation'),('train','test'),('validation','test')])
    checkpoint=ROOT/'dxa_project/outputs/retrained_20260929/heavy_stratified/spine.pt'
    initial=torch.load(checkpoint,map_location='cpu',weights_only=False);size=initial['size']
    flags={}
    for name,prefix in [('manifest.csv',''),('augmented_15000_20260929/manifest.csv','aug/')]:
        with (ROOT/'dxa_project/outputs'/name).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                path=r['relative_path'] if not prefix else prefix+r['image_path'];v=r.get('spine_axis','')
                flags[path]=int(float(v)) if v not in ('',None) else None
    cache={};references={};reference_details={};raws={};geometries={}
    for part,rs in groups.items():
        ds=DxaDataset(rs,size);items=[]
        for i,r in enumerate(rs):
            geometry=json.loads(r.geometry_path.read_text(encoding='utf-8'));dcm=pydicom.dcmread(r.source_path);raw=np.squeeze(dcm.pixel_array)
            geometries[r.relative_path]=geometry;raws[r.relative_path]=raw
            x,t=ds[i];coords,visible=target(r);t['ordered_lines']=torch.tensor(coords[:int(visible.sum())])
            a=analyze_strict(raw,geometry,r.spacing_mm,'dark' if dcm.PhotometricInterpretation=='MONOCHROME1' else 'bright')
            angle=a['global_angle_deg'];t['reference_angle']=angle
            if angle is not None:
                pa,pb=np.asarray(a['global_axis_points']);dxdy=(pb[0]-pa[0])/(pb[1]-pa[1]);h=geometry['image_height']-1;w=geometry['image_width']-1
                t['reference_frame_x']=[float((pa[0]-pa[1]*dxdy)/w),float((pa[0]+(h-pa[1])*dxdy)/w)]
            else:t['reference_frame_x']=[.5,.5]
            references[r.relative_path]=angle;reference_details[r.relative_path]={'angle':angle,'review_required':a['review_required'],'axis_points':a['global_axis_points'],'flag':flags.get(r.relative_path)}
            items.append((x,t))
            if (i+1)%50==0:print(json.dumps({'stage':'reference_geometry','part':part,'processed':i+1,'total':len(rs)}),flush=True)
        cache[part]=items
    configs={'control_c2':{'all':False,'c':2.,'angle':False},'all_peaks_c2':{'all':True,'c':2.,'angle':False},'all_peaks_c05':{'all':True,'c':.5,'angle':False},'angle_c2':{'all':True,'c':2.,'angle':True},'angle_c05':{'all':True,'c':.5,'angle':True}}
    protocol={'sizes':{k:len(v) for k,v in groups.items()},'partitions':previous['partitions'],'epochs':epochs,'configs':configs,'initial_checkpoint':str(checkpoint),'device':str(device),'selection':'validation angle MAE +15deg per undefined reference-eligible prediction; geometry angle for non-head variants, learned global axis for head variants','reference':'strict geometry from manual dividers, pseudo-reference not independent radiologist axis','data_same_as_previous_brightness':True,'test_used_for_selection':False}
    save(DEST/'protocol.json',protocol);save(DEST/'reference_axes.json',reference_details)
    audits={}
    for part,items in cache.items():
        audits[part]={}
        for c in [2.,.5]:
            conflicts=[]
            for _,t in items:
                ys=t['ordered_lines'][:,0].numpy();g=np.diff(ys);minimum=1/(1/g.mean()+c)
                if np.any(g<minimum):conflicts.append(t['relative_path'])
            audits[part][str(c)]={'conflicts':len(conflicts),'n':len(items),'original_conflicts':sum(not p.startswith('aug/') for p in conflicts)}
    save(DEST/'prior_audit.json',audits)
    results={};all_predictions={};validation_outputs={}
    def evaluate(model,part,config,keep_axes=False):
        model.eval();rows=[];line_rows=[];pred={};head=[]
        with torch.no_grad():
            for offset in range(0,len(groups[part]),8):
                ids=list(range(offset,min(offset+8,len(groups[part]))));x,t=collate([cache[part][i] for i in ids]);output=model(x.to(device))
                maps=torch.sigmoid(output['spatial'][:,0]).cpu().numpy();xp=output['frame_x'].cpu().numpy()
                for j,index in enumerate(ids):
                    r=groups[part][index];a=t[j];meta={'width':a['width'],'height':a['height'],'left':a['pad_left'],'top':a['pad_top'],'scale':a['scale'],'size':size,'flipped':False}
                    lines=_spine_lines(maps[j],meta);pred[r.relative_path]=lines;line_rows.append(measure(lines,r))
                    g=copy.deepcopy(geometries[r.relative_path]);g['spine']['disc_lines']=lines
                    axes=analyze_strict(raws[r.relative_path],g,r.spacing_mm)
                    row={'relative_path':r.relative_path,'study':r.study,'reference_angle':references[r.relative_path],'angle':axes['global_angle_deg'],'flag':flags.get(r.relative_path),'review_required':axes['review_required']}
                    ys=[]
                    for line in lines:
                        pa,pb=sorted(line['points'],key=lambda z:z[0]);ys.append((pa[1]+((a['width']-1)/2-pa[0])*(pb[1]-pa[1])/max(pb[0]-pa[0],1e-6))/(a['height']-1))
                    gap=np.diff(sorted(ys));gt=np.diff(a['ordered_lines'][:,0].numpy());minimum=1/(1/gt.mean()+config['c'])
                    row['close']=bool(np.any(gap<minimum)) if len(gap) else False
                    row['holes']=bool(np.any(gap>max(1.5*gt.mean(),gt.max()))) if len(gap) else True
                    if keep_axes:row['axes']=axes
                    rows.append(row)
                    angle,points=axis_from_frame_x(xp[j],a['width'],a['height'],r.spacing_mm)
                    head.append({**row,'angle':angle,'axis_points':points})
        stats=angle_summary(head if config['angle'] else rows,bootstrap=0)
        return {'angle':stats,'geometry_angle':angle_summary(rows,bootstrap=0),'lines':summarize(line_rows),'close_images':sum(z['close'] for z in rows),'hole_images':sum(z['holes'] for z in rows)},rows,head,pred,line_rows
    for name,config in configs.items():
        if (DEST/(name+'_result.json')).exists():
            results[name]=json.loads((DEST/(name+'_result.json')).read_text(encoding='utf-8'));print(json.dumps({'resumed':name}),flush=True);continue
        torch.manual_seed(42);model=AngleSpatial(initial['state_dict']).to(device)
        for key,p in model.named_parameters():p.requires_grad_(key.startswith(('base.up','base.spatial.')) or (config['angle'] and key.startswith('axis_head')))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        gen=torch.Generator().manual_seed(42);best=math.inf;history=[];best_epoch=None
        for epoch in range(epochs):
            model.train()
            for m in [model.base.stem,model.base.layer1,model.base.layer2,model.base.layer3,model.base.layer4]:m.eval()
            order=torch.randperm(len(cache['train']),generator=gen).tolist();losses=[];components=[]
            for offset in range(0,len(order),8):
                x,t=collate([cache['train'][i] for i in order[offset:offset+8]]);output=model(x.to(device));truth=torch.stack([a['line'] for a in t]).to(device)
                old=quadratic_priors(output['spatial'][:,:1],t)
                loss=dice_bce(output['spatial'][:,:1],truth,3)+2*old['count']+2*old['coordinate']
                pieces={k:float(v.detach()) for k,v in old.items()}
                if config['all']:
                    p=all_peak_priors(output['spatial'][:,:1],t,config['c']);loss=loss+8*p['close']+4*p['holes']+4*p['tails'];pieces.update({k:float(v.detach()) for k,v in p.items()})
                else:loss=loss+12*(old['close']+old['ends'])
                if config['angle']:
                    angular,endpoint=frame_angle_loss(output['frame_x'],t);loss=loss+2*angular+8*endpoint;pieces.update(angle=float(angular.detach()),endpoint=float(endpoint.detach()))
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss in {name}')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                if offset%160==0:print(json.dumps({'stage':'training','variant':name,'epoch':epoch+1,'batch_offset':offset}),flush=True)
            stats,_,_,_,_=evaluate(model,'validation',config)
            score=stats['angle']['failure_penalized_mae_deg'];history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':stats,'penalties':{k:float(np.mean([p[k] for p in components])) for k in components[0]} if components else {},'last_batch_penalties':pieces})
            if score<best:
                best=score;best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'size':size,'config':config,'best_epoch':best_epoch,'reference':'manual strict geometry'},DEST/(name+'.pt'))
            save(DEST/(name+'_history.json'),history);print(json.dumps({'stage':'validation','variant':name,'epoch':epoch+1,'stats':stats}),flush=True)
        model.load_state_dict(torch.load(DEST/(name+'.pt'),map_location=device,weights_only=False)['state_dict'])
        stats,rows,head,pred,line_rows=evaluate(model,'test',config,True)
        val,vrows,vhead,vpred,_=evaluate(model,'validation',config)
        results[name]={'config':config,'best_epoch':best_epoch,'validation':val,'test':stats,'test_angles':angle_summary(head if config['angle'] else rows),'geometry_angles':angle_summary(rows),
                      'original_angles':angle_summary([z for z in (head if config['angle'] else rows) if not z['relative_path'].startswith('aug/')]),
                      'synthetic_angles':angle_summary([z for z in (head if config['angle'] else rows) if z['relative_path'].startswith('aug/')]),
                      'rows':rows,'head_rows':head,'line_rows':line_rows,'history':history,'test_lines':pred,'validation_lines':vpred,'validation_rows':vrows}
        save(DEST/(name+'_result.json'),results[name]);save(DEST/'progress_results.json',{k:{a:b for a,b in v.items() if a not in ['rows','head_rows','history','line_rows','test_lines','validation_lines','validation_rows']} for k,v in results.items()})
        del model;torch.cuda.empty_cache();print(json.dumps({'stage':'variant_complete','variant':name,'angle':results[name]['test_angles']}),flush=True)
    finish(results,groups,raws,geometries,references,flags,start,protocol,audits)


def finish(results,groups,raws,geometries,references,flags,start,protocol,audits):
    # Geometry ablation on the *same* selected checkpoints and same reference.
    comparisons={};diagnostic_rows={};bisector_predictions={}
    previous=json.loads((ROOT/'dxa_project/outputs/spine_brightness_study_20260929/predicted_lines.json').read_text(encoding='utf-8'))
    methods={k:v['test_lines'] for k,v in results.items()};methods.update({'previous_raw_old_loss':previous['raw_old_loss'],'previous_raw_new_loss':previous['raw_new_loss']})
    for name,pred in methods.items():
        rows=[];legacy=[];rebuilt=[];rejected=0;equal_error=[];bisector_predictions[name]={}
        for i,r in enumerate(groups['test']):
            g=copy.deepcopy(geometries[r.relative_path]);g['spine']['disc_lines']=pred[r.relative_path]
            a=analyze_strict(raws[r.relative_path],g,r.spacing_mm)
            old=analyze_frame_axes(raws[r.relative_path],g,r.spacing_mm,neighbor_scale=.25,weight_basis='length')
            base={'relative_path':r.relative_path,'study':r.study,'reference_angle':references[r.relative_path],'flag':flags.get(r.relative_path)}
            rows.append({**base,'angle':a['global_angle_deg']});legacy.append({**base,'angle':old['global_angle_deg']})
            new,info=bisect_dividers(g,a,r.spacing_mm)
            if new is None:
                rejected+=1;rebuilt.append({**base,'angle':None,'status':info['status']})
            else:
                b=analyze_strict(raws[r.relative_path],new,r.spacing_mm);rebuilt.append({**base,'angle':b['global_angle_deg'],'status':info['status']})
                equal_error.append(info['max_equal_angle_error_deg']);bisector_predictions[name][r.relative_path]=new['spine']['disc_lines']
            if (i+1)%40==0:print(json.dumps({'stage':'terminal_and_bisector_ablation','variant':name,'processed':i+1}),flush=True)
        comparisons[name]={'weighted_terminals':angle_summary(legacy),'strict_terminals':angle_summary(rows),'strict_plus_bisectors_refit':angle_summary(rebuilt),'bisector_rejections':rejected,'max_equal_angle_error_deg':max(equal_error,default=None)}
        diagnostic_rows[name]={'weighted':legacy,'strict':rows,'bisectors':rebuilt}
        save(DEST/'geometry_comparison_progress.json',comparisons)
    winner=min(results,key=lambda k:results[k]['validation']['angle']['failure_penalized_mae_deg'])
    report={'seconds':time.perf_counter()-start,'protocol':protocol,'prior_audit':audits,'winner_validation':winner,
            'results':{k:{a:b for a,b in v.items() if a not in ['rows','head_rows','history','line_rows','test_lines','validation_lines','validation_rows']} for k,v in results.items()},
            'geometry_comparisons':comparisons,'reference_caveat':'Manual-divider strict-contour axes are pseudo-reference, not independent measured clinical axes. Existing original flags combine different expert definition; reported separately. Previously inspected test is diagnostic, never selection.'}
    save(DEST/'report.json',report);save(DEST/'geometry_comparison_rows.json',diagnostic_rows);save(DEST/'bisector_lines.json',bisector_predictions)
    render(report,results,groups,geometries)
    print(json.dumps({'stage':'complete','minutes':report['seconds']/60,'winner_validation':winner,'report':str(DEST/'report.json')}),flush=True)


def render(report,results,groups,geometries):
    from PIL import Image,ImageDraw
    import html
    page=ROOT/'dxa_project/team_demo/spine_angle_study_20260929.html';assets=page.parent/'spine_angle_assets';assets.mkdir(exist_ok=True)
    def table(headers,rows):return '<table><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
    def n(v):return '—' if v is None else f'{v:.3f}'
    parts=['<h1>Целевая метрика: угол позвоночника относительно вертикали</h1>',f'<p>Выбран по validation: <b>{report["winner_validation"]}</b>. {report["seconds"]/60:.1f} минут, CUDA, 4 эпохи каждого из пяти вариантов. Данные и начальные веса одинаковы с предыдущим пространственным исследованием; энкодер заморожен. Рабочие модели и метки не заменены.</p>',
           '<p>Численная ошибка считается относительно строго построенной оси по ручным линиям: это геометрический псевдоэталон, не независимое измерение рентгенолога. На исходниках отдельно проверяется флаг организаторов &gt;5°. Test ранее просматривался; он используется для диагностики, выбор моделей только по validation. Пропущенный угол штрафуется 15° в критерии выбора, coverage показывается отдельно.</p>',
           '<h2>Контролируемое обучение</h2>',table(['Вариант','Эпоха','Validation MAE с пропусками ↓','Test MAE, ° ↓','95% ДИ','Покрытие','RMSE, °','В пределах 1°'],[[k,v['best_epoch'],n(v['validation']['angle']['failure_penalized_mae_deg']),n(v['test_angles']['mae_deg']),str(v['test_angles'].get('ci95',{}).get('mae_deg')),n(v['test_angles']['coverage']),n(v['test_angles']['rmse_deg']),n(v['test_angles']['within_1deg'])] for k,v in results.items()]),
           '<p>control_c2 / all_peaks: итоговый угол от контурной цепочки; angle_c2 / angle_c05: отдельная обученная общая ось (две координаты пересечений с верхней/нижней рамкой), с квадратичной угловой и координатной потерями. Это разные способы получить целевой угол, а не замена геометрической оценки в таблице без объявления.</p>',
           '<h2>Линии и геометрия</h2>',table(['Вариант','MAE контурного угла','Покрытие контурного угла','Ошибка линий, мм','Recall ≤5 мм','Precision ≤5 мм','Недобор','Сверх эталона','Кластеры /154','Большие пробелы /154'],[[k,n(v['geometry_angles']['mae_deg']),n(v['geometry_angles']['coverage']),n(v['test']['lines']['matched_error_mm']),n(v['test']['lines']['recall_5mm']),n(v['test']['lines']['precision_5mm']),n(v['test']['lines']['under_fraction']),n(v['test']['lines']['over_fraction']),v['test']['close_images'],v['test']['hole_images']] for k,v in results.items()]),
           '<h2>Оригиналы и аугментации отдельно</h2>',table(['Вариант','Оригиналы MAE','Покрытие','F1 по табличному >5°','AUC','Синтетика MAE','F1 по синтетическому >5°'],[[k,n(v['original_angles']['mae_deg']),n(v['original_angles']['coverage']),n((v['original_angles'].get('binary_existing_labels') or {}).get('f1')),n((v['original_angles'].get('binary_existing_labels') or {}).get('roc_auc')),n(v['synthetic_angles']['mae_deg']),n((v['synthetic_angles'].get('binary_existing_labels') or {}).get('f1'))] for k,v in results.items()]),
           '<h2>Перпендикулярные крайние границы и биссектрисы</h2>',table(['Линии','Старые крайние оси MAE','Strict MAE','Strict покрытие','Биссектрисы + повторный fit MAE','Покрытие','Отказы'],[[k,n(v['weighted_terminals']['mae_deg']),n(v['strict_terminals']['mae_deg']),n(v['strict_terminals']['coverage']),n(v['strict_plus_bisectors_refit']['mae_deg']),n(v['strict_plus_bisectors_refit']['coverage']),v['bisector_rejections']] for k,v in report['geometry_comparisons'].items()]),
           '<p>Биссектрисы строятся через общий узел, в миллиметрах, с равными углами к соседним осям. Геометрические оси в первом шаге фиксированы: такое перестроение само по себе не меняет их угол. В таблице отдельно оценено повторное построение контуров по новым разделителям; оно может как помочь, так и разрушить локализацию.</p>',
           '<h2>Конфликт усиленного минимума с эталоном</h2>',table(['Часть','c=2','c=0,5','Из них оригиналы c=0,5'],[[k,v['2.0']['conflicts'],v['0.5']['conflicts'],v['0.5']['original_conflicts']] for k,v in report['prior_audit'].items()]),
           '<p>Равномерность не является абсолютным анатомическим требованием. Есть частично видимые крайние позвонки и специально нарушенные укладки. Увеличивать штраф без проверки противоречий с ручной геометрией нельзя; здесь конфликт количественно измерен, эталоны не удалены.</p>', '<h2>Одни и те же примеры</h2>']
    parts.append('<p>На примерах: зелёные — разделители; голубые — локальные контурные оси; красная — общая контурная ось (в ручной колонке псевдоэталон); оранжевая — общая ось обученной головы. В конце добавлены четыре наибольшие угловые ошибки выбранного метода.</p>')
    reference_axes=json.loads((DEST/'reference_axes.json').read_text(encoding='utf-8'))
    winner=results[report['winner_validation']]
    angle_rows=winner['head_rows'] if winner['config']['angle'] else winner['rows']
    worst=sorted(angle_rows,key=lambda r:abs(r['angle']-r['reference_angle']) if r['angle'] is not None and r['reference_angle'] is not None else 15.,reverse=True)[:4]
    indices={r.relative_path:i for i,r in enumerate(groups['test'])}
    chosen=list(dict.fromkeys(list(range(0,6))+list(range(34,40))+[indices[r['relative_path']] for r in worst]))
    for i in chosen:
        r=groups['test'][i];parts.append('<h3>'+html.escape(r.relative_path)+'</h3><div class="grid">')
        for name in ['manual']+list(results):
            pic=Image.fromarray((read_dicom_image(r.source_path)*255).astype('uint8')).convert('RGB');draw=ImageDraw.Draw(pic)
            if name=='manual':
                lines=geometries[r.relative_path]['spine']['disc_lines'];axis=None
                ref=reference_axes[r.relative_path];label='ручные линии; псевдоэталон '+n(ref['angle'])+'°'
                if ref['axis_points']:draw.line([tuple(p) for p in ref['axis_points']],fill='red',width=2)
            else:
                v=results[name];lines=v['test_lines'][r.relative_path];a=next(z for z in v['rows'] if z['relative_path']==r.relative_path);axis=a['axes'];label=f'{name}: контур {n(a["angle"])}°'
            for line in lines:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
            if axis:
                for z in [axis.get('upper_fragment')]+axis['axes']+[axis.get('lower_fragment')]:
                    if z and z.get('axis_points'):draw.line([tuple(p) for p in z['axis_points']],fill='cyan',width=2)
                    if z:
                        for boundary in z.get('side_boundaries',[]):draw.line([tuple(p) for p in boundary],fill='#71b5ed',width=1)
                if axis.get('global_axis_points'):draw.line([tuple(p) for p in axis['global_axis_points']],fill='red',width=2)
            if name!='manual' and results[name]['config']['angle']:
                head=next(z for z in results[name]['head_rows'] if z['relative_path']==r.relative_path)
                if head['axis_points']:draw.line([tuple(p) for p in head['axis_points']],fill='#ffa500',width=3)
                label+='; общая ось '+n(head['angle'])+'°'
            filename=f'{i}_{name}.png';pic.resize((pic.width*2,pic.height*2)).save(assets/filename)
            parts.append(f'<figure><img loading="lazy" src="spine_angle_assets/{filename}"><figcaption>{label}</figcaption></figure>')
        parts.append('</div>')
    page.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>DXA: угол позвоночника</title><style>body{font:16px system-ui;max-width:1700px;margin:24px auto;padding:18px;background:#101923;color:#eef}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin-bottom:20px}td,th{border:1px solid #567;padding:7px}.grid{display:flex;gap:12px;overflow-x:auto}figure{margin:0;min-width:270px}img{max-width:290px}</style>'+''.join(parts)+'</html>',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=4)
    args=parser.parse_args()
    with warnings.catch_warnings():warnings.simplefilter('ignore');run(args.epochs)
