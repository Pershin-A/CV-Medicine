"""Small isolated GPU experiment: fixed cached encoder, ordered line slots.

No production weights or augmentation labels are overwritten. This is a head
ablation, not a claim that full backbone training or clinical validation is done.
"""
import csv,json,math,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from .data import load_records,load_augmented_records,DxaDataset,collate
from .predict import _load_model,_spine_lines
from .compare_reviewed import ROOT,OUT,read
from .evaluate import with_ci
from .report_reviewed import table,n
from .experimental_spine_axes import analyze_frame_axes
import pydicom

DEST=ROOT/'dxa_project/outputs/spine_penalty_study_20260929'

class Slots(nn.Module):
    def __init__(self,features):
        super().__init__();self.head=nn.Sequential(nn.LayerNorm(features),nn.Linear(features,128),nn.ReLU(),nn.Linear(128,21))
    def forward(self,x):
        a=self.head(x).reshape(-1,7,3)
        # Ordered centers. Presence allows fewer than seven active slots.
        y,order=torch.sigmoid(a[:,:,0]).sort(dim=1)
        z=a.gather(1,order[:,:,None].expand(-1,-1,3))
        return y,.35*torch.tanh(z[:,:,1]),z[:,:,2]

def target(record):
    g=read(record.geometry_path);h,w=g['image_height'],g['image_width']
    values=[]
    for line in g['spine']['disc_lines']:
        a,b=sorted(line['points'],key=lambda p:p[0]);s=(b[1]-a[1])/max(b[0]-a[0],1e-6)
        yc=a[1]+((w-1)/2-a[0])*s
        values.append([yc/(h-1),s*(w-1)/(h-1)])
    values=sorted(values)
    if not 1<=len(values)<=7:raise ValueError('Unexpected line count')
    padded=np.zeros((7,2),np.float32);padded[:len(values)]=values
    visible=np.arange(7)<len(values)
    return padded,visible.astype(np.float32)

def decoded(y,slope,presence,record):
    g=read(record.geometry_path);h,w=g['image_height'],g['image_width']
    result=[]
    for i in np.flatnonzero(presence>=.5):
        yc=y[i]*(h-1);dy=slope[i]*(h-1)/2
        # Intersection with frame, retaining the true fitted slope.
        a=np.array([0.,yc-dy]);b=np.array([w-1.,yc+dy]);points=[a,b]
        for j,p in enumerate(points):
            if p[1]<0 or p[1]>h-1:
                boundary=0 if p[1]<0 else h-1
                t=(boundary-a[1])/(b[1]-a[1]);points[j]=a+t*(b-a)
        result.append({'id':f'slot_{i}','points':[p.tolist() for p in points]})
    return result

def losses(y,slope,logits,gt,visible,axis_ratio,aspect_mm,config):
    p=torch.sigmoid(logits);count=visible.sum(1)
    coordinate=(((y-gt[:,:,0]).abs()+.25*(slope-gt[:,:,1]).abs())*visible).sum()/visible.sum().clamp_min(1)
    presence=F.binary_cross_entropy_with_logits(logits,visible)
    under=F.relu(count-p.sum(1)).square().mean()
    # Physical length surrogate follows normals to neighboring predicted dividers.
    # The complete contour axes are separately measured after decoding.
    gap=y[:,1:]-y[:,:-1];pair=p[:,1:]*p[:,:-1]
    mean_slope=(slope[:,1:]+slope[:,:-1])/2/aspect_mm[:,None]
    physical_gap=gap*torch.sqrt(1+mean_slope.square())
    minimum=1/(1/axis_ratio.clamp_min(1e-4)+config['gap_extra'])
    close=(F.relu(minimum[:,None]-physical_gap).square()*pair).sum()/pair.sum().clamp_min(1)
    terminal=(F.relu(y[:,0]-config['terminal_limit']).square()*p[:,0]+F.relu(1-y[:,-1]-config['terminal_limit']).square()*p[:,-1]).mean()
    half_top=((y[:,0]-.5*gap[:,0]).square()*p[:,0]*p[:,1]).mean()
    # Inactive trailing slots must not hide the lower terminal constraint.
    end=0.
    for i in range(7):
        last=p[:,i]*torch.prod(1-p[:,i+1:],dim=1)
        end=end+(F.relu(1-y[:,i]-config['terminal_limit']).square()*last).mean()
    terminal=terminal+end
    loss=8*coordinate+presence+config['count']*under+config['geometry']*(close+terminal+.25*half_top)
    return loss,{'coordinate':float(coordinate.detach()),'under_surrogate':float(under.detach()),'close':float(close.detach()),'terminal':float(terminal.detach())}

def measure(lines,record):
    gt=read(record.geometry_path)['spine']['disc_lines'];h=read(record.geometry_path)['image_height'];w=read(record.geometry_path)['image_width']
    xs=np.array([.25,.5,.75])*(w-1)
    def ys(line):
        a,b=sorted(line['points'],key=lambda p:p[0]);return a[1]+(xs-a[0])*(b[1]-a[1])/max(b[0]-a[0],1e-6)
    errors=[]
    if gt and lines:
        cost=np.array([[np.mean(abs(ys(a)-ys(b)))*record.spacing_mm[0] for b in lines] for a in gt]);i,j=linear_sum_assignment(cost);errors=cost[i,j].tolist()
    return {'gt_count':len(gt),'pred_count':len(lines),'under':max(len(gt)-len(lines),0),'over':max(len(lines)-len(gt),0),'matched_errors_mm':errors,'recall_5mm':sum(e<=5 for e in errors)/len(gt),'precision_5mm':sum(e<=5 for e in errors)/len(lines) if lines else 0.,'exact_count':len(gt)==len(lines),'study':record.study,'relative_path':record.relative_path}

def summarize(rows):
    errors=[e for r in rows for e in r['matched_errors_mm']]
    return {'images':len(rows),'studies':len({r['study'] for r in rows}),'count_mae':float(np.mean([abs(r['gt_count']-r['pred_count']) for r in rows])),'under_fraction':float(np.mean([r['under']>0 for r in rows])),'under_squared':float(np.mean([r['under']**2 for r in rows])),'over_fraction':float(np.mean([r['over']>0 for r in rows])),'exact_count_fraction':float(np.mean([r['exact_count'] for r in rows])),'matched_error_mm':float(np.mean(errors)) if errors else None,'recall_5mm':float(np.mean([r['recall_5mm'] for r in rows])),'precision_5mm':float(np.mean([r['precision_5mm'] for r in rows]))}

def run(epochs=35):
    start=time.perf_counter();DEST.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    protocol=read(OUT/'heavy_stratified/protocol.json');originals=[r for r in load_records(ROOT) if r.region=='SPINE'];by={r.relative_path:r for r in originals}
    aug=load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',load_records(ROOT));aug=[r for r in aug if r.region=='SPINE']
    rng=np.random.default_rng(934);groups={}
    for part,limit in [('train',768),('validation',120),('test',120)]:
        a=[r for r in aug if protocol['partition_by_path'][r.source_id]==part];rng.shuffle(a)
        # Sampling is deterministic; all original records in each partition are retained.
        groups[part]=[r for r in originals if protocol['partition_by_path'][r.relative_path]==part]+a[:limit]
    assert not any({r.study for r in groups[a]}&{r.study for r in groups[b]} for a,b in [('train','validation'),('train','test'),('validation','test')])
    (DEST/'protocol.json').write_text(json.dumps({'source_protocol':str(OUT/'heavy_stratified/protocol.json'),'partitions':{k:[r.relative_path for r in v] for k,v in groups.items()},'epochs':epochs,'device':str(device),'encoder':'frozen existing heavy spine; same study split; cached layer4 and line-map profiles'},indent=2),encoding='utf-8')
    base,size=_load_model('spine',OUT/'heavy_stratified',device);feature=[]
    hook=base.layer4.register_forward_hook(lambda m,i,o:feature.append(o.detach()))
    cache={};old_lines={};truth={};reference_ratio={};ratio_basis={};aspects={}
    with torch.inference_mode():
        for part,records in groups.items():
            loader=torch.utils.data.DataLoader(DxaDataset(records,size),batch_size=8,shuffle=False,collate_fn=collate)
            for images,targets in loader:
                out=base(images.to(device));spatial=torch.sigmoid(out['spatial'][:,0]);e=feature.pop()
                encoded=F.adaptive_avg_pool2d(e,(4,2)).flatten(1)
                encoded=torch.cat([encoded,spatial.mean(2)],1).cpu()
                for j,t in enumerate(targets):
                    rel=t['relative_path'];r=next(r for r in records if r.relative_path==rel)
                    cache[rel]=encoded[j];truth[rel]=target(r)
                    meta={'width':t['width'],'height':t['height'],'left':t['pad_left'],'top':t['pad_top'],'scale':t['scale'],'size':size,'flipped':False}
                    old_lines[rel]=_spine_lines(spatial[j].cpu().numpy(),meta)
            print(json.dumps({'stage':'features','part':part,'images':len(records)}),flush=True)
    hook.remove();del base;torch.cuda.empty_cache() if device.type=='cuda' else None
    for part,records in groups.items():
        for r in records:
            g=read(r.geometry_path);ds=pydicom.dcmread(str(r.source_path),force=True)
            a=analyze_frame_axes(np.squeeze(ds.pixel_array),g,r.spacing_mm,'dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright', neighbor_scale=1., weight_basis='height')
            lengths=[z['length_mm'] for z in a['axes'] if z.get('valid') and z.get('length_mm')]
            physical_height=(g['image_height']-1)*r.spacing_mm[0]
            if lengths:
                reference_ratio[r.relative_path]=float(np.mean(lengths))/physical_height
                ratio_basis[r.relative_path]='manual_contour_axes'
            else:
                coords,visible=truth[r.relative_path];ys=coords[:int(visible.sum()),0]
                reference_ratio[r.relative_path]=float(np.mean(np.diff(ys))) if len(ys)>1 else .2
                ratio_basis[r.relative_path]='manual_divider_gap_fallback'
            aspects[r.relative_path]=(g['image_width']-1)*r.spacing_mm[1]/physical_height
        print(json.dumps({'stage':'reference_axis_priors','part':part,'files':len(records),'fallback':sum(ratio_basis[r.relative_path]!='manual_contour_axes' for r in records)}),flush=True)
    data={}
    for part,records in groups.items():
        data[part]=(torch.stack([cache[r.relative_path] for r in records]).to(device),torch.tensor(np.stack([truth[r.relative_path][0] for r in records]),device=device),torch.tensor(np.stack([truth[r.relative_path][1] for r in records]),device=device),torch.tensor([reference_ratio[r.relative_path] for r in records],dtype=torch.float32,device=device),torch.tensor([aspects[r.relative_path] for r in records],dtype=torch.float32,device=device))
    prior={}
    for part,records in groups.items():
        rows=[]
        for r in records:
            gt,p=truth[r.relative_path];ys=gt[:int(p.sum()),0];gaps=np.diff(ys)
            ratio=reference_ratio[r.relative_path];physical_gaps=gaps*np.sqrt(1+((gt[1:int(p.sum()),1]+gt[:int(p.sum())-1,1])/2/aspects[r.relative_path])**2)
            rows.append({'relative_path':r.relative_path,'close_c1':bool(np.any(physical_gaps<1/(1/ratio+1))),'close_c2':bool(np.any(physical_gaps<1/(1/ratio+2))),'top_1_6_conflict':bool(ys[0]>1/6),'bottom_1_6_conflict':bool(1-ys[-1]>1/6)})
        prior[part]={'n':len(rows),'close_c1':sum(r['close_c1'] for r in rows),'close_c2':sum(r['close_c2'] for r in rows),'top_1_6_conflicts':sum(r['top_1_6_conflict'] for r in rows),'bottom_1_6_conflicts':sum(r['bottom_1_6_conflict'] for r in rows),'fallbacks':sum(ratio_basis[r.relative_path]!='manual_contour_axes' for r in records)}
    variants={'slots_baseline':{'count':0.,'geometry':0.,'terminal_limit':1/6,'gap_extra':1},'count_only':{'count':2.,'geometry':0.,'terminal_limit':1/6,'gap_extra':1},'count_geometry_c1_1_6':{'count':2.,'geometry':12.,'terminal_limit':1/6,'gap_extra':1},'count_geometry_c2_1_6':{'count':2.,'geometry':12.,'terminal_limit':1/6,'gap_extra':2},'count_geometry_c2_1_7':{'count':2.,'geometry':12.,'terminal_limit':1/7,'gap_extra':2}}
    results={};predictions={}
    for name,config in variants.items():
        torch.manual_seed(42);model=Slots(data['train'][0].shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        best=float('inf');history=[];best_state=None;gen=torch.Generator().manual_seed(42)
        for epoch in range(epochs):
            model.train();order=torch.randperm(len(groups['train']),generator=gen);train=[]
            x,gt,p,ratio,aspect=data['train']
            for ids in order.split(32):
                ids=ids.to(device);y,s,logits=model(x[ids]);loss,components=losses(y,s,logits,gt[ids],p[ids],ratio[ids],aspect[ids],config)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();train.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                y,s,l=model(data['validation'][0]);v=float(losses(y,s,l,*data['validation'][1:],config)[0]);rows=[measure(decoded(y[i].cpu().numpy(),s[i].cpu().numpy(),torch.sigmoid(l[i]).cpu().numpy(),r),r) for i,r in enumerate(groups['validation'])]
            metrics=summarize(rows);criterion=metrics['under_squared']+2*(1-metrics['recall_5mm'])+.1*metrics['count_mae']
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(train)),'validation_loss':v,'selection_value':criterion,'metrics':metrics})
            if criterion<best:best=criterion;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
            if (epoch+1)%10==0:print(json.dumps({'variant':name,'epoch':epoch+1,'validation':metrics}),flush=True)
        model.load_state_dict(best_state);model.eval();torch.save({'state_dict':best_state,'config':config,'features':data['train'][0].shape[1],'best_epoch':best_epoch},DEST/(name+'.pt'))
        (DEST/(name+'_history.json')).write_text(json.dumps(history,indent=2),encoding='utf-8')
        with torch.no_grad():y,s,l=model(data['test'][0])
        pred={r.relative_path:decoded(y[i].cpu().numpy(),s[i].cpu().numpy(),torch.sigmoid(l[i]).cpu().numpy(),r) for i,r in enumerate(groups['test'])};predictions[name]=pred
        rows=[measure(pred[r.relative_path],r) for r in groups['test']]
        results[name]={'config':config,'best_epoch':best_epoch,'validation_selection':best,'test':summarize(rows),'original_test':summarize([r for r in rows if not r['relative_path'].startswith('aug/')]),'augmented_test':summarize([r for r in rows if r['relative_path'].startswith('aug/')]),'details':rows}
        print(json.dumps({'stage':'variant_complete','variant':name,'test':results[name]['test']}),flush=True)
    rows=[measure(old_lines[r.relative_path],r) for r in groups['test']]
    results['previous_heavy_map']={'test':summarize(rows),'original_test':summarize([r for r in rows if not r['relative_path'].startswith('aug/')]),'augmented_test':summarize([r for r in rows if r['relative_path'].startswith('aug/')]),'details':rows}
    predictions['previous_heavy_map']={r.relative_path:old_lines[r.relative_path] for r in groups['test']}
    # Selection only on validation, never using the test scores above.
    winner=min(variants,key=lambda k:results[k]['validation_selection'])
    axes_results={};examples=[]
    with (ROOT/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        label_map={r['relative_path']:r.get('spine_axis','') for r in csv.DictReader(f)}
    with (ROOT/'dxa_project/outputs/augmented_15000_20260929/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        label_map.update({'aug/'+r['image_path']:r.get('spine_axis','') for r in csv.DictReader(f)})
    axis_records=groups['test'][:34]+[r for r in groups['test'] if r.relative_path.startswith('aug/')][:45]
    for name in ['manual', 'previous_heavy_map',winner]:
        rows=[]
        for i,r in enumerate(axis_records):
            g=read(r.geometry_path)
            if name!='manual':g['spine']['disc_lines']=predictions[name][r.relative_path]
            ds=pydicom.dcmread(str(r.source_path),force=True);raw=np.squeeze(ds.pixel_array)
            a=analyze_frame_axes(raw,g,r.spacing_mm,'dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright', neighbor_scale=1., weight_basis='height')
            flag=label_map.get(r.relative_path,'')
            rows.append({'relative_path':r.relative_path,'study':r.study,'truth_flag':int(float(flag)) if flag not in ('',None) else None,'angle':a['global_angle_deg'],'previous_angle':a.get('previous_global_angle_deg'),'review_required':a['review_required'],'terminal_fit_valid':a.get('terminal_fit_valid',False),'line_count':len(g['spine']['disc_lines']),'axis_lengths_mm':[z.get('length_mm') for z in a['axes']],'top_length_mm':(a.get('upper_fragment') or {}).get('length_mm'),'bottom_length_mm':(a.get('lower_fragment') or {}).get('length_mm')})
            if i<5 or (r.relative_path.startswith('aug/') and i<39):examples.append((name,r,g,a,raw))
        axes_results[name]=rows;print(json.dumps({'stage':'axes','variant':name,'files':len(rows)}),flush=True)
    truth_angles={r['relative_path']:r['angle'] for r in axes_results['manual']}
    for name,rows in list(axes_results.items()):
        paired=[(r,truth_angles[r['relative_path']]) for r in rows if r['angle'] is not None and truth_angles[r['relative_path']] is not None]
        labeled=[r for r in rows if r['angle'] is not None and r['truth_flag'] is not None]
        historical=[r for r in rows if r['previous_angle'] is not None and r['truth_flag'] is not None]
        axes_results[name]={'n':len(rows),'defined':sum(r['angle'] is not None for r in rows),'reliable_terminal_fits':sum(r['terminal_fit_valid'] for r in rows),'mean_angle_error_manual_deg':float(np.mean([abs(r['angle']-gt) for r,gt in paired])) if paired else None,'binary_vs_manual':with_ci([{'study':r['study'],'truth':int(abs(gt)>5),'prediction':int(abs(r['angle'])>5),'score':abs(r['angle'])} for r,gt in paired],200),'binary_vs_existing_labels':with_ci([{'study':r['study'],'truth':r['truth_flag'],'prediction':int(abs(r['angle'])>5),'score':abs(r['angle'])} for r in labeled],200),'previous_axis_vs_existing_labels':with_ci([{'study':r['study'],'truth':r['truth_flag'],'prediction':int(abs(r['previous_angle'])>5),'score':abs(r['previous_angle'])} for r in historical],200),'details':rows}
    report={'scope':'Exploratory fixed-encoder ordered-slot head study, not full retraining. Prior tests already inspected. No production integration. Validation-only model selection. Same pretrained heavy encoder and same sampling for all new heads.', 'device':str(device),'seconds':time.perf_counter()-start,'sizes':{k:len(v) for k,v in groups.items()},'prior_audit':prior,'results':results,'winner_validation':winner,'axes':axes_results}
    (DEST/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page=ROOT/'dxa_project/team_demo/spine_penalty_study_20260929.html';assets=page.parent/'spine_penalty_assets';assets.mkdir(exist_ok=True)
    parts=['<h1>Исследование штрафов линий и крайних осей</h1><p>Отдельная экспериментальная ветка. Общий пайплайн не изменён. GPU: '+str(device)+f'; время {report["seconds"]/60:.1f} мин. Обучались только новые головы на замороженных признаках большой модели. Это небольшой тест формулировки потерь, не финальное обучение.</p>', '<p>Новый декодер: семь упорядоченных слотов с положением, наклоном и вероятностью. Все пять вариантов стартуют одинаково, используют одинаковые данные, оптимизатор и 35 эпох. Предыдущая большая карта линий — отдельный исторический контроль на том же тесте, но с другим способом обучения. Малая версия использовала другое разбиение, поэтому её цифры из прежнего отчёта не смешиваются с этой таблицей.</p>', '<h2>Данные и конфликт ограничений с разметкой</h2>',table(['Часть','Файлов','Зазор ниже адаптивного c=1','Зазор ниже адаптивного c=2','Верх >H/6','Низ >H/6','Приближённых эталонных осей'],[[k,v['n'],v['close_c1'],v['close_c2'],v['top_1_6_conflicts'],v['bottom_1_6_conflicts'],v['fallbacks']] for k,v in prior.items()]), '<p>Минимальная относительная длина: 1/(H_мм / средняя эталонная длина оси_мм + c), где c=1 или 2. Для эталонной средней длины использован расчёт контуров по ручным линиям; при сбое явно отмечено приближение средним зазором. При обучении расстояния между линиями — дифференцируемый суррогат физических длин по нормалям к среднему наклону. Эти приоры используют разметку только в функции потерь; предсказание модели не требует эталонных линий.</p>']
    for scope,title in [('test','Весь тест'),('original_test','Исходные тестовые снимки'),('augmented_test','Аугментированные тестовые снимки')]:
        parts+=['<h2>'+title+'</h2>',table(['Вариант','N','Недобор ↓','Перебор','Точное число ↑','MAE числа ↓','Ошибка пар, мм ↓','Полнота линий ≤5 мм ↑','Точность линий ≤5 мм ↑'],[[k,v[scope]['images'],n(v[scope]['under_fraction']),n(v[scope]['over_fraction']),n(v[scope]['exact_count_fraction']),n(v[scope]['count_mae']),n(v[scope]['matched_error_mm']),n(v[scope]['recall_5mm']),n(v[scope]['precision_5mm'])] for k,v in results.items()])]
    parts+=['<p>Ошибка сопоставленных пар не штрафует отсутствующие линии: обязательно читать её вместе с полнотой. Перебор сам по себе не штрафуется квадратичной потерей недобора, но BCE наличия сохраняется, чтобы лишние линии не подменяли анатомию.</p><h2>Общая ось от границы до границы</h2>',table(['Вариант','N','Угол определён','Оба крайних контура построены','Ошибка к ручным линиям, °','F1 к ручному геометрическому правилу'],[[k,v['n'],v['defined'],v['reliable_terminal_fits'],n(v['mean_angle_error_manual_deg']),n((v['binary_vs_manual'].get('metrics') or {}).get('f1'))] for k,v in axes_results.items()]), '<p>Ось начинается на пересечении верхней оси с рамкой и заканчивается на пересечении нижней оси с рамкой. Нижний фрагмент теперь расположен ниже последней линии. Его направление — среднее собственного и соседнего направления с весами высот в мм. При ненадёжном контуре показана экстраполяция соседней оси с флагом проверки, не выдуманная надёжная ось. Ручной геометрический эталон использует тот же алгоритм, поэтому это проверка согласованности, а не независимая истинность.</p>', '<p>Победитель по внутренней валидации: '+winner+'. Оценки теста не использовались для выбора коэффициентов или варианта.</p>']
    parts += ['<h2>Ось против существующих бинарных меток</h2><p>Метрика по смеси исходников и аугментаций; интервалы по исследованиям сохранены в JSON. Метки аугментаций создавались старым определением оси, поэтому расхождение нового геометрического правила с ними не обязательно означает ошибку модели. Отдельно показаны старое и новое определение на тех же предсказанных линиях.</p>',table(['Вариант','Старое: N','Старое: F1','Старое: AUC','Новое: N','Новое: F1','Новое: AUC'],[[k,v['previous_axis_vs_existing_labels']['n'],n((v['previous_axis_vs_existing_labels'].get('metrics') or {}).get('f1')),n((v['previous_axis_vs_existing_labels'].get('metrics') or {}).get('roc_auc')),v['binary_vs_existing_labels']['n'],n((v['binary_vs_existing_labels'].get('metrics') or {}).get('f1')),n((v['binary_vs_existing_labels'].get('metrics') or {}).get('roc_auc'))] for k,v in axes_results.items()])]
    for j,(name,r,g,a,raw) in enumerate(examples):
        lo,hi=np.percentile(raw,[.5,99.5]);pixels=np.clip((raw.astype(float)-lo)/max(hi-lo,1),0,1)
        ds=pydicom.dcmread(str(r.source_path),stop_before_pixels=True,force=True)
        if ds.PhotometricInterpretation=='MONOCHROME1':pixels=1-pixels
        pic=Image.fromarray((pixels*255).astype('uint8')).convert('RGB');draw=ImageDraw.Draw(pic)
        for line in g['spine']['disc_lines']:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
        for axis in [(a.get('upper_fragment') or {})]+a['axes']+[(a.get('lower_fragment') or {})]:
            if axis.get('axis_points'):draw.line([tuple(p) for p in axis['axis_points']],fill='cyan',width=2)
        if a.get('global_axis_points'):draw.line([tuple(p) for p in a['global_axis_points']],fill='red',width=3)
        pic.thumbnail((400,500));filename=f'axis_{j:03}.png';pic.save(assets/filename)
        parts.append(f'<figure><img loading="lazy" src="spine_penalty_assets/{filename}"><figcaption>{name} · {r.relative_path.split("/")[-1]} · линий {len(g["spine"]["disc_lines"])} · угол {n(a["global_angle_deg"])}° · проверка {a["review_required"]}</figcaption></figure>')
    page.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>DXA: штрафы линий</title><style>body{font:16px system-ui;background:#101923;color:#eef;max-width:1450px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #567;padding:8px}p{line-height:1.5}figure{display:inline-block;width:30%;vertical-align:top}img{max-width:100%}</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print(json.dumps({'complete':True,'report':str(page),'minutes':report['seconds']/60,'winner':winner,'test':{k:v['test'] for k,v in results.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
