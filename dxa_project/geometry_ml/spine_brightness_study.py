"""Controlled spatial-decoder ablation: bright inputs and quadratic priors."""
import json,time,csv,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
import pydicom
from .compare_reviewed import ROOT,OUT,read
from .data import load_records,load_augmented_records,DxaDataset,collate,read_dicom_image
from .models import SpatialNet,dice_bce
from .predict import _spine_lines
from .spine_penalty_study import measure,summarize,target
from .experimental_spine_axes import analyze_frame_axes
from .report_reviewed import table,n
from .evaluate import with_ci

DEST=ROOT/'dxa_project/outputs/spine_brightness_study_20260929'

def filter_bright(images,targets,fraction):
    if fraction is None:return images.clone(),[1.]*len(targets)
    mean=torch.tensor([.485,.456,.406],device=images.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=images.device)[None,:,None,None]
    pixels=(images*std+mean).clamp(0,1);mask=torch.zeros_like(pixels[:,:1]);retained=[]
    for i,t in enumerate(targets):
        l,top=t['pad_left'],t['pad_top'];w=round(t['width']*t['scale']);h=round(t['height']*t['scale'])
        area=pixels[i,0,top:top+h,l:l+w];cut=torch.quantile(area.flatten(),1-fraction)
        keep=(area>=cut)&(area>0);mask[i,0,top:top+h,l:l+w]=keep
        retained.append(float(keep.float().mean()))
    # Preserve original retained gray values; no binarization and no padding in quantiles.
    return (pixels*mask-mean)/std,retained

def quadratic_priors(logits,targets):
    counts=[];closes=[];ends=[];coordinates=[]
    for i,t in enumerate(targets):
        gt=t['ordered_lines'].to(logits.device);nlines=len(gt)
        left,top=t['pad_left'],t['pad_top'];dw=round(t['width']*t['scale']);dh=round(t['height']*t['scale'])
        core=logits[i,0,top:top+dh,left:left+dw]
        ys=[];confidence=[];boundaries=torch.cat([gt.new_tensor([0.]),(gt[:-1,0]+gt[1:,0])/2,gt.new_tensor([1.])])
        grid=torch.linspace(0,1,dh,device=logits.device)
        for j in range(nlines):
            # Align each profile to the reference divider's tilt. A horizontal
            # average would punish a correct thin slanted line and reward thick bands.
            x=torch.linspace(.25,.75,24,device=logits.device)
            yy=grid[:,None]+gt[j,1]*(x[None,:]-.5)
            xx=x[None,:].expand(dh,-1);inside=(yy>=0)&(yy<=1)
            sampling=torch.stack([2*xx-1,2*yy-1],-1)[None]
            sampled=F.grid_sample(core[None,None],sampling,align_corners=True)[0,0]
            profile=(sampled*inside).sum(1)/inside.sum(1).clamp_min(1)
            valid=(grid>=boundaries[j])&(grid<=boundaries[j+1])&inside.any(1)
            if not valid.any():valid[torch.argmin(abs(grid-gt[j,0]))]=True
            q=profile[valid]
            mass=torch.softmax(q/.2,0);ys.append((mass*grid[valid]).sum())
            peak=(torch.logsumexp(q/.2,0)-np.log(len(q)))*.2
            confidence.append(torch.sigmoid(peak*2))
        y=torch.stack(ys);p=torch.stack(confidence)
        counts.append(F.relu(nlines-p.sum()).square())
        aspect=t['width']*t['spacing_mm'][1]/(t['height']*t['spacing_mm'][0])
        # Differentiable physical lengths from line-normal geometry; contour axes
        # are not differentiable and are evaluated separately on original pixels.
        slope=gt[:,1]/aspect
        normal=torch.sqrt(1+((slope[:-1]+slope[1:])/2).square())
        gaps=(y[1:]-y[:-1])*normal
        reference=((gt[1:,0]-gt[:-1,0])*normal).mean()
        minimum=1/(1/reference.clamp_min(1e-4)+2)
        closes.append(F.relu(minimum-gaps).square().mean())
        top_length=y[0]*torch.sqrt(1+slope[0].square())
        bottom_length=(1-y[-1])*torch.sqrt(1+slope[-1].square())
        top_limit=gaps[:2].mean();bottom_limit=gaps[-2:].mean()
        ends.append(F.relu(top_length-top_limit).square()+F.relu(bottom_length-bottom_limit).square())
        coordinates.append((y-gt[:,0]).abs().mean())
    return {'count':torch.stack(counts).mean(),'close':torch.stack(closes).mean(),'ends':torch.stack(ends).mean(),'coordinate':torch.stack(coordinates).mean()}

def run(epochs=4):
    DEST.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    previous=read(ROOT/'dxa_project/outputs/spine_penalty_study_20260929/protocol.json')
    originals=load_records(ROOT);all_records=originals+load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',originals)
    by={r.relative_path:r for r in all_records};groups={k:[by[p] for p in paths] for k,paths in previous['partitions'].items()}
    # Keep all train originals and a fixed source-order subset of 384 augmentations.
    groups['train']=[r for r in groups['train'] if not r.relative_path.startswith('aug/')]+[r for r in groups['train'] if r.relative_path.startswith('aug/')][:384]
    assert not any({r.study for r in groups[a]}&{r.study for r in groups[b]} for a,b in [('train','validation'),('train','test'),('validation','test')])
    initial=torch.load(OUT/'heavy_stratified/spine.pt',map_location='cpu',weights_only=False);size=initial['size'];cache={}
    for part,records in groups.items():
        dataset=DxaDataset(records,size);items=[]
        for i,r in enumerate(records):
            image,t=dataset[i];coords,visible=target(r);t['ordered_lines']=torch.tensor(coords[:int(visible.sum())]);items.append((image,t))
        cache[part]=items
    configs={'raw_old_loss':(None,False),'raw_new_loss':(None,True),'bright10_new_loss':(.1,True),'bright15_new_loss':(.15,True),'bright20_new_loss':(.2,True),'bright15_old_loss':(.15,False)}
    results={};all_predictions={};retention={};test_probabilities={}
    (DEST/'protocol.json').write_text(json.dumps({'sizes':{k:len(v) for k,v in groups.items()},'partitions':{k:[r.relative_path for r in v] for k,v in groups.items()},'epochs':epochs,'initial_checkpoint':str(OUT/'heavy_stratified/spine.pt'),'device':str(device),'scope':'Equal warm-start spatial decoder training, encoder frozen. Test unchanged from prior study; training subset smaller. Selection validation only.'},indent=2),encoding='utf-8')
    def evaluate(model,part,fraction):
        rows=[];pred={};kept=[];model.eval()
        with torch.no_grad():
            for ids in np.array_split(np.arange(len(groups[part])),max(1,(len(groups[part])+7)//8)):
                x,t=collate([cache[part][i] for i in ids]);x,k=filter_bright(x.to(device),t,fraction);kept+=k
                probability=torch.sigmoid(model(x)['spatial'][:,:1]).cpu().numpy()
                for j,index in enumerate(ids):
                    r=groups[part][index];a=t[j];meta={'width':a['width'],'height':a['height'],'left':a['pad_left'],'top':a['pad_top'],'scale':a['scale'],'size':size,'flipped':False}
                    lines=_spine_lines(probability[j,0],meta);pred[r.relative_path]=lines;rows.append(measure(lines,r))
        return rows,pred,kept
    for name,(fraction,newloss) in configs.items():
        torch.manual_seed(42);model=SpatialNet('SPINE',False,'resnet50').to(device);model.load_state_dict(initial['state_dict'])
        for key,p in model.named_parameters():p.requires_grad_(key.startswith(('up','spatial.')))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        best=-float('inf');history=[];best_epoch=0;gen=torch.Generator().manual_seed(42)
        for epoch in range(epochs):
            model.train()
            for m in [model.stem,model.layer1,model.layer2,model.layer3,model.layer4]:m.eval()
            losses=[];pieces=[];order=torch.randperm(len(groups['train']),generator=gen).tolist()
            for offset in range(0,len(order),8):
                x,t=collate([cache['train'][i] for i in order[offset:offset+8]]);x,_=filter_bright(x.to(device),t,fraction)
                out=model(x);truth=torch.stack([a['line'] for a in t]).to(device)
                loss=dice_bce(out['spatial'][:,:1],truth,3)
                if newloss:
                    p=quadratic_priors(out['spatial'][:,:1],t);loss=loss+2*p['count']+12*(p['close']+p['ends'])+2*p['coordinate']
                    pieces.append({k:float(v.detach()) for k,v in p.items()})
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
            rows,_,_=evaluate(model,'validation',fraction);v=summarize(rows)
            f1=2*v['recall_5mm']*v['precision_5mm']/max(v['recall_5mm']+v['precision_5mm'],1e-9)
            selection=f1-.05*v['under_fraction']
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':v,'selection_value':selection,'penalties':{k:float(np.mean([p[k] for p in pieces])) for k in pieces[0]} if pieces else {}})
            if selection>best:
                best=selection;best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'size':size,'task':'spine','architecture':'heavy','input_fraction':fraction,'new_loss':newloss},DEST/(name+'.pt'))
            print(json.dumps({'variant':name,'epoch':epoch+1,'validation':v}),flush=True)
        model.load_state_dict(torch.load(DEST/(name+'.pt'),map_location=device,weights_only=False)['state_dict'])
        rows,pred,kept=evaluate(model,'test',fraction);all_predictions[name]=pred;retention[name]=kept
        results[name]={'best_epoch':best_epoch,'validation_selection':best,'fraction':fraction,'new_loss':newloss,'test':summarize(rows),'originals':summarize([r for r in rows if not r['relative_path'].startswith('aug/')]),'augmentations':summarize([r for r in rows if r['relative_path'].startswith('aug/')]),'details':rows,'history':history,'retained_fraction_mean':float(np.mean(kept))}
        (DEST/'progress_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps({'completed':name,'test':results[name]['test']}),flush=True)
        del model;torch.cuda.empty_cache() if device.type=='cuda' else None
    best_new=max([k for k in configs if configs[k][1]],key=lambda k:results[k]['validation_selection'])
    previous_report=read(ROOT/'dxa_project/outputs/spine_penalty_study_20260929/report.json')
    # Axis comparison uses the same unfiltered DICOM for body-center estimation.
    selected=groups['test'][:12]+groups['test'][34:52];axis_results={};examples={}
    for name in ['manual','raw_old_loss',best_new]:
        rows=[];examples[name]=[]
        for i,r in enumerate(selected):
            g=read(r.geometry_path)
            if name!='manual':g['spine']['disc_lines']=all_predictions[name][r.relative_path]
            ds=pydicom.dcmread(str(r.source_path),force=True);raw=np.squeeze(ds.pixel_array);polarity='dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright'
            old=analyze_frame_axes(raw,g,r.spacing_mm,polarity,neighbor_scale=1.,weight_basis='height')
            new=analyze_frame_axes(raw,g,r.spacing_mm,polarity,neighbor_scale=.25,weight_basis='length')
            rows.append({'relative_path':r.relative_path,'study':r.study,'old_angle':old['global_angle_deg'],'new_angle':new['global_angle_deg'],'review':new['review_required'],'axes':new,'old_axes':old})
            if i<4 or 12<=i<16:examples[name].append((r,g,new,old))
        axis_results[name]=rows
        print(json.dumps({'axes_completed':name,'files':len(rows)}),flush=True)
    reference={r['relative_path']:r for r in axis_results['manual']};axis_summary={}
    for name,rows in axis_results.items():
        valid=[r for r in rows if r['new_angle'] is not None and reference[r['relative_path']]['new_angle'] is not None]
        shifts=[abs(r['new_angle']-r['old_angle']) for r in rows if r['new_angle'] is not None and r['old_angle'] is not None]
        continuity=[]
        for r in valid:
            a=r['axes'];chain=[a['upper_fragment']]+a['axes']+[a['lower_fragment']]
            continuity.extend(float(np.linalg.norm(np.asarray(x['axis_points'][1])-y['axis_points'][0])) for x,y in zip(chain,chain[1:]))
        axis_summary[name]={'n':len(rows),'defined':len(valid),'mae_to_manual_deg':float(np.mean([abs(r['new_angle']-reference[r['relative_path']]['new_angle']) for r in valid])) if valid else None,'mean_angle_change_deg':float(np.mean(shifts)) if shifts else None,'max_chain_discontinuity_px':max(continuity,default=None),'review_count':sum(r['review'] for r in rows)}
    rng=np.random.default_rng(42)
    for name,v in results.items():
        rows=v['details'];studies=sorted({r['study'] for r in rows});groups_ci={s:[r for r in rows if r['study']==s] for s in studies};samples=[]
        for _ in range(200):
            sample=[r for j in rng.integers(0,len(studies),len(studies)) for r in groups_ci[studies[j]]]
            samples.append([np.mean([r['under']>0 for r in sample]),np.mean([r['recall_5mm'] for r in sample])])
        ci=np.percentile(samples,[2.5,97.5],axis=0);v['ci95']={'under_fraction':ci[:,0].tolist(),'recall_5mm':ci[:,1].tolist()}
    report={'seconds':time.perf_counter()-start,'device':str(device),'sizes':{k:len(v) for k,v in groups.items()},'epochs':epochs,'winner_validation':best_new,'results':results,'previous_results':{k:v['test'] for k,v in previous_report['results'].items()},'axis_summary':axis_summary,'axis_details':axis_results,'note':'Warm-start decoder study, same split/test, smaller train subset than previous head study. Soft geometric penalties approximate axes; true contour axes use unfiltered raw DICOM. Quantiles computed per image over valid unpadded display pixels, retained gray levels unchanged; ties may retain more than nominal fraction.'}
    (DEST/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'predicted_lines.json').write_text(json.dumps(all_predictions,indent=2),encoding='utf-8')
    render(report,groups['test'],all_predictions,examples,cache['test'])
    from .finalize_brightness_study import run as finalize
    finalize()
    print(json.dumps({'complete':True,'minutes':report['seconds']/60,'winner':best_new,'results':{k:v['test'] for k,v in results.items()},'axes':axis_summary},ensure_ascii=False),flush=True)

def render(report,test,predictions,examples,items):
    page=ROOT/'dxa_project/team_demo/spine_brightness_study_20260929.html';assets=page.parent/'spine_brightness_assets';assets.mkdir(exist_ok=True)
    labels={'raw_old_loss':'Исходник, прежняя потеря','raw_new_loss':'Исходник, новые штрафы','bright10_new_loss':'Яркие 10%, новые штрафы','bright15_new_loss':'Яркие 15%, новые штрафы','bright20_new_loss':'Яркие 20%, новые штрафы','bright15_old_loss':'Яркие 15%, прежняя потеря'}
    rows=[[labels[k],n(v['test']['under_fraction']),n(v['test']['over_fraction']),n(v['test']['matched_error_mm']),n(v['test']['recall_5mm']),n(v['test']['precision_5mm']),v['best_epoch'],n(v['retained_fraction_mean'])] for k,v in report['results'].items()]
    parts=['<h1>DXA: яркие пиксели, квадратичные штрафы, крайние оси</h1>',f'<p>{report["sizes"]["train"]} train, {report["sizes"]["validation"]} validation, {report["sizes"]["test"]} test. GPU: {report["device"]}; {report["seconds"]/60:.1f} мин. Все варианты одинаково дообучают spatial-декодер 4 эпохи, энкодер заморожен, начальные веса и порядок данных одинаковы. Выбор эпох и варианта только по валидации. Рабочий пайплайн и исходные метки не изменяются.</p>', '<p>Фильтр применяется после корректной полярности DICOM, только внутри изображения без padding. Оставляем исходную яркость пикселей выше квантиля, остальные делаем чёрными. При одинаковой яркости сохраняется группа целиком, поэтому фактическая доля немного отличается. Для построения контуров и осей используется исходный DICOM, а не отфильтрованный.</p>', '<h2>Метрики на одинаковом тесте</h2>',table(['Вариант','Недобор ↓','Перебор','Ошибка пар, мм ↓','Полнота ≤5 мм ↑','Точность ≤5 мм ↑','Лучшая эпоха','Фактически сохранено'],rows)]
    for scope in ['originals','augmentations']:
        parts+=['<h3>'+scope+'</h3>',table(['Вариант','N','Недобор','Ошибка пар, мм','Полнота ≤5 мм'],[[labels[k],v[scope]['images'],n(v[scope]['under_fraction']),n(v[scope]['matched_error_mm']),n(v[scope]['recall_5mm'])] for k,v in report['results'].items()])]
    parts+=['<h2>Сравнение с предыдущим исследованием</h2><p>Тест тот же, но предыдущие новые головы обучались на 873 примерах и другой архитектуре; здесь spatial-декодер обучен на 489. Причинный вывод о фильтре делаем по новым одинаковым контрольным вариантам, а не по исторической разнице.</p>',table(['Предыдущий вариант','Недобор','Ошибка, мм','Полнота ≤5 мм'],[[k,n(v['under_fraction']),n(v['matched_error_mm']),n(v['recall_5mm'])] for k,v in report['previous_results'].items()]), '<h2>Новые штрафы</h2><p>Все штрафы перехода через границу квадратичные: ReLU(Lmin−L)², ReLU(Lupper−Ltop)², ReLU(Llower−Lbot)². Ltop — среднее двух первых полных межлинейных осей, Lbot — двух последних (если доступна одна, используется одна). Lmin=H/(H/Lmean_reference+2). Недобор: ReLU(Nreference−sum(presence))². Точное число после декодирования измеряется отдельно; мягкая сумма не является точным детектором количества. Длины в потере — дифференцируемое приближение по положениям линий и нормалям в физических единицах; контуры оцениваются отдельно.</p>', '<h2>Крайние оси и непрерывность</h2><p>Вес своей оси a/(a+0,25b), где a и b — физические длины. При одинаковых длинах вес своей оси 80%, соседней 20%. Общие точки сохраняются, конечные точки лежат на первой пересечённой стороне рамки. Если собственный контур не построен, применяется экстраполяция соседа с флагом проверки.</p>',table(['Вариант','N','Угол определён','MAE к ручным линиям, °','Изменение угла, °','Макс. разрыв цепочки, px','Требуют проверки'],[[k,v['n'],v['defined'],n(v['mae_to_manual_deg']),n(v['mean_angle_change_deg']),n(v['max_chain_discontinuity_px']),v['review_count']] for k,v in report['axis_summary'].items()]), '<p>Эталон угла — этот же алгоритм на ручных линиях, а не независимая разметка истинной оси. Старые табличные метки не пересчитывались.</p>', '<h2>Одни и те же снимки: входы и предсказанные линии</h2>']
    with (ROOT/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(csv.DictReader(f),1)}
    for index in list(range(5))+list(range(34,39)):
        r=test[index];base=Image.fromarray((read_dicom_image(r.source_path)*255).astype('uint8')).convert('RGB')
        parts.append(f'<h3>Исходный №{numbers.get(r.source_id or r.relative_path)} · '+('аугментация' if r.source_id else 'оригинал')+'</h3><div class="grid">')
        for name in ['manual']+list(report['results']):
            if name=='manual':pic=base.copy();lines=read(r.geometry_path)['spine']['disc_lines'];title='Эталон'
            else:
                fraction=report['results'][name]['fraction'];image,t=items[index]
                filtered,_=filter_bright(image[None],[t],fraction)
                a=filtered[0].numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
                l,top=t['pad_left'],t['pad_top'];dw=round(t['width']*t['scale']);dh=round(t['height']*t['scale'])
                a=np.clip(a[:,top:top+dh,l:l+dw],0,1).transpose(1,2,0)
                pic=Image.fromarray((a*255).round().astype('uint8')).resize((t['width'],t['height']),Image.Resampling.NEAREST)
                lines=predictions[name][r.relative_path];title=labels[name]
            draw=ImageDraw.Draw(pic)
            for line in lines:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
            filename=f'{index}_{name}.png';pic.thumbnail((300,430));pic.save(assets/filename)
            parts.append(f'<figure><img loading="lazy" src="spine_brightness_assets/{filename}"><figcaption>{title}; линий {len(lines)}</figcaption></figure>')
        parts.append('</div>')
    parts.append('<h2>Оси: прежние и новые веса на одинаковых линиях</h2>')
    for name,rows in examples.items():
        for j,(r,g,new,old) in enumerate(rows):
            base=Image.fromarray((read_dicom_image(r.source_path)*255).astype('uint8')).convert('RGB');parts.append(f'<h3>{name} · исходный №{numbers.get(r.source_id or r.relative_path)}</h3><div class="grid">')
            for tag,a in [('old',old),('new',new)]:
                pic=base.copy();draw=ImageDraw.Draw(pic)
                for line in g['spine']['disc_lines']:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
                for z in [(a.get('upper_fragment') or {})]+a['axes']+[(a.get('lower_fragment') or {})]:
                    if z.get('axis_points'):draw.line([tuple(p) for p in z['axis_points']],fill='cyan',width=2)
                if a.get('global_axis_points'):draw.line([tuple(p) for p in a['global_axis_points']],fill='red',width=2)
                filename=f'axes_{name}_{j}_{tag}.png';pic.thumbnail((380,500));pic.save(assets/filename);parts.append(f'<figure><img loading="lazy" src="spine_brightness_assets/{filename}"><figcaption>{tag}: {n(a["global_angle_deg"])}°; проверка {a["review_required"]}</figcaption></figure>')
            parts.append('</div>')
    page.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>DXA: яркие пиксели</title><style>body{font:16px system-ui;max-width:1700px;margin:25px auto;padding:20px;background:#101923;color:#eef}p{line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #567;padding:8px}.grid{display:flex;gap:10px;overflow-x:auto}figure{margin:0;min-width:210px}img{max-width:300px}</style>'+''.join(parts)+'</html>',encoding='utf-8')

if __name__=='__main__':run()
