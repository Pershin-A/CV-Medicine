"""Train the complete spine geometry module, selecting on angular/localization metrics."""
import copy,json,random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader
from .data import DxaDataset,collate
from .models import spatial_loss
from .spine_angle_study import all_peak_priors,angle_summary
from .spine_brightness_study import quadratic_priors
from .spine_penalty_study import target,measure,summarize
from .predict import _spine_lines
from .train import train_task


def strict_angle(item):
    from .spine_angle_geometry import analyze_strict
    raw,g,spacing,polarity=item
    a=analyze_strict(raw,g,spacing,polarity)
    return a['global_angle_deg']


class SpineLoss:
    def __init__(self,records,c=2.):
        self.c=c
        self.coords={}
        for r in records:
            if r.region!='SPINE':continue
            values,visible=target(r);self.coords[r.relative_path]=torch.tensor(values[:int(visible.sum())])
    def __call__(self,task,model,images,targets,device):
        for t in targets:t['ordered_lines']=self.coords[t['relative_path']]
        output={k:v.float() for k,v in model(images.to(device)).items()};base=spatial_loss(output,targets,'SPINE')
        old=quadratic_priors(output['spatial'][:,:1],targets);p=all_peak_priors(output['spatial'][:,:1],targets,self.c)
        loss=base+2*old['count']+2*old['coordinate']+8*p['close']+4*p['holes']+4*p['tails']
        return loss,output


class SpineSelection:
    def __init__(self,records,loss,size=384):
        self.records=records;self.loss=loss;self.size=size;self.tasks=[]
        self.pool=ProcessPoolExecutor(max_workers=6)
        for r in records:
            ds=pydicom.dcmread(r.source_path);g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
            self.tasks.append((np.squeeze(ds.pixel_array),g,r.spacing_mm,'dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright'))
        self.reference=list(self.pool.map(strict_angle,self.tasks,chunksize=2))
    def __call__(self,model,device,stats):
        model.eval();pred=[];tasks=[];line_rows=[]
        loader=DataLoader(DxaDataset(self.records,self.size),batch_size=4,collate_fn=collate)
        with torch.no_grad():
            for x,ts in loader:
                maps=torch.sigmoid(model(x.to(device))['spatial'][:,0]).cpu().numpy()
                for heat,t in zip(maps,ts):
                    i=len(pred);r=self.records[i]
                    meta={'width':t['width'],'height':t['height'],'left':t['pad_left'],'top':t['pad_top'],'scale':t['scale'],'size':self.size,'flipped':False}
                    lines=_spine_lines(heat,meta);pred.append(lines);line_rows.append(measure(lines,r))
                    raw,g,spacing,polarity=self.tasks[i];g=copy.deepcopy(g);g['spine']['disc_lines']=lines;tasks.append((raw,g,spacing,polarity))
        angles=list(self.pool.map(strict_angle,tasks,chunksize=2));rows=[{'relative_path':r.relative_path,'study':r.study,'reference_angle':ref,'angle':angle,'flag':None} for r,ref,angle in zip(self.records,self.reference,angles)]
        original=angle_summary([r for r in rows if not r['relative_path'].startswith('aug/')],bootstrap=0)
        synthetic=angle_summary([r for r in rows if r['relative_path'].startswith('aug/')],bootstrap=0)
        lines=summarize(line_rows);recall=lines['recall_5mm'];precision=lines['precision_5mm'];f1=2*recall*precision/max(1e-9,recall+precision)
        score=.7*original['failure_penalized_mae_deg']/5+.3*synthetic['failure_penalized_mae_deg']/5+.2*(1-f1)
        stats.update({'angle_original':original,'angle_synthetic':synthetic,'divider_metrics':lines,'divider_f1_5mm':f1,'target_selection':score})
        return score


def train_final_spine(groups,augmented,protocol,bundle,device):
    from .final_assembly import save,AUG,OUT,clear,event
    # Validation originals are primary. Add balanced source-held synthetic strata.
    import csv
    with (AUG/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:table={f"aug/{r['image_path']}":r for r in csv.DictReader(f)}
    buckets={};rng=random.Random(42)
    for r in augmented:
        if r.region=='SPINE' and protocol['partition_by_path'][r.source_id]=='validation':buckets.setdefault(table[r.relative_path]['generation_group'],[]).append(r)
    selected=[]
    for name,pool in sorted(buckets.items()):rng.shuffle(pool);selected+=pool[:20]
    validation=[r for r in groups['validation'] if r.region=='SPINE']+selected
    loss=SpineLoss(groups['train']+validation);selection=SpineSelection(validation,loss);candidates={}
    try:
        for name,c in [('c2',2.),('c05',.5)]:
            folder=OUT/('spine_'+name);folder.mkdir(exist_ok=True);report_path=folder/'spine_report.json'
            if not report_path.exists():
                clear();torch.manual_seed(42);loss.c=c;event('training',task='spine',candidate=name)
                report=train_task('spine',groups['train'],validation,folder,device,24,384,4,None,True,'heavy',2,2e-5,2e-4,'cosine',6,True,
                                  loss_callback=loss,selection_callback=selection,mixed_precision=True)
                save(report_path,report)
            report=json.loads(report_path.read_text(encoding='utf-8'));best=next(h for h in report['history'] if h['epoch']==report['best_epoch'])
            candidates[name]={'score':best['validation_metrics']['target_selection'],'report':report,'validation':best['validation_metrics']}
    finally:selection.pool.shutdown()
    import shutil
    selected=min(candidates,key=lambda k:candidates[k]['score']);report=candidates[selected]['report']
    shutil.copy2(OUT/('spine_'+selected)/'spine.pt',bundle/'spine.pt')
    checkpoint=torch.load(bundle/'spine.pt',map_location='cpu',weights_only=False);checkpoint['axis_method']='strict_perpendicular_terminals';checkpoint['line_priors']='all_peaks_'+selected
    torch.save(checkpoint,bundle/'spine.pt')
    report['selection']='0.7 original angular MAE/5 +0.3 synthetic angular MAE/5 +0.2*(1-divider F1 <=5mm); undefined eligible angle=15deg'
    save(OUT/'spine_selection.json',{'selected':selected,'validation':{k:{'score':v['score'],'metrics':v['validation']} for k,v in candidates.items()},'test_used':False})
    save(bundle/'spine_report.json',report)
