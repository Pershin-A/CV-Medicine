"""Compare both tested detector families on the same new validation partition."""
import json,shutil,random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .final_assembly import ROOT,OUT,save,event,clear
from .train import train_task
from .data import DxaDataset,collate
from .models import detector_images
from .predict import _load_model
from .evaluate import binary_metrics


def f1(report):
    row=next(h for h in report['history'] if h['epoch']==report['best_epoch']);m=row['validation_metrics']
    p=m['box_precision_iou50'];r=m['box_recall_iou50'];return 2*p*r/max(1e-9,p+r)


def select_and_calibrate(groups,bundle,device):
    heavy=OUT/'artifact_heavy';heavy.mkdir(exist_ok=True)
    report_path=heavy/'artifact_report.json'
    if not report_path.exists():
        clear();event('training',task='artifact',architecture='heavy',scope='common-split candidate comparison')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        report=train_task('artifact',groups['train'],groups['validation'],heavy,device,24,800,4,None,True,'heavy',2,1e-5,3e-5,'cosine',6,True,mixed_precision=True)
        save(report_path,report)
    scores={k:f1(json.loads((OUT/('artifact_'+k)/'artifact_report.json').read_text(encoding='utf-8'))) for k in ('light','heavy')}
    selected=max(scores,key=scores.get);shutil.copy2(OUT/('artifact_'+selected)/'artifact.pt',bundle/'artifact.pt')
    shutil.copy2(OUT/('artifact_'+selected)/'artifact_report.json',bundle/'artifact_report.json');clear()
    model,size=_load_model('artifact',bundle,device);records=[r for r in groups['validation'] if r.region=='SPINE']
    rows=[];model.eval()
    with torch.no_grad():
        for x,ts in DataLoader(DxaDataset(records,size),batch_size=2,collate_fn=collate):
            outputs=model(detector_images(x.to(device)))
            for out,t in zip(outputs,ts):
                rows.append({'truth':int(len(t['artifact_boxes'])>0),'score':float(out['scores'].max()) if len(out['scores']) else 0.})
    best=None
    values=np.unique([0.]+[r['score'] for r in rows])
    for threshold in np.unique([.5,1.]+[v for v in ((values[:-1]+values[1:])/2).tolist() if v>0]):
        m=binary_metrics([{**r,'prediction':int(r['score']>=threshold)} for r in rows]);score=(m['f1']+m['balanced_accuracy'])/2
        if best is None or (score,-abs(float(threshold)-.5))>best[:2]:best=(score,-abs(float(threshold)-.5),float(threshold),m)
    threshold=best[2]
    ckpt=torch.load(bundle/'artifact.pt',map_location='cpu',weights_only=False);ckpt['artifact_threshold']=threshold;torch.save(ckpt,bundle/'artifact.pt')
    save(OUT/'artifact_selection.json',{'box_f1_iou50_validation':scores,'selected':selected,'image_flag_threshold':threshold,
                                      'threshold_validation_metrics':best[3],'threshold_criterion':'mean image F1 and balanced accuracy; originals validation only',
                                      'test_used':False});clear()
