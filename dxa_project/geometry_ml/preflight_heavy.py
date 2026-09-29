"""CPU checks on real labeled DICOM; keeps the GPU available to current training."""
import json
import time
import gc
from pathlib import Path
import torch
from .data import load_records,DxaDataset,collate
from .landmarks import PointsNet,PointDataset,point_loss
from .train import _model,_loss
from .retrain_reviewed import ROOT,OUT,save

def main():
    torch.set_num_threads(2);torch.manual_seed(42);records=load_records(ROOT);reports=[]
    for task in ('spine','hip','hip_points','artifact'):
        start=time.perf_counter()
        rows=[r for r in records if (r.region=='SPINE')==(task in ('spine','artifact'))]
        if task=='artifact':
            positive=next(r for r in rows if json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects'])
            negative=next(r for r in rows if not json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects'])
            rows=[positive,negative]
        else:rows=rows[:2]
        size=800 if task=='artifact' else 512 if task=='hip_points' else 384
        if task=='hip_points':model=PointsNet(False,'shared','heavy');dataset=PointDataset(rows,size,'full')
        else:model=_model(task,False,'heavy');dataset=DxaDataset(rows,size)
        images,targets=collate([dataset[i] for i in range(len(rows))]);model.train()
        loss=point_loss(model(images),targets) if task=='hip_points' else _loss(task,model,images,targets,torch.device('cpu'))[0]
        if not torch.isfinite(loss):raise FloatingPointError(f'{task}: invalid loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5);optimizer.step()
        report=dict(task=task,size=size,batch=len(rows),loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.perf_counter()-start,
                    parameter_count=sum(p.numel() for p in model.parameters()),device='cpu',architecture='heavy',status='passed')
        reports.append(report);save(OUT/'heavy_preflight.json',reports);print(json.dumps(report),flush=True)
        del model,optimizer,images,targets,loss,dataset;gc.collect()

if __name__=='__main__':main()
