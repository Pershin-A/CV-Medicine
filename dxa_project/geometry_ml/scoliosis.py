"""Detect the manually annotated SCOLIOSIS label, independently of global tilt."""
import json, math, time
from collections import Counter
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from .data import read_dicom_image
from .predict import prepare_input
from .evaluate import binary_metrics,with_ci


class ScoliosisNet(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__();self.net=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.net.fc=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,1))
        for name,p in self.net.named_parameters():p.requires_grad_(name.startswith(('layer4.','fc.')))
    def forward(self,x):return self.net(x).flatten()
    def train(self,mode=True):
        super().train(mode)
        for name,m in self.net.named_children():
            if name not in ('layer4','fc'):m.eval()
        return self


class ScoliosisDataset(Dataset):
    def __init__(self,records,labels,size=256):self.records=records;self.labels=labels;self.size=size
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];x,_=prepare_input(read_dicom_image(r.source_path),self.size)
        return x,float(self.labels[r.relative_path]['spine_scoliosis']),i


def summarize(rows,threshold):
    usable=[{**r,'prediction':int(r['score']>=threshold)} for r in rows]
    return with_ci(usable,100)


def choose_threshold(rows):
    if len({r['truth'] for r in rows})<2:raise ValueError('Both scoliosis classes required in validation')
    best=None
    scores=np.unique([r['score'] for r in rows])
    candidates=np.unique([0.,.5,1.]+((scores[:-1]+scores[1:])/2).tolist())
    for threshold in candidates:
        metrics=binary_metrics([{**r,'prediction':int(r['score']>=threshold)} for r in rows])
        value=(metrics['f1']+metrics['balanced_accuracy'])/2
        # Prefer .5 on exact ties, do not use test for calibration.
        candidate=(value,-abs(float(threshold)-.5),float(threshold),metrics)
        if best is None or candidate[:2]>best[:2]:best=candidate
    return best[2],best[3],best[0]


def train_scoliosis(train,validation,labels,output,device,epochs=16):
    train=[r for r in train if labels[r.relative_path].get('spine_scoliosis') in (0,1)]
    validation=[r for r in validation if not r.source_id and labels[r.relative_path].get('spine_scoliosis') in (0,1)]
    if len({labels[r.relative_path]['spine_scoliosis'] for r in train})<2:raise ValueError('Both training classes required')
    ds=ScoliosisDataset(train,labels);vs=ScoliosisDataset(validation,labels)
    keys=[(labels[r.relative_path]['spine_scoliosis'],r.source_id or r.relative_path) for r in train]
    counts=Counter(keys);sources=Counter(k[0] for k in counts)
    sampler=WeightedRandomSampler([1/(counts[k]*sources[k[0]]) for k in keys],len(keys),replacement=True)
    loader=DataLoader(ds,batch_size=16,sampler=sampler,num_workers=2,persistent_workers=True)
    valid_loader=DataLoader(vs,batch_size=16,num_workers=2,persistent_workers=True)
    torch.manual_seed(42);model=ScoliosisNet(True).to(device)
    optimizer=torch.optim.AdamW([{'params':model.net.layer4.parameters(),'lr':2e-5},{'params':model.net.fc.parameters(),'lr':2e-4}],weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)
    best=-math.inf;history=[];stale=0;start=time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train();losses=[]
        for x,y,_ in loader:
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(x.to(device)),y.float().to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite scoliosis loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        model.eval();rows=[]
        with torch.no_grad():
            for x,y,indices in valid_loader:
                scores=torch.sigmoid(model(x.to(device))).cpu().tolist()
                rows.extend({'relative_path':validation[int(i)].relative_path,'study':validation[int(i)].study,'truth':int(t),'score':s} for i,t,s in zip(indices,y,scores))
        threshold,metrics,value=choose_threshold(rows)
        if value>best:
            best=value;stale=0
            torch.save({'task':'scoliosis','size':256,'state_dict':model.state_dict(),'threshold':threshold,'epoch':epoch+1,'label':'local spine_issue=SCOLIOSIS'},output/'scoliosis.pt')
            (output/'scoliosis_validation_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        else:stale+=1
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':metrics,'threshold':threshold,'selection_score':value,'seconds':time.perf_counter()-start})
        (output/'scoliosis_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({'task':'scoliosis',**history[-1]}),flush=True);scheduler.step()
        if stale>=6:break
    report={'train_images':len(train),'validation_originals':len(validation),'selection':'mean positive F1 and balanced accuracy; threshold and epoch use originals validation only','history':history,'seconds':time.perf_counter()-start}
    (output/'scoliosis_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report
