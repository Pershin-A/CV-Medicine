"""Independent iliac landmark module; angle priors must not dictate point selection."""
import json,random,shutil
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import DxaDataset,collate,CRESTS
from .predict import _heatmap_point,_load_model
from .final_assembly import OUT,save,clear,event
from .train import train_task


def point_loss(task,model,images,targets,device):
    output={k:v.float() for k,v in model(images.to(device)).items()}
    truth=torch.stack([t['crest'] for t in targets]).to(device)
    visible=torch.stack([t['crest_present'] for t in targets]).to(device)
    # Reuse the foreground-weighted masked heatmap objective of the best point pipeline.
    valid=torch.ones_like(truth)
    for n,t in enumerate(targets):
        if 'pad_top' not in t:continue
        valid[n].zero_();left,top=t['pad_left'],t['pad_top'];dw=round(t['width']*t['scale']);dh=round(t['height']*t['scale'])
        valid[n,:,top:top+dh,left:left+dw]=1
    loss=10*((torch.sigmoid(output['spatial'][:,1:])-truth).square()*(1+100*truth)*valid).sum()/valid.sum().clamp_min(1)
    loss+=F.binary_cross_entropy_with_logits(output['presence'],visible)
    return loss,output


class CrestSelection:
    def __init__(self,records,size=384):
        self.records=[r for r in records if r.region=='SPINE' and not r.source_id];self.size=size
        self.references={r.relative_path:json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['iliac_crests'] for r in self.records}
    def __call__(self,model,device,stats):
        visible=found=hits=0;errors=[];tp=tn=fp=fn=0;model.eval()
        loader=DataLoader(DxaDataset(self.records,self.size),batch_size=8,collate_fn=collate)
        with torch.no_grad():
            for images,targets in loader:
                output=model(images.to(device));logits=output['spatial'].float().cpu().numpy();maps=torch.sigmoid(output['spatial'].float()).cpu().numpy();presence=torch.sigmoid(output['presence']).cpu().numpy()
                for n,t in enumerate(targets):
                    meta={'width':t['width'],'height':t['height'],'left':t['pad_left'],'top':t['pad_top'],'scale':t['scale'],'size':self.size,'flipped':False}
                    for k,key in enumerate(CRESTS):
                        reference=self.references[t['relative_path']][key];point,peak=_heatmap_point(maps[n,k+1],meta,logits[n,k+1]);guess=point is not None and presence[n,k]>=.5 and peak>=.35
                        if reference is None:
                            fp+=bool(guess);tn+=not bool(guess);continue
                        visible+=1;tp+=bool(guess);fn+=not bool(guess)
                        if not guess:continue
                        delta=np.asarray(reference)-point;d=float(np.hypot(delta[0]*t['spacing_mm'][1],delta[1]*t['spacing_mm'][0]));errors.append(d);found+=1;hits+=d<=10
        pck=hits/max(1,visible);balanced=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp))
        penalized=(sum(errors)+60*(visible-found))/max(1,visible)
        score=1-pck+.1*(1-balanced)+.001*penalized/60
        stats.update(crest_pck_10mm=pck,crest_point_coverage=found/max(1,visible),crest_mean_error_mm=float(np.mean(errors)) if errors else None,
                     crest_presence_balanced_accuracy=balanced,crest_failure_penalized_error_mm=penalized,crest_selection_score=score)
        return score


def train_and_select(groups,bundle,device):
    folder=OUT/'spine_crests';folder.mkdir(exist_ok=True);path=folder/'spine_report.json';selection=CrestSelection(groups['validation'])
    if not path.exists():
        clear();event('training',task='spine_crests',architecture='heavy',scope='point-only loss; original validation PCK10')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        report=train_task('spine',groups['train'],groups['validation'],folder,device,24,384,8,None,True,'heavy',2,2e-5,2e-4,'cosine',6,True,
                          loss_callback=point_loss,selection_callback=selection,mixed_precision=True)
        save(path,report)
    candidate={};baseline={};clear();model,_=_load_model('spine',folder,device);a=selection(model,device,candidate)
    clear();model,_=_load_model('spine',bundle,device);b=selection(model,device,baseline)
    separate=a<b
    if separate:
        shutil.copy2(folder/'spine.pt',bundle/'spine_crests.pt');shutil.copy2(path,bundle/'spine_crests_report.json')
    result={'separate_module_selected':separate,'candidate_validation':candidate,'joint_validation':baseline,
            'criterion':'Original validation: 1-PCK10 +0.1*(1-presence balanced accuracy)+0.001*failure-penalized distance/60; missing visible point=60mm',
            'test_used':False,'initialization':'generic ImageNet only; point-only loss'}
    save(OUT/'crest_selection.json',result);clear();return result
