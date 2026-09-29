"""Dedicated hip landmarks: shared/independent networks and context crops."""
import json
import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from .models import SpatialNet
from .data import read_dicom_image,LANDMARKS
from .decoding import decode_heatmap

class PointsNet(nn.Module):
    def __init__(self,pretrained=True,mode='shared',architecture='light'):
        super().__init__();self.mode=mode
        if mode not in ('shared','independent'):raise ValueError(mode)
        def network(channels):
            m=SpatialNet('HIP',pretrained,'resnet50' if architecture=='heavy' else 'resnet18')
            m.spatial=nn.Conv2d(16,channels,1);m.presence=nn.Linear(m.presence.in_features,channels);m.roi=None
            return m
        self.networks=nn.ModuleList([network(3)] if mode=='shared' else [network(1) for _ in range(3)])
    def forward(self,x):
        result=[m(x) for m in self.networks]
        return {key:torch.cat([r[key] for r in result],dim=1) for key in ('spatial','presence')}

def point_input(image,size,flipped=False,box=None,context=.25):
    h,w=image.shape;ox=oy=0
    if box is not None:
        x1,y1,x2,y2=box;dx=(x2-x1)*context;dy=(y2-y1)*context
        ox=max(0,int(np.floor(x1-dx)));oy=max(0,int(np.floor(y1-dy)))
        endx=min(w,int(np.ceil(x2+dx))+1);endy=min(h,int(np.ceil(y2+dy))+1)
        if endx-ox>=16 and endy-oy>=16:image=image[oy:endy,ox:endx]
        else:ox=oy=0
    h,w=image.shape
    if flipped:image=np.fliplr(image)
    scale=min(size/w,size/h);dw,dh=round(w*scale),round(h*scale)
    left,top=(size-dw)//2,(size-dh)//2
    canvas=Image.new('L',(size,size));canvas.paste(Image.fromarray((image*255).astype(np.uint8)).resize((dw,dh),Image.Resampling.BILINEAR),(left,top))
    rgb=torch.from_numpy(np.repeat(np.asarray(canvas,dtype=np.float32)[None],3,axis=0)/255)
    rgb=(rgb-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    meta={'width':w,'height':h,'left':left,'top':top,'scale':scale,'scale_x':dw/w,'scale_y':dh/h,
          'resized_width':dw,'resized_height':dh,'flipped':flipped,'origin_x':ox,'origin_y':oy,'size':size}
    return rgb,meta

class PointDataset(Dataset):
    def __init__(self,records,size=256,crop='full',predicted_boxes=None):
        self.records=records;self.size=size;self.crop=crop;self.predicted_boxes=predicted_boxes
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
        image=read_dicom_image(r.source_path)
        if image.shape!=(g['image_height'],g['image_width']):raise ValueError('Point geometry and DICOM dimensions differ')
        box=(self.predicted_boxes.get(r.relative_path) if self.predicted_boxes is not None
             else g['hip']['roi_box']) if self.crop=='roi' else None
        tensor,meta=point_input(image,self.size,r.region=='LEG_RIGHT',box)
        s=self.size;maps=np.zeros((3,s,s),np.float32);present=np.zeros(3,np.float32)
        known=np.ones(3,np.float32);coords=np.zeros((3,2),np.float32)
        yy,xx=np.mgrid[:s,:s]
        for k,key in enumerate(LANDMARKS):
            p=g['hip']['landmarks'][key]
            if p is None:continue
            x,y=p[0]-meta['origin_x'],p[1]-meta['origin_y']
            if not(0<=x<meta['width'] and 0<=y<meta['height']):known[k]=0;continue
            if meta['flipped']:x=meta['width']-1-x
            x=meta['left']+x*meta['scale_x'];y=meta['top']+y*meta['scale_y']
            sigma=3*self.size/256
            maps[k]=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma));present[k]=1
            coords[k]=[x/(s-1),y/(s-1)]
        valid=np.zeros((s,s),np.float32)
        valid[meta['top']:meta['top']+meta['resized_height'],meta['left']:meta['left']+meta['resized_width']]=1
        return tensor,{'maps':torch.from_numpy(maps),'present':torch.from_numpy(present),'known':torch.from_numpy(known),
            'coords':torch.from_numpy(coords),'valid':torch.from_numpy(valid),'meta':meta,'record':r}

def point_loss(outputs,targets,kind='heatmap'):
    device=outputs['spatial'].device
    truth=torch.stack([t['maps'] for t in targets]).to(device)
    known=torch.stack([t['known'] for t in targets]).to(device)
    present=torch.stack([t['present'] for t in targets]).to(device)
    valid=torch.stack([t['valid'] for t in targets]).to(device)[:,None]
    mask=valid*known[:,:,None,None]
    loss=10*((torch.sigmoid(outputs['spatial'])-truth).square()*(1+100*truth)*mask).sum()/mask.sum().clamp_min(1)
    loss+=(nn.functional.binary_cross_entropy_with_logits(outputs['presence'],present,reduction='none')*known).sum()/known.sum().clamp_min(1)
    if kind=='coordinate':
        logits=outputs['spatial'].masked_fill(valid==0,-1e4)
        distribution=torch.softmax(logits.flatten(2),dim=-1).view_as(logits)
        s=truth.shape[-1];grid=torch.linspace(0,1,s,device=device)
        xy=torch.stack(((distribution*grid[None,None,None,:]).sum((-2,-1)),(distribution*grid[None,None,:,None]).sum((-2,-1))),dim=-1)
        coords=torch.stack([t['coords'] for t in targets]).to(device)
        loss+=5*((xy-coords).abs().sum(-1)*present*known).sum()/(present*known).sum().clamp_min(1)
    elif kind!='heatmap':raise ValueError(kind)
    return loss

def predict_points(model,image,size,side='LEFT',box=None,decoder='masked_argmax',presence_threshold=.5,peak_threshold=.35):
    tensor,meta=point_input(image,size,side=='RIGHT',box)
    with torch.no_grad():out=model(tensor[None].to(next(model.parameters()).device))
    maps=torch.sigmoid(out['spatial'][0]).cpu().numpy();presence=torch.sigmoid(out['presence'][0]).cpu().numpy()
    logits=out['spatial'][0].cpu().numpy()
    points={};details={}
    for k,key in enumerate(LANDMARKS):
        # Preserve strict ranking when float32 sigmoid saturates at 1.
        method='masked_logit_argmax' if decoder=='masked_argmax' else decoder
        point,peak=decode_heatmap(maps[k],meta,method,raw_logits=logits[k])
        points[key]=point if presence[k]>=presence_threshold and peak>=peak_threshold else None
        details[key]={'presence':float(presence[k]),'peak':peak}
    return points,details
