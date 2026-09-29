"""Batch GPU forwards with the same decoding/rules as the existing pipeline."""
import hashlib
import numpy as np
import torch
from .store import write_json
from dxa_project.geometry_ml.predict import _load_model,prepare_input,_original_point,predict_file
from dxa_project.geometry_ml.data import read_dicom_image,REGIONS
from dxa_project.geometry_ml.landmarks import point_input

def tensor_key(x):
    x=x.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).digest()

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return value

class CachedForward:
    def __init__(self,model,outputs):
        self.point_configuration=getattr(model,'point_configuration',None)
        self.axis_method=getattr(model,'axis_method',None)
        self.scoliosis_threshold=getattr(model,'scoliosis_threshold',None)
        self.artifact_threshold=getattr(model,'artifact_threshold',.5)
        self.outputs=outputs
    def parameters(self):return iter([torch.zeros(1)])
    def __call__(self,x):
        if isinstance(x,list):return [self.outputs[tensor_key(v)] for v in x]
        result=[self.outputs[tensor_key(v)] for v in x]
        if isinstance(result[0],dict):return {k:torch.cat([r[k] for r in result]) for k in result[0]}
        return torch.cat(result)

class InferenceEngine:
    def __init__(self,checkpoints,device='auto',max_batch=8):
        self.checkpoints=checkpoints
        self.device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
        self.max_batch=max_batch;self.batch_sizes={};self.memory_profiles={}
    @torch.inference_mode()
    def forwards(self,task,inputs,cache):
        model,size=_load_model(task,self.checkpoints,self.device)
        profiling=self.device.type=='cuda' and task not in self.batch_sizes
        cap=self.batch_sizes.get(task,1 if profiling else min(self.max_batch,2))
        if profiling:
            baseline=torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        outputs={};index=0
        while index<len(inputs):
            chunk=inputs[index:index+cap]
            try:
                batch=torch.stack(chunk).to(self.device)
                out=model(list(batch) if task=='artifact' else batch)
                if task=='artifact':items=cpu(out)
                elif isinstance(out,dict):items=[{k:cpu(v[i:i+1]) for k,v in out.items()} for i in range(len(chunk))]
                else:items=[cpu(out[i:i+1]) for i in range(len(chunk))]
                for tensor,value in zip(chunk,items):outputs[tensor_key(tensor)]=value
                index+=len(chunk)
                if profiling:
                    torch.cuda.synchronize(self.device)
                    additional=max(1,torch.cuda.max_memory_allocated(self.device)-baseline)
                    free,_=torch.cuda.mem_get_info(self.device)
                    cap=max(1,min(self.max_batch,int(.60*free/additional)))
                    self.memory_profiles[task]={'single_image_peak_extra_bytes':additional,'free_bytes':free,'memory_fraction':.60,'batch_size':cap}
                    profiling=False
            except torch.cuda.OutOfMemoryError:
                if cap==1:raise
                cap=max(1,cap//2);torch.cuda.empty_cache()
        self.batch_sizes[task]=cap
        cache[task]=(CachedForward(model,outputs),size)
        return outputs
    def predict(self,paths):
        images=[read_dicom_image(p) for p in paths];cache={}
        _,size=_load_model('router',self.checkpoints,self.device)
        tensors=[prepare_input(im,size,stretch=True)[0] for im in images]
        out=self.forwards('router',tensors,cache)
        regions=[REGIONS[out[tensor_key(t)].argmax().item()] for t in tensors]
        tasks=['spine','hip','artifact']+[task for task in ('scoliosis','hip_mask','spine_crests') if (self.checkpoints/(task+'.pt')).is_file()]
        for task in tasks:
            ids=[i for i,r in enumerate(regions) if (r=='SPINE')==(task not in ('hip','hip_mask'))]
            if not ids:continue
            _,size=_load_model(task,self.checkpoints,self.device)
            ts=[prepare_input(images[i],size,regions[i]=='LEG_RIGHT')[0] for i in ids]
            if task=='artifact':
                from dxa_project.geometry_ml.models import detector_images
                ts=detector_images(torch.stack(ts))
            self.forwards(task,ts,cache)
        if (self.checkpoints/'hip_points.pt').is_file() and any(r!='SPINE' for r in regions):
            model,psize=_load_model('hip_points',self.checkpoints,self.device)
            config=model.point_configuration;point_tensors=[]
            _,hsize=cache['hip']
            for i,r in enumerate(regions):
                if r=='SPINE':continue
                tensor,meta=prepare_input(images[i],hsize,r=='LEG_RIGHT')
                out=cache['hip'][0](tensor[None]);y1,y2,lateral=(out['roi'][0]*hsize).tolist()
                top=_original_point(0,min(y1,y2),meta)[1];bottom=_original_point(0,max(y1,y2),meta)[1]
                x=_original_point(lateral,0,meta)[0];w=images[i].shape[1]
                box=[0,top,x,bottom] if r=='LEG_LEFT' else [x,top,w-1,bottom]
                point_tensors.append(point_input(images[i],psize,r=='LEG_RIGHT',box if config['crop']=='roi' else None)[0])
                if config['crop']=='roi':point_tensors.append(point_input(images[i],psize,r=='LEG_RIGHT',None)[0])
            self.forwards('hip_points',point_tensors,cache)
        loader=lambda task,*args:cache[task]
        import json,pydicom
        from dxa_project.augmentation.generate import _source_spacing
        calibration=json.loads((self.checkpoints/'evaluation/calibration.json').read_text(encoding='utf-8'))
        area_mm2=calibration.get('rotation_threshold_mm2',calibration['rotation_threshold_px2']*1.05*.6)
        results=[]
        for p in paths:
            ds=pydicom.dcmread(p,stop_before_pixels=True);spacing,basis=_source_spacing(ds,True)
            result=predict_file(p,self.checkpoints,rotation_area_threshold_px2=area_mm2/(spacing[0]*spacing[1]),model_loader=loader,device_override='cpu')
            result.update(spacing_mm=spacing,spacing_basis=basis,rotation_threshold_mm2=area_mm2)
            results.append(result)
        return results

def render_overlay(image,geometry,destination,axes=None):
    from PIL import Image,ImageDraw
    canvas=Image.fromarray((np.clip(image,0,1)*255).astype('uint8')).convert('RGB');draw=ImageDraw.Draw(canvas)
    for line in geometry['spine']['disc_lines']:draw.line([tuple(p) for p in line['points']],fill='#00e870',width=2)
    for obj in geometry['spine']['foreign_objects']:draw.rectangle(obj['bbox'],outline='#ff8220',width=2)
    for points,color in ((geometry['spine']['iliac_crests'],'#ffa500'),(geometry['hip']['landmarks'],'#00d8ff')):
        for key,p in points.items():
            if p is not None:
                x,y=p;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
                labels={'image_left':'L','image_right':'R','greater_trochanter':'1','femoral_neck':'2','ischial_bone':'3'}
                draw.text((min(x+5,canvas.width-10),min(y,canvas.height-12)),labels[key],fill=color)
    if geometry['hip']['roi_box']:draw.rectangle(geometry['hip']['roi_box'],outline='#c864ff',width=2)
    if axes:
        for axis in axes.get('axes',[]):
            points=axis.get('axis_points')
            if points:draw.line([tuple(p) for p in points],fill='#ffff00',width=2)
        global_axis=axes.get('global_axis') or {}
        if global_axis.get('axis_points'):draw.line([tuple(p) for p in global_axis['axis_points']],fill='#ff66dd',width=2)
    mask=np.zeros(image.shape,dtype=np.uint8)
    for x,y in geometry['hip'].get('lesser_trochanter_pixels',[]):
        if 0<=x<mask.shape[1] and 0<=y<mask.shape[0]:mask[y,x]=255
    if mask.any():
        arr=np.array(canvas);arr[mask>0]=[255,0,0];canvas=Image.fromarray(arr)
    else:
        for trace in geometry['hip']['lesser_trochanter_traces']['trochanter']:
            points=trace.get('points',[]) if isinstance(trace,dict) else trace
            for i in range(0,len(points)-1,2):draw.line([tuple(points[i]),tuple(points[i+1])],fill='red',width=2)
    destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)
    Image.fromarray(mask).save(destination.with_name('mask.png'))

def save_prediction(store,id,result,path,model_version):
    folder=store.root/'results'/id;folder.mkdir(parents=True,exist_ok=True)
    render_overlay(read_dicom_image(path),result['geometry'],folder/'overlay.png',result.get('vertebral_axes'))
    result.update(result_id=id,model_version=model_version)
    write_json(folder/'geometry.json',result['geometry'])
    result['geometry']['hip']['lesser_trochanter_pixels']=[]
    result['geometry']['hip']['lesser_trochanter_mask_png']=f'/v1/results/{id}/files/mask.png'
    result['overlay_url']=f'/v1/results/{id}/files/overlay.png'
    result['geometry_url']=f'/v1/results/{id}/files/geometry.json'
    result['score_semantics']='router: softmax; scoliosis: uncalibrated classifier sigmoid with validation-selected threshold; other quality_scores: heuristic ranking'
    result['annotation_legend']={'1':'Большой вертел','2':'Шейка бедра','3':'Седалищная кость','L':'Подвздошная кость слева на снимке','R':'Подвздошная кость справа на снимке','red':'Малый вертел'}
    write_json(folder/'result.json',result);store.put('result',id,result)
    return result
