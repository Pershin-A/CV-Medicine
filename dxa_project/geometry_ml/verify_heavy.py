"""Verify heavier spatial and detection models with finite GPU training steps."""
import json,time,gc
from pathlib import Path
import torch
from .models import SpatialNet,artifact_detector

def run():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results=[]
    for region in ('SPINE','HIP'):
        start=time.perf_counter(); model=SpatialNet(region,False,'resnet50').to(device).train()
        image=torch.rand(2,3,128,128,device=device); result=model(image)
        loss=sum(value.float().square().mean() for value in result.values())
        loss.backward(); assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results.append({'model':'resnet50_unet_'+region,'parameters':sum(p.numel() for p in model.parameters()),
                        'outputs':{k:list(v.shape) for k,v in result.items()},'finite_forward_backward':True,
                        'seconds':time.perf_counter()-start})
        del model,result,loss,image;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    start=time.perf_counter();model=artifact_detector(False,True).to(device).train()
    model.transform.min_size=(256,);model.transform.max_size=256
    losses=model([torch.rand(3,256,256,device=device)],
                 [{'boxes':torch.tensor([[40.,45.,110.,140.]],device=device),'labels':torch.ones(1,dtype=torch.long,device=device)}])
    loss=sum(losses.values());loss.backward();assert torch.isfinite(loss)
    results.append({'model':'fasterrcnn_resnet50_fpn_v2','parameters':sum(p.numel() for p in model.parameters()),
                    'finite_forward_backward':True,'test_detector_resolution':256,
                    'losses':{k:float(v.detach()) for k,v in losses.items()},'seconds':time.perf_counter()-start})
    out=Path(__file__).resolve().parents[1]/'outputs/heavy_model_verification.json'
    out.write_text(json.dumps({'device':str(device),'random_initialization':True,'models':results},indent=2),encoding='utf-8')
    print(out,flush=True)

if __name__=='__main__':run()
