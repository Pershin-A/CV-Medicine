"""Measure real batch-8 training feasibility for costly light point variants."""
from pathlib import Path
import json,time,gc
import torch
from .data import load_records,collate
from .landmarks import PointsNet,PointDataset,point_loss
from .experiments import VARIANTS

def main():
    root=Path(__file__).resolve().parents[2];torch.hub.set_dir(str(root/'dxa_project/outputs/torch_hub'))
    torch.backends.cudnn.benchmark=True;device=torch.device('cuda')
    records=[r for r in load_records(root) if r.region!='SPINE'][:8];report=[]
    for name in ('shared512','independent256'):
        config=VARIANTS[name];model=PointsNet(True,config['mode']).to(device)
        x,t=collate([PointDataset(records,config['size'])[i] for i in range(8)])
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);torch.cuda.reset_peak_memory_stats()
        times=[]
        try:
            for i in range(4):
                torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                loss=point_loss(model(x.to(device)),t,config['loss']);assert torch.isfinite(loss)
                loss.backward();optimizer.step();torch.cuda.synchronize()
                if i:times.append(time.perf_counter()-start)
            result={'variant':name,'batch_size':8,'finite_loss':True,'seconds_per_batch':sum(times)/len(times),
                    'process_peak_memory_GiB':torch.cuda.max_memory_allocated()/2**30}
        except torch.OutOfMemoryError:result={'variant':name,'batch_size':8,'fits':False}
        report.append(result);print(json.dumps(result),flush=True)
        del model,optimizer,x,t;gc.collect();torch.cuda.empty_cache()
    out=root/'dxa_project/outputs/retrained_20260929/point_batch_benchmark.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
