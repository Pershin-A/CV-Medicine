"""Choose among retained point checkpoints using inner validation only."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import load_records,collate
from .landmarks import PointsNet,PointDataset
from .experiments import evaluate

def select(root,out):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    rows=[r for r in load_records(root) if r.region!='SPINE' and protocol['partition_by_path'][r.relative_path]=='validation']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');results=[]
    for variant in ('shared256','coordinate256'):
        for name in ('hip_points.pt','hip_points_last.pt'):
            path=out/'trained'/variant/name;ckpt=torch.load(path,map_location='cpu',weights_only=False)
            model=PointsNet(False,ckpt['mode'],ckpt['architecture']).to(device);model.load_state_dict(ckpt['state_dict'])
            loader=DataLoader(PointDataset(rows,ckpt['size'],ckpt['crop']),batch_size=4,num_workers=2,collate_fn=collate)
            metrics,_,_=evaluate(model,loader,device,'masked_logit_argmax',ckpt['loss'])
            results.append({'variant':variant,'checkpoint':name,'path':str(path),'epoch':ckpt['epoch'],'metrics':metrics})
            del model
    chosen=min(results,key=lambda x:x['metrics']['mean_error_mm'])
    report={'chosen':chosen,'candidates':results,'criterion':'inner validation mean error with deployed raw-logit masked decoder; no outer test used'}
    (out/'point_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return chosen
