"""Exercise the new deployed validation callback on both sides and cropped points."""
import json,gc,random
import torch
from .retrain_reviewed import setup,ROOT,OUT,AUG,protocol,validation_augments,Validation
from .data import load_records,load_augmented_records,collate
from .landmarks import PointsNet,PointDataset,point_loss
from .experiments import VARIANTS

def main():
    device=setup();records=load_records(ROOT);p=protocol(records);aug=load_augmented_records(ROOT,AUG,records)
    originals=[next(r for r in records if p['partition_by_path'][r.relative_path]=='validation' and r.region==region) for region in ('LEG_LEFT','LEG_RIGHT')]
    candidates=validation_augments(aug,p['partition_by_path'])
    synthetic=[r for r in candidates if any(v is None for v in json.loads(r.geometry_path.read_text(encoding='utf-8'))['hip']['landmarks'].values())][:2]
    assert len(synthetic)==2
    rows=originals+synthetic
    boxes={r.relative_path:[0.,0.,float(json.loads(r.geometry_path.read_text(encoding='utf-8'))['image_width']-1),float(json.loads(r.geometry_path.read_text(encoding='utf-8'))['image_height']-1)] for r in rows}
    validation=Validation(originals,synthetic,boxes);reports=[]
    for name,c in VARIANTS.items():
        torch.manual_seed(42);model=PointsNet(False,c['mode']).to(device);ds=PointDataset(rows,c['size'],c['crop'])
        x,t=collate([ds[i] for i in range(2)]);loss=point_loss(model(x.to(device)),t,c['loss']);assert torch.isfinite(loss)
        loss.backward();metrics,result,_=validation(model,c,device)
        assert len(result)==4 and 0<=metrics['selection_score']<=1
        reports.append({'variant':name,'finite_loss':True,'validation_rows':len(result),'both_sides':True,'absent_points_exercised':True,
                        'technical_smoke_only':True,'not_model_quality':True})
        print(json.dumps(reports[-1]),flush=True);del model,loss,x,t;gc.collect();torch.cuda.empty_cache()
    (OUT/'preflight.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
if __name__=='__main__':main()
