"""Verify crop and independent point checkpoints in complete hip inference on CPU."""
from pathlib import Path
import json,os
import torch
from .data import load_records
from .predict import predict_file

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    records=[r for r in load_records(root) if r.region!='SPINE' and protocol['partition_by_path'][r.relative_path]=='validation']
    selected=[next(r for r in records if r.region==region) for region in ('LEG_LEFT','LEG_RIGHT')]
    # Avoid competing for GPU memory with the running long training.
    torch.cuda.is_available=lambda:False
    torch.set_num_threads(2);reports=[]
    for variant in ('roi384','independent256'):
        folder=out/'integration_smoke'/variant;folder.mkdir(parents=True,exist_ok=True)
        for source,name in ((root/'dxa_project/outputs/geometry_ml_augmented_5epochs/hip.pt','hip.pt'),
                            (root/'dxa_project/outputs/geometry_ml_augmented_5epochs/router.pt','router.pt'),
                            (out/'smoke'/variant/'hip_points.pt','hip_points.pt')):
            if not (folder/name).exists():os.link(source,folder/name)
        for record in selected:
            predicted=predict_file(record.source_path,folder,force_region=record.region)
            assert predicted['landmark_model'].startswith('dedicated_')
            assert set(predicted['quality_flags'])=={'hip_position','hip_roi','hip_rotation'}
            reports.append({'variant':variant,'region':record.region,'passed':True,
                            'crop_fallback':predicted.get('point_crop_fallback',False),'quality_flags':predicted['quality_flags']})
    (out/'point_integration_smoke.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports))

if __name__=='__main__':main()
