"""Compare saved hip candidates at the same original-coordinate resolution."""
import json,shutil
import numpy as np
from .final_assembly import OUT,save,clear,event
from .predict import _load_model,predict_file
from .final_test_metrics import box_iou
from dxa_project.augmentation.core import prepare_geometry


def select(groups,bundle,device):
    old=json.loads((OUT/'hip_selection.json').read_text(encoding='utf-8'))
    if old.get('selection_space')=='original DICOM coordinates':return old
    event('module_validation',task='hip',scope='same original-coordinate validation masks/boxes; no test')
    candidates={};rows=[]
    for architecture in ('light','heavy'):
        folder=OUT/('hip_'+architecture);clear()
        loader=lambda task,*args:_load_model(task,bundle if task=='router' else folder,device)
        values=[]
        for r in groups['validation']:
            if r.region=='SPINE':continue
            gt=prepare_geometry(json.loads(r.geometry_path.read_text(encoding='utf-8')),r.region)
            pred=predict_file(r.source_path,folder,force_region=r.region,model_loader=loader,device_override=str(device),spacing_override_mm=r.spacing_mm)
            target={tuple(p) for p in gt['hip'].get('lesser_trochanter_pixels',[])}
            guess={tuple(p) for p in pred['geometry']['hip'].get('lesser_trochanter_pixels',[])}
            dice=2*len(target&guess)/(len(target)+len(guess)) if target or guess else 1.
            iou=box_iou(gt['hip']['roi_box'],pred['geometry']['hip']['roi_box'])
            values.append((dice,iou));rows.append({'architecture':architecture,'relative_path':r.relative_path,'dice':dice,'roi_iou':iou})
        candidates[architecture]={'pixel_dice':float(np.mean([v[0] for v in values])),
                                  'mean_roi_iou':float(np.mean([v[1] for v in values])),'images':len(values)}
    roi=max(candidates,key=lambda k:candidates[k]['mean_roi_iou']);mask=max(candidates,key=lambda k:candidates[k]['pixel_dice'])
    shutil.copy2(OUT/('hip_'+roi)/'hip.pt',bundle/'hip.pt');shutil.copy2(OUT/('hip_'+mask)/'hip.pt',bundle/'hip_mask.pt');clear()
    report={'validation':candidates,'roi':roi,'mask':mask,'test_used':False,'selection_space':'original DICOM coordinates',
            'training_resolution_validation':old['validation'],
            'candidate_epochs':'best saved combined ROI/Dice epoch of each retrained family; no retrospective all-epoch search'}
    save(OUT/'hip_selection.json',report);save(OUT/'hip_original_validation_rows.json',rows);return report
