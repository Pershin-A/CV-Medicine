"""Compare decoded localization on the same correctly routed original files."""
import json
from pathlib import Path
import numpy as np
from .data import load_records,CRESTS
from .train import _box_iou
from dxa_project.augmentation.core import prepare_geometry

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def evaluate(root,folder):
    records={r.relative_path:r for r in load_records(root)}
    stats={'hips':0,'spines':0,'roi_ious':[],'crest_errors':[],'visible_crests':0,'missing_crests':0,
           'intersection':0,'prediction_pixels':0,'truth_pixels':0,'box_matches':0,'truth_boxes':0,'prediction_boxes':0}
    for i,row in enumerate(read(folder/'predictions.json'),1):
        if row['truth_region']!=row['predicted_region']:continue
        record=records[row['relative_path']];g=prepare_geometry(read(record.geometry_path),record.region)
        p=read(folder/f'prediction_{i:03}.json')['geometry']
        if record.region=='SPINE':
            stats['spines']+=1
            for key in CRESTS:
                truth=g['spine']['iliac_crests'][key];pred=p['spine']['iliac_crests'][key]
                if truth is None:continue
                stats['visible_crests']+=1
                if pred is None:stats['missing_crests']+=1
                else:stats['crest_errors'].append(float(np.hypot((pred[0]-truth[0])*record.spacing_mm[1],(pred[1]-truth[1])*record.spacing_mm[0])))
            truth=[b['bbox'] for b in g['spine']['foreign_objects']];pred=[b['bbox'] for b in p['spine']['foreign_objects']]
            stats['truth_boxes']+=len(truth);stats['prediction_boxes']+=len(pred);available=set(range(len(pred)))
            for box in truth:
                if not available:break
                best=max(available,key=lambda k:_box_iou(box,pred[k]))
                if _box_iou(box,pred[best])>=.5:stats['box_matches']+=1;available.remove(best)
        else:
            stats['hips']+=1
            if g['hip']['roi_box'] is not None and p['hip']['roi_box'] is not None:
                stats['roi_ious'].append(_box_iou(g['hip']['roi_box'],p['hip']['roi_box']))
            width=g['image_width'];a={y*width+x for x,y in g['hip']['lesser_trochanter_pixels']}
            b={y*width+x for x,y in p['hip']['lesser_trochanter_pixels']}
            stats['intersection']+=len(a&b);stats['truth_pixels']+=len(a);stats['prediction_pixels']+=len(b)
    inter=stats['intersection'];den=stats['truth_pixels']+stats['prediction_pixels']
    return {'correctly_routed_hips':stats['hips'],'correctly_routed_spines':stats['spines'],
            'mean_roi_iou':float(np.mean(stats['roi_ious'])) if stats['roi_ious'] else None,
            'trochanter_micro_dice':2*inter/max(den,1),'trochanter_micro_iou':inter/max(den-inter,1),
            'mean_crest_error_mm':float(np.mean(stats['crest_errors'])) if stats['crest_errors'] else None,
            'missing_visible_crests':stats['missing_crests'],'visible_crests':stats['visible_crests'],
            'box_recall_iou50':stats['box_matches']/max(stats['truth_boxes'],1),
            'box_precision_iou50':stats['box_matches']/max(stats['prediction_boxes'],1),
            'matched_boxes':stats['box_matches'],'ground_truth_boxes':stats['truth_boxes'],'predicted_boxes':stats['prediction_boxes']}

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    report={name:evaluate(root,folder) for name,folder in [('baseline',root/'dxa_project/outputs/geometry_ml_augmented_5epochs/evaluation'),
                                                         ('improved',out/'pipeline/evaluation')]}
    report['scope']='decoded localization, correctly routed originals only; trochanter reference mask derived from manual curves; nominal physical spacing'
    (out/'localization_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
