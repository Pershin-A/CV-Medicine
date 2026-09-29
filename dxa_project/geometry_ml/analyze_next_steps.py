"""Audit artifact balance, ROI scale, and the two landmark visibility gates."""
from pathlib import Path
import json, csv
import numpy as np
import torch
from .data import load_records,load_augmented_records,read_dicom_image,LANDMARKS
from .train import split_records
from .predict import _load_model,prepare_input,_original_point

def main():
    root=Path(__file__).resolve().parents[2];run=root/'dxa_project/outputs/geometry_ml_augmented_5epochs'
    records=load_records(root);train,valid=split_records(records,0)
    with (root/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        reference={r['relative_path']:r for r in csv.DictReader(f)}
    train_studies={r.study for r in train}
    aug=load_augmented_records(root,root/'dxa_project/outputs/augmented_15000_final',records)
    def artifacts(rows):
        rows=[r for r in rows if r.region=='SPINE']; positive=[];boxes=0;areas=[];disagreements=[];author_positive=0
        for r in rows:
            g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
            b=g['spine']['foreign_objects'];boxes+=len(b)
            if r.relative_path in reference:
                value=reference[r.relative_path].get('spine_artifact','')
                if value not in ('',None):
                    author_positive+=int(float(value))
                    if int(float(value))!=int(bool(b)):disagreements.append(r.relative_path)
            if b:positive.append(r)
            for item in b:
                x1,y1,x2,y2=item['bbox'];areas.append((x2-x1)*(y2-y1)/(g['image_width']*g['image_height']))
        return {'images':len(rows),'positive_images':len(positive),'positive_fraction':len(positive)/len(rows),
                'positive_studies':len({r.study for r in positive}),'all_studies':len({r.study for r in rows}),
                'boxes':boxes,'median_box_area_fraction':float(np.median(areas)) if areas else None,
                'author_positive_originals':author_positive,'visual_author_disagreements_originals':disagreements}
    report={'artifacts':{'train_originals':artifacts(train),'validation_originals':artifacts(valid),
                         'train_with_augmentation':artifacts(train+[r for r in aug if r.study in train_studies])}}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,size=_load_model('hip',run,device)
    points=[];roi=[]
    for r in [r for r in valid if r.region!='SPINE']:
        image=read_dicom_image(r.source_path);tensor,meta=prepare_input(image,size,flipped=r.region=='LEG_RIGHT')
        g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
        with torch.no_grad():out=model(tensor[None].to(device))
        maps=torch.sigmoid(out['spatial'][0,:3]).cpu().numpy()
        presence=torch.sigmoid(out['presence'][0]).cpu().numpy()
        for k,name in enumerate(LANDMARKS):
            visible=g['hip']['landmarks'][name] is not None;peak=float(maps[k].max())
            py,px=np.unravel_index(maps[k].argmax(),maps[k].shape)
            padding=not(meta['left']<=px<meta['left']+round(image.shape[1]*meta['scale'])
                        and meta['top']<=py<meta['top']+round(image.shape[0]*meta['scale']))
            ox,oy=_original_point(float(px),float(py),meta)
            margin=.025*min(image.shape)
            border=not(margin<=ox<=image.shape[1]-1-margin and margin<=oy<=image.shape[0]-1-margin)
            points.append({'name':name,'visible':visible,'presence':float(presence[k]),'peak':peak,
                           'presence_gate':bool(presence[k]>=.5),'peak_gate':peak>=.35,
                           'argmax_in_padding':padding,'border_violation':border})
        box=g['hip']['roi_box']
        if box:
            x1,y1,x2,y2=box;roi.append({'width_model_px':(x2-x1)*meta['scale'],
                    'height_model_px':(y2-y1)*meta['scale'],'area_fraction':(x2-x1)*(y2-y1)/(image.shape[0]*image.shape[1])})
    visible=[p for p in points if p['visible']]
    report['visible_hip_landmarks']={'n':len(visible),'rejected_by_presence':sum(not p['presence_gate'] for p in visible),
            'rejected_by_peak':sum(not p['peak_gate'] for p in visible),
            'rejected_by_either':sum(not(p['presence_gate'] and p['peak_gate']) for p in visible),
            'argmax_in_padding':sum(p['argmax_in_padding'] for p in visible),
            'border_margin_violations':sum(p['border_violation'] for p in visible),
            'presence_quantiles':np.quantile([p['presence'] for p in visible],[0,.1,.5,.9,1]).tolist(),
            'peak_quantiles':np.quantile([p['peak'] for p in visible],[0,.1,.5,.9,1]).tolist(),
            'by_point':{name:{'n':sum(p['name']==name for p in visible),
                'rejected':sum(p['name']==name and not(p['presence_gate'] and p['peak_gate']) for p in visible)} for name in LANDMARKS}}
    report['hip_roi_size']={key:np.quantile([r[key] for r in roi],[0,.1,.5,.9,1]).tolist() for key in roi[0]}
    report['roi_note']='ROI is a prediction target; current input is the whole image letterboxed to 256, not a crop of the annotated ROI'
    (run/'next_steps_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
