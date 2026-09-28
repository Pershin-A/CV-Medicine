"""Exhaustive file/metadata audit and sampled geometry recomputation."""
import argparse,csv,json,time,warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pydicom
from .core import validate_geometry,hip_position_ok,hip_roi_ok
from .generate import _labels,_read_rows

def audit(root,output,samples=90):
    start=time.perf_counter(); rows=_read_rows(root/'manifest.csv')
    problems=[]; counts=Counter(); seen=set(); pixel_shapes=Counter()
    for index,row in enumerate(rows,1):
        try:
            image=root/row['image_path']; geometry=root/row['geometry_path']
            ds=pydicom.dcmread(image); pixels=ds.pixel_array
            g=json.loads(geometry.read_text(encoding='utf-8'))
            validate_geometry(g,int(ds.Columns),int(ds.Rows))
            assert pixels.shape==(g['image_height'],g['image_width'])
            assert np.isfinite(pixels).all()
            uid=str(ds.SOPInstanceUID); assert uid not in seen; seen.add(uid)
            block=ds.private_block(0x0011,'DXA_MANUAL_LABELER',create=False)
            assert json.loads(ds[block.get_tag(9)].value)==g
            targets=json.loads(ds[block.get_tag(10)].value)
            for task in ('spine_position','spine_axis','spine_artifact','hip_position','hip_roi','hip_rotation'):
                if row.get(task,'')!='':assert targets[task]==int(float(row[task]))
            if row['region'].startswith('LEG_'):
                side=row['region'].removeprefix('LEG_'); spacing=(float(row['row_spacing_mm']),float(row['col_spacing_mm']))
                assert int(not hip_position_ok(g))==int(row['hip_position'])
                assert int(not hip_roi_ok(g,True,side,spacing))==int(row['hip_roi'])
            counts[(row['region'],row['generation_group'])]+=1
            pixel_shapes[pixels.shape]+=1
        except Exception as error:
            problems.append({'image':row['image_path'],'error':repr(error)})
        if index%2000==0:print(json.dumps({'audit_files':index,'total':len(rows),'problems':len(problems)}),flush=True)
    rng=np.random.default_rng(42); selected=[]
    for region in ('SPINE','LEG_LEFT','LEG_RIGHT'):
        for group in ('positive','negative_position','negative_axis_or_roi'):
            candidates=[r for r in rows if r['region']==region and r['generation_group']==group]
            if candidates:selected.extend(candidates[i] for i in rng.choice(len(candidates),min(len(candidates),max(1,samples//9)),replace=False))
    recomputed=[]
    for row in selected:
        ds=pydicom.dcmread(root/row['image_path']); g=json.loads((root/row['geometry_path']).read_text(encoding='utf-8'))
        block=ds.private_block(0x0011,'DXA_MANUAL_LABELER',create=False)
        targets=json.loads(ds[block.get_tag(10)].value)
        source={'axis_polarity':'dark' if ds.PhotometricInterpretation=='MONOCHROME1' else 'bright',
                'spine_artifact':str(targets.get('spine_artifact')),
                row['region'].removeprefix('LEG_').lower()+'_hip_rotation':str(targets.get('hip_rotation'))}
        labels=_labels(row['region'],g,{'roi_fully_visible':True},ds.pixel_array,
                       (float(row['row_spacing_mm']),float(row['col_spacing_mm'])),source)
        keys=('spine_position','spine_axis','spine_artifact') if row['region']=='SPINE' else ('hip_position','hip_roi','hip_rotation')
        mismatches=[k for k in keys if labels[k]!=targets[k]]
        recomputed.append({'image':row['image_path'],'mismatches':mismatches})
        if mismatches:problems.append({'image':row['image_path'],'recomputed_label_mismatches':mismatches})
    result={'files':len(rows),'unique_sop_uids':len(seen),'checked_geometry_and_private_metadata':len(rows),
            'counts':{f'{r}/{g}':n for (r,g),n in counts.items()},'recomputed_examples':len(recomputed),
            'recomputation':recomputed,'problems':problems,'seconds':time.perf_counter()-start,
            'scope':'all files readable, finite, aligned geometry and metadata; all hip rules recomputed; stratified sample of spine brightness rules recomputed'}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('recomputation','problems')},ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with warnings.catch_warnings():warnings.simplefilter('ignore');audit(args.root,args.output)
