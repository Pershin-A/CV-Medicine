import hashlib
import json
import math
import random
import shutil
import uuid
from pathlib import Path
import numpy as np
import pydicom
from .store import PROJECT,allowed_path,write_json
from dxa_project.augmentation.core import (prepare_geometry,transform_geometry,warp_image,
    transformed_spacing,Transform)
from dxa_project.augmentation.generate import _source_spacing,_labels,_hip_variant,_spine_variant,_write_dicom
from dxa_project.augmentation.vertebral_axes import analyze_spine
from labeler.geometry import validate_geometry

def image_hash(pixels):
    return hashlib.sha256(str((pixels.shape,pixels.dtype)).encode()+pixels.tobytes()).hexdigest()

def import_image(store,path):
    ds=pydicom.dcmread(path);pixels=ds.pixel_array
    if pixels.ndim!=2:raise ValueError('Only single-frame grayscale DICOM is supported')
    digest=image_hash(pixels)
    identity=str(getattr(ds,'StudyInstanceUID',''))+'\0'+str(getattr(ds,'SOPInstanceUID',''))+'\0'+digest
    id=hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    try:return store.get('image',id)
    except KeyError:pass
    study=str(getattr(ds,'StudyInstanceUID',''))
    if not study:raise ValueError('DICOM has no StudyInstanceUID')
    # Keep benchmark group IDs when a known image has an anonymized DICOM UID.
    from dxa_project.geometry_ml.data import load_records
    try:records=load_records(PROJECT)
    except FileNotFoundError:records=[]
    for r in records:
        if r.source_path.resolve()==Path(path).resolve():study=r.study;break
    folder=store.root/'sources'/id;folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,folder/'image.dcm')
    spacing,basis=_source_spacing(ds,True)
    item=dict(image_id=id,path=str(folder/'image.dcm'),original_path=str(path),study=study,pixel_hash=digest,
              width=pixels.shape[1],height=pixels.shape[0],spacing_mm=spacing,spacing_basis=basis,annotation_version=0)
    store.put('image',id,item);return item

def annotate(store,id,payload):
    image=store.get('image',id)
    if image['annotation_version']!=payload['expected_version']:raise ValueError('Annotation version conflict; reload before editing')
    reviewed=payload['reviewed'];region=payload['region']
    allowed={'spine','artifact','scoliosis','spine_crests'} if region=='SPINE' else {'hip','hip_points','hip_mask'}
    if not set(reviewed)<=allowed:raise ValueError('Reviewed sections do not match the anatomical region')
    target_names={'spine_position','spine_axis','spine_artifact','spine_scoliosis'} if region=='SPINE' else {'hip_position','hip_roi','hip_rotation'}
    if not set(payload['targets'])<=target_names:raise ValueError('Target names do not match the anatomical region')
    if any(value not in (0,1,None) for value in payload['targets'].values()):raise ValueError('Targets must be 0, 1 or null')
    spacing=payload.get('spacing_mm') or image['spacing_mm']
    if len(spacing)!=2 or not all(math.isfinite(s) and s>0 for s in spacing):raise ValueError('Invalid pixel spacing')
    raw=dict(payload['geometry'])
    if payload.get('mask_result_id'):
        from PIL import Image
        result=store.get('result',payload['mask_result_id'])
        source=pydicom.dcmread(allowed_path(result['source'])).pixel_array
        if image_hash(source)!=image['pixel_hash']:raise ValueError('Prediction mask belongs to another source image')
        mask=np.asarray(Image.open(store.artifact(payload['mask_result_id'],'mask.png')).convert('L'))>0
        if mask.shape!=(image['height'],image['width']):raise ValueError('Prediction mask dimensions differ')
        raw['hip']=dict(raw.get('hip') or {})
        raw['hip']['lesser_trochanter_pixels']=[[int(x),int(y)] for y,x in np.argwhere(mask)]
        raw['hip']['lesser_trochanter_mask_ready']=True
    elif (raw.get('hip') or {}).get('lesser_trochanter_mask_png'):
        raise ValueError('Provide mask_result_id or full geometry.json when importing a prediction mask')
    g=validate_geometry(raw,image['width'],image['height'])
    g=prepare_geometry(g,region)
    g['complete']['spine' if region=='SPINE' else 'hip']=('spine' if region=='SPINE' else 'hip') in reviewed
    version=image['annotation_version']+1
    folder=store.root/'annotations'/id/str(version)
    write_json(folder/'geometry.json',g)
    result=dict(image_id=id,version=version,region=region,reviewed=reviewed,targets=payload['targets'],
                spacing_mm=spacing,geometry_path=str(folder/'geometry.json'))
    write_json(folder/'annotation.json',result);store.put('annotation',f'{id}:{version}',result)
    image['annotation_version']=version;store.put('image',id,image)
    return result

def validate_enqueue(store,payload):
    a=store.get('annotation',f"{payload['image_id']}:{payload['annotation_version']}")
    region=a['region'];section='spine' if region=='SPINE' else 'hip'
    if section not in a['reviewed']:raise ValueError(f'Mark {section} annotation reviewed before augmentation')
    supported={'spine_position','spine_axis'} if region=='SPINE' else {'hip_position','hip_roi'}
    config=payload['config']
    for key,value in config['negative_count_by_target'].items():
        if key not in supported or not isinstance(value,int) or not 0<=value<=500:raise ValueError('Unsupported augmentation target or quota')
    return a

def augment(store,job,enqueue=True):
    payload=job['payload'];a=validate_enqueue(store,payload);im=store.get('image',a['image_id'])
    geometry=json.loads(Path(a['geometry_path']).read_text(encoding='utf-8'));region=a['region']
    ds=pydicom.dcmread(im['path']);pixels=ds.pixel_array;spacing=tuple(a['spacing_mm']);config=payload['config']
    source={'axis_polarity':'dark' if str(ds.PhotometricInterpretation)=='MONOCHROME1' else 'bright',
            'spine_artifact':str(a['targets'].get('spine_artifact','')),
            'artifact_annotation_complete':'artifact' in a['reviewed']}
    if region!='SPINE':source[region.removeprefix('LEG_').lower()+'_hip_rotation']=str(a['targets'].get('hip_rotation',''))
    required=('spine_position','spine_axis') if region=='SPINE' else ('hip_position','hip_roi')
    base_labels=_labels(region,geometry,{'roi_fully_visible':True},pixels,spacing,source)
    if payload.get('strict_targets'):
        # Overall positive means no quality violation, including inherited anatomical labels.
        for k,v in a['targets'].items():
            if v in (0,1):base_labels[k]=v
        if region=='SPINE' and 'spine_scoliosis' not in base_labels:base_labels['spine_scoliosis']=a['targets'].get('spine_scoliosis')
        required=('spine_position','spine_axis','spine_artifact','spine_scoliosis') if region=='SPINE' else ('hip_position','hip_roi','hip_rotation')
    violations=[k for k in required if base_labels[k]==1]
    if not violations and any(base_labels[k] is None for k in required):raise ValueError('Cannot determine geometric source labels; review annotation')
    requests={'negative_source':config['negative_source_count']} if violations else {
        'positive':config['positive_count'],**config['negative_count_by_target']}
    rng=random.Random(config['seed']);folder=Path(payload['output_directory']) if payload.get('output_directory') else store.root/'augmentations'/job['id'];rows=[];rejected={};seen=set()
    for strategy,quota in requests.items():
        produced=0
        for attempt in range(quota*config['max_attempts_per_sample']):
            if produced>=quota:break
            try:
                if strategy=='negative_source':
                    scale=rng.uniform(1.01,1.35);w,h=im['width'],im['height']
                    cx=(w-1)/2;cy=(h-1)/2
                    t=Transform(w,h,scale,0,rng.uniform(cx/scale,w-1-cx/scale),rng.uniform(cy/scale,h-1-cy/scale))
                    variant='preserve_source_violation'
                else:
                    group='positive' if strategy=='positive' else 'negative_position' if strategy.endswith('position') else 'negative_axis_or_roi'
                    if region=='SPINE':
                        angle=analyze_spine(pixels,geometry,spacing,polarity=source['axis_polarity'])['global_angle_deg']
                        t,variant=_spine_variant(geometry,group,rng,angle,spacing)
                        # Use full requested negative range while leaving old experiment unchanged.
                        if strategy=='spine_axis':
                            target=rng.choice((rng.uniform(-15,-6),rng.uniform(6,15)))
                            normalized=-angle if t.reflect_x else angle
                            rotation=normalized-target;rad=math.radians(rotation);y,x=spacing
                            scale=max(abs(math.cos(rad))+abs(math.sin(rad))*y/x,
                                      abs(math.cos(rad))+abs(math.sin(rad))*x/y)+.05
                            t=Transform(im['width'],im['height'],scale,rotation,reflect_x=t.reflect_x,physical_spacing_mm=spacing)
                    else:t,variant=_hip_variant(geometry,group,rng,region.removeprefix('LEG_'),spacing)
                if not t.covers_output():continue
                g,info=transform_geometry(geometry,t,region=region);moved=warp_image(pixels,t);new_spacing=transformed_spacing(spacing,t)
                labels=_labels(region,g,info,moved,new_spacing,source)
                if payload.get('strict_targets') and region=='SPINE':
                    labels['spine_scoliosis']=a['targets'].get('spine_scoliosis')
                    if labels['spine_scoliosis']==1 and len(g['spine']['disc_lines'])!=len(geometry['spine']['disc_lines']):continue
                if strategy=='positive':
                    okay=all(labels[k]==0 for k in required)
                    if region=='SPINE':okay=okay and labels['spine_axis_angle_deg'] is not None and abs(labels['spine_axis_angle_deg'])<=4
                elif strategy=='negative_source':okay=any(labels[k]==1 for k in violations)
                else:
                    okay=labels[strategy]==1
                    if strategy=='spine_axis':
                        angle=labels['spine_axis_angle_deg'];okay=okay and angle is not None and 6<=abs(angle)<=15
                if not okay:continue
                digest=image_hash(moved)
                if digest in seen:
                    rejected['duplicate_pixels']=rejected.get('duplicate_pixels',0)+1
                    continue
                seen.add(digest)
                name=uuid.uuid4().hex;output=folder/name;output.mkdir(parents=True,exist_ok=True)
                _write_dicom(ds,moved,t,output/'image.dcm',name,g,labels,region)
                updated=pydicom.dcmread(output/'image.dcm');updated.PixelSpacing=list(new_spacing);updated.save_as(output/'image.dcm',enforce_file_format=True)
                if updated.pixel_array.shape!=moved.shape or not np.array_equal(updated.pixel_array,moved):raise ValueError('Generated DICOM pixels differ')
                write_json(output/'geometry.json',g)
                entry=dict(id=name,job_id=job['id'],image_id=a['image_id'],annotation_version=a['version'],study=im['study'],
                           source_id=a['image_id'],path=str(output/'image.dcm'),geometry_path=str(output/'geometry.json'),
                           region=region,spacing_mm=new_spacing,labels=labels,reviewed=a['reviewed'],strategy=strategy,status='ready',pixel_hash=digest)
                write_json(output/'metadata.json',entry)
                if enqueue:store.put('queue',name,entry)
                rows.append(entry);produced+=1
                if payload.get('manage_job',True):store.update(job['id'],result={'generated':len(rows),'strategy':strategy,'requested':requests})
            except (ValueError,TypeError,IndexError,KeyError) as e:rejected[str(e)]=rejected.get(str(e),0)+1
        if produced<quota:rejected[f'Unmet quota {strategy}']=quota-produced
    report=dict(requested=requests,generated=len(rows),generated_by_strategy={k:sum(r['strategy']==k for r in rows) for k in requests},
                rejected=rejected,queue_ids=[r['id'] for r in rows],source_labels=base_labels)
    write_json(folder/'report.json',report)
    if payload.get('manage_job',True):store.update(job['id'],'completed_with_errors' if any(k.startswith('Unmet quota') for k in rejected) else 'completed',report)
    return report
