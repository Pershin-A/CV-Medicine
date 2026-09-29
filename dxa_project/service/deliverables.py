"""Prediction table and lossless DICOM metadata / rendered-image exports."""
import copy,csv,json,shutil,time,zipfile
from pathlib import Path
import pydicom
from .store import write_json

COLUMNS=('path_to_study','study_uid','image_uid','anatomical_region','quality_class',
         'violation_type','processing_status','time_of_processing')
CREATOR='DXA_MODEL_V1'

def table_row(path,result=None,error=None,elapsed=0.):
    study=image=''
    try:
        ds=pydicom.dcmread(path,stop_before_pixels=True)
        study=str(getattr(ds,'StudyInstanceUID',''));image=str(getattr(ds,'SOPInstanceUID',''))
    except Exception:pass
    flags=(result or {}).get('quality_flags',{})
    unknown=[k for k,v in flags.items() if v is None]
    violations=[k for k,v in flags.items() if v==1]
    failure=bool(error or (unknown and not violations))
    if failure:violations+=['prediction_unavailable' if error else 'undetermined:'+','.join(unknown)]
    return dict(path_to_study=str(path),study_uid=study,image_uid=image,
                anatomical_region=(result or {}).get('region','UNKNOWN'),
                quality_class=int(bool(violations) or failure),violation_type=';'.join(violations),
                processing_status='Failure' if failure else 'Success',time_of_processing=float(elapsed))

def embed(source,destination,result,geometry,row):
    ds=pydicom.dcmread(source)
    original=ds.PixelData
    # Preserve original UIDs and pixel bytes. Private UT values hold UTF-8 JSON.
    ds.SpecificCharacterSet='ISO_IR 192'
    block=ds.private_block(0x0011,CREATOR,create=True)
    block.add_new(0x01,'UT',json.dumps({'schema':'dxa-prediction-v1','table':row,'prediction':result},ensure_ascii=False))
    block.add_new(0x02,'UT',json.dumps(geometry,ensure_ascii=False))
    ds.save_as(destination,enforce_file_format=True)
    if pydicom.dcmread(destination).PixelData!=original:raise ValueError('Export changed original pixel bytes')

def finish(store,job_id,files,result_ids,errors,times):
    root=store.root/'jobs'/job_id/'output'
    for name in ('originals','annotated','temporary'):(root/name).mkdir(parents=True,exist_ok=True)
    results=[store.get('result',rid) for rid in result_ids]
    indexed={str(Path(r['source']).resolve()):r for r in results}
    rows=[]
    for i,path in enumerate(files,1):
        path=Path(path);result=indexed.get(str(path.resolve()))
        error=next((e['error'] for e in errors if e['path']==str(path)),None)
        row=table_row(path,result,error,times.get(str(path),0.));rows.append(row)
        name=f'{i:05d}_{path.stem}'
        if result:
            geometry=json.loads(store.artifact(result['result_id'],'geometry.json').read_text(encoding='utf-8'))
            embed(path,root/'originals'/f'{name}.dcm',result,geometry,row)
            shutil.copy2(store.artifact(result['result_id'],'overlay.png'),root/'annotated'/f'{name}.png')
        else:
            shutil.copy2(path,root/'originals'/f'{name}.dcm')
            write_json(root/'originals'/f'{name}.failure.json',{'table':row,'error':error})
    write_json(root/'table.json',rows)
    with (root/'table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)
    write_json(root/'manifest.json',{'schema':'dxa-predict-export-v1','columns':COLUMNS,
        'private_creator':CREATOR,'metadata_tags':{'prediction':'0011,xx01','geometry':'0011,xx02'},
        'originals':'originals/*.dcm','annotated':'annotated/*.png','errors':errors,
        'failure_quality_class':'1 means reject an unprocessable/undetermined image; consult processing_status'})
    return dict(table=rows,output_directory=str(root),originals_directory=str(root/'originals'),
                annotated_directory=str(root/'annotated'),temporary_directory=str(root/'temporary'),
                table_url=f'/v1/jobs/{job_id}/export?format=csv',download_url=f'/v1/jobs/{job_id}/export?format=zip')

def export(store,job_id,fmt):
    root=store.root/'jobs'/job_id/'output';dest=root.parent/f'predictions.{fmt}'
    if fmt in ('csv','json'):shutil.copy2(root/f'table.{fmt}',dest)
    else:
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and 'temporary' not in p.relative_to(root).parts:z.write(p,p.relative_to(root).as_posix())
    return dest
