import csv
import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from fastapi import FastAPI,UploadFile,HTTPException,Query
from fastapi.responses import FileResponse,JSONResponse
from .schemas import Predict,Annotation,Enqueue,Register,PartialFit
from .store import Store,allowed_path
from .data_ops import import_image,annotate,validate_enqueue
from .training import register,model,validate_rows

def create_app(data_root=None):
    app=FastAPI(title='DXA: модели и разметка',version='2.0.0')
    store=Store(data_root);app.state.store=store;annotation_lock=threading.Lock()

    @app.exception_handler(KeyError)
    async def missing(request,error):return JSONResponse(status_code=404,content={'detail':str(error)})
    @app.exception_handler(ValueError)
    async def invalid(request,error):return JSONResponse(status_code=422,content={'detail':str(error)})

    @app.get('/v1/health')
    def health():
        from .worker import external_training_busy
        try:active=store.get('config','active_model')['version']
        except KeyError:active=None
        return {'status':'ready' if active else 'no_model','active_model':active,'external_training_busy':external_training_busy()}

    @app.post('/v1/model/versions')
    def register_model(payload:Register):return register(store,payload.model_dump())
    @app.get('/v1/model/versions')
    def versions():return store.all('model')
    @app.post('/v1/model/versions/{version}/activate')
    def activate(version:str):
        item=model(store,version)
        if item['status'] not in ('validated','imported'):raise HTTPException(409,'Candidate did not pass validation')
        store.put('config','active_model',{'version':version});return item

    @app.post('/v1/data/import')
    def import_path(path:str):
        return import_image(store,allowed_path(path))
    @app.post('/v1/data/uploads')
    async def upload(file:UploadFile):
        temp=store.root/'uploads'/f'{uuid.uuid4().hex}.dcm';temp.parent.mkdir(parents=True,exist_ok=True)
        try:
            size=0
            with temp.open('wb') as out:
                while chunk:=await file.read(1024*1024):
                    size+=len(chunk)
                    if size>128*1024*1024:raise HTTPException(413,'File exceeds 128 MiB')
                    out.write(chunk)
            try:return import_image(store,temp)
            except Exception as e:raise HTTPException(422,f'Invalid DICOM: {e}') from e
        finally:temp.unlink(missing_ok=True);await file.close()
    @app.get('/v1/data/images/{id}')
    def image(id:str):return store.get('image',id)
    @app.get('/v1/data/images/{id}/preview.png')
    def preview(id:str):
        from PIL import Image
        import numpy as np
        from dxa_project.geometry_ml.data import read_dicom_image
        im=store.get('image',id);path=Path(im['path']).with_name('preview.png')
        if not path.exists():Image.fromarray((np.clip(read_dicom_image(Path(im['path'])),0,1)*255).astype('uint8')).save(path)
        return FileResponse(path,media_type='image/png')
    @app.put('/v1/data/images/{id}/annotation')
    def annotation(id:str,payload:Annotation):
        with annotation_lock:return annotate(store,id,payload.model_dump())
    @app.get('/v1/data/images/{id}/annotation/{version}')
    def annotation_get(id:str,version:int):
        a=store.get('annotation',f'{id}:{version}');return {**a,'geometry':json.loads(Path(a['geometry_path']).read_text(encoding='utf-8'))}
    @app.get('/v1/data/images/{id}/annotation/{version}/overlay.png')
    def annotation_overlay(id:str,version:int):
        from .inference import render_overlay
        from dxa_project.geometry_ml.data import read_dicom_image
        im=store.get('image',id);a=store.get('annotation',f'{id}:{version}')
        path=Path(a['geometry_path']).with_name('overlay.png')
        if not path.exists():render_overlay(read_dicom_image(Path(im['path'])),json.loads(Path(a['geometry_path']).read_text(encoding='utf-8')),path)
        return FileResponse(path,media_type='image/png')

    def enqueue(payload,action):
        p=payload.model_dump();validate_enqueue(store,p);request_id=p.pop('request_id')
        id=store.create_job(action,p,request_id);return JSONResponse(status_code=202,content={'job_id':id,'status':store.job(id)['status']})
    @app.post('/v1/data/queue')
    def add_queue(payload:Enqueue):return enqueue(payload,'enqueue')
    @app.post('/v1/data/augmentation')
    def augmentation(payload:Enqueue):return enqueue(payload,'augmentation')
    @app.get('/v1/data/queue')
    def queue(status:str | None=None):return [r for r in store.all('queue') if status is None or r['status']==status]
    @app.delete('/v1/data/queue/{id}')
    def remove_queue(id:str):
        row=store.get('queue',id)
        if row['status']=='reserved':raise HTTPException(409,'Item is used in an active training snapshot')
        row['status']='excluded';store.put('queue',id,row);return row

    @app.post('/partial_fit',include_in_schema=False)
    @app.post('/v1/model/partial_fit')
    def fit(payload:PartialFit):
        original_request=payload.model_dump()
        p=payload.model_dump();base=model(store,p['base_model_version']);p['base_model_version']=base['version']
        if p.get('human') is not None or p.get('model') is not None:
            from .staged_fit import validate_datasets
            with annotation_lock:
                request_id=p.get('request_id');id=None
                if request_id:
                    with store.connect() as db:
                        old=db.execute('SELECT id,payload FROM jobs WHERE request_key=?',('partial_fit:'+request_id,)).fetchone()
                    if old:
                        if json.loads(old['payload']).get('original_request')!=original_request:raise ValueError('request_id already used with another payload')
                        id=old['id']
                if id is None:
                    p=validate_datasets(store,base,p);p.pop('request_id');p['original_request']=original_request
                    id=store.create_job('partial_fit',p,request_id)
            return JSONResponse(status_code=202,content={'job_id':id,'stages':['human_original','model_original','human_augmented','model_augmented']})
        rows,_,_=validate_rows(store,base,p['queue_ids']);p['queue_ids']=[r['id'] for r in rows]
        request_id=p.pop('request_id');id=store.create_job('partial_fit',p,request_id)
        return JSONResponse(status_code=202,content={'job_id':id,'examples':len(rows)})

    @app.post('/predict',include_in_schema=False)
    @app.post('/v1/model/predict')
    def predict(payload:Predict,wait_seconds:float=Query(default=10,ge=0,le=30)):
        base=model(store,payload.model_version);paths=[]
        for raw in payload.paths:
            path=allowed_path(raw)
            if path.is_dir():
                paths.extend(str(allowed_path(str(p))) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix.lower() in ('.dcm','.dicom'))
            else:paths.append(str(path))
        paths.extend(store.get('image',id)['path'] for id in payload.image_ids)
        paths=list(dict.fromkeys(paths))
        if not paths or len(paths)>10000:raise ValueError('Select between 1 and 10000 DICOM files')
        if payload.mode=='single' and len(paths)!=1:raise ValueError('Use mode=batch for multiple files')
        id=store.create_job('predict',{'paths':paths,'model_version':base['version']},payload.request_id)
        deadline=time.monotonic()+wait_seconds
        while time.monotonic()<deadline:
            job=store.job(id)
            if job['status'] in ('completed','completed_with_errors'):
                if payload.mode=='single' and job['result']['result_ids']:return {**store.get('result',job['result']['result_ids'][0]),'job_id':id,**job['result']}
                return job
            if job['status']=='failed':raise HTTPException(500,job['error'])
            time.sleep(.1)
        return JSONResponse(status_code=202,content={'job_id':id,'status':store.job(id)['status']})

    @app.post('/v1/model/predict/upload')
    async def predict_upload(files:list[UploadFile],wait_seconds:float=Query(default=0,ge=0,le=30)):
        if not 1<=len(files)<=10000:raise ValueError('Select between 1 and 10000 DICOM files')
        ids=[]
        for file in files:ids.append((await upload(file))['image_id'])
        return predict(Predict(image_ids=ids,mode='single' if len(ids)==1 else 'batch'),wait_seconds)

    @app.get('/v1/jobs/{id}')
    def job(id:str):
        item=store.job(id)
        if item['action']=='partial_fit' and item['status']=='running':
            task=(item['result'] or {}).get('task')
            history=store.root/'models'/id/f'{task}_history.json'
            if history.exists():
                try:item['epoch_progress']=json.loads(history.read_text(encoding='utf-8'))[-1]['epoch']
                except (json.JSONDecodeError,IndexError):pass
        return item
    @app.post('/v1/jobs/{id}/cancel')
    def cancel(id:str):
        store.job(id)
        with store.connect() as db:
            updated=db.execute('UPDATE jobs SET status="cancelled",updated=? WHERE id=? AND status="queued"',(time.time(),id)).rowcount
        if not updated:raise HTTPException(409,'Only queued jobs can be cancelled')
        return store.job(id)
    @app.get('/v1/jobs/{id}/logs')
    def job_logs(id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=100,ge=1,le=1000)):
        store.job(id)
        path=store.root/'jobs'/id/('training.jsonl' if store.job(id)['action']=='partial_fit' else 'prediction.jsonl');events=[];total=0
        if path.exists():
            with path.open(encoding='utf-8') as f:
                for line in f:
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    if offset<=total<offset+limit:events.append(event)
                    total+=1
        return {'events':events,'next_offset':offset+len(events),'total':total}
    @app.get('/v1/jobs/{id}/artifacts')
    def artifacts(id:str):
        store.job(id);root=store.root/'jobs'/id/'output'
        return {'files':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
                         'url':f'/v1/jobs/{id}/files/'+p.relative_to(root).as_posix()}
                        for p in sorted(root.rglob('*')) if p.is_file()]}
    @app.get('/v1/jobs/{id}/files/{path:path}')
    def job_file(id:str,path:str):
        store.job(id);root=(store.root/'jobs'/id/'output').resolve();file=(root/path).resolve()
        if not file.is_relative_to(root) or not file.is_file():raise HTTPException(404,'Artifact not found')
        return FileResponse(file)
    @app.get('/v1/results/{id}')
    def result(id:str):return store.get('result',id)
    @app.get('/v1/results/{id}/files/{name}')
    def result_file(id:str,name:str):
        store.get('result',id)
        if name not in ('overlay.png','mask.png','geometry.json','result.json'):raise HTTPException(404)
        return FileResponse(store.artifact(id,name))
    @app.get('/v1/jobs/{id}/export')
    def export(id:str,format:str=Query(default='zip',pattern='^(zip|json|csv)$')):
        job=store.job(id)
        if job['action']!='predict' or job['status'] not in ('completed','completed_with_errors'):raise HTTPException(409,'Prediction job not completed')
        if (store.root/'jobs'/id/'output/table.json').exists():
            from .deliverables import export as export_delivery
            destination=export_delivery(store,id,format)
            return FileResponse(destination,filename=destination.name)
        rows=[store.get('result',rid) for rid in job['result']['result_ids']]
        folder=store.root/'jobs'/id;folder.mkdir(parents=True,exist_ok=True);destination=folder/f'predictions.{format}'
        if format=='json':destination.write_text(json.dumps({'results':rows,'errors':job['result']['errors']},ensure_ascii=False,indent=2),encoding='utf-8')
        elif format=='csv':
            keys=['result_id','source','region','model_version','error','spine_position','spine_axis','spine_artifact','spine_scoliosis','hip_position','hip_roi','hip_rotation']
            with destination.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
                for r in rows:writer.writerow({**{k:r.get(k,'') for k in keys[:4]},**r['quality_flags']})
                for e in job['result']['errors']:writer.writerow({'source':e['path'],'error':e['error']})
        else:
            with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('summary.json',json.dumps(job['result'],ensure_ascii=False))
                for r in rows:
                    for name in ('overlay.png','mask.png','geometry.json','result.json'):z.write(store.artifact(r['result_id'],name),f"{r['result_id']}/{name}")
        return FileResponse(destination,filename=destination.name)
    return app

app=create_app()
