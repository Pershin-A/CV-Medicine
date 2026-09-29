import json
from pathlib import Path
import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,generate_uid,SecondaryCaptureImageStorage
from labeler.geometry import empty_geometry
from dxa_project.service.app import create_app
from dxa_project.service.store import Store
from dxa_project.service.worker import Worker
from dxa_project.service.data_ops import annotate,augment,import_image

def dicom(path,study=None):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=SecondaryCaptureImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID=study or generate_uid();ds.SeriesInstanceUID=generate_uid();ds.Modality='OT'
    ds.Rows=160;ds.Columns=160;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0;ds.PixelSpacing=[1.05,.6]
    ds.PixelData=np.arange(160*160,dtype=np.uint16).reshape(160,160).tobytes();ds.save_as(path,enforce_file_format=True)
    return path

@pytest.fixture
def api(tmp_path,monkeypatch):
    monkeypatch.setenv('DXA_ALLOWED_ROOTS',str(tmp_path))
    app=create_app(tmp_path/'service');return TestClient(app),app.state.store,tmp_path

def annotated(api,negative=False):
    client,store,tmp=api;path=dicom(tmp/'source.dcm')
    response=client.post('/v1/data/import',params={'path':str(path)});assert response.status_code==200
    item=response.json();g=empty_geometry(160,160)
    g['hip']['roi_box']=[0,40,110,110]
    g['hip']['landmarks']={'greater_trochanter':[75,55],'femoral_neck':[95,75],'ischial_bone':None if negative else [105,100]}
    payload=dict(region='LEG_LEFT',geometry=g,reviewed=['hip'],targets={'hip_rotation':1},expected_version=0)
    r=client.put(f"/v1/data/images/{item['image_id']}/annotation",json=payload);assert r.status_code==200,r.text
    return item,r.json()

def test_upload_and_annotation_conflict(api):
    client,store,tmp=api;item,a=annotated(api)
    r=client.put(f"/v1/data/images/{item['image_id']}/annotation",json=dict(region=a['region'],geometry=empty_geometry(160,160),reviewed=['hip'],expected_version=0))
    assert r.status_code==422
    assert client.get(f"/v1/data/images/{item['image_id']}/annotation/1").json()['geometry']['hip']['roi_box'][0]==0
    with (tmp/'source.dcm').open('rb') as f:r=client.post('/v1/data/uploads',files={'file':('test.dcm',f,'application/dicom')})
    assert r.status_code==200 and r.json()['image_id']==item['image_id']

def test_idempotent_enqueue_and_restart(api):
    client,store,tmp=api;im,a=annotated(api)
    p=dict(image_id=im['image_id'],annotation_version=1,request_id='same',config={'positive_count':1,'negative_count_by_target':{'hip_roi':1}})
    first=client.post('/v1/data/queue',json=p);second=client.post('/v1/data/queue',json=p)
    assert first.status_code==202 and first.json()['job_id']==second.json()['job_id']
    assert Store(store.root).job(first.json()['job_id'])['status']=='queued'
    p['config']['positive_count']=2;assert client.post('/v1/data/queue',json=p).status_code==422

def test_unsupported_target_and_unreviewed(api):
    client,store,tmp=api;im,a=annotated(api)
    p=dict(image_id=im['image_id'],annotation_version=1,config={'negative_count_by_target':{'hip_rotation':2}})
    assert client.post('/v1/data/queue',json=p).status_code==422
    r=client.put(f"/v1/data/images/{im['image_id']}/annotation",json=dict(region='LEG_LEFT',geometry=empty_geometry(160,160),reviewed=[],expected_version=1));assert r.status_code==200
    p['annotation_version']=2;p['config']={};assert client.post('/v1/data/queue',json=p).status_code==422

@pytest.mark.parametrize('negative',[False,True])
def test_augmentation_labels_metadata_and_clipping(api,negative):
    client,store,tmp=api;im,a=annotated(api,negative)
    p=dict(image_id=im['image_id'],annotation_version=1,config={'positive_count':1,'negative_source_count':2,'negative_count_by_target':{'hip_position':1,'hip_roi':1},'max_attempts_per_sample':80})
    id=client.post('/v1/data/queue',json=p).json()['job_id'];job=store.claim(('enqueue',));Worker(store,'cpu').run(job)
    report=store.job(id);assert report['status']=='completed',report
    rows=store.all('queue');assert len(rows)==(2 if negative else 3)
    for row in rows:
        ds=pydicom.dcmread(row['path']);g=json.loads(Path(row['geometry_path']).read_text())
        assert ds.pixel_array.shape==(160,160) and g['image_width']==160
        assert list(map(float,ds.PixelSpacing))==pytest.approx(row['spacing_mm'])
        assert row['study']==im['study']
        if row['strategy']=='positive':assert row['labels']['hip_position']==row['labels']['hip_roi']==0
        elif row['strategy']=='negative_source':assert row['labels']['hip_position']==1
        else:assert row['labels'][row['strategy']]==1

def test_path_bounds_cancel_and_no_model(api):
    client,store,tmp=api
    assert client.post('/v1/data/import',params={'path':'C:/Windows/system.ini'}).status_code==422
    assert client.get('/v1/health').json()['status']=='no_model'
    id=store.create_job('predict',{'paths':[]});assert client.post(f'/v1/jobs/{id}/cancel').status_code==200
    assert store.claim(('predict',)) is None
    assert client.get('/v1/results/unknown/files/../state.sqlite').status_code==404

def test_prediction_partial_failure_and_exports(api,monkeypatch):
    client,store,tmp=api;path=dicom(tmp/'source.dcm');bad=tmp/'bad.dcm';bad.write_text('broken')
    store.put('model','base',{'version':'base','path':str(tmp),'status':'imported'});store.put('config','active_model',{'version':'base'})
    class Engine:
        batch_sizes={'router':2}
        def __init__(self,*a):pass
        def predict(self,paths):
            if any(p.name=='bad.dcm' for p in paths):raise ValueError('Bad DICOM')
            return [dict(source=str(p),region='LEG_LEFT',quality_flags={'hip_roi':0},geometry=empty_geometry(160,160)) for p in paths]
    monkeypatch.setattr('dxa_project.service.inference.InferenceEngine',Engine)
    r=client.post('/v1/model/predict?wait_seconds=0',json={'paths':[str(path),str(bad)],'mode':'batch'});assert r.status_code==202
    id=r.json()['job_id'];Worker(store,'cpu').run(store.claim(('predict',)))
    job=client.get(f'/v1/jobs/{id}').json();assert job['status']=='completed_with_errors';assert len(job['result']['errors'])==1
    rid=job['result']['result_ids'][0]
    assert client.get(f'/v1/results/{rid}/files/overlay.png').content.startswith(b'\x89PNG')
    for fmt in ('json','csv','zip'):assert client.get(f'/v1/jobs/{id}/export?format={fmt}').status_code==200

def test_failed_job_is_persistent(api):
    client,store,tmp=api
    id=store.create_job('predict',{'model_version':'missing','paths':[]})
    Worker(store,'cpu').run(store.claim(('predict',)))
    assert Store(store.root).job(id)['status']=='failed'
    assert (store.root/'jobs'/id/'error.log').exists()

def test_batch_oom_reduces_batch(monkeypatch,tmp_path):
    import torch
    from dxa_project.service.inference import InferenceEngine,tensor_key
    calls=[]
    class Net:
        def __call__(self,x):
            calls.append(len(x))
            if len(x)>1:raise torch.cuda.OutOfMemoryError('simulated')
            return x.mean((1,2,3))[:,None]
    monkeypatch.setattr('dxa_project.service.inference._load_model',lambda *a:(Net(),8))
    engine=InferenceEngine(tmp_path,'cpu');inputs=[torch.ones(3,8,8)*v for v in (1,2,3)];cache={}
    result=engine.forwards('router',inputs,cache)
    assert engine.batch_sizes['router']==1 and calls==[2,1,1,1]
    assert [result[tensor_key(x)].item() for x in inputs]==[1,2,3]

def test_heldout_pixels_rejected_even_with_new_study(api,monkeypatch):
    from dxa_project.geometry_ml.data import Record
    from dxa_project.service.training import validate_rows
    from dxa_project.service.data_ops import image_hash
    client,store,tmp=api;path=dicom(tmp/'heldout.dcm')
    item=import_image(store,path)
    record=Record('original',path,tmp/'g.json','heldout-study','LEG_LEFT')
    proto=tmp/'protocol.json';proto.write_text(json.dumps({'partition_by_path':{'original':'validation'}}))
    row=dict(id='new',image_id=item['image_id'],study='new-study',status='ready')
    store.put('queue','new',row)
    monkeypatch.setattr('dxa_project.service.training.load_records',lambda root:[record])
    with pytest.raises(ValueError,match='Held-out'):validate_rows(store,{'protocol':str(proto)})

def test_warm_start_loads_existing_weights(tmp_path,monkeypatch):
    import torch
    import dxa_project.geometry_ml.train as train
    captured=[]
    class Net(torch.nn.Module):
        def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,3)
        def forward(self,x):
            if not captured:captured.append(self.fc.bias.detach().clone())
            return self.fc(x.mean((2,3)))
    class Dataset:
        def __init__(self,*a,**k):pass
        def __len__(self):return 2
        def __getitem__(self,i):return torch.ones(3,8,8),{'region':torch.tensor(0)}
    checkpoint=tmp_path/'router_initial.pt';net=Net();net.fc.bias.data[:]=torch.tensor([10.,-10.,-10.])
    torch.save({'task':'router','architecture':'light','state_dict':net.state_dict()},checkpoint)
    monkeypatch.setattr(train,'_model',lambda *a:Net());monkeypatch.setattr(train,'DxaDataset',Dataset)
    result=train.train_task('router',[object(),object()],[object(),object()],tmp_path,torch.device('cpu'),1,8,2,None,False,initial_checkpoint=checkpoint)
    assert captured[0].tolist()==[10,-10,-10] and result['epochs']==1

def test_prediction_mask_import_and_manual_preview(api):
    from PIL import Image
    client,store,tmp=api;im,a=annotated(api)
    store.put('result','predictionMask',{'source':str(tmp/'source.dcm')})
    path=store.artifact('predictionMask','mask.png');path.parent.mkdir(parents=True)
    mask=np.zeros((160,160),dtype=np.uint8);mask[80:83,90:94]=255;Image.fromarray(mask).save(path)
    raw=empty_geometry(160,160)
    raw['hip']['lesser_trochanter_mask_png']='/v1/results/predictionMask/files/mask.png'
    p=dict(region='LEG_LEFT',geometry=raw,reviewed=['hip'],expected_version=1)
    assert client.put(f"/v1/data/images/{im['image_id']}/annotation",json=p).status_code==422
    p['mask_result_id']='predictionMask'
    r=client.put(f"/v1/data/images/{im['image_id']}/annotation",json=p);assert r.status_code==200,r.text
    g=json.loads(Path(r.json()['geometry_path']).read_text())
    assert len(g['hip']['lesser_trochanter_pixels'])==12
    assert client.get(f"/v1/data/images/{im['image_id']}/preview.png").content.startswith(b'\x89PNG')
    assert client.get(f"/v1/data/images/{im['image_id']}/annotation/2/overlay.png").content.startswith(b'\x89PNG')
