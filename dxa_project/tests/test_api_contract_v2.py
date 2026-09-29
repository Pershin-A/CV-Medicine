import json,zipfile
from pathlib import Path
import numpy as np
import pydicom,pytest,torch
from fastapi.testclient import TestClient
from dxa_project.tests.test_service import dicom
from labeler.geometry import empty_geometry
from dxa_project.service.app import create_app
from dxa_project.service.worker import Worker
from dxa_project.service import staged_fit as fit
from dxa_project.service.deliverables import COLUMNS

def test_predict_table_two_folders_and_lossless_private_metadata(tmp_path,monkeypatch):
    monkeypatch.setenv('DXA_ALLOWED_ROOTS',str(tmp_path));app=create_app(tmp_path/'store');s=app.state.store;c=TestClient(app)
    path=dicom(tmp_path/'input.dcm');bad=tmp_path/'broken.dcm';bad.write_bytes(b'not a DICOM')
    s.put('model','base',dict(version='base',path=str(tmp_path)));s.put('config','active_model',{'version':'base'})
    class Engine:
        batch_sizes={}
        def __init__(self,*a):pass
        def predict(self,paths):
            if any(p.name=='broken.dcm' for p in paths):raise ValueError('Invalid file')
            return [dict(source=str(p),region='LEG_LEFT',geometry=empty_geometry(160,160),quality_flags={'hip_rotation':1,'hip_roi':0,'hip_position':0}) for p in paths]
    monkeypatch.setattr('dxa_project.service.inference.InferenceEngine',Engine)
    j=c.post('/predict?wait_seconds=0',json=dict(paths=[str(path),str(bad)],mode='batch')).json()['job_id']
    Worker(s,'cpu').run(s.claim(('predict',)));job=s.job(j);assert job['status']=='completed_with_errors',job
    rows=job['result']['table'];assert tuple(rows[0])==COLUMNS
    assert rows[0]['image_uid']==str(pydicom.dcmread(path).SOPInstanceUID)
    assert rows[0]['quality_class']==1 and rows[0]['processing_status']=='Success'
    assert rows[1]['processing_status']=='Failure' and rows[1]['time_of_processing']>=0
    exported=next((Path(job['result']['originals_directory'])).glob('00001*.dcm'));out=pydicom.dcmread(exported);original=pydicom.dcmread(path)
    assert out.PixelData==original.PixelData and out.SOPInstanceUID==original.SOPInstanceUID
    b=out.private_block(0x0011,'DXA_MODEL_V1');assert json.loads(out[b.get_tag(1)].value)['table']==rows[0]
    assert json.loads(out[b.get_tag(2)].value)['image_width']==160
    import io
    z=zipfile.ZipFile(io.BytesIO(c.get(f'/v1/jobs/{j}/export?format=zip').content))
    assert 'table.csv' in z.namelist() and any(n.startswith('originals/') for n in z.namelist()) and any(n.startswith('annotated/') for n in z.namelist())
    assert c.get(f'/v1/jobs/{j}/export?format=csv').text.lstrip('\ufeff').splitlines()[0]==','.join(COLUMNS)
    assert c.get(f'/v1/jobs/{j}/files/../state.sqlite').status_code==404

def test_four_stages_order_and_separate_temporary_folders(tmp_path,monkeypatch):
    app=create_app(tmp_path/'store');s=app.state.store;b=tmp_path/'bundle';b.mkdir();(b/'protocol.json').write_text('{"partition_by_path":{}}')
    (b/'router.pt').write_bytes(b'unchanged');base=dict(version='base',path=str(b),protocol=str(b/'protocol.json'),epochs={'router':8})
    s.put('model','base',base);j=s.create_job('partial_fit',dict(base_model_version='base',human={'rows':[]},model={'rows':[]},augmentation={},learning_rate=2e-5));calls=[]
    monkeypatch.setattr(fit,'load_records',lambda _:[])
    def snap(store,job,kind):
        root=store.root/'jobs'/job['id']/'output'
        for name in ('originals','annotated','temporary/augmented'):(root/name/kind).mkdir(parents=True)
        return [dict(id=kind)]
    monkeypatch.setattr(fit,'snapshot',snap)
    def augment(store,job,kind,rows):calls.append('augment_'+kind);return [dict(id=kind+'_aug')]
    monkeypatch.setattr(fit,'augment_set',augment)
    monkeypatch.setattr(fit,'train_stage',lambda store,job,stage,rows,*a:calls.append((stage,rows[0]['id'])) or [])
    monkeypatch.setattr('dxa_project.service.validation.compare',lambda *a:{'passed':True})
    Worker(s,'cpu').run(s.claim(('partial_fit',)))
    assert calls==[('human_original','human'),('model_original','model'),'augment_human','augment_model',('human_augmented','human_aug'),('model_augmented','model_aug')]
    assert s.job(j)['status']=='completed';assert (b/'router.pt').read_bytes()==b'unchanged'
    root=s.root/'jobs'/j/'output';assert (root/'temporary/augmented/human').is_dir() and (root/'temporary/augmented/model').is_dir()
    assert s.get('model',j)['epochs']['router']==8

def test_epoch_budget_logs_and_actual_optimizer_on_isolated_toy(tmp_path,monkeypatch):
    from dxa_project.service.store import Store
    from dxa_project.geometry_ml.data import Record
    s=Store(tmp_path/'store');j=s.create_job('partial_fit',{});folder=tmp_path/'candidate';folder.mkdir()
    net=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3));torch.save({'state_dict':net.state_dict()},folder/'router.pt')
    initial={k:v.clone() for k,v in net.state_dict().items()}
    record=Record('valid',tmp_path/'none',tmp_path/'none','v','SPINE');row=dict(id='x',region='SPINE',reviewed=['spine'],labels={},path='x',geometry_path='x',study='new',source_id='x',spacing_mm=[1,1])
    class Dataset:
        def __init__(self,records,*a,**kw):self.records=records
        def __len__(self):return len(self.records)
        def __getitem__(self,i):return torch.ones(1,2,2),{'region':torch.tensor(0),'relative_path':self.records[i].relative_path}
    monkeypatch.setattr(fit,'DxaDataset',Dataset);monkeypatch.setattr(fit,'_load_model',type('Loader',(),{'__call__':lambda self,*a:(net,2),'cache_clear':lambda self:None})())
    monkeypatch.setattr('dxa_project.geometry_ml.final_protocol.labels_for_originals',lambda *a:{'valid':{}})
    job={'id':j,'payload':{'learning_rate':.001}}
    reports=fit.train_stage(s,job,'human_original',[row],folder,{'epochs':{'router':2}},[record],{'valid':'validation'},torch.device('cpu'))
    assert reports[0]['epochs']==2 and len(reports[0]['history'])==2
    assert any(not torch.equal(initial[k],net.state_dict()[k]) for k in initial)
    events=TestClient(create_app(s.root)).get(f'/v1/jobs/{j}/logs').json()['events']
    assert [e['epoch'] for e in events if e['status']=='started']==[1,2]

def test_real_negative_rotation_never_becomes_positive_and_temp_is_sibling(tmp_path,monkeypatch):
    from dxa_project.service.data_ops import import_image,annotate,augment
    from dxa_project.service.store import Store
    s=Store(tmp_path/'store');p=dicom(tmp_path/'hip.dcm');im=import_image(s,p);g=empty_geometry(160,160)
    g['hip']['roi_box']=[0,40,110,110];g['hip']['landmarks']={'greater_trochanter':[75,55],'femoral_neck':[95,75],'ischial_bone':[105,100]}
    a=annotate(s,im['image_id'],dict(region='LEG_LEFT',geometry=g,reviewed=['hip'],targets={'hip_position':0,'hip_roi':0,'hip_rotation':1},expected_version=0))
    j=s.create_job('partial_fit',{});s.update(j,'running');root=s.root/'jobs'/j/'output/temporary/augmented/human'
    job={'id':j,'payload':dict(image_id=im['image_id'],annotation_version=a['version'],output_directory=str(root),strict_targets=True,manage_job=False,config={'positive_count':3,'negative_count_by_target':{'hip_roi':2},'negative_source_count':1,'seed':42,'max_attempts_per_sample':100})}
    r=augment(s,job,False);assert r['requested']=={'negative_source':1} and r['generated']==1
    item=json.loads(next(root.glob('*/metadata.json')).read_text());assert item['labels']['hip_rotation']==1
    assert s.job(j)['status']=='running'

def test_partial_fit_accepts_two_annotated_sets_without_training_and_rejects_overlap(tmp_path,monkeypatch):
    monkeypatch.setenv('DXA_ALLOWED_ROOTS',str(tmp_path));app=create_app(tmp_path/'store');s=app.state.store;c=TestClient(app)
    b=tmp_path/'base';b.mkdir();(b/'protocol.json').write_text('{"partition_by_path":{}}');(b/'router.pt').write_bytes(b'base unchanged')
    s.put('model','base',dict(version='base',path=str(b),protocol=str(b/'protocol.json'),epochs={'router':8}));s.put('config','active_model',{'version':'base'})
    monkeypatch.setattr(fit,'load_records',lambda _:[])
    a=dicom(tmp_path/'human.dcm');p=dicom(tmp_path/'model.dcm')
    ds=pydicom.dcmread(p);pixels=ds.pixel_array.copy();pixels[0,0]=123;ds.PixelData=pixels.tobytes();ds.save_as(p,enforce_file_format=True)
    g=empty_geometry(160,160);g['hip']['roi_box']=[0,40,110,110]
    sample=lambda path:dict(path=str(path),region='LEG_LEFT',geometry=g,targets={'hip_position':0,'hip_roi':0,'hip_rotation':0})
    payload=dict(human={'images':[sample(a)]},model={'images':[sample(p)]},augmentation={'n_pp':1,'n_pn':1,'n_nn':1},request_id='same-input')
    response=c.post('/partial_fit',json=payload);assert response.status_code==202,response.text
    job=s.job(response.json()['job_id']);assert job['status']=='queued'
    assert c.post('/partial_fit',json=payload).json()['job_id']==job['id']
    assert len(job['payload']['human']['rows'])==len(job['payload']['model']['rows'])==1
    assert job['payload']['human']['rows'][0]['pixel_hash']!=job['payload']['model']['rows'][0]['pixel_hash']
    payload['model']['images']=[sample(a)];assert c.post('/partial_fit',json=payload).status_code==422
    assert (b/'router.pt').read_bytes()==b'base unchanged'

def test_predict_deployment_registration_without_original_dataset(tmp_path,monkeypatch):
    from dxa_project.service.training import register
    from dxa_project.service.store import Store
    monkeypatch.setenv('DXA_ALLOWED_ROOTS',str(tmp_path));s=Store(tmp_path/'store');b=tmp_path/'bundle';b.mkdir()
    for name in fit.MODULES:(b/(name+'.pt')).write_bytes(b'unchanged weights')
    (b/'evaluation').mkdir();(b/'evaluation/calibration.json').write_text('{"rotation_threshold_px2":1}')
    epochs={k:7 for k in fit.MODULES};(b/'training_config.json').write_text(json.dumps({'epochs':epochs}))
    protocol=tmp_path/'protocol.json';protocol.write_text('{"partition_by_path":{"example.dcm":"train"}}')
    def missing(_):raise FileNotFoundError('no original dataset')
    monkeypatch.setattr('dxa_project.service.training.load_records',missing)
    registered=register(s,dict(checkpoints=str(b),protocol=str(protocol),epochs=None,activate=True))
    assert not registered['training_data_available'] and registered['epochs']==epochs
    assert (Path(registered['path'])/'spine_crests.pt').read_bytes()==b'unchanged weights'

def test_identical_pixels_with_different_dicom_uids_remain_two_inputs(tmp_path):
    from dxa_project.service.data_ops import import_image
    from dxa_project.service.store import Store
    s=Store(tmp_path/'store');a=dicom(tmp_path/'a.dcm');b=dicom(tmp_path/'b.dcm')
    first=import_image(s,a);second=import_image(s,b)
    assert first['pixel_hash']==second['pixel_hash'] and first['image_id']!=second['image_id']
    assert pydicom.dcmread(first['path']).SOPInstanceUID!=pydicom.dcmread(second['path']).SOPInstanceUID
