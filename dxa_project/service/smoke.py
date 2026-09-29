"""Real CPU integration check on existing checkpoints; isolated storage and 1-epoch budget."""
import json
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient
from .app import create_app
from .store import ROOT,PROJECT,write_json
from .worker import Worker
from .inference import InferenceEngine
from dxa_project.geometry_ml.data import load_records
from dxa_project.geometry_ml.predict import predict_file,_load_model
from dxa_project.augmentation.core import prepare_geometry,hip_roi_ok,hip_position_ok

def main():
    torch.set_num_threads(2)
    folder=ROOT/'outputs/service_smoke';folder.mkdir(parents=True,exist_ok=True)
    app=create_app(folder/'state');client=TestClient(app);store=app.state.store
    base=ROOT/'outputs/improvements_v2/pipeline';proto=ROOT/'outputs/improvements_v2/protocol.json'
    r=client.post('/v1/model/versions',json={'checkpoints':str(base),'protocol':str(proto),'epochs':{t:1 for t in ('router','spine','hip','artifact','hip_points')}})
    assert r.status_code==200,r.text
    version=r.json()['version'];records=load_records(PROJECT)
    samples=[next(r for r in records if r.region==region) for region in ('SPINE','LEG_LEFT','LEG_RIGHT')]
    expected=[predict_file(r.source_path,base,device_override='cpu') for r in samples]
    _load_model.cache_clear()
    engine=InferenceEngine(base,'cpu');actual=engine.predict([r.source_path for r in samples])
    for a,b in zip(actual,expected):
        assert a['region']==b['region'] and a['quality_flags']==b['quality_flags']
        assert list(a['router_probabilities'].values())==__import__('pytest').approx(list(b['router_probabilities'].values()),abs=1e-5)
        if a['region']!='SPINE':
            assert np.allclose(a['geometry']['hip']['roi_box'],b['geometry']['hip']['roi_box'],atol=.001)
            assert a['geometry']['hip']['landmarks']==b['geometry']['hip']['landmarks']
    id=client.post('/v1/model/predict?wait_seconds=0',json={'paths':[str(r.source_path) for r in samples],'mode':'batch'}).json()['job_id']
    worker=Worker(store,'cpu');worker.run(store.claim(('predict',)))
    job=store.job(id);assert job['status']=='completed',job
    assert client.get(f'/v1/jobs/{id}/export').status_code==200
    mapping=json.loads(proto.read_text(encoding='utf-8'))['partition_by_path']
    source=None
    for record in records:
        if record.region!='LEG_LEFT' or mapping[record.relative_path]!='train':continue
        g=prepare_geometry(json.loads(record.geometry_path.read_text(encoding='utf-8')),record.region)
        if hip_position_ok(g) and hip_roi_ok(g,True,'LEFT',record.spacing_mm):source=record;break
    assert source is not None
    imported=client.post('/v1/data/import',params={'path':str(source.source_path)}).json()
    ann=client.put(f"/v1/data/images/{imported['image_id']}/annotation",json={'region':source.region,
        'geometry':json.loads(source.geometry_path.read_text(encoding='utf-8')),'reviewed':['hip'],
        'targets':{'hip_rotation':0},'expected_version':imported['annotation_version']})
    assert ann.status_code==200,ann.text
    queued=client.post('/v1/data/queue',json={'image_id':imported['image_id'],'annotation_version':ann.json()['version'],
        'config':{'positive_count':1,'negative_source_count':0,'max_attempts_per_sample':100}})
    assert queued.status_code==202,queued.text
    worker.run(store.claim(('enqueue',)))
    aug=store.job(queued.json()['job_id']);assert aug['status']=='completed',aug
    fit=client.post('/v1/model/partial_fit',json={'queue_ids':aug['result']['queue_ids'],'replay_per_task':1})
    assert fit.status_code==202,fit.text
    worker.run(store.claim(('partial_fit',)))
    trained=store.job(fit.json()['job_id']);assert trained['status']=='completed',trained
    assert store.get('config','active_model')['version']==version
    reports=json.loads(Path(trained['result']['model']['report']).read_text(encoding='utf-8'))
    assert all(r['epochs']==1 for r in reports)
    report={'single_batch_same_flags':True,'anatomical_regions':len(samples),'prediction_job':id,
            'augmentation_job':aug['id'],'warm_start_job':trained['id'],'trained_modules':trained['result']['trained_modules'],
            'active_model_unchanged':True,'smoke_epochs':1,'device':'cpu','storage':str(store.root)}
    write_json(folder/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
