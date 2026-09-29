import json
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient
from dxa_project.geometry_ml.final_protocol import assign_groups
from dxa_project.geometry_ml.scoliosis import choose_threshold,ScoliosisNet
from dxa_project.service.app import create_app
from dxa_project.service.worker import Worker


def test_multilabel_split_distributes_both_classes_when_groups_allow_it():
    # Three independent rare positive groups must not all enter training.
    features=np.array([[1,int(i<3),int(i>=3),int(i%2==0),int(i%2==1)] for i in range(18)])
    assignment,cost=assign_groups(features,seed=17,trials=300)
    assert set(assignment)=={0,1,2}
    assert all(features[assignment==part,1:].sum(0).min()>0 for part in range(3))
    assert np.array_equal(assignment,assign_groups(features,seed=17,trials=300)[0])


def test_scoliosis_threshold_is_learned_from_given_validation_rows():
    rows=[{'truth':0,'score':.05},{'truth':0,'score':.1},{'truth':1,'score':.25},{'truth':1,'score':.3}]
    threshold,metrics,score=choose_threshold(rows)
    assert .1<threshold<=.25 and metrics['f1']==1 and score==1


def test_scoliosis_keeps_frozen_batchnorm_statistics():
    model=ScoliosisNet(False);model.train()
    assert not model.net.bn1.training and model.net.layer4.training and model.net.fc.training
    assert not model.net.layer1[0].conv1.weight.requires_grad
    assert model.net.layer4[0].conv1.weight.requires_grad


def test_examples_cover_both_labels_with_exactly_five_cases():
    from dxa_project.geometry_ml.final_examples import choose_examples
    rows=[{'truth':{'axis':i%2,'artifact':int(i==7)},'truth_region':'SPINE','predicted_region':'SPINE','id':i} for i in range(10)]
    chosen,absent=choose_examples(rows)
    assert len(chosen)==5 and not absent
    assert all({r['truth'][key] for r in chosen}=={0,1} for key in ('axis','artifact'))


def test_test_spatial_metrics_include_missing_points_as_pck_failures():
    from dxa_project.geometry_ml.final_test_metrics import aggregate,box_iou
    metrics=aggregate([{'visible':3,'found':2,'distances_mm':[2.,8.],'pck5':1,'pck10':2}])
    assert metrics['pck_10mm']==2/3 and metrics['point_coverage']==2/3
    assert metrics['point_error_mm']==5
    assert box_iou([0,0,10,10],[5,0,15,10])==1/3
    assert box_iou([0,0,10,10],None)==0


def test_iliac_point_loss_does_not_train_the_divider_channel():
    from dxa_project.geometry_ml.final_crests import point_loss
    logits=torch.zeros(1,3,16,16,requires_grad=True);presence=torch.zeros(1,2,requires_grad=True)
    class Model:
        def __call__(self,images):return {'spatial':logits,'presence':presence}
    truth=torch.zeros(2,16,16);truth[:,8,8]=1
    loss,_=point_loss('spine',Model(),torch.zeros(1,3,16,16),[{'crest':truth,'crest_present':torch.ones(2)}],torch.device('cpu'))
    loss.backward()
    assert not torch.count_nonzero(logits.grad[:,:1])
    assert logits.grad[:,1:].abs().sum()>0 and presence.grad.abs().sum()>0


def test_iliac_selection_counts_undecodable_border_peak_as_missing(monkeypatch):
    import dxa_project.geometry_ml.final_crests as module
    selection=module.CrestSelection.__new__(module.CrestSelection);selection.records=[];selection.size=8
    selection.references={'x':{'image_left':[1,1],'image_right':[2,2]}}
    target={'relative_path':'x','width':8,'height':8,'pad_left':0,'pad_top':0,'scale':1.,'spacing_mm':(1.05,.6)}
    monkeypatch.setattr(module,'DxaDataset',lambda *args:None)
    monkeypatch.setattr(module,'DataLoader',lambda *args,**kwargs:[(torch.zeros(1,3,8,8),[target])])
    monkeypatch.setattr(module,'_heatmap_point',lambda *args:(None,.9))
    class Model:
        def eval(self):return self
        def __call__(self,x):return {'spatial':torch.zeros(1,3,8,8),'presence':torch.ones(1,2)*10}
    stats={};score=selection(Model(),torch.device('cpu'),stats)
    assert np.isfinite(score) and stats['crest_pck_10mm']==0 and stats['crest_point_coverage']==0
    assert stats['crest_failure_penalized_error_mm']==60


def test_prediction_logs_include_per_image_failure_and_continue(tmp_path,monkeypatch,caplog):
    import dxa_project.service.inference as inference
    app=create_app(tmp_path/'service');store=app.state.store
    store.put('model','test-model',{'version':'test-model','path':str(tmp_path)})
    store.put('config','active_model',{'version':'test-model'})
    paths=[tmp_path/'one.dcm',tmp_path/'bad.dcm',tmp_path/'three.dcm']
    jobid=store.create_job('predict',{'paths':[str(p) for p in paths],'model_version':'test-model'},None)
    class Engine:
        batch_sizes={}
        def predict(self,ps):
            if any(p.name=='bad.dcm' for p in ps):raise ValueError('bad DICOM')
            return [{'source':str(p)} for p in ps]
    monkeypatch.setattr(inference,'save_prediction',lambda *args:None)
    monkeypatch.setattr('dxa_project.service.deliverables.finish',lambda *args:{'table':[]})
    worker=Worker(store,'cpu');worker.engine=Engine();worker.engine_version='test-model'
    with caplog.at_level('INFO',logger='dxa.prediction'):worker.run(store.job(jobid))
    job=store.job(jobid);assert job['status']=='completed_with_errors'
    assert len(job['result']['result_ids'])==2 and len(job['result']['errors'])==1
    response=TestClient(app).get(f'/v1/jobs/{jobid}/logs').json()
    assert response['total']==6
    assert [r['status'] for r in response['events'] if r['image_index']==2]==['started','failed']
    assert 'Фотография 3/3: готово' in caplog.text
    assert TestClient(app).get(f'/v1/jobs/{jobid}/logs?offset=4&limit=1').json()['next_offset']==5
