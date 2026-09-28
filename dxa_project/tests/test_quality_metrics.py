import pytest
from dxa_project.geometry_ml.evaluate import binary_metrics,with_ci


def test_binary_metrics_use_positive_class_one_and_continuous_scores():
    rows=[{'truth':y,'prediction':p,'score':s,'study':str(i)} for i,(y,p,s) in
          enumerate(((1,1,.9),(1,0,.7),(0,0,.2),(0,1,.3)))]
    metrics=binary_metrics(rows)
    assert metrics['sensitivity']==.5
    assert metrics['specificity']==.5
    assert metrics['balanced_accuracy']==.5
    assert metrics['f1']==.5
    assert metrics['roc_auc']==1
    report=with_ci(rows,40)
    assert report['ci95']['f1'][0]<=.5<=report['ci95']['f1'][1]


def test_auc_undefined_for_single_class():
    metrics=binary_metrics([{'truth':0,'prediction':0,'score':.2,'study':'a'}])
    assert metrics['roc_auc'] is None
    assert metrics['sensitivity'] is None
    assert metrics['specificity']==1
