import numpy as np
from dxa_project.geometry_ml.decoding import decode_heatmap

def test_padding_peak_is_excluded_and_flip_is_restored():
    a=np.zeros((8,8));a[0,0]=1.;a[3,4]=.8
    meta={'width':4,'height':8,'left':2,'top':0,'scale':1.,'flipped':False}
    assert decode_heatmap(a,meta)[0]==[2.,3.]
    assert decode_heatmap(a,dict(meta,flipped=True))[0]==[1.,3.]

def test_local_decoder_stays_in_valid_pixels():
    a=np.zeros((8,8));a[2,2]=.8;a[2,3]=.7;a[:,0]=1
    meta={'width':4,'height':8,'left':2,'top':0,'scale':1.}
    p,peak=decode_heatmap(a,meta,'local_softargmax')
    assert 0<=p[0]<=3 and 0<=p[1]<=7 and peak==.8

def test_raw_logits_preserve_ranking_after_sigmoid_saturation():
    probabilities=np.ones((4,4))
    logits=np.zeros((4,4));logits[2,3]=30
    meta={'width':4,'height':4,'left':0,'top':0,'scale':1.}
    assert decode_heatmap(probabilities,meta,'masked_logit_argmax',raw_logits=logits)[0]==[3.,2.]
