import numpy as np
from types import SimpleNamespace
from dxa_project.geometry_ml.landmarks import point_input
from dxa_project.geometry_ml.decoding import decode_heatmap
from dxa_project.geometry_ml.protocol import source_sampler

def test_crop_flip_coordinates_return_to_original_image():
    image=np.zeros((100,120),np.float32)
    _,meta=point_input(image,128,True,[30,20,80,70],context=0)
    local_x,local_y=15.,25.
    x=meta['left']+(meta['width']-1-local_x)*meta['scale_x']
    y=meta['top']+local_y*meta['scale_y']
    heat=np.zeros((128,128));heat[round(y),round(x)]=1
    point,_=decode_heatmap(heat,meta)
    assert np.linalg.norm(np.asarray(point)-[45,45])<1

def test_source_balancing_does_not_favor_sources_with_more_augmentations():
    rows=[SimpleNamespace(source_id='a',relative_path=str(i)) for i in range(3)]
    rows+=[SimpleNamespace(source_id='b',relative_path='b')]
    sampler=source_sampler(rows)
    assert abs(float(sampler.weights[:3].sum())-float(sampler.weights[3]))<1e-8
