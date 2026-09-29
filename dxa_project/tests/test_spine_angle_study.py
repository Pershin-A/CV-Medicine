import numpy as np
import torch
from dxa_project.geometry_ml.spine_angle_geometry import strict_terminals,bisect_dividers,axis_from_frame_x
from dxa_project.geometry_ml.spine_angle_study import frame_angle_loss,all_peak_priors


def test_strict_terminals_ignore_pelvic_direction_and_are_physical_perpendiculars():
    # In pixel coordinates divider slope=.5; with Y twice X its physical slope=1.
    g={'image_width':101,'image_height':201,'spine':{'disc_lines':[{'points':[[0,40],[100,90]]},{'points':[[0,110],[100,160]]}]}}
    axis={'axis_points':[[50,65],[50,135]],'boundary_parts':{'upper_divider':[[40,60],[60,70]],'lower_divider':[[40,130],[60,140]]},'review_reasons':[]}
    report={'joint_fit':{'success':True},'axes':[axis],'upper_fragment':{'axis_points':[[0,0],[50,65]]},'global_angle_deg':80}
    a=strict_terminals(report,g,(2,1))
    assert abs(a['upper_fragment']['normal_dot_divider'])<1e-10
    assert abs(a['lower_fragment']['normal_dot_divider'])<1e-10
    assert abs(a['lower_fragment']['angle_deg']+45)<1e-9
    assert a['max_chain_discontinuity_px']==0
    assert a['top_ratio']==a['upper_fragment']['length_mm']/a['mean_gap_mm']
    assert a['bottom_ratio']==a['lower_fragment']['length_mm']/a['mean_gap_mm']
    assert report['global_angle_deg']==80


def test_bisector_keeps_node_and_equal_physical_angles():
    g={'image_width':101,'image_height':201,'spine':{'disc_lines':[{'points':[[0,60],[100,60]]},{'points':[[0,140],[100,140]]}]}}
    a={'global_angle_deg':0,'upper_fragment':{'axis_points':[[40,0],[50,60]]},'axes':[{'axis_points':[[50,60],[60,140]]}],'lower_fragment':{'axis_points':[[60,140],[55,200]]}}
    new,info=bisect_dividers(g,a,(2,1))
    assert info['status']=='ok'
    assert info['max_equal_angle_error_deg']<1e-8
    assert len(new['spine']['disc_lines'])==2


def test_angle_loss_penalizes_difference_and_has_finite_gradients():
    t=[{'reference_angle':0.,'reference_frame_x':[.5,.5],'width':101,'height':201,'spacing_mm':(2,1)}]
    equal=torch.tensor([[.5,.5]],requires_grad=True);wrong=torch.tensor([[.2,.8]],requires_grad=True)
    assert frame_angle_loss(equal,t)[0].item()==0
    angular,endpoint=frame_angle_loss(wrong,t);(angular+endpoint).backward()
    assert angular.item()>0 and torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum()>0
    a,p=axis_from_frame_x([.5,.5],101,201,(2,1));assert a==0 and p is not None


def test_all_peak_penalty_detects_extra_close_pair_outside_reference_candidates():
    grid=torch.linspace(0,1,160)
    def logits(centers):
        p=sum(torch.exp(-((grid-c)/.008).square()) for c in centers).clamp(.001,.999)
        return torch.logit(p)[None,None,:,None].expand(1,1,160,80).clone().requires_grad_()
    t=[{'pad_left':0,'pad_top':0,'width':80,'height':160,'scale':1.,'ordered_lines':torch.tensor([[.15,0],[.35,0],[.55,0],[.75,0]])}]
    normal=logits([.15,.35,.55,.75]);extra=logits([.15,.35,.55,.75,.80])
    a=all_peak_priors(normal,t,2.);b=all_peak_priors(extra,t,2.)
    assert b['close'].item()>a['close'].item()
    b['close'].backward();assert torch.isfinite(extra.grad).all() and extra.grad.abs().sum()>0
