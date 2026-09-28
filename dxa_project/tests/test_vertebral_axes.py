import math
import numpy as np
import pytest

from dxa_project.augmentation.vertebral_axes import analyze_spine, placement_from_axes
from labeler.geometry import empty_geometry


def test_removed_annotated_artifacts_become_negative():
    from dxa_project.augmentation.generate import _labels
    image,g=fixture()
    source={"spine_artifact":"1", "artifact_annotation_complete":True}
    labels=_labels("SPINE",g,{},image,(1.05,.6),source)
    assert labels["spine_artifact"]==0
    source.pop("artifact_annotation_complete")
    assert _labels("SPINE",g,{},image,(1.05,.6),source)["spine_artifact"] is None


def fixture(slope=0.,line_half_width=100):
    y,x=np.mgrid[:300,:300]
    image=100+1000*np.exp(-((x-(150+slope*(y-150)))/20)**2)
    g=empty_geometry(300,300)
    for i,level in enumerate((22,67,112,157,202,247)):
        g['spine']['disc_lines'].append({'id':str(i),'points':[
            [150-line_half_width,level],[150+line_half_width,level]]})
    g['spine']['iliac_crests']={'image_left':[20,280],'image_right':[280,280]}
    return image,g


def test_physical_axis_and_partial_upper_vertebra():
    image,g=fixture(.12)
    report=analyze_spine(image,g,(1.05,.6))
    expected=math.degrees(math.atan(.12*.6/1.05))
    assert len(report['axes'])==5
    assert all(a['valid'] for a in report['axes'])
    assert report['global_angle_deg']==pytest.approx(expected,abs=.8)
    assert report['top_ratio']==pytest.approx(22/45,abs=.07)
    assert placement_from_axes(g,report) is True
    assert all(a['valid_sections']>=5 for a in report['axes'])
    assert max(a['mean_absolute_error_mm'] for a in report['axes'])<1


def test_roi_width_does_not_use_annotated_segment_lengths():
    image,g=fixture()
    _,short=fixture(line_half_width=40)
    a,b=analyze_spine(image,g),analyze_spine(image,short)
    for first,second in zip(a['axes'],b['axes']):
        assert first['roi']['half_width_mm']==pytest.approx(second['roi']['half_width_mm'])
        assert first['axis_points']==pytest.approx(np.asarray(second['axis_points']))


def test_blank_image_is_unknown_and_dark_polarity_is_supported():
    image,g=fixture()
    blank=analyze_spine(np.zeros_like(image),g)
    assert blank['global_angle_deg'] is None
    assert placement_from_axes(g,blank) is None
    dark=analyze_spine(image.max()-image,g,polarity='dark')
    assert dark['global_angle_deg']==pytest.approx(0,abs=.5)


def test_crest_on_border_is_visible_and_missing_crest_is_negative():
    image,g=fixture()
    report=analyze_spine(image,g)
    g['spine']['iliac_crests']['image_left']=[0,299]
    assert placement_from_axes(g,report) is True
    g['spine']['iliac_crests']['image_left']=None
    assert placement_from_axes(g,report) is False


def test_roi_cross_sections_do_not_read_outside_image():
    image,g=fixture(.1)
    report=analyze_spine(image,g)
    fragment=report['upper_fragment']
    for section in fragment.get('sections',[]):
        points=np.asarray(section['roi_endpoints'])
        assert np.all(points>=0) and np.all(points<=299)


def test_midpoints_depend_on_borders_not_internal_brightness():
    _,g=fixture()
    y,x=np.mgrid[:300,:300]
    mask=(x>=120)&(x<=180)
    image=np.where(mask,800.,0.)
    brighter=image+np.where(mask&(x>155),300.,0.)
    first,second=analyze_spine(image,g),analyze_spine(brighter,g)
    assert first['method']=='joint_continuous_contour_l1'
    for a,b in zip(first['axes'],second['axes']):
        assert np.asarray(a['axis_points'])==pytest.approx(np.asarray(b['axis_points']),abs=1.)
        for section in b['sections']:
            midpoint=np.mean(section['roi_endpoints'],axis=0)
            assert section['point']==pytest.approx(midpoint)


def test_axis_limit_center_out_order_and_shared_boundaries():
    image,g=fixture(.65)
    report=analyze_spine(image,g)
    assert report['construction_order']==[2,1,3,0,4]
    assert all(abs(a['local_tilt_deg'])<=12.+1e-6 for a in report['axes'] if a['valid'])
    image,g=fixture()
    report=analyze_spine(image,g)
    for above,below in zip(report['axes'],report['axes'][1:]):
        assert above['boundary_parts']['lower_divider']==pytest.approx(np.asarray(below['boundary_parts']['upper_divider']))
    assert any(a['neighbor_boundary_overlap'] is not None for a in report['axes'])


def test_contour_has_variable_width_and_closes_internal_holes():
    _,g=fixture()
    y,x=np.mgrid[:300,:300]
    half=20+7*np.cos((y-22)*2*np.pi/45)
    body=np.abs(x-150)<half
    holes=(x>145)&(x<155)&(y%45>15)&(y%45<24)
    report=analyze_spine(np.where(body&~holes,1000.,0.),g)
    for axis in report['axes']:
        assert axis['valid']
        assert np.ptp([q['width_mm'] for q in axis['sections']])>5
        assert np.asarray(axis['axis_points'])[:,0]==pytest.approx([150,150],abs=1)
        assert len(axis['body_contour'])>20


@pytest.mark.parametrize('top,bottom,count',[(0,150,1),(-30,330,0)])
def test_partial_body_can_have_one_or_no_visible_dividers(top,bottom,count):
    from scipy.ndimage import gaussian_filter
    from dxa_project.augmentation.vertebral_contours import fit_contour
    from dxa_project.augmentation.vertebral_axes import AxisConfig
    y,x=np.mgrid[:300,:300]
    image=gaussian_filter(((x>=120)&(x<=180)).astype(float),1)
    xy=np.asarray([.6,1.05])
    result=fit_contour(image,np.asarray([[0,top],[299,top]])*xy,
                       np.asarray([[0,bottom],[299,bottom]])*xy,xy,AxisConfig(),
                       kind='upper_fragment' if count==1 else 'full')
    assert result['valid']
    assert result['visible_divider_count']==count
    assert np.asarray(result['body_contour']).min()>=0
    assert np.asarray(result['body_contour']).max()<=299
    assert np.asarray(result['axis_points']).min()>=0
    assert np.asarray(result['axis_points']).max()<=299


def test_joint_axes_share_exact_endpoints_and_keep_angle_constraint():
    image,g=fixture(.12)
    report=analyze_spine(image,g)
    assert report['joint_fit']['success']
    chain=[report['upper_fragment']]+report['axes']
    for a,b in zip(chain,chain[1:]):
        assert a['axis_points'][1]==b['axis_points'][0]
    for axis in chain:
        assert abs(axis['local_tilt_deg'])<=12+1e-7
    for point,line in zip(report['joint_fit']['joint_points'][1:],g['spine']['disc_lines']):
        a,b=np.asarray(line['points'])
        assert abs(np.linalg.det(np.vstack((b-a,np.asarray(point)-a))))<1e-5


def test_terminal_weights_and_symmetric_top_contour():
    image,g=fixture(.12)
    report=analyze_spine(image,g)
    assert report['terminal_refinement_applied']
    for axis in (report['upper_fragment'],report['axes'][-1]):
        weights=axis['terminal_refinement']
        assert weights['own_weight']==pytest.approx(weights['height_mm']/(weights['height_mm']+weights['neighbor_length_mm']))
        assert weights['own_weight']+weights['neighbor_weight']==pytest.approx(1)
    top=report['upper_fragment']
    assert top['contour_method']=='bilateral_tissue_symmetric_about_refined_axis'
    for section in top['symmetric_sections']:
        assert np.mean(section['roi_endpoints'],axis=0)==pytest.approx(section['point'])


def test_joint_fit_is_a_global_optimum_for_independent_noisy_centers():
    from copy import deepcopy
    from dxa_project.augmentation.joint_spine_axes import fit_joint_axes
    from dxa_project.augmentation.vertebral_axes import AxisConfig
    image,g=fixture()
    report=analyze_spine(image,g)
    axes=deepcopy(report['axes'])
    for i,axis in enumerate(axes):
        for section in axis['sections']:
            section['point'][0]+=(-1 if i%2 else 1)*5
    xy=np.asarray([.6,1.05])
    lines=[np.asarray(line['points'])*xy for line in g['spine']['disc_lines']]
    result=fit_joint_axes(axes,None,lines,xy,AxisConfig())
    assert result['success']
    assert result['objective_mm']>0
    for a,b in zip(axes,axes[1:]):
        assert a['axis_points'][1]==b['axis_points'][0]
