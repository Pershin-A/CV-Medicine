"""Experimental strict terminal axes and equal-angle divider reconstruction.

Coordinates and perpendiculars are physical (mm), not raw anisotropic pixels.
Production axes and augmentation labels are deliberately not replaced.
"""
import copy
import math
import numpy as np
from .experimental_spine_axes import ray_frame
from dxa_project.augmentation.vertebral_axes import analyze_spine


def _frame_line(point, direction, width, height):
    a=ray_frame(np.asarray(point),-np.asarray(direction),width,height)
    b=ray_frame(np.asarray(point),np.asarray(direction),width,height)
    return None if a is None or b is None else [a[1].tolist(),b[1].tolist()]


def strict_terminals(report, geometry, spacing_mm):
    """Extend both outer fragments along the divider's exact normal.

    Side boundaries start at measured contact points of the neighboring body.
    No independent terminal tissue fit can pull the axis toward the iliac bones.
    Missing internal contours stay undefined rather than being invented.
    """
    result=copy.deepcopy(report);result['method']='strict_perpendicular_terminal_boundaries'
    result['global_angle_deg']=None;result['global_axis_points']=None
    if not result.get('joint_fit',{}).get('success') or not result.get('axes'):
        result['review_required']=True;return result
    xy=np.asarray(spacing_mm[::-1],float)
    width=(geometry['image_width']-1)*xy[0];height=(geometry['image_height']-1)*xy[1]
    lines=sorted(geometry['spine']['disc_lines'],key=lambda z:np.mean(np.asarray(z['points'])[:,1]))
    axes=result['axes']
    for axis in axes:
        if axis.get('joint_axis_points'):
            axis['axis_points']=copy.deepcopy(axis['joint_axis_points'])
        delta=np.diff(np.asarray(axis['axis_points'])*xy,axis=0)[0]
        axis['length_mm']=float(np.linalg.norm(delta))
        axis['angle_deg']=math.degrees(math.atan2(delta[0],delta[1]))
    def terminal(index,outward):
        line=np.asarray(lines[0 if outward<0 else -1]['points'],float)*xy
        along=line[1]-line[0];along/=np.linalg.norm(along)
        normal=np.array([-along[1],along[0]])
        if normal[1]<0:normal=-normal
        neighbor=axes[index];anchor=np.asarray(neighbor['axis_points'][0 if outward<0 else 1])*xy
        hit=ray_frame(anchor,outward*normal,width,height)
        if hit is None:return None
        contacts=np.asarray(neighbor['boundary_parts']['upper_divider' if outward<0 else 'lower_divider'])*xy
        boundary=[]
        for point in contacts:
            endpoint=ray_frame(point,outward*normal,width,height)
            if endpoint is None:return None
            boundary.append([point.tolist(),endpoint[1].tolist()])
        points=[hit[1],anchor] if outward<0 else [anchor,hit[1]]
        return {'axis_points':(np.asarray(points)/xy).tolist(),'length_mm':float(hit[0]),
                'angle_deg':math.degrees(math.atan2(normal[0],normal[1])),
                'valid':True,'review_reasons':[],
                'side_boundaries':(np.asarray(boundary)/xy).tolist(),
                'contact_points':(contacts/xy).tolist(),'direction_mm':normal.tolist(),
                'normal_dot_divider':float(normal@along),'fit_method':'strict_divider_normal',
                'contour_status':'geometric_extension_from_neighbor_contacts'}
    top=terminal(0,-1);bottom=terminal(-1,1)
    result['upper_fragment']=top;result['lower_fragment']=bottom
    if top is None or bottom is None:
        result['review_required']=True;return result
    points=[top['axis_points'][0],bottom['axis_points'][1]]
    delta=np.diff(np.asarray(points)*xy,axis=0)[0]
    result['global_axis_points']=points
    result['global_angle_deg']=math.degrees(math.atan2(delta[0],delta[1]))
    result['global_angle_pixel_deg']=math.degrees(math.atan2(points[1][0]-points[0][0],points[1][1]-points[0][1]))
    # Placement must use the same strict terminal geometry as the reported axis.
    gap=float(np.mean([a['length_mm'] for a in axes]))
    result['mean_gap_mm']=gap
    result['top_ratio']=top['length_mm']/gap if gap>1e-9 else None
    result['bottom_ratio']=bottom['length_mm']/gap if gap>1e-9 else None
    result['axis_spread_deg']=max(a['angle_deg'] for a in axes)-min(a['angle_deg'] for a in axes)
    result['review_required']=any(a.get('review_reasons') for a in axes)
    result['terminal_fit_valid']=True
    chain=[top]+axes+[bottom]
    result['max_chain_discontinuity_px']=max((float(np.linalg.norm(np.asarray(a['axis_points'][1])-b['axis_points'][0])) for a,b in zip(chain,chain[1:])),default=0.)
    return result


def analyze_strict(raw,geometry,spacing_mm=(1.05,.6),polarity='bright'):
    return strict_terminals(analyze_spine(raw,geometry,spacing_mm,polarity=polarity),geometry,spacing_mm)


def bisect_dividers(geometry,report,spacing_mm):
    """Replace each divider by the angular bisector through its shared node.

    The divider is perpendicular to the sum of downward unit axis directions.
    Nodes/axes stay fixed; crossing reconstructed lines are rejected explicitly.
    """
    if report.get('global_angle_deg') is None:return None,{'status':'undefined_axes'}
    chain=[report['upper_fragment']]+report['axes']+[report['lower_fragment']]
    original=sorted(geometry['spine']['disc_lines'],key=lambda z:np.mean(np.asarray(z['points'])[:,1]))
    if len(chain)!=len(original)+1:return None,{'status':'chain_count_mismatch'}
    xy=np.array(spacing_mm[::-1],dtype=float);w=(geometry['image_width']-1)*xy[0];h=(geometry['image_height']-1)*xy[1]
    lines=[];errors=[]
    for index,(a,b) in enumerate(zip(chain,chain[1:])):
        pa=np.asarray(a['axis_points'],dtype=float)*xy;pb=np.asarray(b['axis_points'],dtype=float)*xy
        u=pa[1]-pa[0];u/=np.linalg.norm(u)
        v=pb[1]-pb[0];v/=np.linalg.norm(v)
        normal=u+v;normal/=np.linalg.norm(normal)
        transverse=np.array([normal[1],-normal[0]])
        endpoints=_frame_line(pa[1],transverse,w,h)
        if endpoints is None:return None,{'status':'frame_intersection_failed'}
        endpoints=(np.asarray(endpoints)/xy).tolist()
        lines.append({'id':f'bisector_{index}','points':endpoints})
        # Dividers/axes are unoriented lines: compare acute angles, not rays.
        errors.append(abs(math.degrees(math.acos(np.clip(abs(transverse@u),0,1)))-math.degrees(math.acos(np.clip(abs(transverse@v),0,1)))))
    def y_at(line,x):
        a,b=line['points'];return a[1]+(x-a[0])*(b[1]-a[1])/max(b[0]-a[0],1e-9)
    # Only the body-adjacent central width is required to preserve line order.
    crossing=any(y_at(a,x)>=y_at(b,x) for a,b in zip(lines,lines[1:]) for x in [geometry['image_width']*.25,geometry['image_width']*.5,geometry['image_width']*.75])
    if crossing:return None,{'status':'rejected_crossing_dividers'}
    output=copy.deepcopy(geometry);output['spine']['disc_lines']=lines
    return output,{'status':'ok','max_equal_angle_error_deg':max(errors,default=0.),'fixed_axes':True}


def axis_from_frame_x(x_pair,width,height,spacing_mm):
    xy=np.asarray(spacing_mm[::-1]);physical=np.asarray([[x_pair[0]*(width-1)*xy[0],0.],[x_pair[1]*(width-1)*xy[0],(height-1)*xy[1]]])
    delta=physical[1]-physical[0]
    angle=math.degrees(math.atan2(delta[0],delta[1]))
    middle=physical.mean(0);direction=delta/np.linalg.norm(delta)
    endpoints=_frame_line(middle,direction,(width-1)*xy[0],(height-1)*xy[1])
    return angle,None if endpoints is None else (np.asarray(endpoints)/xy).tolist()
