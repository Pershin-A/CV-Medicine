"""Isolated frame-to-frame axes experiment; production geometry is untouched."""
import copy
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from dxa_project.augmentation.vertebral_axes import analyze_spine, AxisConfig, _intersection
from dxa_project.augmentation.vertebral_contours import fit_contour

def ray_frame(point, direction, width_mm, height_mm):
    """First forward intersection with any of the four physical image edges."""
    candidates=[]
    for axis,limit in ((0,0.),(0,width_mm),(1,0.),(1,height_mm)):
        if abs(direction[axis])<1e-9:continue
        t=(limit-point[axis])/direction[axis]
        q=point+t*direction
        if t>1e-7 and -1e-6<=q[0]<=width_mm+1e-6 and -1e-6<=q[1]<=height_mm+1e-6:
            candidates.append((float(t),q))
    return min(candidates,key=lambda x:x[0]) if candidates else None

def analyze_frame_axes(raw, geometry, spacing_mm=(1.05,.6), polarity='bright', neighbor_scale=.25, weight_basis='length'):
    if not np.isfinite(neighbor_scale) or neighbor_scale<=0:
        raise ValueError('neighbor_scale must be positive')
    report=copy.deepcopy(analyze_spine(raw,geometry,spacing_mm,polarity=polarity))
    report['previous_global_angle_deg']=report['global_angle_deg']
    report['method']='experimental_frame_to_frame_weighted_terminals'
    xy=np.array(spacing_mm[::-1]); h,w=raw.shape
    width,height=(w-1)*xy[0],(h-1)*xy[1]
    report['lower_fragment']=None
    # A fabricated extrapolation is not reported as a reliable contour fit.
    if not report['joint_fit'].get('success') or not report['axes']:
        report['global_angle_deg']=None;report['global_axis_points']=None
        report['review_required']=True;return report
    axes=report['axes']
    def terminal(neighbor,line,outward,existing=None):
        neighbor_points=np.asarray(neighbor['axis_points'])*xy
        shared=neighbor_points[0] if outward<0 else neighbor_points[1]
        delta=neighbor_points[1]-neighbor_points[0];length=np.linalg.norm(delta)
        direction=delta/max(length,1e-9)
        hit=ray_frame(shared,outward*direction,width,height)
        if hit is None:return None
        endpoint=hit[1]
        boundary=np.asarray([[0.,0.],[width,0.]]) if endpoint[1]<1e-5 else np.asarray([[0.,height],[width,height]]) if endpoint[1]>height-1e-5 else np.asarray([[0.,0.],[0.,height]]) if endpoint[0]<1e-5 else np.asarray([[width,0.],[width,height]])
        # Fit independently to the actual terminal band using the neighboring center.
        values=np.asarray(raw,float);lo,hi=np.percentile(values,[.5,99.5]);values=np.clip((values-lo)/max(hi-lo,1e-9),0,1)
        if polarity=='dark':values=1-values
        values=gaussian_filter(values,1.)
        upper,lower=(boundary,line) if outward<0 else (line,boundary)
        own=fit_contour(values,upper,lower,xy,AxisConfig(),kind='upper_fragment' if outward<0 else 'lower_fragment',transverse=line[1]-line[0],initial_center=shared,neighbor=neighbor)
        normal=np.array([-(line[1]-line[0])[1],(line[1]-line[0])[0]],float);normal/=np.linalg.norm(normal)
        if normal[1]<0:normal=-normal
        perpendicular_hit=ray_frame(shared,outward*normal,width,height)
        terminal_height=perpendicular_hit[0] if perpendicular_hit else hit[0]
        if own['valid']:
            own_delta=np.diff(np.asarray(own['axis_points'])*xy,axis=0)[0];own_direction=own_delta/np.linalg.norm(own_delta)
            own_length=float(np.linalg.norm(own_delta))
            a=own_length if weight_basis=='length' else terminal_height
            alpha=a/max(a+neighbor_scale*length,1e-9)
            corrected=alpha*own_direction+(1-alpha)*direction;corrected/=np.linalg.norm(corrected)
        else:
            own_length=None;alpha=0.;corrected=direction
        final=ray_frame(shared,outward*corrected,width,height)
        if final is None:return None
        endpoints=[final[1],shared] if outward<0 else [shared,final[1]]
        own['axis_points']=(np.asarray(endpoints)/xy).tolist();own['length_mm']=float(final[0])
        own['angle_deg']=math.degrees(math.atan2(corrected[0],corrected[1]))
        own['terminal_refinement']={'height_mm':terminal_height,'own_length_mm':own_length,'neighbor_length_mm':float(length),'neighbor_scale':neighbor_scale,'own_weight':alpha,'neighbor_weight':1-alpha,'fit_valid':own['valid'],'frame_edge':'top' if final[1][1]<1e-5 else 'bottom' if final[1][1]>height-1e-5 else 'left' if final[1][0]<1e-5 else 'right'}
        if not own['valid']:own['review_reasons'].append('terminal_neighbor_extrapolation_only')
        return own
    lines=sorted(geometry['spine']['disc_lines'],key=lambda x:np.mean(np.asarray(x['points'])[:,1]))
    physical=[np.asarray(sorted(x['points'],key=lambda p:p[0]))*xy for x in lines]
    # Undo the old refinement of the last *inter-line* axis: bottom refinement
    # now belongs to the fragment below the last divider.
    if axes[-1].get('joint_axis_points'):
        axes[-1]['axis_points']=axes[-1]['joint_axis_points']
        delta=np.diff(np.asarray(axes[-1]['axis_points'])*xy,axis=0)[0]
        axes[-1]['length_mm']=float(np.linalg.norm(delta))
    top=terminal(axes[0],physical[0],-1);bottom=terminal(axes[-1],physical[-1],1)
    report['upper_fragment']=top;report['lower_fragment']=bottom
    if top and bottom:
        report['joint_fit']['previous_joint_points']=report['joint_fit'].get('joint_points',[])
        chain=[top]+axes+[bottom]
        report['joint_fit']['joint_points']=[chain[0]['axis_points'][0]]+[z['axis_points'][1] for z in chain]
        endpoints=[top['axis_points'][0],bottom['axis_points'][1]]
        delta=(np.asarray(endpoints[1])-endpoints[0])*xy
        report['global_axis_points']=endpoints;report['global_angle_deg']=math.degrees(math.atan2(delta[0],delta[1]))
        report['terminal_fit_valid']=top['valid'] and bottom['valid']
        report['review_required']=report['review_required'] or not report['terminal_fit_valid']
    else:
        report['global_angle_deg']=None;report['global_axis_points']=None;report['review_required']=True
    return report
