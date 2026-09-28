"""Height-weighted terminal directions and symmetric independent top contour."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def refine_terminal_axes(report, image, physical_lines, xy_mm, config):
    from .vertebral_axes import _intersection, _sample
    from .core import _clip_polygon
    axes=report['axes']; top=report['upper_fragment']
    if not report.get('joint_fit',{}).get('success') or len(axes)<2:
        return
    frame=np.asarray([[0.,0.],[(image.shape[1]-1)*xy_mm[0],0.]])
    def refine(axis,neighbor,boundary,at_top):
        endpoints=np.asarray(axis['axis_points'])*xy_mm
        shared=endpoints[1] if at_top else endpoints[0]
        segment=endpoints[1]-endpoints[0]
        own=segment/np.linalg.norm(segment)
        other=np.diff(np.asarray(neighbor['axis_points'])*xy_mm,axis=0)[0]
        neighbor_length=float(np.linalg.norm(other)); other/=neighbor_length
        divider=physical_lines[0] if at_top else boundary
        along=divider[1]-divider[0]; along/=np.linalg.norm(along)
        normal=np.asarray([-along[1],along[0]])
        if normal[1]<0: normal=-normal
        if at_top:
            distance=_intersection(shared,-normal,frame)
            height=float(distance) if distance is not None and distance>0 else 0.
        else:
            height=abs(float((shared-divider[0])@normal))
        alpha=height/max(height+neighbor_length,1e-9)
        direction=alpha*own+(1-alpha)*other
        direction/=np.linalg.norm(direction)
        u=np.asarray(axis['reference_u']); v=np.asarray(axis['reference_v'])
        tilt=math.degrees(math.atan2(direction@v,direction@u))
        tilt_clipped=float(np.clip(tilt,-config.max_deviation_deg,config.max_deviation_deg))
        direction=math.cos(math.radians(tilt_clipped))*u+math.sin(math.radians(tilt_clipped))*v
        offset=_intersection(shared,direction,boundary)
        if offset is None or (at_top and offset>=0) or (not at_top and offset<=0):
            axis['review_reasons'].append('terminal_refinement_failed'); return
        free=shared+offset*direction
        corrected=np.asarray([free,shared] if at_top else [shared,free])
        axis['joint_axis_points']=axis['axis_points']
        axis['axis_points']=(corrected/xy_mm).tolist()
        axis['angle_deg']=math.degrees(math.atan2(direction[0],direction[1]))
        axis['local_tilt_deg']=tilt_clipped
        axis['length_mm']=abs(float(offset))
        axis['terminal_refinement']={'height_mm':height,'neighbor_length_mm':neighbor_length,
            'own_weight':alpha,'neighbor_weight':1-alpha,'tilt_clamped':abs(tilt-tilt_clipped)>1e-8}
        axis['fit_method']='joint_l1_then_height_weighted_terminal'
    if top and top['valid']:
        refine(top,axes[0],frame,True)
    refine(axes[-1],axes[-2],physical_lines[-1],False)
    # Re-estimate the upper body's contour *after* its axis is corrected.
    # Each cross-section gets one image-derived radius shared by both sides.
    if top and top['valid']:
        endpoints=np.asarray(top['axis_points'])*xy_mm
        direction=endpoints[1]-endpoints[0]; length=np.linalg.norm(direction); direction/=length
        transverse=np.asarray([direction[1],-direction[0]])
        step=xy_mm.min()/2
        distances=np.arange(0.,min(45.,image.shape[1]*xy_mm[0]*.4),step)
        sections=[]; radii=[]
        for fraction in np.linspace(.03,.97,41):
            center=endpoints[0]+fraction*length*direction
            left,li=_sample(image,center-distances[:,None]*transverse,xy_mm)
            right,ri=_sample(image,center+distances[:,None]*transverse,xy_mm)
            valid=li&ri
            if valid.sum()<12: continue
            profile=gaussian_filter1d(np.minimum(left[valid],right[valid]),1.2)
            ds=distances[valid]
            background=float(np.percentile(profile,15)); contrast=float(np.percentile(profile,95)-background)
            if contrast<.02: continue
            foreground=profile>background+.35*contrast
            # Follow bilateral tissue from the center; ignore isolated ribs
            # after a sustained gap longer than 2 mm.
            last=0; gap=0
            for j,visible in enumerate(foreground):
                if visible: last=j; gap=0
                else: gap+=1
                if gap*step>2. and ds[j]>5.: break
            radius=float(ds[last])
            if radius<config.minimum_half_width_mm: continue
            radii.append(radius); sections.append((center,fraction))
        if len(sections)>=config.minimum_sections:
            radii=gaussian_filter1d(median_filter(radii,size=3,mode='nearest'),.8)
            left=[]; right=[]; mids=[]
            for (center,fraction),radius in zip(sections,radii):
                l=center-radius*transverse; r=center+radius*transverse
                left.append(l); right.append(r)
                mids.append({'point':(center/xy_mm).tolist(),'roi_endpoints':(np.asarray([l,r])/xy_mm).tolist(),
                             'symmetric_radius_mm':float(radius),'fraction':float(fraction)})
            # Closing segments lie on the frame and first divider.
            def closing(center,radius,line):
                pair=[]
                for sign in (-1,1):
                    p=center+sign*radius*transverse
                    t=_intersection(p,direction,line)
                    pair.append(p+t*direction)
                return pair
            upper=closing(endpoints[0],radii[0],frame)
            lower=closing(endpoints[1],radii[-1],physical_lines[0])
            polygon=_clip_polygon((np.asarray([upper[0]]+left+[lower[0],lower[1]]+right[::-1]+[upper[1]])/xy_mm).tolist(),image.shape[1],image.shape[0])
            top['pre_symmetric_body_contour']=top['body_contour']
            top['body_contour']=top['roi_polygon']=polygon
            top['symmetric_sections']=mids
            top['contour_method']='bilateral_tissue_symmetric_about_refined_axis'
            arr=np.asarray(polygon); top['body_bbox']=[*arr.min(axis=0).tolist(),*arr.max(axis=0).tolist()]
        else:
            top['review_reasons'].append('symmetric_top_contour_not_found')
    chain=([top] if top and top['valid'] else [])+axes
    report['joint_fit']['joint_points']=[chain[0]['axis_points'][0]]+[a['axis_points'][1] for a in chain]
    report['terminal_refinement_applied']=True
