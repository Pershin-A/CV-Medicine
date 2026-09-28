"""Image-derived lateral contours and axes through geometric midpoints."""
import math
from dataclasses import replace
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter1d, label, median_filter
from scipy.optimize import linprog


def fit_contour(image, upper, lower, xy_mm, config, *, kind='full',
                transverse=None, initial_center=None, neighbor=None):
    from .vertebral_axes import _intersection, _sample
    from .core import _clip_polygon, _clip_segment
    failed={'kind':kind,'valid':False,'review_reasons':[],'axis_points':None}
    left_mid=(upper[0]+lower[0])/2
    right_mid=(upper[1]+lower[1])/2
    v=right_mid-left_mid if transverse is None else np.asarray(transverse,float)
    if np.linalg.norm(v)<1e-8:
        return {**failed,'review_reasons':['invalid_transverse_direction']}
    v=v/np.linalg.norm(v)
    if v[0]<0:
        v=-v
    u=np.asarray([-v[1],v[0]])
    c=(left_mid+right_mid)/2 if initial_center is None else np.asarray(initial_center,float)
    t1,t2=_intersection(c,u,upper),_intersection(c,u,lower)
    if t1 is None or t2 is None or t2-t1<3:
        return {**failed,'review_reasons':['invalid_interline_geometry']}
    step=xy_mm.min()/2
    reach=min(image.shape[1]*xy_mm[0]*.49,90.)
    s=np.arange(-reach,reach+step/2,step)
    levels=np.linspace(t1+.035*(t2-t1),t2-.035*(t2-t1),config.section_count)
    previous=None
    expected_width=None
    if neighbor and neighbor.get('valid'):
        neighbor_axis=np.asarray(neighbor['axis_points'])*xy_mm
        previous=float(np.dot(neighbor_axis.mean(axis=0)-c,v))
        expected_width=neighbor['roi']['half_width_mm']*2
    # Select a 2D body cluster first. A dark hole within the same body must
    # not split one transverse section into two unrelated half-vertebrae.
    yy,xx=np.mgrid[:image.shape[0],:image.shape[1]]
    physical_points=np.stack((xx*xy_mm[0],yy*xy_mm[1]),axis=-1)
    local_s=(physical_points-c)@v
    local_t=(physical_points-c)@u
    def line_level(line):
        normal=np.asarray([-(line[1]-line[0])[1],(line[1]-line[0])[0]])
        return ((line[0]-c)@normal-local_s*(v@normal))/(u@normal)
    band=(local_t>=line_level(upper)) & (local_t<=line_level(lower)) & (np.abs(local_s)<reach)
    band_values=image[band]
    if len(band_values)<15:
        return {**failed,'review_reasons':['body_roi_not_found']}
    background2=float(np.percentile(band_values,15))
    contrast2=float(np.percentile(band_values,95)-background2)
    if contrast2<.02:
        return {**failed,'review_reasons':['body_roi_not_found']}
    foreground=(image>background2+config.width_threshold_fraction*contrast2) & band
    foreground=binary_closing(foreground,structure=np.ones((3,3)),iterations=2) & band
    clusters,number=label(foreground)
    choices=[]
    anchor2=previous if previous is not None else 0.
    for k in range(1,number+1):
        component=clusters==k
        area=int(component.sum())
        if area<20:
            continue
        delta=float(np.mean(local_s[component])-anchor2)
        score=float(np.mean(image[component])-background2)*math.sqrt(area)/(1+(delta/20.)**2)
        choices.append((score,k))
    if not choices:
        return {**failed,'review_reasons':['body_roi_not_found']}
    body=binary_fill_holes(clusters==max(choices)[1]) & band
    sections=[None]*len(levels)
    # Establish the center of a body first; extend its sides in both directions.
    center_index=len(levels)//2
    order=[center_index]
    for distance in range(1,len(levels)):
        order.extend(i for i in (center_index-distance,center_index+distance) if 0<=i<len(levels))
    for index in order:
        t=levels[index]
        points=c+t*u+s[:,None]*v
        values,inside=_sample(image,points,xy_mm)
        if inside.sum()<15:
            continue
        background=float(np.percentile(values[inside],15))
        contrast=float(np.percentile(values[inside],95)-background)
        if contrast<.02:
            continue
        profile=gaussian_filter1d(np.where(inside,values,background),1.3)
        body_values,body_inside=_sample(body.astype(float),points,xy_mm)
        tissue=inside & body_inside & (body_values>.25)
        # A section of one connected body may contain internal gaps. Its
        # lateral borders are the outermost tissue pixels of that body.
        indices=np.flatnonzero(tissue)
        if not len(indices):
            continue
        tissue[indices[0]:indices[-1]+1]=True
        components,count=label(tissue)
        near=[j for j in range(len(levels)) if sections[j] is not None]
        anchor=previous if previous is not None else 0.
        width_prior=expected_width
        if near:
            nearest=min(near,key=lambda j:abs(j-index))
            anchor=sections[nearest]['center_s_mm']
            width_prior=sections[nearest]['width_mm']
        candidates=[]
        for component in range(1,count+1):
            ids=np.flatnonzero(components==component)
            l,r=float(s[ids[0]]),float(s[ids[-1]])
            width=r-l
            midpoint=(l+r)/2
            if not 2*config.minimum_half_width_mm<=width<=2*config.maximum_half_width_mm:
                continue
            brightness=float(np.mean(profile[ids])-background)
            score=brightness*math.sqrt(width)/(1+((midpoint-anchor)/15.)**2)
            if width_prior:
                score/=1+3*math.log(width/width_prior)**2
            candidates.append((score,l,r))
        if not candidates:
            continue
        _,l,r=max(candidates)
        sections[index]={'t_mm':float(t),'left_s_mm':l,'right_s_mm':r,
                         'center_s_mm':(l+r)/2,'width_mm':r-l,'background':background}
    sections=[section for section in sections if section is not None]
    if len(sections)<config.minimum_sections:
        if config.width_threshold_fraction<.65:
            return fit_contour(image,upper,lower,xy_mm,
                replace(config,width_threshold_fraction=config.width_threshold_fraction+.15),
                kind=kind,transverse=transverse,initial_center=initial_center,neighbor=neighbor)
        return {**failed,'review_reasons':['too_few_boundary_sections'],
                'sections':[],'valid_sections':len(sections)}
    t=np.asarray([section['t_mm'] for section in sections])
    # Remove isolated contour spikes, never reweight the center by brightness.
    left=gaussian_filter1d(median_filter([q['left_s_mm'] for q in sections],size=3,mode='nearest'),.65)
    right=gaussian_filter1d(median_filter([q['right_s_mm'] for q in sections],size=3,mode='nearest'),.65)
    center=(left+right)/2
    width=right-left
    for j,section in enumerate(sections):
        section.update(left_s_mm=float(left[j]),right_s_mm=float(right[j]),
                       center_s_mm=float(center[j]),width_mm=float(width[j]),
                       point=((c+t[j]*u+center[j]*v)/xy_mm).tolist(),
                       roi_endpoints=(np.asarray([c+t[j]*u+left[j]*v,c+t[j]*u+right[j]*v])/xy_mm).tolist())
    trim=(1-config.central_fraction)/2
    fitting=(t>=t1+trim*(t2-t1)) & (t<=t2-trim*(t2-t1))
    if fitting.sum()<config.minimum_sections:
        return {**failed,'review_reasons':['too_few_axis_sections'],'sections':sections}
    tf,m=t[fitting],center[fitting]
    n=len(tf)
    design=np.column_stack((np.ones(n),tf))
    limit=math.tan(math.radians(config.max_deviation_deg))
    solution=linprog(np.r_[0.,0.,np.ones(n)],
        A_ub=np.vstack((np.column_stack((design,-np.eye(n))),np.column_stack((-design,-np.eye(n))))),
        b_ub=np.r_[m,-m],bounds=[(float(left.min()),float(right.max())),(-limit,limit)]+[(0,None)]*n,method='highs')
    if not solution.success:
        return {**failed,'review_reasons':['l1_optimizer_failed']}
    a,b=solution.x[:2]
    origin=c+a*v
    direction=u+b*v
    direction/=np.linalg.norm(direction)
    first,last=_intersection(origin,direction,upper),_intersection(origin,direction,lower)
    if first is None or last is None or last<=first:
        return {**failed,'review_reasons':['axis_intersection_failed']}
    axis=np.asarray([origin+first*direction,origin+last*direction])/xy_mm
    clipped_axis,_=_clip_segment(axis[0].tolist(),axis[1].tolist(),image.shape[1],image.shape[0])
    if clipped_axis is None:
        return {**failed,'review_reasons':['axis_outside_image']}
    axis=np.asarray(clipped_axis)
    def end_pair(line,j):
        endpoints=[]
        for side in (left,right):
            p=c+side[j]*v
            offset=_intersection(p,u,line)
            endpoints.append(p+offset*u)
        return endpoints
    top,bottom=end_pair(upper,0),end_pair(lower,-1)
    continuity=None
    reasons=[]
    if neighbor and neighbor.get('valid'):
        # Compare intervals on the shared divider, rather than 2D IoU of
        # adjacent bodies (which are separated vertically by definition).
        above=np.asarray(neighbor['axis_points']).mean(axis=0)[1]<axis.mean(axis=0)[1]
        other=np.asarray(neighbor['boundary_parts']['lower_divider' if above else 'upper_divider'])*xy_mm
        current=top if above else bottom
        q=np.sort((np.asarray(current)-c)@v)
        p=np.sort((other-c)@v)
        overlap=max(0.,min(q[1],p[1])-max(q[0],p[0]))
        continuity=overlap/max(1e-9,min(q[1]-q[0],p[1]-p[0]))
        if continuity<config.minimum_neighbor_overlap:
            reasons.append('neighbor_boundary_overlap_low')
        # Share the measured interface when the two image-derived estimates
        # agree. Large differences remain visible and generate a warning.
        if continuity>=config.minimum_neighbor_overlap and np.max(np.abs(q-p))<6.:
            if above:
                top=list(other)
            else:
                bottom=list(other)
    lside=[top[0]]+[c+tt*u+ss*v for tt,ss in zip(t,left)]+[bottom[0]]
    rside=[top[1]]+[c+tt*u+ss*v for tt,ss in zip(t,right)]+[bottom[1]]
    polygon=_clip_polygon((np.asarray(lside+list(reversed(rside)))/xy_mm).tolist(),image.shape[1],image.shape[0])
    error=float(np.mean(np.abs(m-(a+b*tf))))
    if error>config.max_error_mm:
        reasons.append('large_l1_residual')
    fitted=a+b*tf
    if np.any(fitted<left[fitting]) or np.any(fitted>right[fitting]):
        reasons.append('axis_leaves_body_contour')
    if np.any(axis[:,0]<0) or np.any(axis[:,0]>image.shape[1]-1):
        reasons.append('axis_endpoint_outside_image')
    polygon_array=np.asarray(polygon)
    bbox=[*polygon_array.min(axis=0),*polygon_array.max(axis=0)] if len(polygon) else None
    bbox_overlap=None
    if bbox and neighbor and neighbor.get('body_bbox'):
        other=neighbor['body_bbox']
        bbox_overlap=max(0.,min(bbox[2],other[2])-max(bbox[0],other[0]))/max(1e-9,min(bbox[2]-bbox[0],other[2]-other[0]))
        if bbox_overlap<config.minimum_neighbor_overlap:
            reasons.append('neighbor_bbox_horizontal_overlap_low')
    return {'kind':kind,'valid':True,'review_reasons':reasons,
            'reference_origin_mm':c.tolist(),'reference_u':u.tolist(),'reference_v':v.tolist(),
            'axis_points':axis.tolist(),'angle_deg':math.degrees(math.atan2(direction[0],direction[1])),
            'shift_mm':float(a),'local_tilt_deg':math.degrees(math.atan(b)),
            'mean_absolute_error_mm':error,'valid_sections':n,'length_mm':float(np.linalg.norm((axis[1]-axis[0])*xy_mm)),
            'roi':{'center_s_mm':float(np.mean(center)),'half_width_mm':float(np.mean(width)/2)},
            'roi_polygon':polygon,'body_contour':polygon,'body_bbox':bbox,
            'neighbor_boundary_overlap':continuity,'neighbor_bbox_horizontal_overlap':bbox_overlap,
            'segmentation_threshold_fraction':config.width_threshold_fraction,'sections':sections,
            'visible_divider_count':sum(_clip_segment(*(np.asarray(pair)/xy_mm).tolist(),image.shape[1],image.shape[0])[0] is not None
                for pair in ([bottom] if kind=='upper_fragment' else [top,bottom])),
            'boundary_parts':{'upper_divider':(np.asarray(top)/xy_mm).tolist(),
                              'lower_divider':(np.asarray(bottom)/xy_mm).tolist(),
                              'left':(np.asarray(lside)/xy_mm).tolist(),
                              'right':(np.asarray(rside)/xy_mm).tolist()}}
