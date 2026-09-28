"""Global constrained L1 fit of a continuous vertebral polyline."""
import math
import numpy as np
from scipy.optimize import linprog


def fit_joint_axes(axes, fragment, physical_lines, xy_mm, config):
    from .vertebral_axes import _intersection
    if not axes or any(not axis['valid'] for axis in axes):
        return {'success':False,'reason':'joint_fit_missing_contour'}
    include_fragment=fragment is not None and fragment['valid']
    chain=([fragment] if include_fragment else [])+axes
    frame=np.asarray(chain[0]['boundary_parts']['upper_divider'])*xy_mm
    lines=([frame] if include_fragment else [])+physical_lines
    origins=np.asarray([line[0] for line in lines])
    directions=np.asarray([(line[1]-line[0])/np.linalg.norm(line[1]-line[0]) for line in lines])
    count=len(lines)
    bounds=[]
    for j in range(count):
        intervals=[]
        if j>0:
            intervals.append((np.asarray(chain[j-1]['boundary_parts']['lower_divider'])*xy_mm-origins[j])@directions[j])
        if j<len(chain):
            intervals.append((np.asarray(chain[j]['boundary_parts']['upper_divider'])*xy_mm-origins[j])@directions[j])
        low=max(float(np.min(interval)) for interval in intervals)
        high=min(float(np.max(interval)) for interval in intervals)
        if high<low:
            return {'success':False,'reason':'joint_fit_disjoint_boundary_intervals'}
        bounds.append((low,high))
    rows=[]; constants=[]; weights=[]; constraints=[]; rhs=[]
    limit=math.tan(math.radians(config.max_deviation_deg))
    for j,axis in enumerate(chain):
        u=np.asarray(axis['reference_u']); v=np.asarray(axis['reference_v'])
        delta0=origins[j+1]-origins[j]
        delta=np.zeros((2,count)); delta[:,j]=-directions[j]; delta[:,j+1]=directions[j+1]
        for normal in (v-limit*u,-v-limit*u,-u):
            constraints.append(normal@delta)
            rhs.append(float(-normal@delta0)-(1e-4 if np.array_equal(normal,-u) else 0))
        selected=[]
        for section in axis['sections']:
            point=np.asarray(section['point'])*xy_mm
            top=_intersection(point,u,lines[j]); bottom=_intersection(point,u,lines[j+1])
            if top is None or bottom is None or bottom-top<1e-6:
                continue
            fraction=-top/(bottom-top)
            trim=(1-config.central_fraction)/2
            if trim<=fraction<=1-trim:
                selected.append((point,fraction))
        if len(selected)<config.minimum_sections:
            return {'success':False,'reason':'joint_fit_too_few_sections'}
        for point,fraction in selected:
            predicted0=(1-fraction)*origins[j]+fraction*origins[j+1]
            for normal in (v,u):
                row=np.zeros(count)
                row[j]=(1-fraction)*float(normal@directions[j])
                row[j+1]=fraction*float(normal@directions[j+1])
                rows.append(row); constants.append(float(normal@(point-predicted0)))
                weights.append(1/len(selected))
    design=np.asarray(rows); targets=np.asarray(constants); n=len(rows)
    inequalities=np.vstack((np.column_stack((design,-np.eye(n))),
                            np.column_stack((-design,-np.eye(n))),
                            np.column_stack((constraints,np.zeros((len(constraints),n))))))
    solution=linprog(np.r_[np.zeros(count),weights],A_ub=inequalities,
        b_ub=np.r_[targets,-targets,rhs],bounds=bounds+[(0,None)]*n,method='highs')
    if not solution.success:
        return {'success':False,'reason':'joint_constraints_infeasible','optimizer_message':solution.message}
    points=origins+solution.x[:count,None]*directions
    pixels=(points/xy_mm).tolist()
    for j,axis in enumerate(chain):
        axis['independent_axis_points']=axis['axis_points']
        axis['axis_points']=[pixels[j],pixels[j+1]]
        delta=points[j+1]-points[j]
        u=np.asarray(axis['reference_u']); v=np.asarray(axis['reference_v'])
        axis['angle_deg']=math.degrees(math.atan2(delta[0],delta[1]))
        axis['local_tilt_deg']=math.degrees(math.atan2(delta@v,delta@u))
        axis['length_mm']=float(np.linalg.norm(delta))
        errors=[]; outside=False
        for section in axis['sections']:
            point=np.asarray(section['point'])*xy_mm
            top=_intersection(point,u,lines[j]); bottom=_intersection(point,u,lines[j+1])
            if top is None or bottom is None or bottom-top<1e-6:
                continue
            fraction=-top/(bottom-top)
            trim=(1-config.central_fraction)/2
            if not trim<=fraction<=1-trim:
                continue
            longitudinal=float((point-points[j])@u/(delta@u))
            if not 0<=longitudinal<=1:
                continue
            fitted=points[j]+longitudinal*delta
            errors.append(abs(float((point-fitted)@v)))
            l,r=np.asarray(section['roi_endpoints'])*xy_mm
            coordinate=float((fitted-l)@v)
            outside|=coordinate < -1e-6 or coordinate>float((r-l)@v)+1e-6
        axis['mean_absolute_error_mm']=float(np.mean(errors))
        axis['valid_sections']=len(errors)
        axis['shift_mm']=float((points[j]-np.asarray(axis['reference_origin_mm']))@v)
        axis['review_reasons']=[r for r in axis['review_reasons'] if r not in ('large_l1_residual','axis_leaves_body_contour')]
        if axis['mean_absolute_error_mm']>config.max_error_mm:
            axis['review_reasons'].append('large_l1_residual')
        if outside:
            axis['review_reasons'].append('axis_leaves_body_contour')
        axis['fit_method']='joint_continuous_l1'
    return {'success':True,'objective_mm':float(solution.fun),'joint_points':pixels,
            'includes_upper_fragment':include_fragment,'max_gap_mm':0.,
            'likelihood_model':'independent Laplace errors in fixed local coordinates; equal total weight per vertebra'}
