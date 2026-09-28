"""Image-derived body contour -> lateral-boundary midpoints -> constrained L1 fit.

All fitting is in physical (x_mm, y_mm) coordinates. Saved endpoints and
diagnostic polygons are converted back to the original image pixel space.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class AxisConfig:
    central_fraction: float = .70
    section_count: int = 41
    minimum_sections: int = 5
    max_deviation_deg: float = 12.
    smoothing_sigma_px: float = 1.
    max_error_mm: float = 3.
    minimum_half_width_mm: float = 5.
    maximum_half_width_mm: float = 45.
    width_threshold_fraction: float = .35
    minimum_neighbor_overlap: float = .65


def _intersection(origin, direction, endpoints):
    a, b = endpoints
    system = np.column_stack((direction, -(b-a)))
    if abs(np.linalg.det(system)) < 1e-9:
        return None
    return float(np.linalg.solve(system, a-origin)[0])


def _sample(image, points, xy_mm):
    pixels = points / xy_mm
    valid = ((pixels[:,0] >= 0) & (pixels[:,0] <= image.shape[1]-1) &
             (pixels[:,1] >= 0) & (pixels[:,1] <= image.shape[0]-1))
    values = map_coordinates(image, (pixels[:,1],pixels[:,0]), order=1,
                             mode="constant", cval=np.nan)
    return values, valid


def analyze_spine(image, geometry, spacing_mm=(1.05,.6), *, polarity="bright",config=None):
    config=config or AxisConfig()
    xy_mm=np.asarray([spacing_mm[1],spacing_mm[0]],float)
    if np.any(xy_mm<=0) or not np.isfinite(xy_mm).all():
        raise ValueError("Positive finite pixel spacing is required")
    values=np.asarray(image,float)
    if values.shape!=(geometry["image_height"],geometry["image_width"]):
        raise ValueError("Image/annotation dimensions differ")
    lo,hi=np.percentile(values[np.isfinite(values)],(.5,99.5))
    values=np.nan_to_num(np.clip((values-lo)/max(hi-lo,1e-9),0,1))
    if polarity=="dark":
        values=1-values
    elif polarity!="bright":
        raise ValueError("Polarity must be bright or dark")
    values=gaussian_filter(values,config.smoothing_sigma_px)
    lines=sorted(geometry["spine"]["disc_lines"],key=lambda line:np.mean(np.asarray(line["points"])[:,1]))
    physical=[np.asarray(sorted(line["points"],key=lambda p:p[0]))*xy_mm for line in lines]
    from .vertebral_contours import fit_contour
    axes=[None]*max(0,len(physical)-1)
    middle=len(axes)//2
    order=([middle] if axes else [])
    for distance in range(1,len(axes)):
        order.extend(i for i in (middle-distance,middle+distance) if 0<=i<len(axes))
    for index in order:
        neighbor=next((axes[j] for j in (index-1,index+1)
                       if 0<=j<len(axes) and axes[j] and axes[j]['valid']),None)
        axes[index]=fit_contour(values,physical[index],physical[index+1],xy_mm,config,neighbor=neighbor)
    report={"spacing_mm_row_col":list(spacing_mm),"config":asdict(config),"axes":axes,
            "upper_fragment":None,"global_axis_points":None,"global_angle_deg":None,
            "global_angle_pixel_deg":None,"top_ratio":None,"mean_gap_mm":None,
            "axis_spread_deg":None,"review_required":True,
            "method":"joint_continuous_contour_l1","construction_order":order}
    valid=[axis for axis in axes if axis["valid"]]
    if valid:
        gap=float(np.mean([axis["length_mm"] for axis in valid]))
        report["mean_gap_mm"]=gap
        c=np.asarray(valid[0]["axis_points"][0])*xy_mm
        # For the partial top vertebra, transverse direction follows D1 exactly.
        upper=np.asarray([[0.,0.],[(image.shape[1]-1)*xy_mm[0],0.]])
        fragment=fit_contour(values,upper,physical[0],xy_mm,config,kind="upper_fragment",
                      transverse=physical[0][1]-physical[0][0],initial_center=c,neighbor=valid[0])
        report["upper_fragment"]=fragment
        if fragment["valid"]:
            report["top_ratio"]=fragment["length_mm"]/gap
        report["axis_spread_deg"]=max(a["angle_deg"] for a in valid)-min(a["angle_deg"] for a in valid)
    from .joint_spine_axes import fit_joint_axes
    joint=fit_joint_axes(axes,report['upper_fragment'],physical,xy_mm,config)
    report['joint_fit']=joint
    if joint['success']:
        from .terminal_spine_axes import refine_terminal_axes
        refine_terminal_axes(report,values,physical,xy_mm,config)
        gap=float(np.mean([axis['length_mm'] for axis in axes]))
        report['mean_gap_mm']=gap
        if report['upper_fragment'] and report['upper_fragment']['valid']:
            report['top_ratio']=report['upper_fragment']['length_mm']/gap
        report['axis_spread_deg']=max(a['angle_deg'] for a in axes)-min(a['angle_deg'] for a in axes)
    else:
        report['top_ratio']=None
        report['mean_gap_mm']=None
        for axis in axes:
            axis['review_reasons'].append(joint['reason'])
            axis['valid']=False
        if report['upper_fragment']:
            report['upper_fragment']['valid']=False
            report['upper_fragment']['review_reasons'].append(joint['reason'])
    if axes and axes[0]["valid"] and axes[-1]["valid"]:
        endpoints=[axes[0]["axis_points"][0],axes[-1]["axis_points"][1]]
        delta=(np.asarray(endpoints[1])-endpoints[0])*xy_mm
        pixel_delta=np.subtract(endpoints[1],endpoints[0])
        report["global_axis_points"]=endpoints
        report["global_angle_deg"]=math.degrees(math.atan2(delta[0],delta[1]))
        report["global_angle_pixel_deg"]=math.degrees(math.atan2(pixel_delta[0],pixel_delta[1]))
    report["review_required"]=(len(valid)!=len(axes) or not axes or
        any(axis["review_reasons"] for axis in axes) or
        report["upper_fragment"] is None or not report["upper_fragment"]["valid"] or
        bool(report["upper_fragment"]["review_reasons"]))
    return report


def placement_from_axes(geometry, report, ratio_range=(.25,.75)):
    h,w=geometry["image_height"],geometry["image_width"]
    points=geometry["spine"]["iliac_crests"].values()
    if len(geometry["spine"]["disc_lines"]) not in range(4,8) or not all(
            p is not None and 0<=p[0]<=w-1 and 0<=p[1]<=h-1 for p in points):
        return False
    ratio=report["top_ratio"]
    if ratio is None or report.get("review_required",False):
        return None  # uncertain brightness fit must not become a fabricated label
    return ratio_range[0]<=ratio<=ratio_range[1]
