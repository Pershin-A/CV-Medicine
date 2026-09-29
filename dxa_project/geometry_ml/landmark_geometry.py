"""Anatomical plausibility features, kept separate from framing criteria."""
import numpy as np

def distance_features(landmarks,spacing_mm):
    from .data import LANDMARKS
    if any(landmarks.get(k) is None for k in LANDMARKS):return None
    xy=np.asarray([landmarks[k] for k in LANDMARKS])*np.asarray([spacing_mm[1],spacing_mm[0]])
    a,b,c=xy;d12=float(np.linalg.norm(a-b));d13=float(np.linalg.norm(a-c));d23=float(np.linalg.norm(b-c))
    u,v=b-a,c-a;area=abs(float(u[0]*v[1]-u[1]*v[0]))/2
    return {'distance_12_mm':d12,'distance_13_mm':d13,'distance_23_mm':d23,
            'ratio_12_13':d12/max(d13,1e-6),'ratio_23_13':d23/max(d13,1e-6),
            'normalized_area':area/max(d13*d13,1e-6)}

def plausibility(features,bounds=None):
    if features is None:return {'status':'unknown','reason':'missing_landmark'}
    if min(features[k] for k in ('distance_12_mm','distance_13_mm','distance_23_mm'))<1e-6:
        return {'status':'implausible','reason':'coincident_points'}
    if bounds is None:return {'status':'not_calibrated','reason':'features_only'}
    bad=[key for key,(lo,hi) in bounds.items() if not lo<=features[key]<=hi]
    return {'status':'implausible' if bad else 'plausible','outside_training_ranges':bad}

def fit_geometry_ranges(records):
    import json
    from dxa_project.augmentation.core import hip_position_ok
    samples=[]
    for r in records:
        g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
        if hip_position_ok(g):
            f=distance_features(g['hip']['landmarks'],r.spacing_mm)
            if f is not None:samples.append(f)
    if len(samples)<10:raise ValueError('At least ten complete training examples required')
    bounds={}
    for key in ('ratio_12_13','ratio_23_13','normalized_area'):
        lo,hi=np.quantile([x[key] for x in samples],[.005,.995]);margin=.1*max(hi-lo,1e-3)
        bounds[key]=[float(max(0,lo-margin)),float(hi+margin)]
    return {'bounds':bounds,'n':len(samples),'source':'training originals only','role':'advisory or abstention, not a replacement for anatomical visibility'}
