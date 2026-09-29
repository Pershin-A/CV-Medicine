"""Landmark decoding on actual image pixels, excluding letterbox padding."""
import numpy as np

def decode_heatmap(heatmap,meta,method='masked_argmax',radius=3,raw_logits=None):
    if method not in ('legacy','masked_argmax','local_softargmax','masked_logit_argmax','local_logit_softargmax'):
        raise ValueError(method)
    use_logits=method in ('masked_logit_argmax','local_logit_softargmax')
    if use_logits and raw_logits is None:raise ValueError('Raw-logit decoder requires raw_logits')
    h,w=heatmap.shape
    left,top=int(meta['left']),int(meta['top'])
    dw=int(meta.get('resized_width',round(meta['width']*meta['scale'])))
    dh=int(meta.get('resized_height',round(meta['height']*meta['scale'])))
    valid=np.zeros((h,w),dtype=bool);valid[top:top+dh,left:left+dw]=True
    ranking=raw_logits if use_logits else heatmap
    values=np.where(valid,ranking,-np.inf) if method!='legacy' else ranking
    if not np.isfinite(values).any():return None,0.
    y,x=np.unravel_index(values.argmax(),values.shape);peak=float(heatmap[y,x])
    if method in ('local_softargmax','local_logit_softargmax'):
        x0,x1=max(left,x-radius),min(left+dw,x+radius+1)
        y0,y1=max(top,y-radius),min(top+dh,y+radius+1)
        local=ranking[y0:y1,x0:x1].astype(float)
        weights=np.exp((local-local.max())/(1. if use_logits else .05))
        yy,xx=np.mgrid[y0:y1,x0:x1]
        x=float((weights*xx).sum()/weights.sum());y=float((weights*yy).sum()/weights.sum())
    sx=meta.get('scale_x',meta['scale']);sy=meta.get('scale_y',meta['scale'])
    px=(x-left)/sx;py=(y-top)/sy
    if meta.get('flipped'):px=meta['width']-1-px
    if method!='legacy' and not(0<=px<=meta['width']-1 and 0<=py<=meta['height']-1):
        return None,peak
    px+=meta.get('origin_x',0);py+=meta.get('origin_y',0)
    if method=='legacy':
        px=float(np.clip(px,0,meta['width']-1));py=float(np.clip(py,0,meta['height']-1))
    return [float(px),float(py)],peak
