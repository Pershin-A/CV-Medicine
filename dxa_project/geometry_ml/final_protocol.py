"""Multilabel balancing of indivisible study / identical-pixel groups."""
import csv, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pydicom
from .evaluate import truth_flags

PARTS=('train','validation','test')


def labels_for_originals(root, records):
    def table(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return {r['relative_path'].replace('\\','/'):r for r in csv.DictReader(f)}
    local=table(root/'Размеченные/labels.csv');author=table(root/'dxa_project/outputs/manifest.csv')
    result={}
    for r in records:
        g=json.loads(r.geometry_path.read_text(encoding='utf-8'))
        row=truth_flags(author[r.relative_path],r.region,g)
        if r.region=='SPINE':
            issue=local[r.relative_path].get('spine_issue','')
            row['spine_scoliosis']=int(issue=='SCOLIOSIS') if issue in ('SCOLIOSIS','NONE','LUMBARIZATION') else None
            row['spine_artifact']=int(bool(g['spine']['foreign_objects']))
            row['spine_lumbarization']=int(issue=='LUMBARIZATION') if issue else None
        else:
            for k in ('metal','fracture'):
                v=local[r.relative_path].get(k,'');row['hip_'+k]=int(v) if v in ('0','1') else None
        result[r.relative_path]=row
    return result


def assign_groups(features, fractions=(.64,.16,.20), seed=20260929, trials=15000):
    """Balance both classes in each feature, penalizing preventable empty cells."""
    x=np.asarray(features,float);f=np.asarray(fractions);rng=np.random.default_rng(seed)
    total=x.sum(0);expected=f[:,None]*total;support=(x>0).sum(0)
    def score(a):
        actual=np.array([x[a==i].sum(0) for i in range(3)])
        cost=float(np.square((actual-expected)/np.maximum(expected,2)).mean())
        cost+=10*int(np.sum((actual==0)&(support[None,:]>=3)))
        return cost
    n=len(x);nval=max(1,round(n*f[1]));ntest=max(1,round(n*f[2]));best=None
    for _ in range(trials):
        order=rng.permutation(n);a=np.zeros(n,int);a[order[:ntest]]=2;a[order[ntest:ntest+nval]]=1
        cost=score(a)
        if best is None or cost<best[0]:best=(cost,a.copy())
    a=best[1]
    for _ in range(4000):
        i,j=rng.choice(n,2,replace=False)
        if a[i]==a[j]:continue
        b=a.copy();b[i],b[j]=a[j],a[i];cost=score(b)
        if cost<best[0]:best=(cost,b);a=b
    return best[1],best[0]


def make_final_protocol(root, records, destination):
    if destination.exists():return json.loads(destination.read_text(encoding='utf-8'))
    labels=labels_for_originals(root,records);parents={r.study:r.study for r in records};digests={};conflicts=[]
    def find(s):
        while parents[s]!=s:parents[s]=parents[parents[s]];s=parents[s]
        return s
    for r in records:
        a=pydicom.dcmread(r.source_path).pixel_array
        h=hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()
        if h in digests:
            old=digests[h];parents[find(r.study)]=find(old.study)
            if labels[r.relative_path].get('spine_scoliosis') != labels[old.relative_path].get('spine_scoliosis'):
                conflicts.append([old.relative_path,r.relative_path])
        else:digests[h]=r
    for pair in conflicts:
        for p in pair:labels[p]['spine_scoliosis']=None
    groups=defaultdict(list)
    for r in records:groups[find(r.study)].append(r)
    ids=sorted(groups);tasks=sorted({k for row in labels.values() for k in row})
    columns=['images']+['region:'+k for k in ('SPINE','LEG_LEFT','LEG_RIGHT')]+[f'{k}:{v}' for k in tasks for v in (0,1)]
    features=[]
    for gid in ids:
        rs=groups[gid]
        features.append([len(rs)]+[sum(r.region==k for r in rs) for k in ('SPINE','LEG_LEFT','LEG_RIGHT')]+[sum(labels[r.relative_path].get(k)==v for r in rs) for k in tasks for v in (0,1)])
    a,cost=assign_groups(features);parts={r.relative_path:PARTS[int(a[i])] for i,gid in enumerate(ids) for r in groups[gid]}
    summary={}
    for part in PARTS:
        rs=[r for r in records if parts[r.relative_path]==part]
        summary[part]={'images':len(rs),'studies':len({r.study for r in rs}),'groups':sum(a==PARTS.index(part)).item(),
                       'regions':{k:sum(r.region==k for r in rs) for k in ('SPINE','LEG_LEFT','LEG_RIGHT')},
                       'labels':{k:{'negative':sum(labels[r.relative_path].get(k)==0 for r in rs),'positive':sum(labels[r.relative_path].get(k)==1 for r in rs),'unknown':sum(k in labels[r.relative_path] and labels[r.relative_path][k] is None for r in rs)} for k in tasks}}
    support={columns[i]:int(sum(np.asarray(features)[:,i]>0)) for i in range(len(columns))}
    report={'split_strategy':'all_violations_stratified_study_pixel_groups','seed':20260929,'target_fractions':dict(zip(PARTS,(.64,.16,.20))),
            'partition_by_path':parts,'summary':summary,'independent_groups':len(ids),'group_overlap':0,'cross_partition_exact_duplicates':0,
            'feature_group_support':support,'unstratifiable_rare_classes':[k for k,v in support.items() if v<3],
            'scoliosis_pixel_label_conflicts':conflicts,'original_labels':labels,'balance_cost':cost,
            'scope':'Internal holdout; all prior project originals have been inspected. Fresh ImageNet initialization avoids prior trained-weight leakage.'}
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
