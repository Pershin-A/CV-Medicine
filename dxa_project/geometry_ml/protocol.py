"""Fixed outer benchmark and group-balanced inner model-selection split."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json,hashlib
import numpy as np
from torch.utils.data import WeightedRandomSampler

def source_sampler(records,artifact=False):
    labels=[int(bool(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects'])) if artifact else 0 for r in records]
    keys=[(label,r.source_id or r.relative_path) for label,r in zip(labels,records)]
    counts=Counter(keys);classes=Counter(k[0] for k in counts)
    weights=[1/(classes[k[0]]*counts[k]) for k in keys]
    return WeightedRandomSampler(weights,len(weights),replacement=True)

def make_protocol(root,records,outer_train,outer_test,path,seed=42):
    # Existing outer fold is retained for an honest before/after comparison;
    # it has been inspected and is NOT claimed to be a pristine external test.
    if path.exists():
        report=json.loads(path.read_text(encoding='utf-8'))
        all_paths={r.relative_path for r in records}
        assert all_paths==set(report['partition_by_path'])
        parts=report['partition_by_path']
        assert set(parts.values())=={'train','validation','test'}
        groups={p:{r.study for r in records if parts[r.relative_path]==p} for p in ('train','validation','test')}
        if any(groups[a]&groups[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
            raise ValueError('Reused protocol no longer separates original study groups')
        if report.get('split_strategy') != 'artifact_stratified_study_groups' and {r.relative_path for r in outer_test}!={r.relative_path for r in records if parts[r.relative_path]=='test'}:
            raise ValueError('Reused protocol outer benchmark differs from the requested fold')
        return report
    with (root/'dxa_project/outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f:reference={r['relative_path']:r for r in csv.DictReader(f)}
    groups=defaultdict(list)
    for r in outer_train:groups[r.study].append(r)
    ids=sorted(groups);rng=np.random.default_rng(seed)
    def feature(rows):
        x=np.zeros(11)
        for r in rows:
            x[0]+=1;x[{'SPINE':1,'LEG_LEFT':2,'LEG_RIGHT':3}[r.region]]+=1
            if r.region=='SPINE':
                x[4]+=bool(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects'])
                for index,key in ((5,'spine_position'),(6,'spine_axis')):
                    if reference[r.relative_path].get(key,'') not in ('',None):x[index]+=float(reference[r.relative_path][key])
            else:
                prefix='left_' if r.region=='LEG_LEFT' else 'right_'
                for index,key in ((7 if r.region=='LEG_LEFT' else 9,'hip_roi'),(8 if r.region=='LEG_LEFT' else 10,'hip_rotation')):
                    if reference[r.relative_path].get(prefix+key,'') not in ('',None):x[index]+=float(reference[r.relative_path][prefix+key])
        return x
    features=np.array([feature(groups[s]) for s in ids]);total=features.sum(0)
    n=max(5,round(.2*len(ids)));fraction=n/len(ids);best=None
    for _ in range(4000):
        indices=rng.choice(len(ids),n,replace=False);f=features[indices].sum(0)
        cost=float((((f-fraction*total)/np.maximum(2,fraction*total))**2).sum())
        if f[4]==0 or total[4]-f[4]==0:cost+=1000
        if best is None or cost<best[0]:best=(cost,indices)
    inner={ids[i] for i in best[1]};test={r.study for r in outer_test}
    partitions={r.relative_path:'test' if r.study in test else 'validation' if r.study in inner else 'train' for r in records}
    def summary(part):
        rows=[r for r in records if partitions[r.relative_path]==part];s=[r for r in rows if r.region=='SPINE']
        pos=sum(bool(json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects']) for r in s)
        return {'images':len(rows),'studies':len({r.study for r in rows}),'spine_images':len(s),'artifact_positive':pos,'artifact_fraction':pos/len(s)}
    sets={p:{r.study for r in records if partitions[r.relative_path]==p} for p in ('train','validation','test')}
    assert not(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    # Equal pixels cannot cross model-selection boundaries either.
    import pydicom
    pixels={}
    for r in records:
        digest=hashlib.sha256(pydicom.dcmread(r.source_path).pixel_array.tobytes()).hexdigest()
        part=partitions[r.relative_path]
        if digest in pixels and pixels[digest]!=part:raise ValueError('Exact pixel duplicate crossed partitions')
        pixels[digest]=part
    report={'seed':seed,'outer_scope':'previously inspected fold 0 benchmark; not a new independent test',
            'partition_by_path':partitions,'summary':{p:summary(p) for p in sets},'group_overlap':0,'cross_partition_exact_duplicates':0}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
