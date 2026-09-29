"""Independent leakage checks for the next-training protocol; no model training."""
import hashlib
import json
from pathlib import Path
import pydicom
from .data import load_records, load_augmented_records
from .train import split_records
from .protocol import make_protocol


def main():
    root=Path(__file__).resolve().parents[2]
    path=root/'dxa_project/outputs/next_training/protocol.json'
    records=load_records(root)
    a,b=split_records(records,0)
    report=make_protocol(root,records,a,b,path)
    mapping=report['partition_by_path']
    study_parts={}
    pixel_parts={}
    for r in records:
        p=mapping[r.relative_path]
        assert study_parts.setdefault(r.study,p)==p
        pixels=pydicom.dcmread(r.source_path).pixel_array
        key=hashlib.sha256(str((pixels.shape,pixels.dtype)).encode()+pixels.tobytes()).hexdigest()
        assert pixel_parts.setdefault(key,p)==p
    augmented=load_augmented_records(root,root/'dxa_project/outputs/augmented_15000_20260929',records)
    counts={p:0 for p in ('train','validation','test')}
    for r in augmented:
        counts[study_parts[r.study]]+=1
    fractions=[report['summary'][p]['artifact_fraction'] for p in counts]
    assert max(fractions)-min(fractions)<.05
    assert sum(counts.values())==15000
    result=dict(original_images=len(records),studies=len(study_parts),unique_pixel_arrays=len(pixel_parts),
                study_overlap=0,cross_partition_pixel_duplicates=0,
                augmented_images_by_source_partition=counts,artifact_fraction_spread=max(fractions)-min(fractions))
    (path.parent/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
