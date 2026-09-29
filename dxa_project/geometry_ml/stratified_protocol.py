"""Artifact-stratified study split for future runs; never changes an existing run."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom

STRATEGY = 'artifact_stratified_study_groups'
PARTS = ('train', 'validation', 'test')


def make_stratified_protocol(records, path, seed=42, trials=6000):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite a saved split: {path}')
    parents = {r.study: r.study for r in records}

    def find(s):
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    # Duplicate images from different studies also belong to one indivisible group.
    hashes = {}
    for r in records:
        pixels = pydicom.dcmread(r.source_path).pixel_array
        digest = hashlib.sha256(str((pixels.shape, pixels.dtype)).encode() + pixels.tobytes()).hexdigest()
        if digest in hashes:
            parents[find(r.study)] = find(hashes[digest])
        hashes[digest] = r.study
    groups = defaultdict(list)
    for r in records:
        groups[find(r.study)].append(r)
    ids = sorted(groups)
    if len(ids) < 10:
        raise ValueError('At least ten independent study groups are required')
    positives = {r.relative_path: r.region == 'SPINE' and bool(
        json.loads(r.geometry_path.read_text(encoding='utf-8'))['spine']['foreign_objects']) for r in records}
    features = np.array([[len(rows), sum(r.region == 'SPINE' for r in rows),
                          sum(positives[r.relative_path] for r in rows),
                          sum(r.region == 'LEG_LEFT' for r in rows),
                          sum(r.region == 'LEG_RIGHT' for r in rows)] for rows in (groups[s] for s in ids)])
    strata = [np.flatnonzero((features[:, 2] > 0) == flag) for flag in (False, True)]
    if min(map(len, strata)) < 3:
        raise ValueError('Each artifact stratum needs at least three independent study groups')
    rng = np.random.default_rng(seed)
    fractions = np.array([.64, .16, .20])
    total = features.sum(0)
    best = None
    for _ in range(trials):
        assignment = np.zeros(len(ids), dtype=int)
        for stratum in strata:
            shuffled = rng.permutation(stratum)
            ntest = max(1, round(.20 * len(stratum)))
            nval = max(1, round(.16 * len(stratum)))
            assignment[shuffled[:ntest]] = 2
            assignment[shuffled[ntest:ntest+nval]] = 1
        actual = np.array([features[assignment == i].sum(0) for i in range(3)])
        expected = fractions[:, None] * total
        cost = np.square((actual - expected) / np.maximum(1, expected)).sum()
        # Balance both presence and absence among spine scans, not hips.
        negatives = actual[:, 1] - actual[:, 2]
        if np.any(actual[:, 2] == 0) or np.any(negatives == 0):
            continue
        if best is None or cost < best[0]:
            best = cost, assignment.copy()
    if best is None:
        raise ValueError('Cannot create all three partitions with both artifact classes')
    mapping = {r.relative_path: PARTS[best[1][i]] for i, s in enumerate(ids) for r in groups[s]}
    summary = {}
    for p in PARTS:
        rows = [r for r in records if mapping[r.relative_path] == p]
        spine = [r for r in rows if r.region == 'SPINE']
        pos = sum(positives[r.relative_path] for r in spine)
        summary[p] = dict(images=len(rows), studies=len({r.study for r in rows}),
                          spine_images=len(spine), artifact_positive=pos,
                          artifact_fraction=pos / len(spine),
                          artifact_positive_groups=sum(best[1][i] == PARTS.index(p) and features[i, 2] > 0 for i in range(len(ids))))
        summary[p]['artifact_positive_groups'] = int(summary[p]['artifact_positive_groups'])
    report = dict(seed=seed, split_strategy=STRATEGY, target_fractions=dict(zip(PARTS, fractions)),
                  outer_scope='new stratified internal holdout; not external validation',
                  partition_by_path=mapping, summary=summary, group_overlap=0,
                  cross_partition_exact_duplicates=0, independent_groups=len(ids),
                  grouping='reference_study_uid plus connected exact-pixel duplicates')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    from .data import load_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = make_stratified_protocol(load_records(args.root), args.output, args.seed)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
