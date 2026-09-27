"""Summarize target balance and completeness of the labeler geometry export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare import TARGETS, read_reference


def audit(reference: Path, output: Path, geometry: Path | None = None) -> dict:
    data = read_reference(reference)
    summary = {'studies': len(data), 'targets': {}}
    for name in TARGETS:
        values = data[name]
        summary['targets'][name] = {'normal_0': int(values.eq(0).sum()),
                                    'violation_1': int(values.eq(1).sum()),
                                    'missing': int(values.isna().sum())}
    summary['geometry'] = {'exported_images': 0, 'spine_complete': 0, 'hip_complete': 0}
    if geometry and geometry.exists():
        records = [json.loads(line) for line in geometry.read_text(encoding='utf-8').splitlines() if line.strip()]
        summary['geometry']['exported_images'] = len(records)
        summary['geometry']['spine_complete'] = sum(bool(r['geometry']['complete']['spine']) for r in records)
        summary['geometry']['hip_complete'] = sum(bool(r['geometry']['complete']['hip']) for r in records)
        summary['geometry']['disc_lines'] = sum(len(r['geometry']['spine']['disc_lines']) for r in records)
        summary['geometry']['hip_landmarks'] = sum(sum(p is not None for p in r['geometry']['hip']['landmarks'].values()) for r in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/audit.json'))
    args = parser.parse_args()
    print(json.dumps(audit(args.reference, args.output, args.geometry), ensure_ascii=False, indent=2))
