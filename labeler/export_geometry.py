"""Convert the original labeler registry + geometry sidecars to ML-ready JSONL/CSV.

Usage: python export_geometry.py --output-root /output
All output coordinates refer to original DICOM pixel space, never preview size.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from geometry import read_geometry, geometry_counts


def export(output_root: Path, jsonl_path: Path, csv_path: Path) -> int:
    labels = Path(output_root) / 'labels.csv'
    if not labels.exists():
        raise FileNotFoundError(labels)
    registry = pd.read_csv(labels, dtype=str, keep_default_na=False)
    rows = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open('w', encoding='utf-8') as stream:
        for row in registry.to_dict('records'):
            geo_rel = row.get('geometry_path', '')
            if not geo_rel:
                continue
            # Guard against malicious CSV entries escaping the output tree.
            path = (Path(output_root) / geo_rel).resolve()
            if not path.is_relative_to(Path(output_root).resolve()):
                raise ValueError(f'Invalid geometry path for {row.get("relative_path")}')
            if not path.is_file():
                raise FileNotFoundError(path)
            raw = json.loads(path.read_text(encoding='utf-8'))
            geo = read_geometry(path, raw['image_width'], raw['image_height'])
            record = {
                'relative_path': row.get('relative_path', ''),
                'label': row.get('label', ''),
                'side': row.get('side', ''),
                'metal': row.get('metal', ''),
                'fracture': row.get('fracture', ''),
                'spine_issue': row.get('spine_issue', ''),
                'annotator': row.get('annotator', ''),
                'annotated_at': row.get('annotated_at', ''),
                'geometry': geo,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts = geometry_counts(geo)
            rows.append({**{k:v for k,v in record.items() if k!='geometry'},
                         'image_width':geo['image_width'], 'image_height':geo['image_height'],
                         **counts, 'spine_complete':geo['complete']['spine'],
                         'hip_complete':geo['complete']['hip'],
                         'geometry_path':geo_rel})
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('/output'))
    parser.add_argument('--jsonl', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None)
    args = parser.parse_args()
    output = args.output_root
    count = export(output, args.jsonl or output/'annotations_geometry.jsonl',
                   args.csv or output/'annotations_geometry_summary.csv')
    print(f'Exported {count} annotated images')
