"""Merge independent labeler registries by their exact DICOM relative path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import pandas as pd


FIELDS = ('label', 'side', 'metal', 'fracture', 'spine_issue')


def merge(root: Path, output: Path) -> dict:
    sources = [root / name for name in ('Размеченные', 'Размеченные_25_09', 'Размеченные_my')]
    rows = []
    for folder in sources:
        csv = folder / 'labels.csv'
        if not csv.is_file():
            continue
        data = pd.read_csv(csv, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        if not {'relative_path', 'label', 'side'}.issubset(data.columns):
            raise ValueError(f'Missing columns in {csv}')
        for row in data.to_dict('records'):
            rel = str(row['relative_path']).replace('\\', '/').strip().lstrip('./')
            parts = PurePosixPath(rel).parts
            if not rel or '..' in parts or PurePosixPath(rel).is_absolute():
                raise ValueError(f'Invalid relative_path in {csv}: {rel!r}')
            image = root / 'Исследования' / Path(*parts)
            if not image.is_file():
                # A label can be valid for a removed/unavailable image, but it
                # cannot be used to train on the current data tree.
                continue
            row['relative_path'] = rel
            row['source'] = folder.name
            rows.append(row)
    if not rows:
        raise ValueError('No matching images in the three labels.csv registries')
    frame = pd.DataFrame(rows).fillna('')
    for field in FIELDS:
        if field not in frame:
            frame[field] = ''
        frame[field] = frame[field].astype(str).str.strip().str.upper()
    accepted, conflicts = [], []
    for rel, group in frame.groupby('relative_path', sort=True):
        # Empty values represent no decision, not a contradictory vote.
        disputed = {field: sorted(set(v for v in group[field] if v))
                    for field in FIELDS if len(set(v for v in group[field] if v)) > 1}
        if disputed:
            conflicts.append({'relative_path': rel, 'sources': group.source.tolist(), 'disputed': disputed})
            continue
        record = {'relative_path': rel, 'sources': '|'.join(group.source)}
        record.update({field: next((v for v in group[field] if v), '') for field in FIELDS})
        accepted.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted, columns=['relative_path', 'sources', *FIELDS]).to_csv(output, index=False, encoding='utf-8-sig')
    conflict_path = output.with_name('label_conflicts.json')
    conflict_path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {'registries': [p.name for p in sources if (p / 'labels.csv').is_file()],
              'matching_rows': len(rows), 'unique_images': frame.relative_path.nunique(),
              'accepted_images': len(accepted), 'conflicts': len(conflicts),
              'conflict_report': str(conflict_path)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, default=Path('dxa_project/outputs/merged_labels.csv'))
    a = p.parse_args()
    merge(a.root.resolve(), (a.root / a.output).resolve())
