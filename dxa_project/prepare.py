"""Join study labels to DICOM files and export a study-safe train/val/test manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGETS = {
    'spine_position': 2, 'spine_axis': 3, 'spine_artifact': 4,
    'right_hip_rotation': 5, 'right_hip_roi': 6,
    'left_hip_rotation': 7, 'left_hip_roi': 8,
}
REGIONS = {'SPINE': (2, 3, 4), 'LEG_RIGHT': (5, 6), 'LEG_LEFT': (7, 8)}


def read_reference(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=[0, 1], dtype={1: str})
    data = pd.DataFrame({'study_uid': raw.iloc[:, 1].astype('string').str.strip()})
    for name, index in TARGETS.items():
        data[name] = pd.to_numeric(raw.iloc[:, index], errors='coerce')
        invalid = data[name].notna() & ~data[name].isin((0, 1))
        if invalid.any():
            raise ValueError(f'Non-binary values in {name}: {invalid.sum()}')
    data = data[data.study_uid.notna() & data.study_uid.ne('')].copy()
    if data.study_uid.duplicated().any():
        raise ValueError('Duplicate study_uid in reference workbook')
    return data


def scan_dicoms(root: Path) -> pd.DataFrame:
    import pydicom

    rows = []
    validation_mode = pydicom.config.settings.reading_validation_mode
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.dcm'):
        try:
            # Supplied anonymized UIDs may exceed the DICOM UI length limit.
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            study = str(ds.get('StudyInstanceUID', '')).strip()
            uid = str(ds.get('SOPInstanceUID', '')).strip()
            if not study or not uid:
                continue
            rows.append({'path': str(path.resolve()), 'relative_path': path.relative_to(root).as_posix(),
                         'study_uid': study, 'image_uid': uid,
                         'patient_id': str(ds.get('PatientID', '')).strip(),
                         'rows': int(ds.get('Rows', 0)), 'columns': int(ds.get('Columns', 0))})
        except Exception:
            continue
    pydicom.config.settings.reading_validation_mode = validation_mode
    return pd.DataFrame(rows, columns=['path', 'relative_path', 'study_uid', 'image_uid', 'patient_id', 'rows', 'columns'])


def split_studies(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    unique = frame[['study_uid', 'patient_id']].drop_duplicates('study_uid').copy()
    # An anonymization placeholder shared by many studies cannot identify a
    # patient. Fall back to study-level grouping and disclose the limitation.
    placeholder = unique.patient_id.str.casefold().isin({'anonymized', 'anonymous', 'anon', 'unknown', 'none', 'na'})
    unique['group'] = np.where(unique.patient_id.ne('') & ~placeholder,
                               unique.patient_id, unique.study_uid)
    groups = unique.group.nunique()
    if groups < 5:
        raise ValueError('At least five distinct patient/study groups needed for a 60/20/20 split')
    first = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed)
    train, holdout = next(first.split(unique, groups=unique.group))
    unique['split'] = 'train'
    rest = unique.iloc[holdout]
    second = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed + 1)
    val, test = next(second.split(rest, groups=rest.group))
    unique.loc[rest.iloc[val].index, 'split'] = 'val'
    unique.loc[rest.iloc[test].index, 'split'] = 'test'
    return frame.merge(unique[['study_uid', 'split']], on='study_uid', validate='many_to_one')


def apply_image_overrides(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Preserve reviewed scan-level corrections when rebuilding study labels."""
    if not path.is_file():
        return frame
    overrides = pd.read_csv(path, dtype=str, keep_default_na=False)
    if overrides.duplicated(['relative_path', 'target']).any():
        raise ValueError('Duplicate per-image target override')
    frame = frame.copy()
    for row in overrides.to_dict('records'):
        if row['target'] not in TARGETS or row['value'] not in ('0', '1'):
            raise ValueError(f'Invalid target override: {row}')
        mask = frame.relative_path.eq(row['relative_path'].replace('\\', '/'))
        if int(mask.sum()) != 1:
            raise ValueError(f'Override source absent/duplicated: {row["relative_path"]}')
        frame.loc[mask, row['target']] = int(row['value'])
    return frame


def build_manifest(reference: Path, dicoms: Path, output: Path,
                   overrides: Path | None = None) -> None:
    labels = read_reference(reference)
    files = scan_dicoms(dicoms)
    if files.empty:
        raise ValueError('No DICOM with StudyInstanceUID/SOPInstanceUID found; no training possible')
    if files.image_uid.duplicated().any():
        raise ValueError('Duplicate SOPInstanceUID within Исследования: review before training')
    # The workbook's "study" matches the top-level source folder, whereas the
    # anonymized DICOM StudyInstanceUID is a different value in this dataset.
    files['reference_study_uid'] = files.relative_path.str.split('/').str[0]
    if files.groupby('reference_study_uid').study_uid.nunique().gt(1).any():
        raise ValueError('A reference folder contains multiple DICOM studies')
    labels = labels.rename(columns={'study_uid': 'reference_study_uid'})
    joined = files.merge(labels, on='reference_study_uid', how='left', indicator=True, validate='many_to_one')
    if not joined['_merge'].eq('both').any():
        raise ValueError('No Excel study identifier matches a DICOM source folder')
    joined['has_reference'] = joined['_merge'].eq('both')
    joined.drop(columns='_merge', inplace=True)
    joined = split_studies(joined)
    overrides = overrides or reference.parent / 'dxa_project' / 'reference_overrides.csv'
    joined = apply_image_overrides(joined, overrides)
    # The workbook is study-level; multiple images of the same anatomical area
    # must be reviewed before applying its target to any individual image.
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False, encoding='utf-8-sig')
    print(json.dumps({'dicoms': len(joined), 'studies': joined.study_uid.nunique(),
                      'with_excel_reference': int(joined.has_reference.sum()),
                      'split_studies': joined.groupby('split').study_uid.nunique().to_dict(),
                      'target_counts': {k: int(joined[k].notna().sum()) for k in TARGETS},
                      'split_grouping': 'patient_id where informative; otherwise study_uid',
                      'patient_leakage_caveat': 'Shared anonymized PatientID cannot prove patient-disjoint splits.',
                      'warning': 'Excel has study-level targets; link each image to a region using labeler labels.csv before image-level training.'},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dicoms', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/manifest.csv'))
    parser.add_argument('--overrides', type=Path,
                        help='Per-image reviewed target overrides (auto-detected by default)')
    args = parser.parse_args()
    build_manifest(args.reference, args.dicoms, args.output, args.overrides)


if __name__ == '__main__':
    main()
