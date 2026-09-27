"""Train an image-level ResNet18 anatomy router and masked quality heads.

Requires image-level region assignments in the labeler's labels.csv. The small
100-study reference alone cannot supply trustworthy image-level region labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from prepare import TARGETS

NAMES = list(TARGETS)
CLASSES = ['SPINE', 'LEG_LEFT', 'LEG_RIGHT']
CLASS_TARGETS = {0: [0, 1, 2], 1: [5, 6], 2: [3, 4]}


def image(ds):
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    a = np.asarray(apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds), dtype=np.float32)
    if a.ndim != 2:
        raise ValueError('Only single-frame 2D DXA images supported')
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        a = 1 - a
    return Image.fromarray((a * 255).astype(np.uint8)).convert('RGB')


class DXA(Dataset):
    def __init__(self, frame, transform):
        self.rows, self.transform = frame.to_dict('records'), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import pydicom
        row = self.rows[i]
        x = self.transform(image(pydicom.dcmread(row['path'])))
        region = CLASSES.index(row['region'])
        y = np.array([row[k] for k in NAMES], dtype=np.float32)
        mask = np.isfinite(y).astype(np.float32)
        allowed = np.zeros(len(y), dtype=np.float32)
        allowed[CLASS_TARGETS[region]] = 1
        return x, region, torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(mask * allowed)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        n = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.anatomy = nn.Linear(n, 3)
        self.quality = nn.Linear(n, len(NAMES))

    def forward(self, x):
        z = self.backbone(x)
        return self.anatomy(z), self.quality(z)


def load_rows(manifest, labels, dicoms_root):
    m = pd.read_csv(manifest, dtype={'study_uid': str, 'image_uid': str})
    l = pd.read_csv(labels, dtype=str, keep_default_na=False)
    if 'relative_path' not in l or 'label' not in l:
        raise ValueError('labels.csv needs relative_path and label')
    base = dicoms_root
    # labeler relative_path identifies the image relative to the DICOM directory;
    # matching by filename alone could silently assign the wrong side of a study.
    lookup = {str(Path(p).resolve()): i for i, p in enumerate(m.path)}
    regions = [None] * len(m)
    for rec in l.to_dict('records'):
        resolved = str((base / rec['relative_path']).resolve())
        if resolved not in lookup:
            continue
        label, side = rec['label'].upper(), rec.get('side', '').upper()
        region = 'SPINE' if label == 'SPINE' else f'LEG_{side}' if label == 'LEG' and side in ('LEFT', 'RIGHT') else None
        regions[lookup[resolved]] = region
    m['region'] = regions
    missing = int(m.region.isna().sum())
    print(f'Region labeled: {len(m) - missing}/{len(m)} images; unlabeled images excluded from supervised training')
    m = m[m.region.notna()].copy()
    if m.empty:
        raise ValueError('No image with a known region and side matched relative_path in labels.csv')
    for name in NAMES:
        m[name] = pd.to_numeric(m[name], errors='coerce')
    # Excel grades a study, not each of its images. If two images cover the
    # same region, no per-image quality target is identifiable from Excel.
    duplicated_region = m.groupby(['study_uid', 'region']).region.transform('size').gt(1)
    for region, names in [('SPINE', NAMES[:3]), ('LEG_LEFT', NAMES[5:]), ('LEG_RIGHT', NAMES[3:5])]:
        m.loc[duplicated_region & m.region.eq(region), names] = np.nan
    print(f'Ambiguous repeated study/region images: {int(duplicated_region.sum())}; corresponding quality labels masked')
    return m


def evaluate(model, loader, device):
    model.eval()
    truth, region, prob, pred_region, masks = [], [], [], [], []
    with torch.no_grad():
        for x, r, y, mask in loader:
            a, q = model(x.to(device))
            region.extend(r.numpy()); pred_region.extend(a.argmax(1).cpu().numpy())
            truth.extend(y.numpy()); prob.extend(q.sigmoid().cpu().numpy()); masks.extend(mask.numpy())
    truth, prob, masks = map(np.asarray, (truth, prob, masks))
    report = {'anatomy_balanced_accuracy': float(balanced_accuracy_score(region, pred_region)), 'targets': {}}
    for i, name in enumerate(NAMES):
        keep = masks[:, i].astype(bool)
        yi, pi = truth[keep, i].astype(int), prob[keep, i]
        if not len(yi):
            report['targets'][name] = {'n': 0, 'f1': None, 'balanced_accuracy': None, 'roc_auc': None}
            continue
        diverse = len(np.unique(yi)) == 2
        report['targets'][name] = {'n': int(len(yi)), 'positives': int(yi.sum()),
            'f1': float(f1_score(yi, pi >= .5, zero_division=0)) if diverse else None,
            'balanced_accuracy': float(balanced_accuracy_score(yi, pi >= .5)) if diverse else None,
            'roc_auc': float(roc_auc_score(yi, pi)) if diverse else None}
    return report


def main():
    from torchvision import transforms as T
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--dicoms-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/model'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    m = load_rows(args.manifest, args.labels, args.dicoms_root)
    if m.split.nunique() != 3 or m.study_uid.nunique() < 5:
        raise ValueError('At least five studies and labeled images in all three study-level splits required')
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    loaders = {s: DataLoader(DXA(m[m.split.eq(s)], tf), batch_size=args.batch_size, shuffle=s == 'train', num_workers=0)
               for s in ('train', 'val', 'test')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for x, r, y, mask in loaders['train']:
            a, q = model(x.to(device))
            ce = nn.functional.cross_entropy(a, r.to(device))
            bce = nn.functional.binary_cross_entropy_with_logits(q, y.to(device), reduction='none')
            mask = mask.to(device)
            loss = ce + (bce * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        val = evaluate(model, loaders['val'], device)
        valid_f1 = [v['f1'] for v in val['targets'].values() if v['f1'] is not None]
        score = val['anatomy_balanced_accuracy'] + (sum(valid_f1) / len(valid_f1) if valid_f1 else 0)
        if score > best:
            best = score
            torch.save({'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'target_names': NAMES,
                        'classes': CLASSES, 'seed': args.seed}, args.output / 'best.pt')
        print(f'epoch={epoch + 1} val_score={score:.3f}')
    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    report = {'validation': evaluate(model, loaders['val'], device), 'held_out_test': evaluate(model, loaders['test'], device)}
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
