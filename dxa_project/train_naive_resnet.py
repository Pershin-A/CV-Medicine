"""Frozen ImageNet ResNet18 + seven independent study-level linear classifiers.

Excel targets are study-level. Images of the same anatomy in one study are
mean-pooled before fitting a head. OOF predictions use stratified study folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from prepare import TARGETS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "naive_resnet18"
TASK_REGION = {
    "spine_position": "SPINE", "spine_axis": "SPINE", "spine_artifact": "SPINE",
    "right_hip_rotation": "LEG_RIGHT", "right_hip_roi": "LEG_RIGHT",
    "left_hip_rotation": "LEG_LEFT", "left_hip_roi": "LEG_LEFT",
}
SEED = 42


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_table(manifest: Path, labels: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest, dtype={"study_uid": str, "reference_study_uid": str})
    l = pd.read_csv(labels, dtype=str, keep_default_na=False)
    if m.relative_path.duplicated().any() or l.relative_path.duplicated().any():
        raise ValueError("Duplicate relative_path in manifest or labels")
    m = m.merge(l[["relative_path", "label", "side"]], on="relative_path", how="inner",
                validate="one_to_one")
    m["region"] = np.where(m.label.eq("SPINE"), "SPINE",
                    np.where(m.label.eq("LEG") & m.side.isin(["LEFT", "RIGHT"]),
                             "LEG_" + m.side, "UNKNOWN"))
    m = m[m.region.ne("UNKNOWN") & m.has_reference].copy()
    if m.empty:
        raise ValueError("No labeled DICOM with Excel reference")
    return m


class Dicoms(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        ds = pydicom.dcmread(self.paths[index], force=True)
        pixels = np.squeeze(np.asarray(ds.pixel_array))
        if pixels.ndim != 2:
            raise ValueError(f"Expected 2D DICOM: {self.paths[index]}")
        image = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float32)
        try:
            image = np.asarray(apply_voi_lut(image, ds), dtype=np.float32)
        except Exception:
            pass
        finite = np.isfinite(image)
        if not finite.any():
            raise ValueError(f"No finite pixel values: {self.paths[index]}")
        lo, hi = np.percentile(image[finite], [.5, 99.5])
        if hi <= lo:
            lo, hi = float(image[finite].min()), float(image[finite].max())
        image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            image = 1 - image
        pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        return self.transform(pil)


def extract_features(paths: list[str], out: Path, signature: dict, batch_size: int) -> np.ndarray:
    cache = out / "image_features.npz"
    meta = out / "feature_cache.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == signature:
        data = np.load(cache)
        features = data["features"]
        if len(features) == len(paths) and features.shape[1] == 512:
            print(f"Reusing {len(features)} cached ResNet18 embeddings")
            return features
    torch.hub.set_dir(str((out.parent / "torch_hub").resolve()))
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    loader = DataLoader(Dicoms(paths), batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            batches.append(model(batch).numpy())
            if step % 5 == 0 or step == len(loader):
                print(f"Embedded {min(step * batch_size, len(paths))}/{len(paths)} DICOM")
    features = np.concatenate(batches).astype(np.float32)
    np.savez_compressed(cache, features=features)
    meta.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return features


def study_bags(images: pd.DataFrame, features: np.ndarray, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    part = images[images.region.eq(TASK_REGION[task])].copy()
    part[task] = pd.to_numeric(part[task], errors="coerce")
    part = part[part[task].isin([0, 1])]
    rows, vectors = [], []
    for study, group in part.groupby("reference_study_uid", sort=True):
        labels = group[task].unique()
        if len(labels) != 1:
            raise ValueError(f"Conflicting {task} labels within study {study}")
        rows.append({"reference_study_uid": study, "region": TASK_REGION[task],
                     "target": int(labels[0]), "n_images": len(group)})
        vectors.append(features[group.index.to_numpy()].mean(axis=0))
    return pd.DataFrame(rows), np.stack(vectors) if vectors else np.empty((0, 512))


def new_head():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED))


def evaluate(task: str, rows: pd.DataFrame, x: np.ndarray, out: Path) -> tuple[dict, pd.DataFrame]:
    y = rows.target.to_numpy(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if min(positives, negatives) < 2:
        raise ValueError(f"{task}: too few studies in one class ({positives}/{len(y)})")
    folds = StratifiedKFold(n_splits=min(5, positives, negatives), shuffle=True, random_state=SEED)
    prob = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for fold, (train, test) in enumerate(folds.split(x, y)):
        model = new_head().fit(x[train], y[train])
        prob[test] = model.predict_proba(x[test])[:, 1]
        fold_id[test] = fold
    if not np.isfinite(prob).all():
        raise RuntimeError(f"Incomplete OOF predictions for {task}")
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = (int(z) for z in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    recall = tp / (tp + fn)
    ci = binomtest(tp, tp + fn).proportion_ci(method="exact")
    report = {"task": task, "region": TASK_REGION[task], "n_studies": len(y),
              "n_images": int(rows.n_images.sum()), "positives": positives,
              "folds": folds.n_splits, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
              "sensitivity": recall, "sensitivity_ci95_low": float(ci.low),
              "sensitivity_ci95_high": float(ci.high),
              "specificity": tn / (tn + fp), "balanced_accuracy": .5 * (recall + tn / (tn + fp)),
              "f1": float(f1_score(y, pred, zero_division=0)),
              "roc_auc": float(roc_auc_score(y, prob)),
              "average_precision": float(average_precision_score(y, prob)),
              "prevalence": positives / len(y)}
    fitted = new_head().fit(x, y)
    joblib.dump(fitted, out / f"{task}.joblib")
    result = rows.copy()
    result.insert(0, "task", task)
    result["oof_probability"] = prob
    result["oof_prediction_05"] = pred
    result["fold"] = fold_id
    return report, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "dxa_project" / "outputs" / "manifest.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "dxa_project" / "outputs" / "merged_labels.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = image_table(args.manifest, args.labels).reset_index(drop=True)
    image_stats = [(path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                   for path in images.path]
    images_fingerprint = hashlib.sha256(json.dumps(image_stats, ensure_ascii=False).encode("utf-8")).hexdigest()
    # The cache is invalidated whenever manifest, anatomy labels, or image paths change.
    signature = {"manifest_sha256": file_hash(args.manifest), "labels_sha256": file_hash(args.labels),
                 "images_fingerprint": images_fingerprint, "image_count": len(images),
                 "weights": "ResNet18_Weights.IMAGENET1K_V1",
                 "preprocessing": "modality/VOI LUT; percentile 0.5/99.5; 224x224 ImageNet RGB"}
    features = extract_features(images.path.tolist(), args.output, signature, args.batch_size)
    reports, predictions = [], []
    for task in TARGETS:
        rows, x = study_bags(images, features, task)
        report, pred = evaluate(task, rows, x, args.output)
        reports.append(report)
        predictions.append(pred)
        print(task, f"{report['tp']}/{report['positives']} TP",
              f"balanced_accuracy={report['balanced_accuracy']:.3f}",
              f"average_precision={report['average_precision']:.3f}")
    pd.DataFrame(reports).to_csv(args.output / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "study_oof_predictions.csv", index=False,
                                                      encoding="utf-8-sig")
    (args.output / "run.json").write_text(json.dumps({
        "encoder": "frozen ImageNet ResNet18 (512 features)", "head": "L2 logistic regression, C=0.1",
        "pooling": "mean embedding across images of the same study and region",
        "evaluation": "stratified study-level out-of-fold, threshold=0.5",
        "seed": SEED, "feature_signature": signature,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
