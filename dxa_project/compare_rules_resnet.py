"""Paired study-level comparison of heuristic rules and frozen ResNet18 heads.

Both approaches use the same studies and outer CV folds. Rule thresholds are
selected on each training fold only. Previously selected rule *families* make
this an exploratory comparison, not an untouched final test.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rule_experiments import candidate_predictions, objective


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs" / "paired_comparison"
NAIVE = PROJECT / "outputs" / "naive_resnet18"
CACHE = ROOT / "prototype_output" / "target_tuning"
TASK_REGION = {
    "spine_position": "SPINE", "spine_axis": "SPINE", "spine_artifact": "SPINE",
    "right_hip_rotation": "LEG_RIGHT", "right_hip_roi": "LEG_RIGHT",
    "left_hip_rotation": "LEG_LEFT", "left_hip_roi": "LEG_LEFT",
}
RULE_SPEC = {
    "spine_position": ("low", "separator_median_gap_px"),
    "spine_axis": ("fixed_5_degrees", "axis_deviation_vertical_deg_boundary"),
    "spine_artifact": ("artifact_or",),
    "right_hip_rotation": ("high", "trochanter_best_offset_ratio"),
    "left_hip_rotation": ("high", "trochanter_best_offset_ratio"),
    "right_hip_roi": ("or_low", "roi_bottom_margin_ratio", "roi_lateral_margin_ratio"),
    "left_hip_roi": ("or_low", "roi_bottom_margin_ratio", "roi_lateral_margin_ratio"),
}
HIP_TARGET = {"right_hip_rotation": "hip_position_rotation", "left_hip_rotation": "hip_position_rotation",
              "right_hip_roi": "hip_roi", "left_hip_roi": "hip_roi"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bags() -> tuple[dict, pd.DataFrame]:
    manifest = PROJECT / "outputs" / "manifest.csv"
    labels = PROJECT / "outputs" / "merged_labels.csv"
    signature = json.loads((NAIVE / "feature_cache.json").read_text(encoding="utf-8"))
    if signature["manifest_sha256"] != sha256(manifest) or signature["labels_sha256"] != sha256(labels):
        raise ValueError("ResNet18 feature cache does not match the current manifest and labels")
    features = np.load(NAIVE / "image_features.npz")["features"]
    m = pd.read_csv(manifest, dtype={"reference_study_uid": str})
    l = pd.read_csv(labels, dtype=str, keep_default_na=False)
    m = m.merge(l[["relative_path", "label", "side"]], on="relative_path", how="inner", validate="one_to_one")
    m["region"] = np.where(m.label.eq("SPINE"), "SPINE",
                    np.where(m.label.eq("LEG") & m.side.isin(["LEFT", "RIGHT"]), "LEG_" + m.side, "UNKNOWN"))
    m = m[m.region.ne("UNKNOWN") & m.has_reference].copy().reset_index(drop=True)
    if len(m) != len(features):
        raise ValueError("Image count differs from cached ResNet18 features")
    bags = {}
    for (study, region), group in m.groupby(["reference_study_uid", "region"]):
        bags[(study, region)] = features[group.index.to_numpy()].mean(axis=0)
    return bags, m


def load_rule_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(CACHE / "spine_features.csv")
    h = pd.read_csv(CACHE / "hip_features.csv")
    identity = pd.read_csv(CACHE / "dicom_identity.csv")[["source_row_id", "rows", "columns"]]
    h = h.merge(identity, on="source_row_id", validate="many_to_one")
    h["trochanter_best_offset_ratio"] = h.trochanter_best_offset / np.minimum(h["rows"], h["columns"])
    s["region"] = "SPINE"
    h["region"] = "LEG_" + h.side
    return s, h


def artifact_candidates(frame: pd.DataFrame, train: np.ndarray):
    columns = ["artifact_extra_component_fraction", "artifact_outside_bright_fraction",
               "artifact_border_bright_fraction", "artifact_orientation_concentration"]
    values = [frame[c].to_numpy(float) for c in columns]
    quantiles = ((.55, .75, .90, .96), (.55, .75, .90, .96), (.65, .85, .95), (.65, .85, .95))
    grids = [np.unique(np.quantile(v[train], q)) for v, q in zip(values, quantiles)]
    for thresholds in itertools.product(*grids):
        pred = np.logical_or.reduce([v > t for v, t in zip(values, thresholds)])
        yield {k: float(v) for k, v in zip(columns, thresholds)}, pred


def train_rule(frame: pd.DataFrame, task: str, train: np.ndarray) -> tuple[np.ndarray, dict]:
    spec = RULE_SPEC[task]
    if spec[0] == "fixed_5_degrees":
        return frame[spec[1]].to_numpy(float) > 5, {"angle_degrees": 5}
    if spec[0] == "artifact_or":
        candidates = artifact_candidates(frame, train)
    else:
        candidates = ((params, pred) for _, params, pred in candidate_predictions(frame, spec, train))
    best = None
    y = frame.target.to_numpy(int)
    for params, pred in candidates:
        rank = (objective(y[train], pred[train].astype(int)), -int(pred[train].sum()))
        if best is None or rank > best[0]:
            best = (rank, pred, params)
    if best is None:
        raise ValueError(f"No rule candidates for {task}")
    return best[1], best[2]


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(z) for z in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {"n": len(y), "positives": int(y.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "f1": float(f1_score(y, pred, zero_division=0))}


def paired_delta_interval(y: np.ndarray, rules: np.ndarray, resnet: np.ndarray,
                          seed: int = 42, repeats: int = 3000) -> tuple[float, float]:
    """Stratified paired bootstrap interval for ResNet minus rule BA."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(repeats)
    for i in range(repeats):
        p = rng.choice(positive, len(positive), replace=True)
        n = rng.choice(negative, len(negative), replace=True)
        deltas[i] = .5 * ((resnet[p] == 1).mean() - (rules[p] == 1).mean()
                           + (resnet[n] == 0).mean() - (rules[n] == 0).mean())
    return tuple(float(v) for v in np.quantile(deltas, [.025, .975]))


def compare_task(task: str, bags: dict, spine: pd.DataFrame, hip: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, dict]:
    region = TASK_REGION[task]
    source = spine if region == "SPINE" else hip.loc[hip.region.eq(region)]
    cached_target = {"spine_position": "spine_coverage", "spine_axis": "spine_alignment",
                     "spine_artifact": "spine_artifacts"}.get(task, HIP_TARGET.get(task))
    frame = source[source[cached_target].isin([0, 1])].copy()
    frame = frame.rename(columns={"selection_study": "reference_study_uid", cached_target: "target"})
    frame = frame[frame.apply(lambda r: (r.reference_study_uid, region) in bags, axis=1)]
    required = RULE_SPEC[task][1:] if RULE_SPEC[task][0] not in ("artifact_or",) else [
        "artifact_extra_component_fraction", "artifact_outside_bright_fraction",
        "artifact_border_bright_fraction", "artifact_orientation_concentration"]
    frame = frame[frame[list(required)].notna().all(axis=1)].sort_values("reference_study_uid").reset_index(drop=True)
    if frame.reference_study_uid.duplicated().any():
        raise ValueError(f"Multiple rule rows for one study and task: {task}")
    manifest = pd.read_csv(PROJECT / "outputs" / "manifest.csv", dtype={"reference_study_uid": str})
    expected = manifest.groupby("reference_study_uid")[task].first()
    if not frame.target.eq(frame.reference_study_uid.map(expected)).all():
        raise ValueError(f"Cached labels do not match current Excel manifest: {task}")
    x = np.stack([bags[(study, region)] for study in frame.reference_study_uid])
    y = frame.target.to_numpy(int)
    folds = StratifiedKFold(n_splits=min(5, int(y.sum()), int((1 - y).sum())),
                            shuffle=True, random_state=42)
    rule_pred = np.full(len(y), -1, dtype=int)
    cnn_prob = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_parameters = []
    for fold, (train, test) in enumerate(folds.split(x, y)):
        pred, params = train_rule(frame, task, train)
        rule_pred[test] = pred[test].astype(int)
        head = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=42))
        head.fit(x[train], y[train])
        cnn_prob[test] = head.predict_proba(x[test])[:, 1]
        fold_id[test] = fold
        fold_parameters.append({"fold": fold, **params})
    if (rule_pred < 0).any() or not np.isfinite(cnn_prob).all():
        raise RuntimeError(f"Incomplete OOF comparison for {task}")
    cnn_pred = (cnn_prob >= .5).astype(int)
    delta_ci = paired_delta_interval(y, rule_pred, cnn_pred)
    common = {"task": task, "n_studies": len(y), "positives": int(y.sum()),
              "folds": folds.n_splits}
    results = [{**common, "method": "rules", **scores(y, rule_pred), "roc_auc": None,
                "average_precision": None},
               {**common, "method": "resnet18_head", **scores(y, cnn_pred),
                "delta_ba_vs_rules": scores(y, cnn_pred)["balanced_accuracy"] - scores(y, rule_pred)["balanced_accuracy"],
                "delta_ba_ci95_low": delta_ci[0], "delta_ba_ci95_high": delta_ci[1],
                "roc_auc": float(roc_auc_score(y, cnn_prob)),
                "average_precision": float(average_precision_score(y, cnn_prob))}]
    predictions = frame[["reference_study_uid", "region", "target"]].copy()
    predictions.insert(0, "task", task)
    predictions["fold"] = fold_id
    predictions["rule_prediction"] = rule_pred
    predictions["resnet18_probability"] = cnn_prob
    predictions["resnet18_prediction"] = cnn_pred
    return results, predictions, {"task": task, "rule_spec": RULE_SPEC[task], "folds": fold_parameters}


def main() -> None:
    bags, _ = load_bags()
    spine, hip = load_rule_features()
    reports, predictions, parameters = [], [], []
    for task in TASK_REGION:
        rows, pred, params = compare_task(task, bags, spine, hip)
        reports.extend(rows)
        predictions.append(pred)
        parameters.append(params)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "rule_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
