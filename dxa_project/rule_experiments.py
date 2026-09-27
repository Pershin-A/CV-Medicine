"""Exploratory, study-grouped validation of rules on cached notebook features.

Run from the repository root. Thresholds are refit inside each held-out fold;
the CSV is diagnostic, not an independent final benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedGroupKFold

from prepare import read_reference


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "prototype_output" / "target_tuning"
OUT = Path(__file__).resolve().parent / "outputs" / "rule_experiments"


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f2 = float(fbeta_score(y, pred, beta=2, zero_division=0)) if tp + fn else None
    return {"n": len(y), "positives": int(y.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sensitivity": sensitivity, "specificity": specificity, "f2": f2,
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None}


def objective(y: np.ndarray, pred: np.ndarray) -> float:
    m = metrics(y, pred)
    if m["balanced_accuracy"] is None or m["f2"] is None:
        return -1.0
    return (m["balanced_accuracy"] + m["f2"]) / 2


def threshold_grid(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    return np.unique(np.quantile(values, np.linspace(.08, .92, 15)))


def candidate_predictions(frame: pd.DataFrame, spec: tuple, train_idx: np.ndarray):
    """Yield name, parameter description, and predictions for all frame rows."""
    kind, *columns = spec
    x = [pd.to_numeric(frame[c], errors="coerce").to_numpy(float) for c in columns]
    if kind in ("high", "low"):
        for t in threshold_grid(x[0][train_idx]):
            yield f"{kind}:{columns[0]}", {"threshold": float(t)}, x[0] > t if kind == "high" else x[0] < t
    elif kind == "outside":
        grid = threshold_grid(x[0][train_idx])[::2]
        for lo in grid:
            for hi in grid:
                if hi > lo:
                    yield f"outside:{columns[0]}", {"low": float(lo), "high": float(hi)}, (x[0] < lo) | (x[0] > hi)
    elif kind in ("or_high", "or_low"):
        a, b = (threshold_grid(v[train_idx])[::2] for v in x)
        for ta in a:
            for tb in b:
                pred = (x[0] > ta) | (x[1] > tb) if kind == "or_high" else (x[0] < ta) | (x[1] < tb)
                yield f"{kind}:{columns[0]}+{columns[1]}", {"first": float(ta), "second": float(tb)}, pred


SPECS = {
    "spine_coverage": [
        ("high", "t12_half_ratio"), ("high", "bright_threshold"),
        ("low", "separator_median_gap_px"), ("low", "iliac_left_image_area_fraction"),
        ("low", "separator_median_gap_ratio"),
    ],
    "spine_alignment": [
        ("high", "axis_deviation_vertical_deg_boundary"),
        ("high", "axis_deviation_vertical_deg_symmetry"),
        ("high", "axis_deviation_vertical_deg_brightness"),
    ],
    "spine_artifacts": [
        ("high", "artifact_border_bright_fraction"),
        ("low", "artifact_orientation_concentration"),
        ("or_high", "artifact_border_bright_fraction", "artifact_outside_bright_fraction"),
    ],
    "hip_position_rotation": [
        ("low", "rotation_prominence_ratio"),
        ("outside", "rotation_prominence_ratio"),
        ("high", "trochanter_adjacent_thickening_ratio"),
        ("low", "trochanter_min_width_px"),
        ("high", "trochanter_best_offset"),
        ("high", "trochanter_best_offset_ratio"),
        ("low", "rotation_bump_x"),
        ("high", "neck_adjacent_thickening_ratio"),
    ],
    "hip_roi": [
        ("low", "roi_bottom_margin_ratio"),
        ("low", "roi_top_margin_ratio"),
        ("low", "roi_lateral_margin_ratio"),
        ("or_low", "roi_bottom_margin_ratio", "roi_lateral_margin_ratio"),
        ("high", "ischial_gap_ratio"),
        ("low", "trochanter_min_width_px"),
    ],
}


def previous_rule(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Published full-data notebook choice, or the 5° clinical axis threshold."""
    v = lambda col: pd.to_numeric(frame[col], errors="coerce")
    if target == "spine_coverage":
        ok = (v("separator_count").between(5, 6) & v("t12_half_ratio").between(.2, .8)
              & v("iliac_left_image_area_fraction").between(.002, .15)
              & v("iliac_right_image_area_fraction").between(.002, .15)
              & v("iliac_left_image_top_y_norm").between(.42, .86)
              & v("iliac_right_image_top_y_norm").between(.42, .86)
              & v("iliac_pair_iou").le(.1))
        return (~ok).to_numpy()
    if target == "spine_alignment":
        return v("axis_deviation_vertical_deg_boundary").gt(5).to_numpy()
    if target == "spine_artifacts":
        return (v("artifact_extra_component_fraction").gt(.00014549266247379435)
                | v("artifact_outside_bright_fraction").gt(.00015303983228511495)
                | v("artifact_border_bright_fraction").gt(.0008444806173509377)
                | v("artifact_orientation_concentration").gt(.3324838458110516)).to_numpy()
    if target == "hip_position_rotation":
        ok = (v("four_boundary_longest_span_ratio").between(.06, .35)
              & v("ischial_width_ratio").between(.06, .24)
              & v("neck_adjacent_thickening_ratio").ge(1.3)
              & v("rotation_prominence_ratio").le(.25))
        return (~ok).to_numpy()
    if target == "hip_roi":
        ok = (v("roi_top_margin_ratio").ge(.06) & v("roi_bottom_margin_ratio").ge(.06)
              & v("roi_lateral_margin_ratio").ge(.04))
        return (~ok).to_numpy()
    raise KeyError(target)


def run_task(frame: pd.DataFrame, target: str) -> tuple[list[dict], list[dict]]:
    specs = SPECS[target]
    columns = list(dict.fromkeys(c for spec in specs for c in spec[1:]))
    frame = frame.loc[frame[target].isin([0, 1]) & frame[columns].notna().all(axis=1)].copy().reset_index(drop=True)
    y = frame[target].astype(int).to_numpy()
    groups = frame["selection_study"].astype(str).to_numpy()
    positive_groups = frame.loc[frame[target].eq(1), "selection_study"].nunique()
    if positive_groups < 2:
        return [], []
    folds = list(StratifiedGroupKFold(n_splits=min(5, positive_groups), shuffle=True, random_state=42)
                 .split(frame, y, groups))
    report, predictions = [], []
    baseline = previous_rule(frame, target).astype(int)
    baseline_name = "fixed_5_degrees" if target == "spine_alignment" else "notebook_full_data_params"
    report.append({"target": target, "rule": baseline_name,
                   "evaluation": "fixed_rule" if target == "spine_alignment" else "full_data_tuned_baseline",
                   **metrics(y, baseline),
                   "n_groups": int(frame.selection_study.nunique()), "positive_groups": int(positive_groups),
                   "fold_parameters": "fixed; no fold tuning"})
    for i in range(len(frame)):
        predictions.append({"target": target, "rule": baseline_name, "selection_study": groups[i],
                            "dicom_path": frame.loc[i, "dicom_path"], "truth": int(y[i]),
                            "prediction": int(baseline[i])})
    for spec in specs:
        pred_oof = np.full(len(frame), -1, dtype=int)
        parameters = []
        full_best = None
        for name, params, pred in candidate_predictions(frame, spec, np.arange(len(frame))):
            rank = (objective(y, pred.astype(int)), -int(pred.sum()))
            if full_best is None or rank > full_best[0]:
                full_best = (rank, params)
        for fold, (train_idx, test_idx) in enumerate(folds):
            best = None
            for name, params, pred in candidate_predictions(frame, spec, train_idx):
                score = objective(y[train_idx], pred[train_idx].astype(int))
                # Prefer fewer flags when scores tie.
                rank = (score, -int(pred[train_idx].sum()))
                if best is None or rank > best[0]:
                    best = (rank, name, params, pred)
            if best is None:
                continue
            pred_oof[test_idx] = best[3][test_idx].astype(int)
            parameters.append({"fold": fold, **best[2]})
        keep = pred_oof >= 0
        if not keep.any():
            continue
        result = {"target": target, "rule": best[1], "evaluation": "study_group_oof",
                  **metrics(y[keep], pred_oof[keep]),
                  "n_groups": int(frame.selection_study.nunique()), "positive_groups": int(positive_groups),
                  "full_fit_parameters": json.dumps(full_best[1], ensure_ascii=False),
                  "fold_parameters": json.dumps(parameters, ensure_ascii=False)}
        report.append(result)
        for i in np.flatnonzero(keep):
            predictions.append({"target": target, "rule": best[1], "selection_study": groups[i],
                                "dicom_path": frame.loc[i, "dicom_path"], "truth": int(y[i]),
                                "prediction": int(pred_oof[i])})
    return report, predictions


def main() -> None:
    all_metrics, all_predictions = [], []
    identity = pd.read_csv(CACHE / "dicom_identity.csv")[["source_row_id", "rows", "columns"]]
    reference = read_reference(ROOT / "разметка.xlsx").set_index("study_uid")
    for file, targets in (("spine_features.csv", list(SPECS)[:3]),
                          ("hip_features.csv", list(SPECS)[3:])):
        frame = pd.read_csv(CACHE / file).merge(identity, on="source_row_id", validate="many_to_one")
        if not frame.selection_study.isin(reference.index).all():
            raise ValueError(f"Cached {file} contains studies absent from the current Excel file")
        if file.startswith("spine"):
            for cached, current in (("spine_coverage", "spine_position"),
                                    ("spine_alignment", "spine_axis"),
                                    ("spine_artifacts", "spine_artifact")):
                expected = frame.selection_study.map(reference[current])
                if not frame[cached].fillna(-1).eq(expected.fillna(-1)).all():
                    raise ValueError(f"Cached {cached} labels differ from the current Excel file")
        else:
            for cached, suffix in (("hip_position_rotation", "hip_rotation"), ("hip_roi", "hip_roi")):
                expected = pd.Series(np.where(frame.side.eq("LEFT"),
                    frame.selection_study.map(reference[f"left_{suffix}"]),
                    frame.selection_study.map(reference[f"right_{suffix}"])), index=frame.index)
                if not frame[cached].fillna(-1).eq(expected.fillna(-1)).all():
                    raise ValueError(f"Cached {cached} labels differ from the current Excel file")
        if file.startswith("spine"):
            frame["separator_median_gap_ratio"] = frame.separator_median_gap_px / frame.rows
        else:
            frame["trochanter_best_offset_ratio"] = frame.trochanter_best_offset / np.minimum(frame["rows"], frame["columns"])
        for target in targets:
            report, predictions = run_task(frame, target)
            all_metrics.extend(report)
            all_predictions.extend(predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(all_metrics)
    result.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_predictions).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    print(result.drop(columns=["fold_parameters", "full_fit_parameters"]).to_string(index=False))


if __name__ == "__main__":
    main()
