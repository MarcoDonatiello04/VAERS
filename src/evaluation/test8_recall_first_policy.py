#!/usr/bin/env python3
"""
TEST8 - Recall-first threshold policy with precision floor.

Goal:
- Increase recall versus the previous production threshold.
- Keep precision above a hard safety floor (>= 0.50 on test).

Method:
1) Load existing tuned LightGBM artifact from Test4.
2) Select threshold on validation by maximizing recall with a validation precision floor.
3) Evaluate on test and save summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET = "IS_SEVERE"
THRESH_GRID = np.round(np.arange(0.50, 0.91, 0.01), 2)
PRECISION_FLOOR_VAL = 0.60
PRECISION_FLOOR_TEST = 0.50


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    root = Path("/Users/marcodonatiello/PycharmProjects/JupyterProject")
    model_path = root / "models" / "production" / "severe_model_test4_best_tactic.pkl"
    val_path = root / "data" / "interim" / "splits" / "val_step6.csv"
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"
    reports_dir = root / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    old_threshold = float(artifact["threshold"])
    features = artifact["features_order"]

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    x_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET].astype(int).to_numpy()
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int).to_numpy()

    for col in features:
        if col not in x_val.columns:
            x_val[col] = 0
            x_test[col] = 0

    x_val = x_val[features]
    x_test = x_test[features]

    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    # Baseline at old threshold.
    baseline_val = compute_metrics(y_val, p_val, old_threshold)
    baseline_test = compute_metrics(y_test, p_test, old_threshold)

    # Recall-first selection on validation under precision floor.
    candidates = []
    for t in THRESH_GRID:
        m_val = compute_metrics(y_val, p_val, float(t))
        m_test = compute_metrics(y_test, p_test, float(t))
        if m_val["precision"] >= PRECISION_FLOOR_VAL:
            candidates.append((m_val, m_test))

    if not candidates:
        raise RuntimeError(
            f"No threshold satisfies validation precision floor >= {PRECISION_FLOOR_VAL:.2f}"
        )

    # Max recall on validation; tie-break on validation precision.
    best_val, best_test = max(
        candidates,
        key=lambda pair: (pair[0]["recall"], pair[0]["precision"]),
    )

    # Hard requirement requested by user on test precision.
    if best_test["precision"] < PRECISION_FLOOR_TEST:
        raise RuntimeError(
            "Selected threshold violates test precision floor. "
            "Increase PRECISION_FLOOR_VAL or tighten threshold grid."
        )

    delta = {
        "precision": best_test["precision"] - baseline_test["precision"],
        "recall": best_test["recall"] - baseline_test["recall"],
        "f1": best_test["f1"] - baseline_test["f1"],
        "fn_reduction": baseline_test["fn"] - best_test["fn"],
        "tp_gain": best_test["tp"] - baseline_test["tp"],
    }

    summary = {
        "policy": "recall_first_with_precision_guardrail",
        "selection_rule_val": f"max_recall_with_precision>={PRECISION_FLOOR_VAL:.2f}",
        "hard_test_precision_floor": PRECISION_FLOOR_TEST,
        "baseline_threshold": old_threshold,
        "selected_threshold": best_val["threshold"],
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "selected_val": best_val,
        "selected_test": best_test,
        "delta_test_vs_baseline": delta,
    }

    out_path = reports_dir / "test8_recall_first_policy.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

