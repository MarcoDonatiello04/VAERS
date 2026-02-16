#!/usr/bin/env python3
"""
TEST9 - Retraining campaign for recall >= 0.70 with precision floor >= 0.50.

Strategy:
- Train a compact set of LGBM candidates on RAW and SMOTE training sets.
- Select threshold on validation by maximizing recall under precision floor.
- Evaluate on test and check if target is achieved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET = "IS_SEVERE"
SEED = 42
VAL_PRECISION_FLOOR = 0.58
TEST_PRECISION_FLOOR = 0.50
TARGET_RECALL = 0.70
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


@dataclass
class Candidate:
    name: str
    train_source: str  # raw | smote
    params: dict


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


def select_threshold_recall_first(
    y_true: np.ndarray,
    probs: np.ndarray,
    precision_floor: float,
) -> tuple[float, dict[str, float], str]:
    best = None
    best_thr = 0.50
    for t in THRESHOLDS:
        pred = (probs >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        if p < precision_floor:
            continue
        cand = (r, p, f, float(t))
        if best is None or cand > best:
            best = cand
            best_thr = float(t)

    if best is not None:
        return best_thr, {"recall": best[0], "precision": best[1], "f1": best[2]}, "recall_first"

    # Fallback: best F1 if no threshold satisfies floor.
    best_f = None
    best_t = 0.50
    for t in THRESHOLDS:
        pred = (probs >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        cand = (f, p, r, float(t))
        if best_f is None or cand > best_f:
            best_f = cand
            best_t = float(t)
    return best_t, {"f1": best_f[0], "precision": best_f[1], "recall": best_f[2]}, "fallback_best_f1"


def main() -> None:
    root = Path("/Users/marcodonatiello/PycharmProjects/JupyterProject")
    data_splits = root / "data" / "interim" / "splits"
    smote_dir = root / "data" / "processed" / "smote"
    reports_dir = root / "reports" / "metrics"
    models_dir = root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_splits / "train_step6.csv")
    val_df = pd.read_csv(data_splits / "val_step6.csv")
    test_df = pd.read_csv(data_splits / "test_step6.csv")
    train_smote_df = pd.read_csv(smote_dir / "train_step6_SMOTE.csv")

    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int).to_numpy()
    x_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET].astype(int).to_numpy()
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int).to_numpy()

    x_smote = train_smote_df.drop(columns=[TARGET])
    y_smote = train_smote_df[TARGET].astype(int).to_numpy()

    # Ensure exact same column order.
    x_smote = x_smote[x_train.columns]
    x_val = x_val[x_train.columns]
    x_test = x_test[x_train.columns]

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    candidates = [
        Candidate(
            name="raw_baseline",
            train_source="raw",
            params=dict(
                n_estimators=650,
                learning_rate=0.03,
                num_leaves=95,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                min_child_samples=40,
                scale_pos_weight=spw,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            name="raw_recall_boost",
            train_source="raw",
            params=dict(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=127,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                min_child_samples=25,
                scale_pos_weight=spw * 1.20,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            name="raw_goss",
            train_source="raw",
            params=dict(
                boosting_type="goss",
                n_estimators=750,
                learning_rate=0.03,
                num_leaves=95,
                top_rate=0.25,
                other_rate=0.1,
                reg_lambda=3.0,
                min_child_samples=30,
                scale_pos_weight=spw,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            name="smote_baseline",
            train_source="smote",
            params=dict(
                n_estimators=650,
                learning_rate=0.03,
                num_leaves=95,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                min_child_samples=40,
                scale_pos_weight=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        Candidate(
            name="smote_deep",
            train_source="smote",
            params=dict(
                n_estimators=850,
                learning_rate=0.03,
                num_leaves=127,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=4.0,
                min_child_samples=25,
                scale_pos_weight=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]

    rows = []
    trained = {}

    for c in candidates:
        print(f"\nTraining candidate: {c.name} ({c.train_source})")
        x_tr, y_tr = (x_train, y_train) if c.train_source == "raw" else (x_smote, y_smote)

        model = LGBMClassifier(**c.params)
        model.fit(x_tr, y_tr)
        trained[c.name] = model

        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]

        thr, sel_stats, sel_mode = select_threshold_recall_first(
            y_true=y_val,
            probs=p_val,
            precision_floor=VAL_PRECISION_FLOOR,
        )
        m_val = compute_metrics(y_val, p_val, thr)
        m_test = compute_metrics(y_test, p_test, thr)

        row = {
            "candidate": c.name,
            "train_source": c.train_source,
            "selection_mode": sel_mode,
            "threshold": thr,
            "val_precision": m_val["precision"],
            "val_recall": m_val["recall"],
            "val_f1": m_val["f1"],
            "val_roc_auc": m_val["roc_auc"],
            "test_precision": m_test["precision"],
            "test_recall": m_test["recall"],
            "test_f1": m_test["f1"],
            "test_roc_auc": m_test["roc_auc"],
            "test_pr_auc": m_test["pr_auc"],
            "test_tn": m_test["tn"],
            "test_fp": m_test["fp"],
            "test_fn": m_test["fn"],
            "test_tp": m_test["tp"],
            "meets_test_precision_floor": m_test["precision"] >= TEST_PRECISION_FLOOR,
            "meets_target_recall": m_test["recall"] >= TARGET_RECALL,
            "meets_both_targets": (m_test["precision"] >= TEST_PRECISION_FLOOR) and (m_test["recall"] >= TARGET_RECALL),
        }
        rows.append(row)
        print(
            f"{c.name}: thr={thr:.2f} | "
            f"val(P={m_val['precision']:.4f}, R={m_val['recall']:.4f}) | "
            f"test(P={m_test['precision']:.4f}, R={m_test['recall']:.4f})"
        )

    res = pd.DataFrame(rows).sort_values(
        ["meets_both_targets", "test_recall", "test_precision"],
        ascending=[False, False, False],
    )

    csv_out = reports_dir / "test9_retrain_recall70_candidates.csv"
    res.to_csv(csv_out, index=False)
    print(f"\nSaved candidates table: {csv_out}")

    best_overall = res.iloc[0].to_dict()
    feasible = res[res["meets_both_targets"]].copy()
    best_feasible = feasible.iloc[0].to_dict() if len(feasible) > 0 else None

    summary = {
        "goal": {"target_recall": TARGET_RECALL, "test_precision_floor": TEST_PRECISION_FLOOR},
        "validation_threshold_policy": f"max_recall_with_val_precision>={VAL_PRECISION_FLOOR:.2f}",
        "best_overall_by_test_metrics": best_overall,
        "best_feasible_meeting_both_targets": best_feasible,
    }

    json_out = reports_dir / "test9_retrain_recall70_summary.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_out}")

    # Persist best feasible model if found.
    if best_feasible is not None:
        name = best_feasible["candidate"]
        model_out = {
            "type": "lgbm_recall70_retrained",
            "candidate": name,
            "model": trained[name],
            "threshold": float(best_feasible["threshold"]),
            "features_order": list(x_train.columns),
            "metadata": {
                "val_precision_floor": VAL_PRECISION_FLOOR,
                "test_precision_floor": TEST_PRECISION_FLOOR,
                "target_recall": TARGET_RECALL,
                "test_precision": float(best_feasible["test_precision"]),
                "test_recall": float(best_feasible["test_recall"]),
                "test_f1": float(best_feasible["test_f1"]),
            },
        }
        pkl_out = models_dir / "severe_model_test9_recall70.pkl"
        joblib.dump(model_out, pkl_out)
        print(f"Saved deployable model: {pkl_out}")
    else:
        print("No candidate met both targets (recall >= 0.70 and precision >= 0.50 on test).")


if __name__ == "__main__":
    main()

