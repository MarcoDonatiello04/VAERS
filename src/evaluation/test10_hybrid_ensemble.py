#!/usr/bin/env python3
"""
TEST10 - Hybrid Ensemble from current production models.

Goal:
- Start from the current single-model champion (LGBM Test4 best tactic).
- Blend it with the existing production stacking model.
- Tune blend weight + decision threshold on validation.
- Compare against the single model baseline and optionally save a deployable bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a hybrid ensemble from production artifacts."
    )
    parser.add_argument(
        "--objective",
        choices=("precision_at_recall", "f1"),
        default="precision_at_recall",
        help="Validation objective used to select threshold.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.60,
        help="Recall constraint when objective=precision_at_recall.",
    )
    parser.add_argument("--thr-start", type=float, default=0.05)
    parser.add_argument("--thr-end", type=float, default=0.95)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.05,
        help="Weight step for LGBM in blended probability: w*lgbm + (1-w)*stack.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save best bundle to models/production/severe_model_hybrid_ensemble.pkl",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    return cwd


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    return np.round(np.arange(start, end + (step / 2.0), step), 6)


def evaluate_thresholds(
    y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "false_negatives": int(((y_true == 1) & (pred == 0)).sum()),
                "false_positives": int(((y_true == 0) & (pred == 1)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def pick_best_threshold(
    thr_df: pd.DataFrame, objective: str, min_recall: float
) -> tuple[pd.Series, str]:
    if objective == "f1":
        row = thr_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
        return row, "best_f1"

    feasible = thr_df[thr_df["recall"] >= min_recall]
    if not feasible.empty:
        row = feasible.sort_values(["precision", "f1"], ascending=False).iloc[0]
        return row, f"best_precision_with_recall>={min_recall:.2f}"

    row = thr_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
    return row, "fallback_best_f1_no_recall_threshold"


def split_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def align_columns(x: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = x.copy()
    for col in features:
        if col not in out.columns:
            out[col] = 0
    return out[features]


def stack_prod_predict_proba(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    rf_prob = bundle["rf_model"].predict_proba(x)[:, 1]
    xgb_prob = bundle["xgb_model"].predict_proba(x)[:, 1]
    lgbm_prob = bundle["lgbm_model"].predict_proba(x)[:, 1]

    meta_x = pd.DataFrame(
        {
            "rf_prob": rf_prob,
            "xgb_prob": xgb_prob,
            "lgbm_prob": lgbm_prob,
        }
    )
    return bundle["meta_model"].predict_proba(meta_x)[:, 1]


def main() -> None:
    args = parse_args()
    root = find_project_root()

    val_path = root / "data" / "interim" / "splits" / "val_step6.csv"
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"

    lgbm_path = root / "models" / "production" / "severe_model_test4_best_tactic.pkl"
    stack_path = root / "models" / "production" / "severe_model_stacking_production.pkl"

    reports_dir = root / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.objective == "precision_at_recall":
        run_tag = f"{args.objective}_r{args.min_recall:.2f}"
    else:
        run_tag = args.objective
    run_tag = run_tag.replace(".", "p")

    if not lgbm_path.exists():
        raise FileNotFoundError(f"Missing artifact: {lgbm_path}")
    if not stack_path.exists():
        raise FileNotFoundError(f"Missing artifact: {stack_path}")

    print("Loading data splits...")
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()
    x_val_raw = val_df.drop(columns=[TARGET])
    x_test_raw = test_df.drop(columns=[TARGET])

    print("Loading production artifacts...")
    lgbm_artifact = joblib.load(lgbm_path)
    stack_artifact = joblib.load(stack_path)

    lgbm_features = list(lgbm_artifact["features_order"])
    stack_features = list(stack_artifact["features_order"])
    common_features = lgbm_features

    x_val_lgbm = align_columns(x_val_raw, lgbm_features)
    x_test_lgbm = align_columns(x_test_raw, lgbm_features)
    x_val_stack = align_columns(x_val_raw, stack_features)
    x_test_stack = align_columns(x_test_raw, stack_features)

    if lgbm_features != stack_features:
        print("Warning: feature order differs between artifacts. Using each model's own order.")
    else:
        print(f"Feature order aligned: {len(common_features)} columns")

    print("Generating base probabilities...")
    p_val_lgbm = lgbm_artifact["model"].predict_proba(x_val_lgbm)[:, 1]
    p_test_lgbm = lgbm_artifact["model"].predict_proba(x_test_lgbm)[:, 1]
    p_val_stack = stack_prod_predict_proba(stack_artifact, x_val_stack)
    p_test_stack = stack_prod_predict_proba(stack_artifact, x_test_stack)

    thresholds = threshold_grid(args.thr_start, args.thr_end, args.thr_step)
    weights = np.round(np.arange(0.0, 1.0 + (args.blend_step / 2.0), args.blend_step), 6)

    summary_rows: list[dict[str, Any]] = []
    threshold_tables: dict[str, pd.DataFrame] = {}

    # Baseline row: single model only.
    baseline_thr_df = evaluate_thresholds(y_val, p_val_lgbm, thresholds)
    baseline_best, baseline_reason = pick_best_threshold(
        baseline_thr_df, args.objective, args.min_recall
    )
    baseline_thr = float(baseline_best["threshold"])
    baseline_val = split_metrics(y_val, p_val_lgbm, baseline_thr)
    baseline_test = split_metrics(y_test, p_test_lgbm, baseline_thr)
    baseline_row = {
        "candidate": "lgbm_single_baseline",
        "blend_weight_lgbm": 1.0,
        "selection_reason": baseline_reason,
        "selection_score": float(
            baseline_best["precision"] if args.objective == "precision_at_recall" else baseline_best["f1"]
        ),
        "threshold": baseline_thr,
        "val_precision": baseline_val["precision"],
        "val_recall": baseline_val["recall"],
        "val_f1": baseline_val["f1"],
        "val_pr_auc": baseline_val["pr_auc"],
        "val_roc_auc": baseline_val["roc_auc"],
        "test_precision": baseline_test["precision"],
        "test_recall": baseline_test["recall"],
        "test_f1": baseline_test["f1"],
        "test_pr_auc": baseline_test["pr_auc"],
        "test_roc_auc": baseline_test["roc_auc"],
        "test_tn": baseline_test["tn"],
        "test_fp": baseline_test["fp"],
        "test_fn": baseline_test["fn"],
        "test_tp": baseline_test["tp"],
    }
    summary_rows.append(baseline_row)
    threshold_tables["lgbm_single_baseline"] = baseline_thr_df

    # Hybrid blend rows.
    for w in weights:
        candidate = f"hybrid_blend_w{w:.2f}"
        p_val = (w * p_val_lgbm) + ((1.0 - w) * p_val_stack)
        p_test = (w * p_test_lgbm) + ((1.0 - w) * p_test_stack)

        thr_df = evaluate_thresholds(y_val, p_val, thresholds)
        best_row, reason = pick_best_threshold(thr_df, args.objective, args.min_recall)
        thr = float(best_row["threshold"])
        val_metrics = split_metrics(y_val, p_val, thr)
        test_metrics = split_metrics(y_test, p_test, thr)

        score = float(
            best_row["precision"] if args.objective == "precision_at_recall" else best_row["f1"]
        )
        summary_rows.append(
            {
                "candidate": candidate,
                "blend_weight_lgbm": float(w),
                "selection_reason": reason,
                "selection_score": score,
                "threshold": thr,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_tn": test_metrics["tn"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tp": test_metrics["tp"],
            }
        )
        threshold_tables[candidate] = thr_df

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selection_score", "val_f1"], ascending=False
    )
    best = summary_df.iloc[0].to_dict()

    baseline = summary_df[summary_df["candidate"] == "lgbm_single_baseline"].iloc[0].to_dict()
    delta_test_vs_baseline = {
        "precision": float(best["test_precision"] - baseline["test_precision"]),
        "recall": float(best["test_recall"] - baseline["test_recall"]),
        "f1": float(best["test_f1"] - baseline["test_f1"]),
        "roc_auc": float(best["test_roc_auc"] - baseline["test_roc_auc"]),
        "pr_auc": float(best["test_pr_auc"] - baseline["test_pr_auc"]),
        "fn_reduction": int(baseline["test_fn"] - best["test_fn"]),
        "tp_gain": int(best["test_tp"] - baseline["test_tp"]),
    }

    summary_path = reports_dir / f"test10_hybrid_ensemble_summary_{run_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Save threshold tables for reproducibility.
    for name, table in threshold_tables.items():
        safe_name = name.replace(".", "_")
        table_path = reports_dir / f"test10_threshold_table_{safe_name}_{run_tag}.csv"
        table.to_csv(table_path, index=False)

    report = {
        "objective": args.objective,
        "min_recall": args.min_recall,
        "search": {
            "thr_start": args.thr_start,
            "thr_end": args.thr_end,
            "thr_step": args.thr_step,
            "blend_step": args.blend_step,
        },
        "baseline": baseline,
        "best_candidate": best,
        "delta_test_vs_baseline": delta_test_vs_baseline,
        "top_candidates": summary_df.head(5).to_dict(orient="records"),
    }
    report_path = reports_dir / f"test10_hybrid_ensemble_report_{run_tag}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {report_path}")

    if args.save_model:
        model_out = {
            "type": "hybrid_blend_lgbm_plus_stack_prod",
            "lgbm_model": lgbm_artifact["model"],
            "stack_models": {
                "rf_model": stack_artifact["rf_model"],
                "xgb_model": stack_artifact["xgb_model"],
                "lgbm_model": stack_artifact["lgbm_model"],
                "meta_model": stack_artifact["meta_model"],
            },
            "blend_weight_lgbm": float(best["blend_weight_lgbm"]),
            "threshold": float(best["threshold"]),
            "features_order_lgbm": lgbm_features,
            "features_order_stack": stack_features,
            "metadata": {
                "objective": args.objective,
                "min_recall": args.min_recall,
                "baseline_test": baseline,
                "best_test": best,
                "delta_test_vs_baseline": delta_test_vs_baseline,
            },
        }
        model_path = root / "models" / "production" / f"severe_model_hybrid_ensemble_{run_tag}.pkl"
        joblib.dump(model_out, model_path)
        print(f"Saved model: {model_path}")

    print("\nTop candidates:")
    print(
        summary_df[
            [
                "candidate",
                "blend_weight_lgbm",
                "threshold",
                "val_precision",
                "val_recall",
                "val_f1",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
