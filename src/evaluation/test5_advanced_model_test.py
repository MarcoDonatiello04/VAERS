#!/usr/bin/env python3
"""
Advanced evaluation and model selection for IS_SEVERE.

This script compares multiple candidate pipelines and selects the best one
on validation data using an explicit objective:

- precision_at_recall (default): maximize precision with recall >= min_recall
- f1: maximize F1

Candidates:
- lgbm_weighted: LightGBM on imbalanced train with scale_pos_weight
- lgbm_smote: LightGBM on SMOTE train
- blend_weighted: weighted blend of lgbm_weighted + xgb_weighted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from xgboost import XGBClassifier


TARGET = "IS_SEVERE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and optimize severe reaction models."
    )
    parser.add_argument(
        "--objective",
        choices=("precision_at_recall", "f1"),
        default="precision_at_recall",
        help="Threshold selection objective on validation set.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.60,
        help="Used when objective=precision_at_recall.",
    )
    parser.add_argument(
        "--candidates",
        default="lgbm_weighted,lgbm_smote,blend_weighted",
        help="Comma separated candidate list.",
    )
    parser.add_argument("--thr-start", type=float, default=0.10)
    parser.add_argument("--thr-end", type=float, default=0.95)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.05,
        help="Blend weight step for lgbm in blend_weighted candidate.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save best candidate model bundle into models/production.",
    )
    return parser.parse_args()


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    return np.round(np.arange(start, end + (step / 2.0), step), 6)


def evaluate_thresholds(
    y_true: pd.Series, probs: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = y_true.to_numpy(dtype=int)
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "false_negatives": int(((y == 1) & (pred == 0)).sum()),
                "false_positives": int(((y == 0) & (pred == 1)).sum()),
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


def split_metrics(
    y_true: pd.Series, probs: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y = y_true.to_numpy(dtype=int)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, probs),
        "roc_auc": roc_auc_score(y, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_lgbm_weighted(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = safe_divide(neg, pos)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=80,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model, {"scale_pos_weight": scale_pos_weight}


def train_lgbm_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict]:
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=80,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model, {}


def train_xgb_weighted(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = safe_divide(neg, pos)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, {"scale_pos_weight": scale_pos_weight}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]

    train_path = root / "data" / "interim" / "splits" / "train_step6.csv"
    train_smote_path = root / "data" / "processed" / "smote" / "train_step6_SMOTE.csv"
    val_path = root / "data" / "interim" / "splits" / "val_step6.csv"
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"

    reports_dir = root / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    thresholds = threshold_grid(args.thr_start, args.thr_end, args.thr_step)
    blend_weights = np.round(np.arange(0.0, 1.0 + (args.blend_step / 2.0), args.blend_step), 6)
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]

    print("Loading validation and test splits...")
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    y_val = val_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    X_val = val_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    print("Loading original train split...")
    train_df = pd.read_csv(train_path)
    y_train = train_df[TARGET].astype(int)
    X_train = train_df.drop(columns=[TARGET])
    features_order = list(X_train.columns)

    if list(X_val.columns) != features_order or list(X_test.columns) != features_order:
        raise ValueError("Feature mismatch between train/val/test columns.")

    trained: dict[str, Any] = {}
    notes: dict[str, dict[str, Any]] = {}
    val_probs_cache: dict[str, np.ndarray] = {}
    test_probs_cache: dict[str, np.ndarray] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    def ensure_lgbm_weighted() -> None:
        if "lgbm_weighted" in trained:
            return
        print("Training lgbm_weighted...")
        model, info = train_lgbm_weighted(X_train, y_train)
        trained["lgbm_weighted"] = model
        notes["lgbm_weighted"] = info
        val_probs_cache["lgbm_weighted"] = model.predict_proba(X_val)[:, 1]
        test_probs_cache["lgbm_weighted"] = model.predict_proba(X_test)[:, 1]

    def ensure_xgb_weighted() -> None:
        if "xgb_weighted" in trained:
            return
        print("Training xgb_weighted (for blend candidate)...")
        model, info = train_xgb_weighted(X_train, y_train)
        trained["xgb_weighted"] = model
        notes["xgb_weighted"] = info
        val_probs_cache["xgb_weighted"] = model.predict_proba(X_val)[:, 1]
        test_probs_cache["xgb_weighted"] = model.predict_proba(X_test)[:, 1]

    def ensure_lgbm_smote() -> None:
        if "lgbm_smote" in trained:
            return
        print("Loading SMOTE train split...")
        train_smote_df = pd.read_csv(train_smote_path)
        y_train_smote = train_smote_df[TARGET].astype(int)
        X_train_smote = train_smote_df.drop(columns=[TARGET])
        if list(X_train_smote.columns) != features_order:
            raise ValueError("Feature mismatch between train_step6 and train_step6_SMOTE.")

        print("Training lgbm_smote...")
        model, info = train_lgbm_smote(X_train_smote, y_train_smote)
        trained["lgbm_smote"] = model
        notes["lgbm_smote"] = info
        val_probs_cache["lgbm_smote"] = model.predict_proba(X_val)[:, 1]
        test_probs_cache["lgbm_smote"] = model.predict_proba(X_test)[:, 1]

    def add_summary_row(
        name: str,
        val_probs: np.ndarray,
        test_probs: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thr_df = evaluate_thresholds(y_val, val_probs, thresholds)
        best_row, reason = pick_best_threshold(thr_df, args.objective, args.min_recall)
        threshold = float(best_row["threshold"])

        val_metrics = split_metrics(y_val, val_probs, threshold)
        test_metrics = split_metrics(y_test, test_probs, threshold)
        select_score = (
            float(best_row["precision"])
            if args.objective == "precision_at_recall"
            else float(best_row["f1"])
        )

        row = {
            "candidate": name,
            "selection_reason": reason,
            "selection_score": select_score,
            "threshold": threshold,
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
        if extra:
            row.update(extra)
        summary_rows.append(row)
        threshold_tables[name] = thr_df
        return row

    for candidate in candidates:
        if candidate == "lgbm_weighted":
            ensure_lgbm_weighted()
            add_summary_row(
                "lgbm_weighted",
                val_probs_cache["lgbm_weighted"],
                test_probs_cache["lgbm_weighted"],
                notes.get("lgbm_weighted"),
            )
            continue

        if candidate == "lgbm_smote":
            ensure_lgbm_smote()
            add_summary_row(
                "lgbm_smote",
                val_probs_cache["lgbm_smote"],
                test_probs_cache["lgbm_smote"],
                notes.get("lgbm_smote"),
            )
            continue

        if candidate == "blend_weighted":
            ensure_lgbm_weighted()
            ensure_xgb_weighted()
            lgb_val = val_probs_cache["lgbm_weighted"]
            xgb_val = val_probs_cache["xgb_weighted"]
            lgb_test = test_probs_cache["lgbm_weighted"]
            xgb_test = test_probs_cache["xgb_weighted"]

            best_blend: dict[str, Any] | None = None
            for w in blend_weights:
                val_probs = (w * lgb_val) + ((1.0 - w) * xgb_val)
                thr_df = evaluate_thresholds(y_val, val_probs, thresholds)
                best_row, _ = pick_best_threshold(thr_df, args.objective, args.min_recall)
                score = (
                    float(best_row["precision"])
                    if args.objective == "precision_at_recall"
                    else float(best_row["f1"])
                )
                if best_blend is None or score > best_blend["score"]:
                    best_blend = {
                        "weight_lgbm": float(w),
                        "score": score,
                        "val_probs": val_probs,
                    }

            if best_blend is None:
                raise RuntimeError("Blend optimization failed.")

            w = best_blend["weight_lgbm"]
            blend_val_probs = best_blend["val_probs"]
            blend_test_probs = (w * lgb_test) + ((1.0 - w) * xgb_test)

            add_summary_row(
                "blend_weighted",
                blend_val_probs,
                blend_test_probs,
                {"weight_lgbm": w},
            )
            continue

        raise ValueError(
            f"Unknown candidate '{candidate}'. "
            "Allowed: lgbm_weighted,lgbm_smote,blend_weighted"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selection_score", "val_f1"], ascending=False
    )

    best = summary_df.iloc[0].to_dict()
    best_name = str(best["candidate"])

    summary_path = reports_dir / "advanced_model_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved benchmark summary: {summary_path}")

    for name, table in threshold_tables.items():
        table_path = reports_dir / f"threshold_table_{name}.csv"
        table.to_csv(table_path, index=False)

    best_report = {
        "objective": args.objective,
        "min_recall": args.min_recall,
        "best_candidate": best,
        "candidates": summary_df.to_dict(orient="records"),
    }
    best_json_path = reports_dir / "advanced_model_benchmark_report.json"
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_report, f, indent=2)
    print(f"Saved benchmark report: {best_json_path}")

    if args.save_model:
        out_path = root / "models" / "production" / "severe_model_optimized.pkl"
        if best_name == "lgbm_weighted":
            bundle = {
                "type": "lgbm_weighted",
                "model": trained["lgbm_weighted"],
                "threshold": float(best["threshold"]),
                "features_order": features_order,
                "metadata": notes.get("lgbm_weighted", {}),
            }
        elif best_name == "lgbm_smote":
            bundle = {
                "type": "lgbm_smote",
                "model": trained["lgbm_smote"],
                "threshold": float(best["threshold"]),
                "features_order": features_order,
                "metadata": notes.get("lgbm_smote", {}),
            }
        elif best_name == "blend_weighted":
            bundle = {
                "type": "blend_weighted",
                "lgbm_model": trained["lgbm_weighted"],
                "xgb_model": trained["xgb_weighted"],
                "weight_lgbm": float(best.get("weight_lgbm", 0.5)),
                "threshold": float(best["threshold"]),
                "features_order": features_order,
                "metadata": {
                    "lgbm": notes.get("lgbm_weighted", {}),
                    "xgb": notes.get("xgb_weighted", {}),
                },
            }
        else:
            raise RuntimeError(f"Unsupported best candidate for save: {best_name}")
        joblib.dump(bundle, out_path)
        print(f"Saved optimized model bundle: {out_path}")

    print("\nTop candidates:")
    print(summary_df[["candidate", "threshold", "val_precision", "val_recall", "val_f1", "test_precision", "test_recall", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
