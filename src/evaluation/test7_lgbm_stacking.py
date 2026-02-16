#!/usr/bin/env python3
# ======================================================
# TEST7 - LGBM ZOO STACKING
# Objective: Stacking with only LightGBM variants + RidgeCV meta-learner
# ======================================================

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ======================================================
# CONFIG
# ======================================================

TARGET = "IS_SEVERE"
SEED = 42
N_SPLITS = 5
BENCHMARK_AUC = 0.8921
MIN_RECALL_FOR_THRESHOLD = 0.58


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    return cwd


def build_lgbm_zoo(scale_pos_weight: float) -> dict[str, dict]:
    # The 5 LGBM variants requested.
    return {
        "lgbm_gbdt": {
            "boosting_type": "gbdt",
            "n_estimators": 650,
            "learning_rate": 0.03,
            "num_leaves": 95,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 3.0,
            "min_child_samples": 40,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgbm_dart": {
            "boosting_type": "dart",
            "xgboost_dart_mode": True,
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 4.0,
            "drop_rate": 0.1,
            "skip_drop": 0.5,
            "min_child_samples": 40,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgbm_goss": {
            "boosting_type": "goss",
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "top_rate": 0.2,
            "other_rate": 0.1,
            "reg_lambda": 3.0,
            "min_child_samples": 40,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgbm_deep": {
            "boosting_type": "gbdt",
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 128,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 10.0,
            "min_child_samples": 25,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
        "lgbm_shallow": {
            "boosting_type": "gbdt",
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 3.0,
            "min_child_samples": 100,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        },
    }


def threshold_search(y_true: np.ndarray, probs: np.ndarray, min_recall: float) -> tuple[float, dict, str]:
    best = None
    best_thr = 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= thr).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        if best is None or (prec > best["precision"]) or (np.isclose(prec, best["precision"]) and f1 > best["f1"]):
            best = {"precision": prec, "recall": rec, "f1": f1}
            best_thr = float(thr)

    if best is not None:
        return best_thr, best, "precision_constrained_by_recall"

    # fallback
    fallback = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    fallback_thr = 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= thr).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > fallback["f1"]:
            fallback = {"precision": prec, "recall": rec, "f1": f1}
            fallback_thr = float(thr)
    return fallback_thr, fallback, "fallback_best_f1"


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
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
    root = find_project_root()
    data_dir = root / "data" / "interim" / "splits"
    models_dir = root / "models"
    reports_dir = root / "reports" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_step6.csv"
    val_path = data_dir / "val_step6.csv"
    test_path = data_dir / "test_step6.csv"

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int).to_numpy()
    x_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET].astype(int).to_numpy()
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int).to_numpy()

    print(f"Train shape: {x_train.shape}")
    print(f"Val shape  : {x_val.shape}")
    print(f"Test shape : {x_test.shape}")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    zoo_params = build_lgbm_zoo(scale_pos_weight=scale_pos_weight)
    model_names = list(zoo_params.keys())

    # -------------------------
    # Level 1 - OOF generation
    # -------------------------
    print("\nGenerating OOF predictions with LGBM Zoo...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(x_train), dtype=float) for name in model_names}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train), start=1):
        print(f"Fold {fold}/{N_SPLITS}")
        x_tr = x_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_va = y_train[va_idx]

        for name in model_names:
            model = LGBMClassifier(**zoo_params[name])
            model.fit(x_tr, y_tr)
            oof[name][va_idx] = model.predict_proba(x_va)[:, 1]

        fold_auc = {name: roc_auc_score(y_va, oof[name][va_idx]) for name in model_names}
        print("  Fold AUC:", ", ".join([f"{k}={v:.4f}" for k, v in fold_auc.items()]))

    x_meta_train = np.column_stack([oof[name] for name in model_names])
    y_meta_train = y_train

    # Correlation diagnostics.
    corr_df = pd.DataFrame({name: oof[name] for name in model_names}).corr(method="pearson")
    print("\n=== Pearson Correlation (OOF, 5x LGBM) ===")
    print(corr_df.round(5))

    corr_path = reports_dir / "test7_lgbm_zoo_oof_correlation.csv"
    corr_df.to_csv(corr_path)
    print(f"Saved correlation matrix: {corr_path}")

    # -------------------------
    # Level 1 - Full fit
    # -------------------------
    print("\nTraining full L1 models...")
    fitted_models = {}
    val_raw = {}
    test_raw = {}
    for name in model_names:
        m = LGBMClassifier(**zoo_params[name])
        m.fit(x_train, y_train)
        fitted_models[name] = m
        val_raw[name] = m.predict_proba(x_val)[:, 1]
        test_raw[name] = m.predict_proba(x_test)[:, 1]

    x_meta_val = np.column_stack([val_raw[name] for name in model_names])
    x_meta_test = np.column_stack([test_raw[name] for name in model_names])

    # -------------------------
    # Level 2 - RidgeCV
    # -------------------------
    print("\nTraining RidgeCV meta-learner...")
    meta_model = RidgeCV(alphas=np.logspace(-3, 2, 12))
    meta_model.fit(x_meta_train, y_meta_train)

    val_probs = np.clip(meta_model.predict(x_meta_val), 0.0, 1.0)
    test_probs = np.clip(meta_model.predict(x_meta_test), 0.0, 1.0)

    print(f"RidgeCV alpha: {float(meta_model.alpha_):.6f}")
    print("Meta coefficients:", np.round(meta_model.coef_, 6))

    # -------------------------
    # Evaluation
    # -------------------------
    thr, thr_stats, thr_mode = threshold_search(y_val, val_probs, MIN_RECALL_FOR_THRESHOLD)
    print(f"\nThreshold mode: {thr_mode}")
    print(
        f"Selected threshold={thr:.2f} | "
        f"Val precision={thr_stats['precision']:.4f} | "
        f"Val recall={thr_stats['recall']:.4f} | "
        f"Val f1={thr_stats['f1']:.4f}"
    )

    val_metrics = metrics_at_threshold(y_val, val_probs, thr)
    test_metrics = metrics_at_threshold(y_test, test_probs, thr)

    print("\n===== TEST7 LGBM STACKING REPORT (TEST) =====")
    test_pred = (test_probs >= thr).astype(int)
    print(classification_report(y_test, test_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    print("\n--- Test ROC-AUC (single L1) ---")
    for name in model_names:
        auc = roc_auc_score(y_test, test_raw[name])
        print(f"{name}: {auc:.4f}")

    final_auc = float(test_metrics["roc_auc"])
    delta_vs_benchmark = final_auc - BENCHMARK_AUC
    status = "above" if delta_vs_benchmark >= 0 else "below"
    print(f"\nFinal STACK ROC-AUC: {final_auc:.4f}")
    print(f"Benchmark ROC-AUC  : {BENCHMARK_AUC:.4f}")
    print(f"Delta vs benchmark : {delta_vs_benchmark:+.4f} ({status})")

    # Persist artifacts.
    artifact = {
        "type": "test7_lgbm_zoo_stacking",
        "target": TARGET,
        "seed": SEED,
        "l1_model_names": model_names,
        "l1_models": fitted_models,
        "l1_params": zoo_params,
        "oof_correlation": corr_df,
        "meta_model": meta_model,
        "threshold": float(thr),
        "benchmark_auc": BENCHMARK_AUC,
        "delta_vs_benchmark": float(delta_vs_benchmark),
    }
    model_out = models_dir / "severe_model_test7_lgbm_zoo_stacking.pkl"
    joblib.dump(artifact, model_out)
    print(f"Saved model artifact: {model_out}")

    summary = {
        "threshold_mode": thr_mode,
        "threshold": float(thr),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "benchmark_auc": BENCHMARK_AUC,
        "delta_vs_benchmark": float(delta_vs_benchmark),
        "meta_alpha": float(meta_model.alpha_),
        "meta_coefficients": meta_model.coef_.tolist(),
    }
    summary_path = reports_dir / "test7_lgbm_stacking_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
