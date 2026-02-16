#!/usr/bin/env python3
# ======================================================
# TEST6 - ORTHOGONAL STACKING (LGBM + LINEAR + KNN + MLP)
# ======================================================

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# ======================================================
# CONFIG
# ======================================================

TARGET = "IS_SEVERE"
SEED = 42
N_SPLITS = 5

# L1 diversity settings
KNN_TOP_FEATURES = 25
KNN_NEIGHBORS = 151
# Training KNN on full folds can be prohibitively expensive on very large data.
# Set None to disable subsampling.
KNN_MAX_TRAIN_SAMPLES = 180_000

# L2 settings
META_MODEL_KIND = "ridgecv"  # alternatives: "ridgecv", "lgbm_shallow"
MIN_RECALL_FOR_THRESHOLD = 0.58

# LGBM anchor (aligned with test4 champion family)
LGBM_PARAMS = {
    "n_estimators": 650,
    "learning_rate": 0.03,
    "num_leaves": 95,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 3.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# No class_weight='balanced' by design (calibration-first).
LOGREG_PARAMS = {
    "C": 0.7,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1200,
    "random_state": SEED,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 512,
    "learning_rate_init": 8e-4,
    "max_iter": 180,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 12,
    "random_state": SEED,
}


# ======================================================
# HELPERS
# ======================================================

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    return cwd


def build_non_tree_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def safe_prob(model, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    # fallback for decision-function models, though not expected here
    s = model.decision_function(x)
    s = np.asarray(s, dtype=float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return s


def stratified_subsample(
    x: pd.DataFrame, y: pd.Series, max_samples: int | None, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples is None or len(x) <= max_samples:
        return x, y
    x_sub, _, y_sub, _ = train_test_split(
        x,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=seed,
    )
    return x_sub, y_sub


def select_top_features_from_lgbm(
    model: LGBMClassifier, columns: pd.Index, k: int
) -> list[str]:
    gains = pd.Series(model.feature_importances_, index=columns)
    # If gain-based importances are all zero, fallback to first k columns.
    if (gains > 0).sum() == 0:
        return list(columns[:k])
    return list(gains.sort_values(ascending=False).head(k).index)


def apply_rank_gauss(
    train_meta_raw: np.ndarray,
    val_meta_raw: np.ndarray,
    test_meta_raw: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, QuantileTransformer]]:
    train_meta = np.zeros_like(train_meta_raw, dtype=float)
    val_meta = np.zeros_like(val_meta_raw, dtype=float)
    test_meta = np.zeros_like(test_meta_raw, dtype=float)
    transformers: dict[str, QuantileTransformer] = {}

    for j, name in enumerate(feature_names):
        n_quant = min(1000, train_meta_raw.shape[0])
        qt = QuantileTransformer(
            n_quantiles=n_quant,
            output_distribution="normal",
            random_state=SEED,
        )
        train_meta[:, j] = qt.fit_transform(train_meta_raw[:, [j]]).ravel()
        val_meta[:, j] = qt.transform(val_meta_raw[:, [j]]).ravel()
        test_meta[:, j] = qt.transform(test_meta_raw[:, [j]]).ravel()
        transformers[name] = qt

    return train_meta, val_meta, test_meta, transformers


def threshold_search(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_recall: float = 0.58,
) -> tuple[float, dict[str, float], str]:
    best_thr = 0.5
    best_metrics: dict[str, float] | None = None
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        if best_metrics is None:
            best_thr = float(thr)
            best_metrics = {"precision": prec, "recall": rec, "f1": f1}
            continue
        if (prec > best_metrics["precision"]) or (
            np.isclose(prec, best_metrics["precision"]) and f1 > best_metrics["f1"]
        ):
            best_thr = float(thr)
            best_metrics = {"precision": prec, "recall": rec, "f1": f1}

    if best_metrics is not None:
        return best_thr, best_metrics, "precision_constrained_by_recall"

    # fallback if no threshold satisfies recall constraint
    fallback_thr = 0.5
    fallback = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > fallback["f1"]:
            fallback_thr = float(thr)
            fallback = {"precision": prec, "recall": rec, "f1": f1}
    return fallback_thr, fallback, "fallback_best_f1"


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict[str, float | int]:
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


# ======================================================
# MAIN
# ======================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    y_train = train_df[TARGET].astype(int)
    x_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET].astype(int)
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    print(f"Train shape: {x_train.shape}")
    print(f"Val shape  : {x_val.shape}")
    print(f"Test shape : {x_test.shape}")

    # Keep anchor consistent with your tuned setup.
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params["scale_pos_weight"] = scale_pos_weight

    l1_names = ["lgbm", "linear", "knn", "mlp"]
    oof_preds = {name: np.zeros(len(x_train), dtype=float) for name in l1_names}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    print("\nGenerating OOF predictions (Orthogonal L1)...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train), start=1):
        print(f"\nFold {fold}/{N_SPLITS}")
        x_tr = x_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        # 1) Anchor LGBM (tree, no scaling/imputation pipeline)
        lgbm_fold = LGBMClassifier(**lgbm_params)
        lgbm_fold.fit(x_tr, y_tr)
        oof_preds["lgbm"][va_idx] = lgbm_fold.predict_proba(x_va)[:, 1]

        # Top-K features for KNN selected from current fold (no leakage).
        top_features_fold = select_top_features_from_lgbm(
            lgbm_fold,
            x_train.columns,
            KNN_TOP_FEATURES,
        )

        # 2) Linear model (imputer + scaler + logistic)
        linear_fold = build_non_tree_pipeline(LogisticRegression(**LOGREG_PARAMS))
        linear_fold.fit(x_tr, y_tr)
        oof_preds["linear"][va_idx] = safe_prob(linear_fold, x_va)

        # 3) KNN (imputer + scaler) on top-k features only
        x_tr_knn, y_tr_knn = stratified_subsample(
            x_tr[top_features_fold],
            y_tr,
            KNN_MAX_TRAIN_SAMPLES,
            seed=SEED + fold,
        )
        knn_model = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
        knn_fold = build_non_tree_pipeline(knn_model)
        knn_fold.fit(x_tr_knn, y_tr_knn)
        oof_preds["knn"][va_idx] = safe_prob(knn_fold, x_va[top_features_fold])

        # 4) Neural net (MLP) with imputer + scaler
        mlp_fold = build_non_tree_pipeline(MLPClassifier(**MLP_PARAMS))
        mlp_fold.fit(x_tr, y_tr)
        oof_preds["mlp"][va_idx] = safe_prob(mlp_fold, x_va)

        # Fold diagnostic
        fold_auc = {
            m: roc_auc_score(y_va, oof_preds[m][va_idx]) for m in l1_names
        }
        print(
            "Fold ROC-AUC -> "
            + ", ".join([f"{m}:{fold_auc[m]:.4f}" for m in l1_names])
        )

    # Pearson correlation matrix across OOF predictions (diversity check).
    oof_df = pd.DataFrame({m: oof_preds[m] for m in l1_names})
    corr = oof_df.corr(method="pearson")
    print("\n=== L1 OOF Pearson Correlation Matrix ===")
    print(corr.round(4))

    off_diag = corr.to_numpy().copy()
    np.fill_diagonal(off_diag, np.nan)
    max_corr = float(np.nanmax(np.abs(off_diag)))
    print(f"Max |off-diagonal correlation|: {max_corr:.4f} (target < 0.95)")

    corr_path = reports_dir / "test6_orthogonal_l1_oof_correlation.csv"
    corr.to_csv(corr_path)
    print(f"Saved correlation matrix: {corr_path}")

    # Train final L1 models on full training split.
    print("\nTraining final L1 models on full train...")
    final_lgbm = LGBMClassifier(**lgbm_params)
    final_lgbm.fit(x_train, y_train)
    top_features_full = select_top_features_from_lgbm(
        final_lgbm,
        x_train.columns,
        KNN_TOP_FEATURES,
    )

    final_linear = build_non_tree_pipeline(LogisticRegression(**LOGREG_PARAMS))
    final_linear.fit(x_train, y_train)

    x_train_knn, y_train_knn = stratified_subsample(
        x_train[top_features_full],
        y_train,
        KNN_MAX_TRAIN_SAMPLES,
        seed=SEED + 999,
    )
    final_knn = build_non_tree_pipeline(
        KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
    )
    final_knn.fit(x_train_knn, y_train_knn)

    final_mlp = build_non_tree_pipeline(MLPClassifier(**MLP_PARAMS))
    final_mlp.fit(x_train, y_train)

    # Raw L1 probabilities for val/test.
    val_raw = {
        "lgbm": final_lgbm.predict_proba(x_val)[:, 1],
        "linear": safe_prob(final_linear, x_val),
        "knn": safe_prob(final_knn, x_val[top_features_full]),
        "mlp": safe_prob(final_mlp, x_val),
    }
    test_raw = {
        "lgbm": final_lgbm.predict_proba(x_test)[:, 1],
        "linear": safe_prob(final_linear, x_test),
        "knn": safe_prob(final_knn, x_test[top_features_full]),
        "mlp": safe_prob(final_mlp, x_test),
    }

    # Meta features: RankGauss-normalized (not raw probs).
    x_meta_train_raw = np.column_stack([oof_preds[m] for m in l1_names])
    x_meta_val_raw = np.column_stack([val_raw[m] for m in l1_names])
    x_meta_test_raw = np.column_stack([test_raw[m] for m in l1_names])

    (
        x_meta_train,
        x_meta_val,
        x_meta_test,
        rankgauss_transformers,
    ) = apply_rank_gauss(
        x_meta_train_raw,
        x_meta_val_raw,
        x_meta_test_raw,
        l1_names,
    )

    print("\nRankGauss applied on meta-features.")
    print(f"Meta train shape: {x_meta_train.shape}")

    # L2 meta-learner.
    print(f"\nTraining meta-learner: {META_MODEL_KIND}")
    if META_MODEL_KIND == "ridgecv":
        meta_model = RidgeCV(alphas=np.logspace(-3, 2, 12))
        meta_model.fit(x_meta_train, y_train.to_numpy())
        val_probs = np.clip(meta_model.predict(x_meta_val), 0.0, 1.0)
        test_probs = np.clip(meta_model.predict(x_meta_test), 0.0, 1.0)
        print("RidgeCV best alpha:", float(meta_model.alpha_))
        print("RidgeCV coefficients:", np.round(meta_model.coef_, 4))
    elif META_MODEL_KIND == "lgbm_shallow":
        meta_model = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=2,
            num_leaves=3,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        meta_model.fit(x_meta_train, y_train.to_numpy())
        val_probs = meta_model.predict_proba(x_meta_val)[:, 1]
        test_probs = meta_model.predict_proba(x_meta_test)[:, 1]
    else:
        raise ValueError(f"Unsupported META_MODEL_KIND: {META_MODEL_KIND}")

    # Threshold tuning on validation.
    thr, thr_metrics, thr_mode = threshold_search(
        y_true=y_val.to_numpy(),
        probs=val_probs,
        min_recall=MIN_RECALL_FOR_THRESHOLD,
    )
    print("\nThreshold strategy:", thr_mode)
    print(
        f"Selected threshold: {thr:.2f} | "
        f"Val Precision: {thr_metrics['precision']:.4f} | "
        f"Val Recall: {thr_metrics['recall']:.4f} | "
        f"Val F1: {thr_metrics['f1']:.4f}"
    )

    # Final evaluation.
    val_metrics = compute_metrics(y_val.to_numpy(), val_probs, thr)
    test_metrics = compute_metrics(y_test.to_numpy(), test_probs, thr)

    test_pred = (test_probs >= thr).astype(int)
    print("\n===== TEST6 ORTHOGONAL STACKING - TEST REPORT =====")
    print(classification_report(y_test.to_numpy(), test_pred, digits=4))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test.to_numpy(), test_pred))

    # Single model diagnostic on test.
    print("\n--- L1 Test ROC-AUC ---")
    for name in l1_names:
        auc = roc_auc_score(y_test, test_raw[name])
        print(f"{name.upper():>8}: {auc:.4f}")
    print(f"{'STACK':>8}: {test_metrics['roc_auc']:.4f}")

    # Persist artifacts.
    artifact = {
        "type": "orthogonal_stacking_test6",
        "target": TARGET,
        "seed": SEED,
        "l1_names": l1_names,
        "l1_models": {
            "lgbm": final_lgbm,
            "linear": final_linear,
            "knn": final_knn,
            "mlp": final_mlp,
        },
        "knn_top_features": top_features_full,
        "rankgauss": rankgauss_transformers,
        "meta_model_kind": META_MODEL_KIND,
        "meta_model": meta_model,
        "threshold": float(thr),
        "correlation_matrix": corr,
        "config": {
            "n_splits": N_SPLITS,
            "knn_top_features": KNN_TOP_FEATURES,
            "knn_neighbors": KNN_NEIGHBORS,
            "knn_max_train_samples": KNN_MAX_TRAIN_SAMPLES,
            "min_recall_for_threshold": MIN_RECALL_FOR_THRESHOLD,
            "scale_pos_weight_anchor_lgbm": scale_pos_weight,
        },
    }

    model_out = models_dir / "severe_model_test6_orthogonal_stacking.pkl"
    joblib.dump(artifact, model_out)
    print(f"\nSaved model artifact: {model_out}")

    summary = {
        "threshold_mode": thr_mode,
        "threshold": float(thr),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "max_abs_offdiag_corr": max_corr,
        "l1_test_auc": {name: float(roc_auc_score(y_test, test_raw[name])) for name in l1_names},
    }
    summary_path = reports_dir / "test6_orthogonal_stacking_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
