#!/usr/bin/env python3
"""Generate report-ready figures for the VAERS severe reaction project.

Canonical outputs are written to report/figures (used by report/main.tex).
A legacy mirror is also written to reports/figures for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

TARGET = "IS_SEVERE"
RANDOM_STATE = 42


def load_artifact(root: Path) -> tuple[object, list[str], float]:
    model_path = root / "models" / "production" / "severe_model_test4_best_tactic.pkl"
    artifact = joblib.load(model_path)

    model = artifact.get("lgbm_model") or artifact.get("model")
    if model is None:
        raise RuntimeError("No valid model key found in artifact.")

    features = artifact.get("features_order")
    if not features:
        raise RuntimeError("features_order missing in production artifact.")

    threshold = float(artifact.get("threshold", 0.5))

    policy_path = root / "reports" / "metrics" / "test8_recall_first_policy.json"
    if policy_path.exists():
        with policy_path.open("r", encoding="utf-8") as f:
            policy = json.load(f)
        threshold = float(policy.get("selected_threshold", threshold))

    return model, list(features), threshold


def align_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = 0
    return out[features]


def ensure_figure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def save_pr_curve(y: np.ndarray, probs: np.ndarray, out: Path) -> None:
    precision, recall, _ = precision_recall_curve(y, probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision, lw=2.5, color="#006d77", label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall (Test Set)")
    plt.legend(loc="lower left")
    plt.savefig(out / "precision_recall_curve.png")
    plt.close()


def save_roc_curve(y: np.ndarray, probs: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, lw=2.5, color="#1d3557", label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC (Test Set)")
    plt.legend(loc="lower right")
    plt.savefig(out / "roc_curve.png")
    plt.close()


def save_confusion_matrix(y: np.ndarray, probs: np.ndarray, threshold: float, out: Path) -> None:
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"Matrice di Confusione (soglia={threshold:.2f})")
    plt.xlabel("Predizione")
    plt.ylabel("Valore Reale")
    plt.savefig(out / "confusion_matrix_recall_first.png")
    plt.close()


def save_feature_importance(model: object, features: list[str], out: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    imp = pd.DataFrame(
        {
            "feature": features,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values("importance", ascending=False)

    top = imp.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"], color="#2a9d8f")
    plt.xlabel("Importanza")
    plt.title("Top 20 Feature Importance - LightGBM")
    plt.savefig(out / "feature_importance_lgbm.png")
    plt.close()


def save_threshold_tradeoff(root: Path, out: Path) -> None:
    scan_path = root / "reports" / "metrics" / "test4_lgbm_threshold_scan.csv"
    if not scan_path.exists():
        return

    df = pd.read_csv(scan_path)
    keep = ["threshold", "precision", "recall", "f1"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["threshold"], df["precision"], label="Precision", color="#264653", lw=2)
    plt.plot(df["threshold"], df["recall"], label="Recall", color="#e76f51", lw=2)
    plt.plot(df["threshold"], df["f1"], label="F1", color="#2a9d8f", lw=2)
    plt.xlabel("Soglia")
    plt.ylabel("Valore metrica")
    plt.title("Trade-off Precision/Recall/F1 al variare della soglia")
    plt.legend(loc="best")
    plt.savefig(out / "threshold_tradeoff_curve.png")
    plt.close()


def save_probability_distributions(y: np.ndarray, probs: np.ndarray, out: Path) -> None:
    plot_df = pd.DataFrame({"target": y, "prob": probs})

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=plot_df[plot_df["target"] == 0],
        x="prob",
        bins=40,
        stat="density",
        element="step",
        fill=True,
        alpha=0.25,
        label="Classe 0",
        color="#5e6472",
    )
    sns.histplot(
        data=plot_df[plot_df["target"] == 1],
        x="prob",
        bins=40,
        stat="density",
        element="step",
        fill=True,
        alpha=0.25,
        label="Classe 1",
        color="#bc4749",
    )
    plt.xlabel("Probabilità predetta P(IS_SEVERE=1)")
    plt.ylabel("Densità")
    plt.title("Distribuzione delle probabilità predette per classe")
    plt.legend()
    plt.savefig(out / "probability_distribution_by_class.png")
    plt.close()


def save_calibration(y: np.ndarray, probs: np.ndarray, out: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")

    plt.figure(figsize=(8.5, 6))
    plt.plot(mean_pred, frac_pos, marker="o", lw=2, color="#457b9d", label="Modello")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfetta calibrazione")
    plt.xlabel("Probabilità media predetta")
    plt.ylabel("Frazione positiva osservata")
    plt.title("Curva di calibrazione (Reliability Plot)")
    plt.legend(loc="upper left")
    plt.savefig(out / "calibration_curve.png")
    plt.close()


def save_top_symptoms(root: Path, out: Path) -> None:
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"
    if not test_path.exists():
        return

    usecols = ["NUMERO_SINTOMI", TARGET]
    df = pd.read_csv(test_path, usecols=usecols)
    if "NUMERO_SINTOMI" not in df.columns:
        return

    tmp = df[["NUMERO_SINTOMI", TARGET]].copy()
    tmp["NUMERO_SINTOMI"] = pd.to_numeric(tmp["NUMERO_SINTOMI"], errors="coerce")
    tmp = tmp.dropna()

    plt.figure(figsize=(9, 6))
    sns.boxplot(
        data=tmp,
        x=TARGET,
        y="NUMERO_SINTOMI",
        hue=TARGET,
        palette=["#8ecae6", "#fb8500"],
        dodge=False,
        legend=False,
    )
    plt.xlabel("Classe target")
    plt.ylabel("Numero sintomi")
    plt.title("Numero sintomi per classe (VAERS)")
    plt.savefig(out / "num_symptoms_by_class_boxplot.png")
    plt.close()


def save_onset_distribution(test_df: pd.DataFrame, out: Path) -> None:
    if "NUMDAYS" not in test_df.columns:
        return
    tmp = pd.to_numeric(test_df["NUMDAYS"], errors="coerce").dropna()
    if tmp.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(tmp, bins=50, color="#3a86ff")
    plt.xlim(0, np.nanpercentile(tmp, 99))
    plt.xlabel("NUMDAYS")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione dei giorni all'insorgenza dei sintomi (Test Set)")
    plt.savefig(out / "onset_distribution_numdays.png")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "report" / "figures"
    out_legacy = root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    out_legacy.mkdir(parents=True, exist_ok=True)

    ensure_figure_style()

    model, features, threshold = load_artifact(root)
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"
    test_df = pd.read_csv(test_path)

    y = test_df[TARGET].astype(int).to_numpy()
    x = align_features(test_df.drop(columns=[TARGET]), features)
    probs = model.predict_proba(x)[:, 1]

    save_pr_curve(y, probs, out)
    save_roc_curve(y, probs, out)
    save_confusion_matrix(y, probs, threshold, out)
    save_feature_importance(model, features, out)
    save_threshold_tradeoff(root, out)
    save_probability_distributions(y, probs, out)
    save_calibration(y, probs, out)
    save_top_symptoms(root, out)
    save_onset_distribution(test_df, out)

    # Mirror to legacy path to keep older scripts/docs working.
    for png in out.glob("*.png"):
        shutil.copy2(png, out_legacy / png.name)

    print("Generated canonical figures in:", out)
    print("Updated legacy mirror in:", out_legacy)


if __name__ == "__main__":
    main()
