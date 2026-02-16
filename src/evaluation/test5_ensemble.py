# ======================================================
# TEST5 - THE AVENGERS ENSEMBLE (LITE) - SCRIPT VERSION
# Objective: Combine LGBM, XGB and MLP (CatBoost removed due to Py3.14 issue)
# ======================================================

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ======================================================
# 1) CONFIG & HYPERPARAMETERS
# ======================================================

TARGET = "IS_SEVERE"
N_SPLITS = 5
RANDOM_STATE = 42

print("Config loaded.")

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "data").exists() and (p / "models").exists():
            return p
    return cwd

ROOT = find_project_root()
data_dir = ROOT / "data" / "interim" / "splits"

train_path = data_dir / "train_step6.csv"
val_path = data_dir / "val_step6.csv"
test_path = data_dir / "test_step6.csv"

print(f"Loading data from {data_dir}...")
try:
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

X_train = df_train.drop(columns=[TARGET])
y_train = df_train[TARGET]

X_val   = df_val.drop(columns=[TARGET])
y_val   = df_val[TARGET]

X_test  = df_test.drop(columns=[TARGET])
y_test  = df_test[TARGET]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ======================================================
# 2) MODEL DEFINITIONS
# ======================================================

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# A. LightGBM (Champion from Test 4)
lgbm_params = {
    "n_estimators": 650,
    "learning_rate": 0.03,
    "num_leaves": 95,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 3.0,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1
}

# B. XGBoost (Diverse tree structure)
xgb_params = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "logloss",
    "n_jobs": -1
}

# D. MLP (Neural Network) - Needs Scaling!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

mlp_params = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 256,
    "learning_rate_init": 0.001,
    "max_iter": 200,
    "random_state": RANDOM_STATE,
    "early_stopping": True
}

print("Models configured (LGBM, XGB, MLP).")

# ======================================================
# 3) TRAINING BASE LEARNERS
# ======================================================

models = {}
preds_val = {}
preds_test = {}

# --- LightGBM ---
print("Training LightGBM...")
lgbm = LGBMClassifier(**lgbm_params)
lgbm.fit(X_train, y_train)
preds_val['lgbm'] = lgbm.predict_proba(X_val)[:, 1]
preds_test['lgbm'] = lgbm.predict_proba(X_test)[:, 1]

# --- XGBoost ---
print("Training XGBoost...")
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train, y_train)
preds_val['xgb'] = xgb.predict_proba(X_val)[:, 1]
preds_test['xgb'] = xgb.predict_proba(X_test)[:, 1]

# --- MLP ---
print("Training MLP...")
mlp = MLPClassifier(**mlp_params)
mlp.fit(X_train_scaled, y_train)
preds_val['mlp'] = mlp.predict_proba(X_val_scaled)[:, 1]
preds_test['mlp'] = mlp.predict_proba(X_test_scaled)[:, 1]

print("All models trained.")

# ======================================================
# 4) ENSEMBLE OPTIMIZATION
# ======================================================

print("Optimizing Ensemble Weights...")

model_names = ['lgbm', 'xgb', 'mlp']
val_matrix = np.column_stack([preds_val[m] for m in model_names])
test_matrix = np.column_stack([preds_test[m] for m in model_names])

def loss_func(weights):
    # Normalize weights
    w = weights / np.sum(weights)
    final_prob = np.dot(val_matrix, w)
    # Maximize ROC-AUC
    return -roc_auc_score(y_val, final_prob)

init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1)] * len(model_names)
cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})

res = minimize(loss_func, init_weights, bounds=bounds, constraints=cons, method='SLSQP')
best_weights = res.x / np.sum(res.x)

print("Best Weights (LGBM, XGB, MLP):")
print(best_weights)

# Combine Predictions
final_test_prob = np.dot(test_matrix, best_weights)
final_val_prob = np.dot(val_matrix, best_weights)

# ======================================================
# 5) THRESHOLD TUNING & EVALUATION
# ======================================================

THRESHOLDS = np.arange(0.1, 0.95, 0.01)
best_thr = 0.5
best_prec = 0
best_f1 = 0
target_recall = 0.58

# Strategy: Maximize Precision s.t. Recall >= 0.58
for t in THRESHOLDS:
    pred = (final_val_prob >= t).astype(int)
    rec = recall_score(y_val, pred, zero_division=0)
    prec = precision_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)
    
    if rec >= target_recall:
        if prec > best_prec:
            best_prec = prec
            best_f1 = f1
            best_thr = t

print(f"Selected Threshold: {best_thr:.2f} (Val Precision: {best_prec:.4f}, Val F1: {best_f1:.4f})")

final_pred_test = (final_test_prob >= best_thr).astype(int)

print("\n===== CLASSIFICATION REPORT (TEST - ENSEMBLE) =====")
print(classification_report(y_test, final_pred_test, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_pred_test))

# Individual Models Performance for Comparison
print("\n--- Single Models AUC ---")
for m in model_names:
    auc = roc_auc_score(y_test, preds_test[m])
    print(f"{m.upper()}: {auc:.4f}")
    
print(f"ENSEMBLE: {roc_auc_score(y_test, final_test_prob):.4f}")
