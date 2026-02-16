#!/usr/bin/env python3
"""
Apply targeted scaling on Step6 datasets.

Requested transformations:
- AGE_YRS -> MinMax scaling
- NUMERO_SINTOMI, NUMDAYS -> Standardization (z-score)

Fit is performed ONLY on train_step6, then applied to val/test and SMOTE train.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


AGE_COL = "AGE_YRS"
STD_COLS = ["NUMERO_SINTOMI", "NUMDAYS"]


def describe_cols(df: pd.DataFrame, cols: list[str], title: str) -> None:
    print(f"\n{title}")
    print(df[cols].describe().loc[["mean", "std", "min", "max"]])


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    train_path = root / "data" / "interim" / "splits" / "train_step6.csv"
    val_path = root / "data" / "interim" / "splits" / "val_step6.csv"
    test_path = root / "data" / "interim" / "splits" / "test_step6.csv"
    smote_path = root / "data" / "processed" / "smote" / "train_step6_SMOTE.csv"

    paths = [train_path, val_path, test_path, smote_path]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    smote_df = pd.read_csv(smote_path)

    required = [AGE_COL, *STD_COLS]
    for name, df in [
        ("train_step6", train_df),
        ("val_step6", val_df),
        ("test_step6", test_df),
        ("train_step6_SMOTE", smote_df),
    ]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    describe_cols(train_df, required, "Train stats BEFORE scaling")

    # Fit on TRAIN ONLY.
    age_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    age_scaler.fit(train_df[[AGE_COL]])
    std_scaler.fit(train_df[STD_COLS])

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[AGE_COL] = age_scaler.transform(out[[AGE_COL]])
        out[STD_COLS] = std_scaler.transform(out[STD_COLS])
        return out

    train_scaled = transform(train_df)
    val_scaled = transform(val_df)
    test_scaled = transform(test_df)
    smote_scaled = transform(smote_df)

    describe_cols(train_scaled, required, "Train stats AFTER scaling")

    # One-time backups of original files.
    for p in paths:
        backup = p.with_name(f"{p.stem}_backup_before_target_scaling.csv")
        if not backup.exists():
            p.replace(backup)
            print(f"Backup created: {backup}")
        else:
            print(f"Backup already exists, keeping it: {backup}")

    # Write transformed datasets to original paths.
    train_scaled.to_csv(train_path, index=False)
    val_scaled.to_csv(val_path, index=False)
    test_scaled.to_csv(test_path, index=False)
    smote_scaled.to_csv(smote_path, index=False)

    scaler_out = root / "models" / "checkpoints" / "step6_target_scalers.pkl"
    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "age_scaler": age_scaler,
            "std_scaler": std_scaler,
            "age_col": AGE_COL,
            "std_cols": STD_COLS,
        },
        scaler_out,
    )
    print(f"Saved scalers: {scaler_out}")
    print("\nDone. Updated datasets in-place with requested scaling.")


if __name__ == "__main__":
    main()
