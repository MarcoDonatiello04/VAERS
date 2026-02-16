from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "production" / "severe_model_stacking_production.pkl"

try:
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        print("KEYS:", list(obj.keys()))
    else:
        print("TYPE:", type(obj))
except Exception as e:
    print("ERROR:", e)
