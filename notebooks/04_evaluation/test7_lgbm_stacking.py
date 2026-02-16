#!/usr/bin/env python3
from runpy import run_path
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    target = root / "src" / "evaluation" / Path(__file__).name
    run_path(str(target), run_name="__main__")
