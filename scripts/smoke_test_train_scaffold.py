from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import write_parquet
from src.utils.train_scaffold import run_training_scaffold


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="train_scaffold_smoke_") as tmp:
        root = Path(tmp)
        dirs = {
            "marts_dir": root / "marts",
            "models_dir": root / "models",
            "backtests_dir": root / "backtests",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        mart = pd.DataFrame(
            {
                "game_pk": [1, 2, 3, 4],
                "target_dummy": [0, 1, 0, 1],
                "num_with_nan": [1.0, np.nan, 3.0, 4.0],
                "obj_col": ["a", "b", "c", "d"],
                "inf_col": [0.0, np.inf, 2.0, -np.inf],
            }
        )
        mart_path = dirs["marts_dir"] / "smoke.parquet"
        write_parquet(mart, mart_path)

        report = run_training_scaffold("smoke_engine", "smoke.parquet", "target_dummy", dirs)
        assert report["status"] == "TRAINED", f"Expected TRAINED status, got {report}"

        model_dir = dirs["models_dir"] / "smoke_engine"
        report_dir = dirs["backtests_dir"] / "smoke_engine"
        assert list(model_dir.glob("*.joblib")), "Expected a trained model joblib artifact"
        assert list(report_dir.glob("*.json")), "Expected a backtest/report json artifact"

    print("Smoke test passed: train scaffold handles NaN/inf/object and persists artifacts.")


if __name__ == "__main__":
    run_smoke()
