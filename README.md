# Joe Plumber MLB Engine v1.0 (Unified Stack)

Joe Plumber MLB Engine v1.0 is a deterministic, reproducible MLB scaffold pipeline for feature engineering, marts, training, and slate outputs.

## Architecture Principles

- **GitHub stores only code/config/docs** (no datasets, models, logs, outputs committed).
- **Google Drive is the data lake** for all heavy artifacts:
  - raw/processed tables
  - feature marts
  - models
  - backtests/reports
  - outputs/logs
- **One centralized path resolver** (`resolve_data_dirs`) controls all IO roots.
- **Leakage-safe rolling features** use `shift(1)` before any rolling window operations.
- **Colab-first with local fallback**: if Drive is unavailable, pipeline writes to local repo `data/`.

## Default Drive Data Root

`/content/drive/MyDrive/joeplumber-mlb/data`

(Resolved from config key `drive_data_root: "joeplumber-mlb/data"`.)

## Colab Bootstrap

### 1) Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2) Clone repo
```bash
!git clone https://github.com/joeplumberpicks/joeplumber-mlb.git
%cd joeplumber-mlb
```

### 3) Install dependencies
```bash
!pip install -r requirements.txt
```

### 4) Print resolved data dirs
```python
from pathlib import Path
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

config = load_config(Path('configs/project.yaml'))
dirs = resolve_data_dirs(config=config, prefer_drive=True)
for k, v in dirs.items():
    print(k, v)
```

### 5) Run scaffold pipeline for season 2024
```bash
!python scripts/build_spine.py --season 2024 --force
!python scripts/build_features_rolling.py
!python scripts/build_marts.py
```

## Expected Outputs

- `processed/by_season/games_<season>.parquet`
- `processed/by_season/pa_<season>.parquet`
- `processed/by_season/weather_game_<season>.parquet`
- `processed/by_season/parks_<season>.parquet`
- `processed/model_spine_game.parquet`
- `processed/batter_game_rolling.parquet`
- `processed/pitcher_game_rolling.parquet`
- `marts/hr_features.parquet`
- `marts/nrfi_features.parquet`
- `marts/moneyline_features.parquet`
- `marts/hitter_props_features.parquet`
- `marts/pitcher_props_features.parquet`
- `models/<engine>/*.joblib`
- `backtests/<engine>/*.json`
- `outputs/<engine>/*.csv`

## Script Entry Points

- `python scripts/build_spine.py --season 2024`
- `python scripts/build_features_rolling.py`
- `python scripts/build_marts.py`
- `python scripts/train/train_hr_ranker.py`
- `python scripts/train/train_nrfi_xgb.py`
- `python scripts/train/train_moneyline_sim.py`
- `python scripts/train/train_hitter_props.py`
- `python scripts/train/train_pitcher_props.py`
- `python scripts/slate/run_slate.py`
