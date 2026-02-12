# joeplumber-mlb

Colab-first MLB data and feature pipeline for game-level and player-prop modeling.
Milestone 1 foundation for a Colab-first MLB modeling repo that writes spine tables to Google Drive.

## Colab quickstart

1. **Mount Drive in a notebook cell (required before scripts):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Clone the repo:**
   ```bash
   !git clone <your-repo-url>
   %cd joeplumber-mlb
   ```
3. **Install dependencies:**
   ```bash
   !pip install -r requirements.txt
   ```

## Backbone workflow

Run scripts in this exact order:

1. Build schedule spine
   ```bash
   !python scripts/build_spine.py --season 2024 --start 2024-03-20 --end 2024-03-22
   ```
2. Build plate-appearance events + game run summaries
   ```bash
   !python scripts/build_events.py --season 2024 --start 2024-03-20 --end 2024-03-22 --max_games 20 --force
   ```
3. Build aggregated marts
   ```bash
   !python scripts/build_marts.py --season 2024 --start 2024-03-20 --end 2024-03-22
   ```
4. Build rolling features
   ```bash
   !python scripts/build_features_rolling.py --season 2024
4. **Build the spine:**
   ```bash
   !python scripts/build_spine.py --season 2024 --start 2024-03-20 --end 2024-09-30
   ```

## Output location

All outputs are written under Drive root configured in `config/project.yaml`:

- `/content/drive/MyDrive/joeplumber-mlb/data/processed/`

Key files produced by Milestone 1 + 2A include:

- `games.parquet`, `parks.parquet`, `pa.parquet`, `weather_game.parquet`
- `events_pa.parquet`, `game_runs.parquet`
- `batter_game.parquet`, `pitcher_game.parquet`, `team_game.parquet`, `model_spine_game.parquet`
- `batter_rolling.parquet`, `pitcher_rolling.parquet`, `team_rolling.parquet`
Spine outputs are written to the processed folder under Drive root configured in `config/project.yaml`:

- `/content/drive/MyDrive/joeplumber-mlb/data/processed/games.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/pa.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/weather_game.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/parks.parquet`


## Statcast pitch ingestion

Pull pitch-by-pitch Statcast data for a season:

```bash
python scripts/pull_statcast_pitches.py --season 2024
```

Useful flags:
- `--start YYYY-MM-DD --end YYYY-MM-DD` to scope a date window
- `--resume` to skip chunk parquet files that already exist
- `--force` to re-pull and overwrite chunk parquet files

## Statcast feature layers (Pitch Mix, Hitter Pitch-Type, PPMI)

These steps build pitch-level Statcast feature tables used for pitch-type matchups and pitch-profile matching.

### 1) Pull Statcast pitches (pitch-by-pitch)

```bash
python scripts/pull_statcast_pitches.py --season 2024
```

### 2) Build pitcher pitch mix + quality features

```bash
python scripts/build_pitcher_mix.py --season 2024
```

### 3) Build hitter pitch-type performance features

```bash
python scripts/build_hitter_pitchtype.py --season 2024
```

### 4) Build PPMI matchup table

```bash
python scripts/build_ppmi_matchup.py --season 2024
```

Outputs are written to:
- `data/processed/statcast/pitcher_mix_{season}.parquet`
- `data/processed/statcast/hitter_pitchtype_{season}.parquet`
- `data/processed/statcast/ppmi_matchup_{season}.parquet`


### 5) Build Statcast game context (starter-level)

```bash
python scripts/build_statcast_game_context.py --season 2024
```

Useful flags:
- `--max-games N` for quick validation runs
- `--allow-partial` to bypass hard fail on low starter coverage


## Build everything for season

Run the full orchestration pipeline (core processed tables, canonical spine, and Statcast layers):

```bash
python scripts/build_season_all.py --season 2024
```

Optional flags:
- `--start YYYY-MM-DD --end YYYY-MM-DD` to constrain date range
- `--max-games N` for quick test runs
- `--skip-statcast` to build core + canonical spine only

Then merge canonical spine + starter context into a shared game-level feature table:

```bash
python scripts/build_model_features_game.py --season 2024
```

Build bullpen / relief context features:

```bash
python scripts/build_bullpen_context.py --season 2024
```

Build model features with bullpen merge enabled:

```bash
python scripts/build_model_features_game.py --season 2024 --with-bullpen
```

Build offense discipline features:

```bash
python scripts/build_offense_discipline.py --season 2024
```

Build model features with bullpen + offense merges enabled:

```bash
python scripts/build_model_features_game.py --season 2024 --with-bullpen --with-offense
```

Build standardized game-level targets:

```bash
python scripts/build_game_targets.py --season 2024
```


Train the first NRFI/YRFI model:

```bash
python scripts/train_yrfi.py --season 2024
```

Run inference with calibrated probabilities:

```bash
python scripts/infer_yrfi.py --season 2024
```

Train Moneyline (home win) model:

```bash
python scripts/train_moneyline.py --season 2024
```

Run Moneyline inference with calibrated probabilities:

```bash
python scripts/infer_moneyline.py --season 2024
```

Train Totals (O/U) model:

```bash
python scripts/train_totals.py --season 2024
```

Run Totals inference:

```bash
python scripts/infer_totals.py --season 2024
```


Build pitcher-game targets:

```bash
python scripts/build_pitcher_game_targets.py --season 2024
```

Build starter pitcher-game features:

```bash
python scripts/build_pitcher_game_features.py --season 2024 --allow-partial
```

Train pitcher Ks model:

```bash
python scripts/train_pitcher_ks.py --season 2024
```

Run pitcher Ks inference:

```bash
python scripts/infer_pitcher_ks.py --season 2024
```


Train pitcher BB model:

```bash
python scripts/train_pitcher_bb.py --season 2024
```

Run pitcher BB inference:

```bash
python scripts/infer_pitcher_bb.py --season 2024
```

Train pitcher Outs model:

```bash
python scripts/train_pitcher_outs.py --season 2024
```

Run pitcher Outs inference:

```bash
python scripts/infer_pitcher_outs.py --season 2024
```

Train pitcher ER model:

```bash
python scripts/train_pitcher_er.py --season 2024
```

Run pitcher ER inference:

```bash
python scripts/infer_pitcher_er.py --season 2024
```


Build hitter-game targets:

```bash
python scripts/build_hitter_game_targets.py --season 2024
```

Build hitter-game features:

```bash
python scripts/build_hitter_game_features.py --season 2024 --allow-partial
```

Train hitter HR model:

```bash
python scripts/train_hitter_hr.py --season 2024
```

Run hitter HR inference:

```bash
python scripts/infer_hitter_hr.py --season 2024
```

Train hitter 1+ Hit model:

```bash
python scripts/train_hitter_hit1p.py --season 2024
```

Run hitter 1+ Hit inference:

```bash
python scripts/infer_hitter_hit1p.py --season 2024
```

Train hitter Total Bases model:

```bash
python scripts/train_hitter_tb.py --season 2024
```

Run hitter Total Bases inference:

```bash
python scripts/infer_hitter_tb.py --season 2024
```

Train hitter RBI model:

```bash
python scripts/train_hitter_rbi.py --season 2024
```

Run hitter RBI inference:

```bash
python scripts/infer_hitter_rbi.py --season 2024
```

Train hitter Walks model:

```bash
python scripts/train_hitter_bb.py --season 2024
```

Run hitter Walks inference:

```bash
python scripts/infer_hitter_bb.py --season 2024
```


## Weather layer

Weather park mapping is season-aware and supports per-game venue overrides.
Default mappings live in `data/reference/mlb_stadiums.csv`, and special-series overrides
(e.g., Athletics home series in Las Vegas) can be set in `data/reference/park_overrides.csv`.

Build game-level weather features (hour nearest first pitch):

```bash
python scripts/build_weather_game.py --season 2024
```

Train weather effects models and create per-game weather factors:

```bash
python scripts/train_weather_factors.py --season 2024
```

Merge shared game features with weather + weather factors:

```bash
python scripts/build_model_features_game.py --season 2024 --with-weather --with-weather-factors
```



## Park layer

Build deterministic per-game park reference table:

```bash
python scripts/build_park_game.py --season 2024
```

Train leakage-safe per-game park factors (HR / runs / YRFI):

```bash
python scripts/train_park_factors.py --season 2024
```

Merge shared game features with park metadata + park factors:

```bash
python scripts/build_model_features_game.py --season 2024 --with-park --with-park-factors
```

Default outputs:
- `data/processed/park_game_{season}.parquet`
- `data/processed/park_factors_game_{season}.parquet`
