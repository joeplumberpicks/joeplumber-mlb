# joeplumber-mlb

Colab-first MLB data and feature pipeline for game-level and player-prop modeling.

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
   ```

## Output location

All outputs are written under Drive root configured in `config/project.yaml`:

- `/content/drive/MyDrive/joeplumber-mlb/data/processed/`

Key files produced by Milestone 1 + 2A include:

- `games.parquet`, `parks.parquet`, `pa.parquet`, `weather_game.parquet`
- `events_pa.parquet`, `game_runs.parquet`
- `batter_game.parquet`, `pitcher_game.parquet`, `team_game.parquet`, `model_spine_game.parquet`
- `batter_rolling.parquet`, `pitcher_rolling.parquet`, `team_rolling.parquet`
