# JoePlumber MLB Pipeline Commands

## Single season (2024)

```bash
python scripts/ingest/run_ingest_parks.py --season 2024 --force
python scripts/ingest/run_ingest_season.py --season 2024 --chunk-days 7 --force
python scripts/build_spine.py --season 2024 --force --allow-partial
python scripts/build_game_logs_from_pitches.py --season 2024 --force
python scripts/build_features_rolling.py
python scripts/build_marts.py
python scripts/validate_season_artifacts.py --season 2024
```

## Multi-season loop

```bash
for s in 2019 2020 2021 2022 2023 2024 2025 2026; do
  python scripts/ingest/run_ingest_parks.py --season "$s" --force
  python scripts/ingest/run_ingest_season.py --season "$s" --chunk-days 7
  python scripts/build_spine.py --season "$s" --allow-partial
  python scripts/build_game_logs_from_pitches.py --season "$s"
done
python scripts/build_features_rolling.py
python scripts/build_marts.py
```

## Expected outputs
- `processed/by_season/batter_game_<season>.parquet`
- `processed/by_season/pitcher_game_<season>.parquet`
- `processed/by_season/pitch_agg_matchup_<season>.parquet`
- `processed/batter_game_rolling.parquet`
- `processed/pitcher_game_rolling.parquet`
- `processed/model_spine_game.parquet`
