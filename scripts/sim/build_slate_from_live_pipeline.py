%%bash
cd /content/joeplumber-mlb

cat > scripts/sim/build_slate_from_live_pipeline.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any of:\n" + "\n".join(str(p) for p in paths))


def team_col(df: pd.DataFrame) -> str:
    for col in ["team", "team_abbr", "batting_team"]:
        if col in df.columns:
            return col
    raise ValueError(f"No team column found. Columns: {list(df.columns)}")


def order_col(df: pd.DataFrame) -> str:
    for col in ["batting_order", "lineup_slot", "slot"]:
        if col in df.columns:
            return col
    raise ValueError(f"No batting order column found. Columns: {list(df.columns)}")


def hitter_id_col(df: pd.DataFrame) -> str:
    for col in ["player_id", "batter_id", "mlb_id"]:
        if col in df.columns:
            return col
    raise ValueError(f"No hitter id column found. Columns: {list(df.columns)}")


def pitcher_id_col(df: pd.DataFrame) -> str:
    for col in ["pitcher_id", "player_id", "mlb_id"]:
        if col in df.columns:
            return col
    raise ValueError(f"No pitcher id column found. Columns: {list(df.columns)}")


def pick_lineup(df_conf, df_proj, game_pk, team):
    if df_conf is not None:
        tc = team_col(df_conf)
        sub = df_conf[(df_conf["game_pk"] == game_pk) & (df_conf[tc] == team)].copy()
        if len(sub) >= 9:
            sub["_lineup_source"] = "confirmed"
            return sub

    if df_proj is not None:
        tc = team_col(df_proj)
        sub = df_proj[(df_proj["game_pk"] == game_pk) & (df_proj[tc] == team)].copy()
        if len(sub) >= 9:
            sub["_lineup_source"] = "projected"
            return sub

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    raw_dir = Path(dirs["raw_dir"])
    outputs_dir = Path(dirs["outputs_dir"])

    live_processed = processed_dir / "live"
    live_raw = raw_dir / "live"

    spine_path = live_processed / f"model_spine_game_{args.season}_{args.date}.parquet"
    spine = read_parquet(spine_path)

    proj_path = first_existing([
        live_raw / f"projected_lineups_{args.season}_{args.date}.parquet",
        live_processed / f"projected_lineups_{args.season}_{args.date}.parquet",
    ])
    projected = read_parquet(proj_path)

    confirmed = None
    for path in [
        live_raw / f"confirmed_lineups_{args.season}_{args.date}.parquet",
        live_processed / f"confirmed_lineups_{args.season}_{args.date}.parquet",
    ]:
        if path.exists():
            confirmed = pd.read_parquet(path)
            break

    starters_path = first_existing([
        live_raw / f"starting_pitchers_{args.season}_{args.date}.parquet",
        live_processed / f"starting_pitchers_{args.season}_{args.date}.parquet",
    ])
    starters = read_parquet(starters_path)

    rows = []

    for _, game in spine.iterrows():
        game_pk = game["game_pk"]
        away_team = game["away_team"]
        home_team = game["home_team"]

        away_lu = pick_lineup(confirmed, projected, game_pk, away_team)
        home_lu = pick_lineup(confirmed, projected, game_pk, home_team)

        if away_lu is None or home_lu is None:
            print(f"[SKIP] Missing lineup: {away_team} @ {home_team}")
            continue

        away_order = order_col(away_lu)
        home_order = order_col(home_lu)
        away_hit_col = hitter_id_col(away_lu)
        home_hit_col = hitter_id_col(home_lu)

        away_ids = (
            away_lu.sort_values(away_order)[away_hit_col]
            .dropna()
            .astype(int)
            .astype(str)
            .tolist()
        )
        home_ids = (
            home_lu.sort_values(home_order)[home_hit_col]
            .dropna()
            .astype(int)
            .astype(str)
            .tolist()
        )

        if len(away_ids) < 9 or len(home_ids) < 9:
            print(f"[SKIP] Missing hitter IDs: {away_team} @ {home_team} away={len(away_ids)} home={len(home_ids)}")
            continue

        st_team_col = team_col(starters)
        st_pid_col = pitcher_id_col(starters)

        away_sp = starters[(starters["game_pk"] == game_pk) & (starters[st_team_col] == away_team)]
        home_sp = starters[(starters["game_pk"] == game_pk) & (starters[st_team_col] == home_team)]

        if away_sp.empty or home_sp.empty:
            print(f"[SKIP] Missing starter: {away_team} @ {home_team}")
            continue

        away_pid = away_sp.iloc[0][st_pid_col]
        home_pid = home_sp.iloc[0][st_pid_col]

        if pd.isna(away_pid) or pd.isna(home_pid):
            print(f"[SKIP] Unresolved starter ID: {away_team} @ {home_team}")
            continue

        rows.append({
            "game_pk": game_pk,
            "game_date": args.date,
            "away_team": away_team,
            "home_team": home_team,
            "away_batters": ",".join(away_ids[:9]),
            "home_batters": ",".join(home_ids[:9]),
            "away_pitcher_id": int(away_pid),
            "home_pitcher_id": int(home_pid),
            "lineup_source_away": away_lu["_lineup_source"].iloc[0],
            "lineup_source_home": home_lu["_lineup_source"].iloc[0],
        })

    out_dir = outputs_dir / "slate_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(rows)
    out_path = out_dir / f"slate_input_{args.date}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved slate input: {out_path}")
    print(f"Sim-ready games: {len(out)}")
    if len(out):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
PY

chmod +x scripts/sim/build_slate_from_live_pipeline.py
echo "Created scripts/sim/build_slate_from_live_pipeline.py"
