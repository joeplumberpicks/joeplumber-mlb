from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header
from src.utils.team_ids import normalize_team_abbr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build standardized live lineups for hit props.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _clean_player_name(x: object) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    s = str(x).strip()
    if not s:
        return None

    # Remove common junk prefixes seen in dirty projected lineup rows
    s = re.sub(r"^\s*NONE\s+", "", s, flags=re.IGNORECASE)

    # Remove leading position tags like "LF ", "CF ", "1B ", "DH ", etc.
    s = re.sub(
        r"^\s*(C|1B|2B|3B|SS|LF|CF|RF|DH|SP|RP)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Filter obvious non-player labels
    bad_tokens = {
        "confirmed lineup",
        "expected lineup",
        "projected lineup",
        "starting pitcher intel",
        "none",
    }
    if s.lower() in bad_tokens:
        return None

    return s or None


def _safe_series(df: pd.DataFrame, col: str | None, default=None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col]


def _normalize_team_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: normalize_team_abbr(x) if pd.notna(x) else x)


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_live_lineups.log")
    log_header("scripts/live/build_live_lineups.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(
            f"Live spine not found: {spine_path}. Build it first with scripts/live/build_spine_from_schedule.py"
        )

    live_dir = dirs["processed_dir"] / "live"
    lineup_candidates = [
        live_dir / f"confirmed_lineups_{args.season}_{args.date}.parquet",
        live_dir / f"projected_lineups_{args.season}_{args.date}.parquet",
        live_dir / f"lineups_{args.season}_{args.date}.parquet",
    ]
    src = next((p for p in lineup_candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"No lineup source found for {args.season} {args.date}. Expected one of: {[str(p) for p in lineup_candidates]}"
        )

    lu = pd.read_parquet(src).copy()
    spine = pd.read_parquet(spine_path).copy()

    logging.info("live_lineups source=%s raw_rows=%s source_cols=%s", src, len(lu), sorted(lu.columns))

    game_pk_col = _pick(list(lu.columns), ["game_pk", "game_id"])
    batter_id_col = _pick(list(lu.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(lu.columns), ["player_name", "batter_name", "name"])
    team_col = _pick(list(lu.columns), ["batter_team", "team", "batting_team", "team_abbrev", "team_name"])
    slot_col = _pick(list(lu.columns), ["lineup_slot", "batting_order", "order", "lineup_position"])

    if team_col is None:
        raise ValueError(f"Lineup source missing required team column. columns={sorted(lu.columns)}")

    out = pd.DataFrame(index=lu.index)

    if game_pk_col:
        out["game_pk"] = pd.to_numeric(lu[game_pk_col], errors="coerce").astype("Int64")
    else:
        out["game_pk"] = pd.Series(pd.NA, index=lu.index, dtype="Int64")

    if batter_id_col:
        out["batter_id"] = pd.to_numeric(lu[batter_id_col], errors="coerce").astype("Int64")
    else:
        out["batter_id"] = pd.Series(pd.NA, index=lu.index, dtype="Int64")

    raw_name = _safe_series(lu, name_col, default=None)
    out["player_name"] = raw_name.map(_clean_player_name)

    out["batter_team"] = _normalize_team_series(_safe_series(lu, team_col, default=None).astype("object"))

    if slot_col:
        out["lineup_slot"] = pd.to_numeric(lu[slot_col], errors="coerce")
    else:
        out["lineup_slot"] = np.nan

    spine_view = spine[
        [c for c in ["game_pk", "game_date", "season", "home_team", "away_team"] if c in spine.columns]
    ].copy()
    spine_view["game_pk"] = pd.to_numeric(spine_view["game_pk"], errors="coerce").astype("Int64")

    for c in ["home_team", "away_team"]:
        if c in spine_view.columns:
            spine_view[c] = _normalize_team_series(spine_view[c].astype("object"))

    spine_view = spine_view.drop_duplicates(subset=["game_pk"], keep="last")

    out = out.merge(spine_view, on="game_pk", how="left")

    # fallback game_pk from team match if missing in lineup source
    if out["game_pk"].isna().any() and {"home_team", "away_team"}.issubset(spine_view.columns):
        team_games = pd.concat(
            [
                spine_view[["game_pk", "home_team"]].rename(columns={"home_team": "batter_team"}),
                spine_view[["game_pk", "away_team"]].rename(columns={"away_team": "batter_team"}),
            ],
            ignore_index=True,
            sort=False,
        ).drop_duplicates(subset=["batter_team"], keep="last")

        missing_mask = out["game_pk"].isna() & out["batter_team"].notna()
        if missing_mask.any():
            mapped = out.loc[missing_mask, ["batter_team"]].merge(team_games, on="batter_team", how="left")
            out.loc[missing_mask, "game_pk"] = pd.to_numeric(mapped["game_pk"], errors="coerce").astype("Int64").values

        out = out.drop(columns=[c for c in ["game_date", "season", "home_team", "away_team"] if c in out.columns], errors="ignore")
        out = out.merge(spine_view, on="game_pk", how="left")

    home_team_series = _safe_series(out, "home_team", default=None).astype("object")
    away_team_series = _safe_series(out, "away_team", default=None).astype("object")

    out["home_away"] = np.where(
        out["batter_team"].astype("object") == home_team_series,
        1.0,
        np.where(
            out["batter_team"].astype("object") == away_team_series,
            0.0,
            np.nan,
        ),
    )

    out["opponent_team"] = np.where(
        out["home_away"] == 1.0,
        away_team_series,
        np.where(out["home_away"] == 0.0, home_team_series, None),
    )

    out["game_date"] = pd.to_datetime(_safe_series(out, "game_date", default=None), errors="coerce").dt.strftime("%Y-%m-%d")
    out["season"] = pd.to_numeric(_safe_series(out, "season", default=args.season), errors="coerce").fillna(args.season).astype("Int64")

    out["lineup_is_usable"] = (
        out["game_pk"].notna()
        & out["batter_team"].notna()
        & out["player_name"].notna()
    )

    debug_cols = [
        "game_pk",
        "game_date",
        "season",
        "batter_id",
        "player_name",
        "batter_team",
        "opponent_team",
        "lineup_slot",
        "home_away",
        "lineup_is_usable",
    ]
    debug_out = out[debug_cols].copy()

    debug_path = live_dir / f"lineups_debug_{args.season}_{args.date}.parquet"
    debug_out.to_parquet(debug_path, index=False)

    logging.info(
        "live_lineups pre-drop rows=%s missing_game_pk=%s missing_batter_id=%s missing_player_name=%s missing_batter_team=%s usable_rows=%s debug_out=%s",
        len(debug_out),
        int(debug_out["game_pk"].isna().sum()),
        int(debug_out["batter_id"].isna().sum()),
        int(debug_out["player_name"].isna().sum()),
        int(debug_out["batter_team"].isna().sum()),
        int(debug_out["lineup_is_usable"].sum()),
        debug_path,
    )

    cols = [
        "game_pk",
        "game_date",
        "season",
        "batter_id",
        "player_name",
        "batter_team",
        "opponent_team",
        "lineup_slot",
        "home_away",
    ]

    # Keep the main output contract strict for downstream hitters workflows.
    final_out = (
        out[cols]
        .dropna(subset=["game_pk", "batter_id", "batter_team"])
        .drop_duplicates(subset=["game_pk", "batter_id"], keep="last")
        .copy()
    )

    out_path = live_dir / f"lineups_{args.season}_{args.date}.parquet"
    final_out.to_parquet(out_path, index=False)

    pct_slot = float(pd.to_numeric(final_out["lineup_slot"], errors="coerce").notna().mean()) if len(final_out) else 0.0
    logging.info(
        "live_lineups built rows=%s unique_games=%s unique_hitters=%s pct_with_lineup_slot=%.4f src=%s out=%s",
        len(final_out),
        int(final_out["game_pk"].nunique()) if len(final_out) else 0,
        int(final_out["batter_id"].nunique()) if len(final_out) else 0,
        pct_slot,
        src,
        out_path,
    )

    print(f"lineups_debug_out={debug_path}")
    print(f"lineups_out={out_path}")


if __name__ == "__main__":
    main()
