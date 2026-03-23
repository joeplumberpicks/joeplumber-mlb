from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build projected live lineups for early-day hit prop runs.")
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


def _norm_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _resolve_player_ids(out: pd.DataFrame, batter_path: Path, slate_date: pd.Timestamp) -> pd.DataFrame:
    if not batter_path.exists() or out.empty:
        return out

    try:
        batter = pd.read_parquet(batter_path).copy()
    except Exception:
        logging.exception("projected lineups failed loading batter rolling file for id resolution: %s", batter_path)
        return out

    team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name"])
    if team_col is None or bid_col is None or name_col is None:
        return out

    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter = batter[batter["game_date"] < slate_date].copy()
    if batter.empty:
        return out

    batter["team"] = batter[team_col].astype(str)
    batter["player_id"] = pd.to_numeric(batter[bid_col], errors="coerce").astype("Int64")
    batter["name_norm"] = _norm_name(batter[name_col])
    lookup = (
        batter.sort_values(["game_date"])
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["team", "name_norm"], keep="last")[["team", "name_norm", "player_id"]]
    )
    if lookup.empty:
        return out

    out["name_norm"] = _norm_name(out["player_name"])
    to_fill = pd.to_numeric(out["player_id"], errors="coerce").isna()
    if to_fill.any():
        resolved = out.loc[to_fill, ["team", "name_norm"]].merge(lookup, on=["team", "name_norm"], how="left")["player_id"]
        out.loc[to_fill, "player_id"] = pd.to_numeric(resolved, errors="coerce").astype("Int64").values
    return out.drop(columns=["name_norm"], errors="ignore")


def _find_projected_source_files(raw_live_dir: Path, processed_live_dir: Path, season: int, slate_date: str) -> list[Path]:
    candidates = [
        processed_live_dir / f"projected_lineups_raw_{season}_{slate_date}.parquet",
        raw_live_dir / f"projected_lineups_{season}_{slate_date}.parquet",
        raw_live_dir / f"lineups_projected_{season}_{slate_date}.parquet",
        raw_live_dir / f"projected_lineups_{slate_date}.parquet",
        raw_live_dir / f"lineups_projected_{slate_date}.parquet",
    ]
    found: list[Path] = [p for p in candidates if p.exists()]
    if found:
        return found

    glob_hits = sorted(raw_live_dir.glob(f"*{season}*{slate_date}*projected*lineup*.parquet"))
    if not glob_hits:
        glob_hits = sorted(raw_live_dir.glob(f"*{slate_date}*projected*lineup*.parquet"))
    return glob_hits


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_projected_lineups.log")
    log_header("scripts/live/build_projected_lineups.py", repo_root, config_path, dirs)

    processed_live_dir = dirs["processed_dir"] / "live"
    raw_live_dir = dirs["raw_dir"] / "live"
    processed_live_dir.mkdir(parents=True, exist_ok=True)

    src_files = _find_projected_source_files(raw_live_dir, processed_live_dir, args.season, args.date)
    if not src_files:
        raise FileNotFoundError(
            f"No projected lineup source found for {args.season} {args.date} under {raw_live_dir}."
        )

    frames: list[pd.DataFrame] = []
    for src in src_files:
        frame = pd.read_parquet(src).copy()
        frame["__source_file"] = str(src)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)

    game_pk_col = _pick(list(raw.columns), ["game_pk", "game_id"])
    game_date_col = _pick(list(raw.columns), ["game_date", "date"])
    team_col = _pick(list(raw.columns), ["team", "batter_team", "team_abbrev", "team_name", "batting_team"])
    player_col = _pick(list(raw.columns), ["player_name", "batter_name", "name"])
    player_id_col = _pick(list(raw.columns), ["player_id", "batter_id", "batter"])
    order_col = _pick(list(raw.columns), ["bat_order", "lineup_slot", "batting_order", "order", "lineup_position"])
    position_col = _pick(list(raw.columns), ["position", "pos", "field_position"])
    bats_col = _pick(list(raw.columns), ["bats", "bat_side", "stand"])
    source_col = _pick(list(raw.columns), ["source", "provider"])
    source_ts_col = _pick(list(raw.columns), ["source_timestamp", "updated_at", "timestamp", "created_at"])

    if team_col is None or player_col is None:
        raise ValueError(f"Projected lineup source missing required team/player columns. columns={sorted(raw.columns)}")

    out = pd.DataFrame(index=raw.index)
    out["game_pk"] = pd.to_numeric(raw[game_pk_col], errors="coerce").astype("Int64") if game_pk_col else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["game_date"] = pd.to_datetime(raw[game_date_col], errors="coerce") if game_date_col else slate_date
    out["game_date"] = out["game_date"].fillna(slate_date).dt.strftime("%Y-%m-%d")
    out["team"] = raw[team_col].astype(str)
    out["player_name"] = raw[player_col].astype(str)
    out["player_id"] = pd.to_numeric(raw[player_id_col], errors="coerce").astype("Int64") if player_id_col else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["bat_order"] = pd.to_numeric(raw[order_col], errors="coerce") if order_col else pd.NA
    out["position"] = raw[position_col].astype(str) if position_col else pd.NA
    out["bats"] = raw[bats_col].astype(str) if bats_col else pd.NA
    out["lineup_status"] = "projected"
    out["source"] = raw[source_col].astype(str) if source_col else "projected_lineups_ingest"
    out["source_timestamp"] = pd.to_datetime(raw[source_ts_col], errors="coerce", utc=True) if source_ts_col else pd.NaT
    out["source_timestamp"] = out["source_timestamp"].fillna(pd.Timestamp(datetime.now(timezone.utc)))
    out["source_timestamp"] = out["source_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.loc[(out["source"].isna()) | (out["source"] == ""), "source"] = raw["__source_file"]

    out = _resolve_player_ids(out, dirs["processed_dir"] / "batter_game_rolling.parquet", slate_date)
    out = out.dropna(subset=["team", "player_name"])
    out = out.drop_duplicates(subset=["game_pk", "team", "player_name"], keep="last")

    out_path = processed_live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"
    out = out[
        [
            "game_pk",
            "game_date",
            "team",
            "player_name",
            "player_id",
            "bat_order",
            "position",
            "bats",
            "lineup_status",
            "source",
            "source_timestamp",
        ]
    ].copy()
    out.to_parquet(out_path, index=False)

    logging.info(
        "projected_lineups built rows=%s games=%s teams=%s source_files=%s out=%s",
        len(out),
        int(out["game_pk"].nunique(dropna=True)) if len(out) else 0,
        int(out["team"].nunique()) if len(out) else 0,
        [str(p) for p in src_files],
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
