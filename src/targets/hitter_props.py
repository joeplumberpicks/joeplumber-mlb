from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.targets.paths import target_output_path
from src.utils.io import read_parquet, write_parquet

_BATTER_ID_CANDIDATES = ["batter_id", "batter", "mlbam_batter_id", "player_id"]
_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_BB_BUCKET_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_json_with_retries(url: str, timeout_s: int = 20, retries: int = 3, backoff_s: float = 1.5) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"Failed GET {url} after {retries} attempts: {last_exc}")


def _parse_boxscore_game(game_pk: int, payload: dict, game_date: object, home_team: object, away_team: object) -> list[dict[str, object]]:
    teams = payload.get("teams", {}) if isinstance(payload, dict) else {}
    rows: list[dict[str, object]] = []
    for side in ["home", "away"]:
        side_obj = teams.get(side, {}) if isinstance(teams, dict) else {}
        players = side_obj.get("players", {}) if isinstance(side_obj, dict) else {}
        for player_obj in players.values():
            if not isinstance(player_obj, dict):
                continue
            person = player_obj.get("person", {})
            batting = ((player_obj.get("stats", {}) or {}).get("batting", {}) or {})
            batter_id = pd.to_numeric(person.get("id"), errors="coerce")
            if pd.isna(batter_id):
                continue
            row = {
                "game_pk": int(game_pk),
                "batter_id": int(batter_id),
                "rbi": int(pd.to_numeric(batting.get("rbi"), errors="coerce") if pd.notna(pd.to_numeric(batting.get("rbi"), errors="coerce")) else 0),
                "h": int(pd.to_numeric(batting.get("hits"), errors="coerce") if pd.notna(pd.to_numeric(batting.get("hits"), errors="coerce")) else 0),
                "tb": pd.to_numeric(batting.get("totalBases"), errors="coerce"),
                "bb": int(pd.to_numeric(batting.get("baseOnBalls"), errors="coerce") if pd.notna(pd.to_numeric(batting.get("baseOnBalls"), errors="coerce")) else 0),
                "ab": pd.to_numeric(batting.get("atBats"), errors="coerce"),
                "pa": pd.to_numeric(batting.get("plateAppearances"), errors="coerce"),
                "so": pd.to_numeric(batting.get("strikeOuts"), errors="coerce"),
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
            }
            rows.append(row)
    return rows


def build_batter_boxscore(dirs: dict[str, Path], season: int, force: bool = False) -> Path:
    games_path = dirs["processed_dir"] / "by_season" / f"games_{season}.parquet"
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games file: {games_path.resolve()}")

    games = read_parquet(games_path)
    if "game_pk" not in games.columns:
        raise ValueError("games parquet missing game_pk")

    out_path = dirs["processed_dir"] / "by_season" / f"batter_boxscore_{season}.parquet"
    cache_dir = dirs["raw_dir"] / "boxscores" / "by_season" / str(season)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    game_rows = games[[c for c in ["game_pk", "game_date", "home_team", "away_team"] if c in games.columns]].copy()
    game_rows["game_pk"] = pd.to_numeric(game_rows["game_pk"], errors="coerce").astype("Int64")
    game_rows = game_rows.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"]).sort_values("game_pk")

    for idx, rec in enumerate(game_rows.to_dict("records"), start=1):
        game_pk = int(rec["game_pk"])
        cache_file = cache_dir / f"{game_pk}.json"
        if cache_file.exists() and not force:
            with cache_file.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        else:
            url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            payload = _get_json_with_retries(url=url)
            with cache_file.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp)

        rows.extend(
            _parse_boxscore_game(
                game_pk=game_pk,
                payload=payload,
                game_date=rec.get("game_date"),
                home_team=rec.get("home_team"),
                away_team=rec.get("away_team"),
            )
        )
        if idx % 50 == 0:
            logging.info("batter_boxscore progress season=%s games=%s/%s", season, idx, len(game_rows))

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(columns=["game_pk", "batter_id", "rbi", "h", "tb", "bb", "ab", "pa", "so", "game_date", "home_team", "away_team"])
    else:
        out_df["game_pk"] = pd.to_numeric(out_df["game_pk"], errors="coerce").astype("Int64")
        out_df["batter_id"] = pd.to_numeric(out_df["batter_id"], errors="coerce").astype("Int64")
        for c in ["rbi", "h", "bb"]:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0).astype("Int64")
        for c in ["tb", "ab", "pa", "so"]:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    write_parquet(out_df, out_path)
    logging.info("batter_boxscore rows=%s path=%s", len(out_df), out_path.resolve())
    return out_path


def build_hitter_prop_targets(dirs: dict[str, Path], season: int, force: bool = False) -> Path:
    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
    box_path = dirs["processed_dir"] / "by_season" / f"batter_boxscore_{season}.parquet"
    out_path = target_output_path(dirs["processed_dir"], "hitter_props", season)

    if out_path.exists() and not force:
        logging.info("targets_hitter_props exists and force=False: %s", out_path.resolve())
        return out_path

    if not pa_path.exists():
        raise FileNotFoundError(f"Missing PA parquet for target build: {pa_path.resolve()}")
    if not box_path.exists():
        raise FileNotFoundError(f"Missing batter boxscore parquet for target build: {box_path.resolve()}")

    pa = read_parquet(pa_path)
    if "game_pk" not in pa.columns:
        raise ValueError("pa parquet missing game_pk")
    batter_col = _pick_col(pa, _BATTER_ID_CANDIDATES)
    if batter_col is None:
        raise ValueError(f"pa parquet missing batter id column, candidates={_BATTER_ID_CANDIDATES}")

    pa = pa.copy()
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["batter_id"] = pd.to_numeric(pa[batter_col], errors="coerce").astype("Int64")
    event_col = "events" if "events" in pa.columns else ("event_type" if "event_type" in pa.columns else None)
    if event_col is None:
        raise ValueError("pa parquet missing events/event_type")
    ev = pa[event_col].astype(str).str.lower()

    pa["_hit"] = ev.isin(_HIT_EVENTS).astype(int)
    pa["_tb"] = (
        (ev == "single").astype(int)
        + 2 * (ev == "double").astype(int)
        + 3 * (ev == "triple").astype(int)
        + 4 * (ev == "home_run").astype(int)
    )
    pa["_bb_bucket"] = ev.isin(_BB_BUCKET_EVENTS).astype(int)

    pa_agg = (
        pa.groupby(["game_pk", "batter_id"], dropna=False)
        .agg(hits=("_hit", "sum"), tb=("_tb", "sum"), bb_bucket=("_bb_bucket", "sum"))
        .reset_index()
    )

    box = read_parquet(box_path)
    if "game_pk" not in box.columns or "batter_id" not in box.columns:
        raise ValueError("batter_boxscore parquet missing game_pk/batter_id")
    box = box.copy()
    box["game_pk"] = pd.to_numeric(box["game_pk"], errors="coerce").astype("Int64")
    box["batter_id"] = pd.to_numeric(box["batter_id"], errors="coerce").astype("Int64")
    box["rbi"] = pd.to_numeric(box.get("rbi"), errors="coerce").fillna(0)
    rbi_agg = box.groupby(["game_pk", "batter_id"], dropna=False)["rbi"].sum().reset_index(name="rbi")

    tgt = pa_agg.merge(rbi_agg, on=["game_pk", "batter_id"], how="left")
    tgt["rbi"] = pd.to_numeric(tgt["rbi"], errors="coerce").fillna(0)

    tgt["target_hit1p"] = (tgt["hits"] >= 1).astype("Int64")
    tgt["target_tb2p"] = (tgt["tb"] >= 2).astype("Int64")
    tgt["target_bb1p"] = (tgt["bb_bucket"] >= 1).astype("Int64")
    tgt["target_rbi1p"] = (tgt["rbi"] >= 1).astype("Int64")

    meta_cols = [c for c in ["game_pk", "game_date"] if c in pa.columns]
    meta = pa[meta_cols].drop_duplicates(subset=["game_pk"]) if "game_pk" in meta_cols else pd.DataFrame(columns=["game_pk"])
    out = tgt[["game_pk", "batter_id", "target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]].copy()
    out = out.merge(meta, on="game_pk", how="left")
    out = out[["game_pk", "game_date", "batter_id", "target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]]
    write_parquet(out, out_path)
    for col in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
        nn = float(out[col].notna().mean()) if len(out) else 0.0
        pr = float(pd.to_numeric(out[col], errors="coerce").fillna(0).mean()) if len(out) else 0.0
        logging.info("%s nonnull=%.4f prevalence=%.4f", col, nn, pr)
    logging.info("targets_hitter_props rows=%s path=%s", len(out), out_path.resolve())
    return out_path
