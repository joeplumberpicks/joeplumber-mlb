#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.ingest.io import log_kv, log_section, write_parquet
from src.ingest.lineups import build_starting_pitchers
from src.providers import rotowire
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

from scripts.live._live_lineup_helpers import (
    enrich_with_schedule,
    load_schedule_for_date,
    print_quality,
)

ROTOWIRE_FALLBACK_URL = "https://www.rotowire.com/baseball/daily-lineups.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build starting pitchers from Rotowire.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _normalize_name(name: object) -> str | None:
    if name is None or pd.isna(name):
        return None
    s = str(name).strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_team(team: object) -> str | None:
    if team is None or pd.isna(team):
        return None
    s = str(team).strip().upper()
    if not s:
        return None
    aliases = {
        "AZ": "ARI",
        "ARZ": "ARI",
        "ATH": "OAK",
        "CWS": "CHW",
        "KCR": "KC",
        "KAN": "KC",
        "LAD": "LAD",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "TAM": "TB",
        "WAS": "WSH",
    }
    return aliases.get(s, s)


def _initial_last(name: object) -> str | None:
    s = _normalize_name(name)
    if not s:
        return None
    parts = s.split()
    if len(parts) < 2:
        return None
    return f"{parts[0][0]} {parts[-1]}"


def _last_only(name: object) -> str | None:
    s = _normalize_name(name)
    if not s:
        return None
    parts = s.split()
    if not parts:
        return None
    return parts[-1]


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _build_lookup_from_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["lookup_key", "lookup_type", "team_norm", "pitcher_id"])

    id_col = _pick_first_existing(
        df,
        [
            "pitcher_id",
            "player_id",
            "mlbam_id",
            "mlb_id",
            "id",
        ],
    )
    name_col = _pick_first_existing(
        df,
        [
            "pitcher_name",
            "player_name",
            "name",
            "full_name",
            "pitcher",
            "player",
        ],
    )
    team_col = _pick_first_existing(
        df,
        [
            "team",
            "pitching_team",
            "player_team",
            "team_abbr",
            "tm",
        ],
    )

    if id_col is None or name_col is None:
        return pd.DataFrame(columns=["lookup_key", "lookup_type", "team_norm", "pitcher_id"])

    base = df[[c for c in [id_col, name_col, team_col] if c is not None]].copy()
    base = base.rename(columns={id_col: "pitcher_id", name_col: "name_raw"})
    if team_col is not None:
        base = base.rename(columns={team_col: "team_raw"})
    else:
        base["team_raw"] = pd.NA

    base["pitcher_id"] = pd.to_numeric(base["pitcher_id"], errors="coerce").astype("Int64")
    base["name_norm"] = base["name_raw"].map(_normalize_name)
    base["team_norm"] = base["team_raw"].map(_normalize_team)

    base = base.dropna(subset=["pitcher_id", "name_norm"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["lookup_key", "lookup_type", "team_norm", "pitcher_id"])

    exact_team = base[["pitcher_id", "team_norm", "name_norm"]].copy()
    exact_team["lookup_key"] = exact_team["name_norm"]
    exact_team["lookup_type"] = f"{source_name}_exact_team"

    exact_global = base[["pitcher_id", "name_norm"]].copy()
    exact_global["lookup_key"] = exact_global["name_norm"]
    exact_global["lookup_type"] = f"{source_name}_exact_global"
    exact_global["team_norm"] = pd.NA

    init_team = base[["pitcher_id", "team_norm", "name_raw"]].copy()
    init_team["lookup_key"] = init_team["name_raw"].map(_initial_last)
    init_team["lookup_type"] = f"{source_name}_initial_last_team"
    init_team = init_team.dropna(subset=["lookup_key"])

    last_team = base[["pitcher_id", "team_norm", "name_raw"]].copy()
    last_team["lookup_key"] = last_team["name_raw"].map(_last_only)
    last_team["lookup_type"] = f"{source_name}_last_team"
    last_team = last_team.dropna(subset=["lookup_key"])

    out = pd.concat(
        [
            exact_team[["lookup_key", "lookup_type", "team_norm", "pitcher_id"]],
            exact_global[["lookup_key", "lookup_type", "team_norm", "pitcher_id"]],
            init_team[["lookup_key", "lookup_type", "team_norm", "pitcher_id"]],
            last_team[["lookup_key", "lookup_type", "team_norm", "pitcher_id"]],
        ],
        ignore_index=True,
    )

    out = out.drop_duplicates()
    return out


def _build_master_pitcher_lookup(processed_dir: Path) -> pd.DataFrame:
    frames = []

    lookup_parquet = _safe_read_parquet(processed_dir / "player_id_lookup_pitchers.parquet")
    if not lookup_parquet.empty:
        if "lookup_key" in lookup_parquet.columns:
            temp = lookup_parquet.copy()
            id_col = _pick_first_existing(temp, ["pitcher_id", "player_id", "mlbam_id", "mlb_id", "id"])
            if id_col is not None:
                temp["lookup_key"] = temp["lookup_key"].map(_normalize_name)
                temp["team_norm"] = temp[_pick_first_existing(temp, ["team", "team_abbr", "tm"])].map(_normalize_team) if _pick_first_existing(temp, ["team", "team_abbr", "tm"]) else pd.NA
                temp["pitcher_id"] = pd.to_numeric(temp[id_col], errors="coerce").astype("Int64")
                temp["lookup_type"] = temp["lookup_type"].astype("string")
                temp = temp[["lookup_key", "lookup_type", "team_norm", "pitcher_id"]].dropna(subset=["lookup_key", "pitcher_id"])
                frames.append(temp)

        frames.append(_build_lookup_from_frame(lookup_parquet, "lookup_parquet"))

    frames.append(_build_lookup_from_frame(_safe_read_parquet(processed_dir / "pitcher_game_statcast.parquet"), "game_statcast"))
    frames.append(_build_lookup_from_frame(_safe_read_parquet(processed_dir / "pitcher_statcast_rolling.parquet"), "rolling_statcast"))
    frames.append(_build_lookup_from_frame(_safe_read_parquet(processed_dir / "pitcher_game_rolling.parquet"), "game_rolling"))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["lookup_key", "lookup_type", "team_norm", "pitcher_id"])

    out["lookup_key"] = out["lookup_key"].map(_normalize_name)
    out["team_norm"] = out["team_norm"].map(_normalize_team)
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["lookup_key", "pitcher_id"]).drop_duplicates().reset_index(drop=True)
    return out


def _resolve_pitcher_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["pitcher_id_resolution_method"] = pd.Series(dtype="string")
        return out

    lookup = _build_master_pitcher_lookup(processed_dir)

    out = df.copy()
    out["team_norm"] = out["team"].map(_normalize_team)
    out["pitcher_name_norm"] = out["pitcher_name"].map(_normalize_name)
    out["pitcher_name_initial_last"] = out["pitcher_name"].map(_initial_last)
    out["pitcher_name_last"] = out["pitcher_name"].map(_last_only)

    if "pitcher_id" not in out.columns:
        out["pitcher_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    out["pitcher_id_resolution_method"] = pd.Series("unresolved", index=out.index, dtype="string")

    def _apply_match(
        current: pd.DataFrame,
        lookup_type_pattern: str,
        left_key: str,
        use_team: bool,
        method_name: str,
    ) -> pd.DataFrame:
        lkp = lookup[lookup["lookup_type"].astype("string").str.contains(lookup_type_pattern, na=False)].copy()
        if lkp.empty:
            return current

        lkp = lkp.rename(columns={"lookup_key": left_key, "pitcher_id": "pitcher_id_lkp"})
        merge_cols = [left_key] + (["team_norm"] if use_team else [])

        lkp = lkp[merge_cols + ["pitcher_id_lkp"]].drop_duplicates()

        merged = current.merge(lkp, on=merge_cols, how="left")
        mask = merged["pitcher_id"].isna() & merged["pitcher_id_lkp"].notna()
        merged.loc[mask, "pitcher_id"] = merged.loc[mask, "pitcher_id_lkp"].astype("Int64")
        merged.loc[mask, "pitcher_id_resolution_method"] = method_name
        merged = merged.drop(columns=["pitcher_id_lkp"])
        return merged

    # Most strict -> loosest
    out = _apply_match(out, "_exact_team$", "pitcher_name_norm", True, "exact_team")
    out = _apply_match(out, "_exact_global$", "pitcher_name_norm", False, "exact_global")
    out = _apply_match(out, "_initial_last_team$", "pitcher_name_initial_last", True, "initial_last_team")
    out = _apply_match(out, "_last_team$", "pitcher_name_last", True, "last_name_team")

    out = out.drop(columns=["team_norm", "pitcher_name_norm", "pitcher_name_initial_last", "pitcher_name_last"])
    return out


def _pull_starters(config: dict) -> pd.DataFrame:
    rw_cfg = config.get("rotowire", {})

    url = (
        str(rw_cfg.get("starting_pitchers_url", "")).strip()
        or str(rw_cfg.get("confirmed_lineups_url", "")).strip()
        or str(rw_cfg.get("lineups_url", "")).strip()
        or str(rw_cfg.get("daily_lineups_url", "")).strip()
        or ROTOWIRE_FALLBACK_URL
    )

    request_timeout = int(rw_cfg.get("request_timeout", 30))

    # First try table reader
    try:
        tables = rotowire.read_html_tables(
            url=url,
            request_timeout=request_timeout,
            verbose=True,
        )

        frames = []
        for idx, table in enumerate(tables):
            try:
                extracted = rotowire.extract_starting_pitchers(table, starter_status="probable")
            except Exception:
                continue

            if extracted is None or extracted.empty:
                continue

            extracted = extracted.copy()
            extracted["source_table_idx"] = idx
            frames.append(extracted)

        if frames:
            return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    except Exception:
        pass

    # Then try token parser if your current rotowire provider has it
    if hasattr(rotowire, "read_html_text") and hasattr(rotowire, "parse_daily_lineups_from_text"):
        try:
            html = rotowire.read_html_text(url=url, request_timeout=request_timeout)
            parsed = rotowire.parse_daily_lineups_from_text(html, verbose=True)
            if isinstance(parsed, dict):
                starters = parsed.get("starting_pitchers")
                if starters is not None and not starters.empty:
                    return starters.reset_index(drop=True)
        except Exception:
            pass

    # Last fallback: empty frame
    return pd.DataFrame()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/build_starting_pitchers_rotowire.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_dir = Path(dirs["processed_dir"])
    schedule_df = load_schedule_for_date(raw_live_dir, args.season, args.date)

    provider_df = _pull_starters(config)
    provider_df = enrich_with_schedule(provider_df, schedule_df, args.date)

    if not provider_df.empty:
        provider_df = _resolve_pitcher_ids(provider_df, processed_dir)

    out_df = build_starting_pitchers(
        records=provider_df,
        starter_status="probable",
        source="rotowire",
        validate=True,
        verbose=True,
    )

    print_quality(out_df, "starting_pitchers_out")

    if "pitcher_id_resolution_method" in provider_df.columns:
        print(provider_df["pitcher_id_resolution_method"].fillna("missing").value_counts(dropna=False).to_string())

    latest_out = raw_live_dir / f"starting_pitchers_{args.season}.parquet"
    dated_out = raw_live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"
    debug_provider = raw_live_dir / f"DEBUG_starting_pitchers_provider_{args.season}_{args.date}.parquet"
    debug_enriched = raw_live_dir / f"DEBUG_starting_pitchers_enriched_{args.season}_{args.date}.parquet"

    provider_df.to_parquet(debug_provider, index=False)
    out_df.to_parquet(debug_enriched, index=False)

    write_parquet(out_df, latest_out)
    write_parquet(out_df, dated_out)

    print(f"starting_pitchers_out={dated_out}")
    print(f"debug_provider_out={debug_provider}")
    print(f"debug_enriched_out={debug_enriched}")


if __name__ == "__main__":
    main()
    