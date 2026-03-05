from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header
from src.utils.team_ids import normalize_team_abbr

MODEL_VERSION = "nrfi_xgb_v1.0"
A_TIER = {"A+", "A", "A-"}
GRADE_THRESHOLDS = [
    ("A+", 0.70),
    ("A", 0.67),
    ("A-", 0.64),
    ("B+", 0.61),
    ("B", 0.58),
    ("B-", 0.55),
    ("C+", 0.53),
    ("C", 0.51),
    ("C-", 0.50),
]
TEAM_ID_PREFER = ["team", "team_abbrev", "team_name", "team_id", "batter_team", "pitcher_team"]

def _fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "--.-%"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "--.-%"


def _print_nrfi_board(daily: pd.DataFrame, date_str: str, top_n: int = 15) -> None:
    """Pretty console board for quick sanity-checking."""
    if daily is None or daily.empty:
        print(f"\nJOE PLUMBER NRFI BOARD — {date_str}\n-----------------------------------\n(no games)\n")
        return

    needed = {"away_team", "home_team", "pick", "pick_prob", "grade", "p_nrfi", "p_yrfi"}
    missing = sorted(list(needed - set(daily.columns)))
    if missing:
        print(f"\nJOE PLUMBER NRFI BOARD — {date_str}\n-----------------------------------")
        print(f"(missing columns: {missing})\n")
        return

    view = daily.copy()
    view["pick_prob"] = pd.to_numeric(view["pick_prob"], errors="coerce")
    view["p_nrfi"] = pd.to_numeric(view["p_nrfi"], errors="coerce")
    view = view.sort_values(["pick_prob", "p_nrfi"], ascending=[False, False]).head(top_n)

    print(f"\nJOE PLUMBER NRFI BOARD — {date_str}")
    print("-----------------------------------")
    for _, r in view.iterrows():
        away = str(r.get("away_team", "") or "")
        home = str(r.get("home_team", "") or "")
        pick = str(r.get("pick", "") or "")
        prob = float(r.get("pick_prob")) if pd.notna(r.get("pick_prob")) else float("nan")
        grade = str(r.get("grade", "") or "")
        matchup = f"{away} @ {home}"
        print(f"{matchup:<26} {pick:<4} {_fmt_pct(prob):>6}   {grade}")
    print("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily NRFI v1.0 scoring with optional schedule-spine auto-build.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--auto-build", action="store_true")
    p.add_argument("--allow-mart-miss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--no-board", action="store_true", help="Disable printing the console NRFI board.")
    p.add_argument("--board-top", type=int, default=15, help="How many games to show on the console board.")
    return p.parse_args()


def _grade_from_prob(prob: float) -> str:
    p = float(prob)
    for grade, threshold in GRADE_THRESHOLDS:
        if p >= threshold:
            return grade
    return "C-"


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    logging.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def _has_targets(df: pd.DataFrame) -> bool:
    return "target_nrfi" in df.columns


def _load_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    feats = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not feats:
        raise ValueError(f"Feature list empty: {path}")
    return feats


def _coerce_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _pick_team_identifier(df: pd.DataFrame, *, side: str | None = None) -> str:
    for col in TEAM_ID_PREFER:
        if col in df.columns and df[col].notna().any():
            return col

    if side in {"home", "away"} and "home_team" in df.columns and "away_team" in df.columns:
        return "home_team" if side == "home" else "away_team"

    existing_candidates = [c for c in TEAM_ID_PREFER if c in df.columns]
    raise ValueError(
        "No usable team identifier column found. "
        f"preferred={TEAM_ID_PREFER}, existing_preferred={existing_candidates}, "
        f"has_home_team={'home_team' in df.columns}, has_away_team={'away_team' in df.columns}"
    )


def _build_batting_rollup(bat: pd.DataFrame, target_date: pd.Timestamp, team_col: str) -> pd.DataFrame:
    if team_col not in bat.columns:
        raise ValueError(f"batter_game_rolling.parquet missing required column: {team_col}")
    bat = bat.copy()
    bat["game_date"] = pd.to_datetime(bat.get("game_date"), errors="coerce")
    bat = bat[(bat["game_date"].notna()) & (bat["game_date"] < target_date)].copy()
    if bat.empty:
        return pd.DataFrame(columns=[team_col])

    metric_cols = [
        c
        for c in bat.columns
        if c not in {"game_pk", "batter_id", "game_date", "home_team", "away_team", "batter_team", team_col}
        and pd.api.types.is_numeric_dtype(bat[c])
    ]

    latest_dates = bat.groupby(team_col, as_index=False)["game_date"].max().rename(columns={"game_date": "_max_date"})
    bat_latest = bat.merge(latest_dates, on=team_col, how="inner")
    bat_latest = bat_latest[bat_latest["game_date"] == bat_latest["_max_date"]].copy()

    # team-level aggregation on latest date: mean across batter rows
    roll = bat_latest.groupby(team_col, as_index=False)[metric_cols].mean(numeric_only=True)
    roll = roll.rename(columns={c: f"bat_{c}" for c in metric_cols})
    return roll


def _build_pitching_rollup(pit: pd.DataFrame, target_date: pd.Timestamp, key_col: str) -> pd.DataFrame:
    if key_col not in pit.columns:
        raise ValueError(f"pitcher_game_rolling.parquet missing required column: {key_col}")
    pit = pit.copy()
    pit["game_date"] = pd.to_datetime(pit.get("game_date"), errors="coerce")
    pit = pit[(pit["game_date"].notna()) & (pit["game_date"] < target_date)].copy()
    if pit.empty:
        return pd.DataFrame(columns=[key_col])

    metric_cols = [
        c
        for c in pit.columns
        if c not in {"game_pk", "pitcher_id", "game_date", "home_team", "away_team", key_col}
        and pd.api.types.is_numeric_dtype(pit[c])
    ]

    pit = pit.sort_values([key_col, "game_date"], kind="mergesort")
    latest = pit.groupby(key_col, as_index=False).tail(1)
    keep_cols = [key_col] + metric_cols
    latest = latest[keep_cols].copy()
    latest = latest.rename(columns={c: f"pit_{c}" for c in metric_cols})
    return latest




def _attach_live_feature_blocks(out: pd.DataFrame, data_root: Path, season: int, date_str: str) -> pd.DataFrame:
    pa_path = data_root / "raw" / "by_season" / f"pa_{season}.parquet"
    if not pa_path.exists() or out.empty:
        return out
    try:
        pa = pd.read_parquet(pa_path)
    except Exception:
        logging.exception("Failed reading PA source for live feature blocks.")
        return out

    inning_col = "inning" if "inning" in pa.columns else None
    half_col = next((c for c in ["inning_topbot", "inning_half", "topbot", "inning_top_bot"] if c in pa.columns), None)
    if inning_col is None or half_col is None:
        return out

    d = pa.copy()
    d["game_date"] = pd.to_datetime(d.get("game_date"), errors="coerce")
    target_date = pd.to_datetime(date_str, errors="coerce")
    d = d[d["game_date"].notna() & (d["game_date"] < target_date)].copy()
    if d.empty:
        return out

    d["game_pk"] = pd.to_numeric(d.get("game_pk"), errors="coerce").astype("Int64")
    d["pitcher_id"] = pd.to_numeric(d.get("pitcher"), errors="coerce").astype("Int64") if "pitcher" in d.columns else pd.NA
    d["batter_id"] = pd.to_numeric(d.get("batter"), errors="coerce").astype("Int64") if "batter" in d.columns else pd.NA
    d["_half"] = d[half_col].astype(str).str.lower()
    d = d[pd.to_numeric(d[inning_col], errors="coerce") == 1].copy()
    if d.empty:
        return out

    # SP first-inning suppression approximation from last 30 starts in inning 1
    d["rbi"] = pd.to_numeric(d.get("rbi"), errors="coerce").fillna(0.0)
    fi = d.groupby(["game_pk", "pitcher_id"], dropna=False).agg(fi_runs=("rbi", "sum"), fi_pa=("game_pk", "size"), game_date=("game_date", "max")).reset_index()
    fi["fi_allowed_yrfi"] = (fi["fi_runs"] >= 1).astype(float)
    fi = fi.sort_values(["pitcher_id", "game_date"], kind="mergesort")
    grp = fi.groupby("pitcher_id", dropna=False)
    n_prev = grp["fi_allowed_yrfi"].shift(1).rolling(30, min_periods=1).count()
    y_prev = grp["fi_allowed_yrfi"].shift(1).rolling(30, min_periods=1).sum()
    fi["sp_fi_yrfi_rate_w30"] = (y_prev / n_prev).fillna(0.0)
    fi["sp_fi_runs_per_start_w30"] = grp["fi_runs"].shift(1).rolling(30, min_periods=1).mean().fillna(0.0)
    fi["sp_fi_pa_w30"] = grp["fi_pa"].shift(1).rolling(30, min_periods=1).sum().fillna(0.0)
    lg = float(fi["fi_allowed_yrfi"].mean()) if len(fi) else 0.27
    fi["sp_fi_yrfi_rate_shrunk_w30"] = ((y_prev.fillna(0.0) + 10.0 * lg) / (n_prev.fillna(0.0) + 10.0)).fillna(lg)
    fi_latest = fi.sort_values(["pitcher_id", "game_date"]).groupby("pitcher_id", dropna=False).tail(1)

    away = fi_latest.rename(columns={"pitcher_id": "away_sp_id", **{c: f"away_sp_{c}" for c in ["sp_fi_yrfi_rate_w30", "sp_fi_runs_per_start_w30", "sp_fi_pa_w30", "sp_fi_yrfi_rate_shrunk_w30"]}})
    home = fi_latest.rename(columns={"pitcher_id": "home_sp_id", **{c: f"home_sp_{c}" for c in ["sp_fi_yrfi_rate_w30", "sp_fi_runs_per_start_w30", "sp_fi_pa_w30", "sp_fi_yrfi_rate_shrunk_w30"]}})
    out = out.merge(away[[c for c in away.columns if c.startswith("away_sp_") or c == "away_sp_id"]], on="away_sp_id", how="left")
    out = out.merge(home[[c for c in home.columns if c.startswith("home_sp_") or c == "home_sp_id"]], on="home_sp_id", how="left")

    # Top-3 fallback from most frequent first-inning hitters over last 10 team games
    if not {"home_team", "away_team"}.issubset(d.columns):
        return out
    d["bat_team"] = np.where(d["_half"].str.contains("top", na=False), d.get("away_team"), d.get("home_team"))
    order_col = "at_bat_number" if "at_bat_number" in d.columns else None
    if order_col:
        d[order_col] = pd.to_numeric(d[order_col], errors="coerce")
        d = d.sort_values(["game_date", "game_pk", "_half", order_col], kind="mergesort")
    d = d.dropna(subset=["batter_id", "bat_team"])
    first3 = d.drop_duplicates(subset=["game_pk", "_half", "batter_id"], keep="first").copy()
    first3["slot"] = first3.groupby(["game_pk", "_half"]).cumcount() + 1
    first3 = first3[first3["slot"] <= 3]
    recent = first3.sort_values("game_date").groupby(["bat_team", "batter_id"], dropna=False).tail(10)
    top_ids = (
        recent.groupby(["bat_team", "batter_id"], dropna=False).size().rename("n").reset_index().sort_values(["bat_team", "n"], ascending=[True, False]).groupby("bat_team").head(3)
    )

    hand_col = "p_throws" if "p_throws" in pa.columns else None
    if hand_col is None:
        return out
    all_pa = pa.copy()
    all_pa["game_date"] = pd.to_datetime(all_pa.get("game_date"), errors="coerce")
    all_pa = all_pa[all_pa["game_date"].notna() & (all_pa["game_date"] < target_date)].copy()
    all_pa["batter_id"] = pd.to_numeric(all_pa.get("batter"), errors="coerce").astype("Int64")
    all_pa["opp_hand"] = all_pa[hand_col].astype(str).str.upper()
    all_pa["pa"] = 1.0
    ev = all_pa.get("events", pd.Series(index=all_pa.index, dtype=object)).astype(str).str.lower()
    all_pa["bb"] = ev.isin(["walk", "intent_walk"]).astype(float)
    all_pa["so"] = ev.isin(["strikeout", "strikeout_double_play"]).astype(float)
    all_pa["h"] = ev.isin(["single", "double", "triple", "home_run"]).astype(float)
    all_pa["tb"] = ev.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0).astype(float)
    b = all_pa.groupby(["batter_id", "opp_hand"], dropna=False).agg(pa=("pa", "sum"), bb=("bb", "sum"), so=("so", "sum"), h=("h", "sum"), tb=("tb", "sum")).reset_index()
    b["woba"] = (0.69 * b["bb"] + 0.89 * b["h"]) / b["pa"].replace(0, np.nan)
    b["iso"] = (b["tb"] - b["h"]) / b["pa"].replace(0, np.nan)
    b["k_rate"] = b["so"] / b["pa"].replace(0, np.nan)
    b["bb_rate"] = b["bb"] / b["pa"].replace(0, np.nan)
    b = b.fillna(0.0)

    hand_map = all_pa.assign(pitcher_id=pd.to_numeric(all_pa.get("pitcher"), errors="coerce").astype("Int64"))[["pitcher_id", "opp_hand"]].dropna().drop_duplicates("pitcher_id", keep="last")
    out = out.merge(hand_map.rename(columns={"pitcher_id": "home_sp_id", "opp_hand": "home_sp_hand"}), on="home_sp_id", how="left")
    out = out.merge(hand_map.rename(columns={"pitcher_id": "away_sp_id", "opp_hand": "away_sp_hand"}), on="away_sp_id", how="left")

    def _agg(team_col, hand_col_name, prefix):
        keys = out[[team_col, hand_col_name, "game_pk"]].rename(columns={team_col: "bat_team", hand_col_name: "opp_hand"})
        z = keys.merge(top_ids[["bat_team", "batter_id"]], on="bat_team", how="left")
        z = z.merge(b[["batter_id", "opp_hand", "woba", "iso", "k_rate", "bb_rate", "pa"]], on=["batter_id", "opp_hand"], how="left")
        g = z.groupby("game_pk", dropna=False).agg(
            **{
                f"{prefix}_top3_woba_vs_sp_hand_w30": ("woba", "mean"),
                f"{prefix}_top3_iso_vs_sp_hand_w30": ("iso", "mean"),
                f"{prefix}_top3_k_rate_vs_sp_hand_w30": ("k_rate", "mean"),
                f"{prefix}_top3_bb_rate_vs_sp_hand_w30": ("bb_rate", "mean"),
                f"{prefix}_top3_pa_w30": ("pa", "sum"),
            }
        ).reset_index()
        return g

    away_g = _agg("away_team", "home_sp_hand", "away")
    home_g = _agg("home_team", "away_sp_hand", "home")
    out = out.merge(away_g, on="game_pk", how="left").merge(home_g, on="game_pk", how="left")
    return out


def _build_live_features(data_root: Path, season: int, date_str: str) -> pd.DataFrame:
    spine_path = data_root / "processed" / "live" / f"model_spine_game_{season}_{date_str}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(
            f"Live schedule spine not found: {spine_path}. Run with --auto-build or build schedule spine first."
        )

    spine = pd.read_parquet(spine_path).copy()
    if spine.empty:
        raise RuntimeError(f"Live schedule spine is empty for date={date_str}: {spine_path}")

    for c in ["game_pk", "away_team", "home_team"]:
        if c not in spine.columns:
            raise ValueError(f"Live schedule spine missing required column: {c}")

    bat_path = data_root / "processed" / "batter_game_rolling.parquet"
    pit_path = data_root / "processed" / "pitcher_game_rolling.parquet"
    if not bat_path.exists() or not pit_path.exists():
        raise FileNotFoundError(f"Missing rolling source tables: batter={bat_path.exists()} pitcher={pit_path.exists()}")

    target_date = pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise")
    bat = pd.read_parquet(bat_path)
    pit = pd.read_parquet(pit_path)

    live = spine.copy()

    away_bat_key = _pick_team_identifier(bat, side="away")
    home_bat_key = _pick_team_identifier(bat, side="home")
    logging.info("Selected batting team identifier columns: away=%s home=%s", away_bat_key, home_bat_key)

    live["away_team_key"] = live["away_team"].map(normalize_team_abbr)
    live["home_team_key"] = live["home_team"].map(normalize_team_abbr)
    bat["_away_team_key"] = bat[away_bat_key].map(normalize_team_abbr)
    bat["_home_team_key"] = bat[home_bat_key].map(normalize_team_abbr)

    if os.environ.get("JOE_DEBUG_TEAMS") == "1":
        away_live_vals = live["away_team_key"].dropna().astype(str).unique().tolist()[:20]
        home_live_vals = live["home_team_key"].dropna().astype(str).unique().tolist()[:20]
        away_roll_vals = bat["_away_team_key"].dropna().astype(str).unique().tolist()[:20]
        home_roll_vals = bat["_home_team_key"].dropna().astype(str).unique().tolist()[:20]
        away_match = int(live["away_team_key"].isin(set(bat["_away_team_key"].dropna())).sum())
        home_match = int(live["home_team_key"].isin(set(bat["_home_team_key"].dropna())).sum())
        logging.info(
            "JOE_DEBUG_TEAMS batting keys away_col=%s home_col=%s live_away=%s live_home=%s roll_away=%s roll_home=%s away_matches=%s/%s home_matches=%s/%s",
            away_bat_key,
            home_bat_key,
            away_live_vals,
            home_live_vals,
            away_roll_vals,
            home_roll_vals,
            away_match,
            len(live),
            home_match,
            len(live),
        )

    away_team_n = int(bat["_away_team_key"].replace("UNK", pd.NA).dropna().nunique())
    home_team_n = int(bat["_home_team_key"].replace("UNK", pd.NA).dropna().nunique())
    logging.warning(
        "Live rollup join keys: away_bat_key=%s home_bat_key=%s away_teams=%s home_teams=%s",
        away_bat_key,
        home_bat_key,
        away_team_n,
        home_team_n,
    )
    if away_team_n == 0 and home_team_n == 0:
        raise RuntimeError("Batting rollups not found; likely team key mismatch or missing rollup file.")

    bat_roll_away = _build_batting_rollup(bat, target_date, away_bat_key)
    bat_roll_home = _build_batting_rollup(bat, target_date, home_bat_key)

    if "pitcher_id" in pit.columns and pit["pitcher_id"].notna().any():
        away_pit_key = "pitcher_id"
        home_pit_key = "pitcher_id"
    else:
        away_pit_key = _pick_team_identifier(pit, side="away")
        home_pit_key = _pick_team_identifier(pit, side="home")
    logging.info("Selected pitching identifier columns: away=%s home=%s", away_pit_key, home_pit_key)
    pit_roll_away = _build_pitching_rollup(pit, target_date, away_pit_key)
    pit_roll_home = _build_pitching_rollup(pit, target_date, home_pit_key)

    away_bat = bat_roll_away.rename(columns={away_bat_key: "away_team_raw", **{c: f"away_{c}" for c in bat_roll_away.columns if c != away_bat_key}})
    away_bat["away_team_key"] = away_bat["away_team_raw"].map(normalize_team_abbr)
    home_bat = bat_roll_home.rename(columns={home_bat_key: "home_team_raw", **{c: f"home_{c}" for c in bat_roll_home.columns if c != home_bat_key}})
    home_bat["home_team_key"] = home_bat["home_team_raw"].map(normalize_team_abbr)
    live = live.merge(away_bat.drop(columns=["away_team_raw"], errors="ignore"), on="away_team_key", how="left")
    live = live.merge(home_bat.drop(columns=["home_team_raw"], errors="ignore"), on="home_team_key", how="left")

    if "away_sp_id" in live.columns:
        live["away_sp_id"] = _coerce_int64(live["away_sp_id"])
    else:
        live["away_sp_id"] = pd.Series([pd.NA] * len(live), dtype="Int64")
        logging.warning("away_sp_id not present in live spine; away pitching features will be zeros where missing.")

    if "home_sp_id" in live.columns:
        live["home_sp_id"] = _coerce_int64(live["home_sp_id"])
    else:
        live["home_sp_id"] = pd.Series([pd.NA] * len(live), dtype="Int64")
        logging.warning("home_sp_id not present in live spine; home pitching features will be zeros where missing.")

    if away_pit_key == "pitcher_id":
        away_pit = pit_roll_away.rename(columns={"pitcher_id": "away_sp_id", **{c: f"away_{c}" for c in pit_roll_away.columns if c != "pitcher_id"}})
        if "away_sp_id" in away_pit.columns:
            away_pit["away_sp_id"] = _coerce_int64(away_pit["away_sp_id"])
        live = live.merge(away_pit, on="away_sp_id", how="left")
    else:
        away_pit = pit_roll_away.rename(columns={away_pit_key: "away_team", **{c: f"away_{c}" for c in pit_roll_away.columns if c != away_pit_key}})
        live = live.merge(away_pit, on="away_team", how="left")

    if home_pit_key == "pitcher_id":
        home_pit = pit_roll_home.rename(columns={"pitcher_id": "home_sp_id", **{c: f"home_{c}" for c in pit_roll_home.columns if c != "pitcher_id"}})
        if "home_sp_id" in home_pit.columns:
            home_pit["home_sp_id"] = _coerce_int64(home_pit["home_sp_id"])
        live = live.merge(home_pit, on="home_sp_id", how="left")
    else:
        home_pit = pit_roll_home.rename(columns={home_pit_key: "home_team", **{c: f"home_{c}" for c in pit_roll_home.columns if c != home_pit_key}})
        live = live.merge(home_pit, on="home_team", how="left")

    logging.info(
        "Starter ID dtypes after coercion: live.away_sp_id=%s live.home_sp_id=%s away_pit.away_sp_id=%s home_pit.home_sp_id=%s",
        live["away_sp_id"].dtype if "away_sp_id" in live.columns else None,
        live["home_sp_id"].dtype if "home_sp_id" in live.columns else None,
        away_pit["away_sp_id"].dtype if "away_sp_id" in away_pit.columns else None,
        home_pit["home_sp_id"].dtype if "home_sp_id" in home_pit.columns else None,
    )

    # combine side features into model-style bat_*/pit_* columns (mean of away/home where both present)
    out = live.copy()
    side_cols = [c for c in out.columns if c.startswith("away_bat_") or c.startswith("home_bat_") or c.startswith("away_pit_") or c.startswith("home_pit_")]
    base_names = set()
    for c in side_cols:
        if c.startswith("away_"):
            base_names.add(c[len("away_") :])
        elif c.startswith("home_"):
            base_names.add(c[len("home_") :])

    derived_cols: dict[str, pd.Series] = {}
    for base in sorted(base_names):
        ac = f"away_{base}"
        hc = f"home_{base}"
        if ac in out.columns and hc in out.columns:
            derived_cols[base] = (pd.to_numeric(out[ac], errors="coerce").fillna(0.0) + pd.to_numeric(out[hc], errors="coerce").fillna(0.0)) / 2.0
        elif ac in out.columns:
            derived_cols[base] = pd.to_numeric(out[ac], errors="coerce").fillna(0.0)
        elif hc in out.columns:
            derived_cols[base] = pd.to_numeric(out[hc], errors="coerce").fillna(0.0)
    if derived_cols:
        out = pd.concat([out, pd.DataFrame(derived_cols, index=out.index)], axis=1)

    # smoke-style diagnostics
    n_games = len(out)
    starter_present = ((pd.to_numeric(out["away_sp_id"], errors="coerce").notna()) & (pd.to_numeric(out["home_sp_id"], errors="coerce").notna())).mean() if n_games else 0.0
    away_bat_found = int(out[[c for c in out.columns if c.startswith("away_bat_")]].notna().any(axis=1).sum()) if n_games else 0
    home_bat_found = int(out[[c for c in out.columns if c.startswith("home_bat_")]].notna().any(axis=1).sum()) if n_games else 0
    logging.info(
        "live feature smoke | games=%s pct_with_starters=%.2f away_bat_rollups_found=%s home_bat_rollups_found=%s",
        n_games,
        float(starter_present) * 100.0,
        away_bat_found,
        home_bat_found,
    )
    print(f"live_feature_smoke games={n_games} pct_with_starters={float(starter_present)*100.0:.2f} away_bat_found={away_bat_found} home_bat_found={home_bat_found}")

    out = _attach_live_feature_blocks(out, data_root, season, date_str)

    # sanity for new feature blocks
    new_cols = [c for c in out.columns if ("_top3_" in c or "_sp_sp_fi_" in c)]
    if new_cols:
        non_null = {c: float(out[c].notna().mean()) for c in new_cols}
        logging.info("live new-feature non_null_pct: %s", non_null)

    out["season"] = season
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    return out


def main() -> None:
    args = parse_args()
    date_ts = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    season = int(args.season) if args.season is not None else int(date_ts.year)

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_daily_nrfi_v1_root.log")
    log_header("scripts/run_daily_nrfi_v1.py", repo_root, config_path, dirs)

    data_root = dirs["data_root"]

    if args.auto_build:
        _run_cmd(
            [
                sys.executable,
                "scripts/ingest/ingest_schedule_games.py",
                "--date",
                args.date,
                "--season",
                str(season),
                "--game-types",
                "S,R",
                "--config",
                str(args.config),
            ],
            cwd=repo_root,
        )
        _run_cmd(
            [
                sys.executable,
                "scripts/live/build_spine_from_schedule.py",
                "--season",
                str(season),
                "--date",
                args.date,
                "--config",
                str(args.config),
                "--force",
            ],
            cwd=repo_root,
        )

    mart_path = data_root / "marts" / "by_season" / f"nrfi_features_{season}.parquet"
    if not mart_path.exists():
        fallback = data_root / "marts" / "by_season" / "nrfi_features_2025.parquet"
        if fallback.exists():
            logging.warning(
                "Season mart missing for %s, using fallback: %s. Live carryover joins use normalized team keys away_team_key/home_team_key.",
                season,
                fallback,
            )
            mart_path = fallback
        else:
            raise FileNotFoundError(f"No mart found for season={season} and no 2025 fallback at {fallback}")

    model_path = data_root / "models" / "nrfi_xgb" / "releases" / "v1.0" / "nrfi_model.json"
    features_path = data_root / "models" / "nrfi_xgb" / "releases" / "v1.0" / "features_240.txt"

    daily_out_path = data_root / "outputs" / "nrfi_xgb" / "v1.0" / "daily" / f"{args.date}_predictions.csv"
    public_out_path = data_root / "outputs" / "nrfi_xgb" / "v1.0" / "public" / f"{args.date}_A_tier_picks.csv"
    ledger_path = data_root / "public_ledgers" / "nrfi_xgb" / "v1.0" / "ledger.csv"

    daily_out_path.parent.mkdir(parents=True, exist_ok=True)
    public_out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    mart = pd.read_parquet(mart_path)
    if "game_date" not in mart.columns:
        raise ValueError(f"Mart missing game_date column: {mart_path}")
    mart["game_date"] = pd.to_datetime(mart["game_date"], errors="coerce")
    daily = mart[mart["game_date"].dt.date == date_ts.date()].copy()

    used_live_fallback = False
    if daily.empty:
        if not args.allow_mart_miss:
            raise RuntimeError(f"No games found in mart={mart_path} for date={args.date}")
        logging.warning("No rows in mart for date; building live features from rolling carryover.")
        daily = _build_live_features(data_root=data_root, season=season, date_str=args.date)
        used_live_fallback = True

    features = _load_features(features_path)
    X = daily.reindex(columns=features).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(X, feature_names=features)

    p_nrfi = booster.predict(dmat)
    p_yrfi = 1.0 - p_nrfi
    pick = pd.Series(p_nrfi, index=daily.index).ge(0.5).map({True: "NRFI", False: "YRFI"})
    pick_prob = pd.concat(
        [pd.Series(p_nrfi, index=daily.index), pd.Series(p_yrfi, index=daily.index)], axis=1
    ).max(axis=1)
    grade = pick_prob.map(_grade_from_prob)
    daily = daily.assign(p_nrfi=p_nrfi, p_yrfi=p_yrfi, pick=pick, pick_prob=pick_prob, grade=grade)

    if (not used_live_fallback) and _has_targets(daily):
        actual = daily["target_nrfi"].map(lambda x: "NRFI" if pd.notna(x) and int(x) == 1 else "YRFI")
        win = (daily["pick"] == actual).astype(int)
        daily = daily.assign(actual=actual, win=win)
    else:
        daily = daily.assign(actual="", win="")

    daily["game_date"] = pd.to_datetime(daily.get("game_date"), errors="coerce").dt.strftime("%Y-%m-%d")

    required = [
        "game_date",
        "game_pk",
        "away_team",
        "home_team",
        "p_nrfi",
        "p_yrfi",
        "pick",
        "pick_prob",
        "grade",
        "actual",
        "win",
    ]
    for c in required:
        if c not in daily.columns:
            daily[c] = ""
    daily_out = daily[required].copy()
    daily_out.to_csv(daily_out_path, index=False)

    public_cols = ["game_date", "away_team", "home_team", "pick", "grade", "pick_prob", "p_nrfi", "p_yrfi", "game_pk"]
    a_tier = daily[daily["grade"].isin(A_TIER)].copy()
    for c in public_cols:
        if c not in a_tier.columns:
            a_tier[c] = ""
    a_tier = a_tier[public_cols].sort_values("pick_prob", ascending=False, kind="mergesort")
    a_tier.to_csv(public_out_path, index=False)

    ledger_add = daily_out.copy()
    ledger_add["run_date"] = args.date
    ledger_add["model_version"] = MODEL_VERSION
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = pd.concat([ledger, ledger_add], ignore_index=True, sort=False)
    else:
        ledger = ledger_add

    ledger = ledger.drop_duplicates(subset=["model_version", "game_pk", "game_date"], keep="last")
    ledger["_gd"] = pd.to_datetime(ledger["game_date"], errors="coerce")
    ledger["_pk"] = pd.to_numeric(ledger["game_pk"], errors="coerce")
    ledger = ledger.sort_values(["_gd", "_pk"], kind="mergesort").drop(columns=["_gd", "_pk"])
    ledger.to_csv(ledger_path, index=False)

    if not args.no_board:
        try:
            _print_nrfi_board(daily=daily, date_str=args.date, top_n=int(args.board_top))
        except Exception:
            logging.exception("Failed printing NRFI board (non-fatal).")

    logging.info(
        "daily nrfi run complete date=%s season=%s rows=%s daily_out=%s public_out=%s ledger=%s",
        args.date,
        season,
        len(daily_out),
        daily_out_path,
        public_out_path,
        ledger_path,
    )
    print(f"daily_out={daily_out_path}")
    print(f"public_out={public_out_path}")
    print(f"ledger_out={ledger_path}")


if __name__ == "__main__":
    main()
