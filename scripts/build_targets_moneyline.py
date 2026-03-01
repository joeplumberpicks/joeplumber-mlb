from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_output_path
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _scores_from_games(games: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    home_col = _pick(games, ["home_score", "home_final_score", "home_runs", "post_home_score", "final_home_score"])
    away_col = _pick(games, ["away_score", "away_final_score", "away_runs", "post_away_score", "final_away_score"])
    if home_col is None or away_col is None:
        return None
    return pd.to_numeric(games[home_col], errors="coerce"), pd.to_numeric(games[away_col], errors="coerce")


def _scores_from_pa_method_a(pa: pd.DataFrame) -> pd.DataFrame | None:
    required = {"game_pk", "inning_topbot", "bat_score"}
    if not required.issubset(pa.columns):
        return None
    work = pa[["game_pk", "inning_topbot", "bat_score"]].copy()
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce").astype("Int64")
    work["bat_score"] = pd.to_numeric(work["bat_score"], errors="coerce")
    half = work["inning_topbot"].astype(str).str.strip().str.lower()
    top = work[half.str.startswith("top")]
    bot = work[half.str.startswith("bot") | half.str.startswith("bottom")]
    away = top.groupby("game_pk", dropna=False)["bat_score"].max().reset_index(name="away_score")
    home = bot.groupby("game_pk", dropna=False)["bat_score"].max().reset_index(name="home_score")
    return home.merge(away, on="game_pk", how="outer")


def _scores_from_pa_method_b(pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame | None:
    required = {"game_pk", "batting_team", "bat_score"}
    if not required.issubset(pa.columns):
        return None
    if not {"game_pk", "home_team", "away_team"}.issubset(games.columns):
        return None
    work = pa[["game_pk", "batting_team", "bat_score"]].copy()
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce").astype("Int64")
    work["bat_score"] = pd.to_numeric(work["bat_score"], errors="coerce")
    max_bt = work.groupby(["game_pk", "batting_team"], dropna=False)["bat_score"].max().reset_index()

    gm = games[["game_pk", "home_team", "away_team"]].copy()
    gm["game_pk"] = pd.to_numeric(gm["game_pk"], errors="coerce").astype("Int64")

    home_scores = gm.merge(max_bt, left_on=["game_pk", "home_team"], right_on=["game_pk", "batting_team"], how="left")
    away_scores = gm.merge(max_bt, left_on=["game_pk", "away_team"], right_on=["game_pk", "batting_team"], how="left")
    out = gm[["game_pk"]].drop_duplicates().copy()
    out["home_score"] = pd.to_numeric(home_scores["bat_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(away_scores["bat_score"], errors="coerce")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build moneyline targets by season")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_targets_moneyline.log")
    log_header("scripts/build_targets_moneyline.py", repo_root, config_path, dirs)

    games_path = dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet"
    games = read_parquet(games_path)

    out = games[[c for c in ["game_pk", "game_date", "home_team", "away_team"] if c in games.columns]].copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    scores = _scores_from_games(games)
    method_used = "games"
    if scores is not None:
        out["home_score"] = scores[0]
        out["away_score"] = scores[1]
    else:
        pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
        if not pa_path.exists():
            raise ValueError(
                f"Cannot find score columns in {games_path.resolve()} and missing fallback PA file: {pa_path.resolve()}"
            )
        pa = read_parquet(pa_path)

        score_df = _scores_from_pa_method_a(pa)
        method_used = "pa_method_a_inning_topbot"
        if score_df is None:
            score_df = _scores_from_pa_method_b(pa, games)
            method_used = "pa_method_b_batting_team"

        if score_df is None:
            raise ValueError(
                "Cannot derive final scores from fallback PA file. "
                f"Tried Method A requires {{game_pk, inning_topbot, bat_score}} and Method B requires "
                f"{{game_pk, batting_team, bat_score}} + games {{game_pk, home_team, away_team}}. "
                f"PA columns={sorted(pa.columns)} file={pa_path.resolve()}"
            )

        score_df["game_pk"] = pd.to_numeric(score_df["game_pk"], errors="coerce").astype("Int64")
        out = out.merge(score_df[["game_pk", "home_score", "away_score"]], on="game_pk", how="left")

    hs = pd.to_numeric(out["home_score"], errors="coerce")
    aw = pd.to_numeric(out["away_score"], errors="coerce")

    out["target_home_win"] = pd.NA
    out.loc[hs > aw, "target_home_win"] = 1
    out.loc[hs < aw, "target_home_win"] = 0
    out["target_home_win"] = pd.to_numeric(out["target_home_win"], errors="coerce").astype("Int64")

    tie_mask = (hs == aw) & hs.notna() & aw.notna()
    tie_count = int(tie_mask.sum())
    missing_score_count = int((hs.isna() | aw.isna()).sum())

    out = out[["game_pk", "game_date", "home_team", "away_team", "home_score", "away_score", "target_home_win"]]
    out = out.drop_duplicates(subset=["game_pk"])
    before_drop = len(out)
    out = out[~tie_mask.reindex(out.index, fill_value=False)].copy()
    out = out.dropna(subset=["target_home_win"]).copy()
    out["target_home_win"] = pd.to_numeric(out["target_home_win"], errors="coerce").astype("Int64")

    remaining_ties = int(((pd.to_numeric(out["home_score"], errors="coerce") == pd.to_numeric(out["away_score"], errors="coerce")) & out["home_score"].notna() & out["away_score"].notna()).sum())
    if remaining_ties > 0:
        raise RuntimeError(f"Moneyline target build invariant failed: ties remain after filtering ({remaining_ties})")

    out_path = target_output_path(dirs["processed_dir"], "moneyline", args.season)
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(out, out_path)

    logging.info(
        "targets_moneyline method=%s rows=%s unique_games=%s date_min=%s date_max=%s tie_count=%s dropped_tie_or_unlabeled=%s missing_score_rows=%s null_rate=%.4f pos_rate=%.4f path=%s",
        method_used,
        len(out),
        int(out["game_pk"].nunique()) if "game_pk" in out.columns else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        tie_count,
        before_drop - len(out),
        missing_score_count,
        float(out["target_home_win"].isna().mean()) if len(out) else 0.0,
        float(pd.to_numeric(out["target_home_win"], errors="coerce").fillna(0).mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_moneyline -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
