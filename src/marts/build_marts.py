from __future__ import annotations

from pathlib import Path
import logging
import re

import pandas as pd

from src.marts.build_hitter_batter_features import build_hitter_batter_features
from src.marts.build_pitcher_game_features import build_pitcher_game_features
from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

MART_SCHEMAS = {
    "hr_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_hr"],
    "nrfi_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_nrfi"],
    "moneyline_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_home_win"],
    "hitter_props_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_hitter_prop"],
    "pitcher_props_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_pitcher_prop"],
}


def _game_level_rollups(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "game_pk" not in df.columns:
        return pd.DataFrame(columns=["game_pk"])
    numeric = [c for c in df.select_dtypes(include=["number"]).columns if c != "game_pk"]
    if not numeric:
        return df[["game_pk"]].drop_duplicates().assign(**{f"{prefix}_feature_count": 0})
    agg = df.groupby("game_pk", dropna=False)[numeric].mean().reset_index()
    return agg.rename(columns={c: f"{prefix}_{c}" for c in numeric})


def _base_mart(spine: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    base_cols = [
        c
        for c in ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key"]
        if c in spine.columns
    ]
    df = spine[base_cols].copy() if not spine.empty else pd.DataFrame(columns=base_cols)
    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA
    return df[schema]


def _load_moneyline_targets(processed_dir: Path) -> pd.DataFrame:
    target_files = sorted(processed_dir.glob("targets_moneyline_*.parquet"))
    if not target_files:
        logging.warning("No targets_moneyline_{season}.parquet files found under %s", processed_dir.resolve())
        return pd.DataFrame(columns=["game_pk", "season", "target_home_win"])

    frames: list[pd.DataFrame] = []
    for path in target_files:
        df = read_parquet(path)
        if "game_pk" not in df.columns or "target_home_win" not in df.columns:
            logging.warning("Skipping malformed targets file %s (missing game_pk/target_home_win)", path.resolve())
            continue

        m = re.search(r"targets_moneyline_(\d{4})\.parquet$", path.name)
        season_from_name = int(m.group(1)) if m else None

        slim = df[["game_pk", "target_home_win"]].copy()
        slim["game_pk"] = pd.to_numeric(slim["game_pk"], errors="coerce").astype("Int64")
        slim["target_home_win"] = pd.to_numeric(slim["target_home_win"], errors="coerce").astype("Int64")
        slim["season"] = season_from_name if season_from_name is not None else pd.NA

        if "season" in df.columns:
            season_series = pd.to_numeric(df["season"], errors="coerce")
            slim["season"] = season_series.fillna(slim["season"]).astype("Int64")
        else:
            slim["season"] = pd.Series([slim["season"].iloc[0]] * len(slim), index=slim.index, dtype="Int64")

        frames.append(slim.dropna(subset=["game_pk"]))

    if not frames:
        logging.warning("No usable moneyline target rows loaded from %s", processed_dir.resolve())
        return pd.DataFrame(columns=["game_pk", "season", "target_home_win"])

    targets = pd.concat(frames, ignore_index=True)
    targets = targets.drop_duplicates(subset=["game_pk", "season"], keep="last")
    return targets


def _merge_moneyline_targets(mart_df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    targets = _load_moneyline_targets(processed_dir)
    if targets.empty:
        return mart_df

    out = mart_df.copy()
    row_count = len(out)
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    merge_keys = ["game_pk"]
    if "season" in out.columns and "season" in targets.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
        targets["season"] = pd.to_numeric(targets["season"], errors="coerce").astype("Int64")
        merge_keys = ["game_pk", "season"]

    merged = out.merge(targets, on=merge_keys, how="left", suffixes=("", "_targets"))

    if "target_home_win" not in merged.columns and "target_home_win_targets" in merged.columns:
        merged = merged.rename(columns={"target_home_win_targets": "target_home_win"})
    elif "target_home_win_targets" in merged.columns:
        # --- dtype stabilization before combine_first ---
        merged["target_home_win"] = pd.to_numeric(
            merged["target_home_win"], errors="coerce"
        )

        merged["target_home_win_targets"] = pd.to_numeric(
            merged["target_home_win_targets"], errors="coerce"
        )

        merged["target_home_win"] = (
            merged["target_home_win"]
                .fillna(merged["target_home_win_targets"])
                .astype("Int64")
        )
        merged = merged.drop(columns=["target_home_win_targets"])

    if "target_home_win" not in merged.columns:
        merged["target_home_win"] = pd.NA

    null_rate = float(merged["target_home_win"].isna().mean()) if len(merged) else 0.0
    pos_rate = float(pd.to_numeric(merged["target_home_win"], errors="coerce").fillna(0).mean()) if len(merged) else 0.0
    logging.info("moneyline target_home_win null rate after merge: %.6f", null_rate)
    logging.info("moneyline target_home_win positive rate after merge: %.6f", pos_rate)

    if len(merged) != row_count:
        logging.warning("moneyline target merge changed row count from %s to %s", row_count, len(merged))

    return merged


def _moneyline_side_offense_features(batter_roll: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    if batter_roll.empty or "game_pk" not in batter_roll.columns or spine.empty:
        return pd.DataFrame(columns=["game_pk"])

    team_col = next(
        (
            c
            for c in ["bat_team", "team", "batting_team", "batter_team", "offense_team"]
            if c in batter_roll.columns
        ),
        None,
    )
    if team_col is None:
        logging.warning("moneyline side offense features skipped: no batter team column found in batter rolling")
        return pd.DataFrame(columns=["game_pk"])

    spine_teams = spine[["game_pk", "home_team", "away_team"]].copy()
    tmp = batter_roll.merge(spine_teams, on="game_pk", how="left")

    row_team = tmp[team_col].astype(str)
    tmp["side"] = pd.NA
    tmp.loc[row_team == tmp["home_team"].astype(str), "side"] = "home"
    tmp.loc[row_team == tmp["away_team"].astype(str), "side"] = "away"
    tmp = tmp[tmp["side"].isin(["home", "away"])].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["game_pk"])

    exclude_numeric = {
        "game_pk",
        "batter",
        "batter_id",
        "mlbam_batter_id",
        "player_id",
        "pitcher",
        "pitcher_id",
        "mlbam_pitcher_id",
        "season",
    }
    num_cols = [
        c
        for c in tmp.select_dtypes(include=["number"]).columns
        if c not in exclude_numeric
    ]
    if not num_cols:
        return pd.DataFrame(columns=["game_pk"])

    agg = tmp.groupby(["game_pk", "side"], dropna=False)[num_cols].mean().reset_index()
    wide = agg.pivot(index="game_pk", columns="side", values=num_cols)
    wide.columns = [f"{side}_off_{col}" for col, side in wide.columns]
    return wide.reset_index()


def build_marts(dirs: dict[str, Path], season: int | None = None) -> dict[str, Path]:
    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"
    require_files([spine_path], "mart_build_model_spine")
    spine = read_parquet(spine_path)

    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pitcher_roll_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    batter_roll = read_parquet(batter_roll_path) if batter_roll_path.exists() else pd.DataFrame()
    pitcher_roll = read_parquet(pitcher_roll_path) if pitcher_roll_path.exists() else pd.DataFrame()

    batter_game_rollup = _game_level_rollups(batter_roll, "bat")
    pitcher_game_rollup = _game_level_rollups(pitcher_roll, "pit")

    outputs: dict[str, Path] = {}
    for filename, schema in MART_SCHEMAS.items():
        mart_df = _base_mart(spine, schema)
        if not batter_game_rollup.empty:
            mart_df = mart_df.merge(batter_game_rollup, on="game_pk", how="left")
        if not pitcher_game_rollup.empty:
            mart_df = mart_df.merge(pitcher_game_rollup, on="game_pk", how="left")

        if filename == "moneyline_features.parquet":
            bat_df = read_parquet(dirs["processed_dir"] / "batter_game_rolling.parquet")
            team_col = next(
                (
                    c
                    for c in ["bat_team", "team", "batting_team", "batter_team", "offense_team"]
                    if c in bat_df.columns
                ),
                None,
            )
            offense_agg_skipped = False
            if team_col is None:
                team_like = [c for c in bat_df.columns if "team" in c.lower()]
                logging.warning(
                    "moneyline offense aggregation skipped: no batting team column found in batter_game_rolling. Team-like columns: %s",
                    team_like,
                )
                offense_agg_skipped = True

            exclude_cols = {
                "game_pk",
                "batter_id",
                "bat_batter_id",
                "pitcher_id",
                "season",
                "park_id",
                "home_team",
                "away_team",
                "game_date",
            }
            if not offense_agg_skipped:
                feat_cols = [
                    c
                    for c in bat_df.select_dtypes(include=["number"]).columns
                    if c not in exclude_cols
                ]
                bat_team_agg = bat_df.groupby(["game_pk", team_col])[feat_cols].mean().reset_index() if feat_cols else pd.DataFrame(columns=["game_pk", team_col])

                home_agg = bat_team_agg.rename(columns={c: f"home_off_{c}" for c in feat_cols})
                mart_df = mart_df.merge(
                    home_agg,
                    left_on=["game_pk", "home_team"],
                    right_on=["game_pk", team_col],
                    how="left",
                )
                if team_col in mart_df.columns:
                    mart_df = mart_df.drop(columns=[team_col])

                away_agg = bat_team_agg.rename(columns={c: f"away_off_{c}" for c in feat_cols})
                mart_df = mart_df.merge(
                    away_agg,
                    left_on=["game_pk", "away_team"],
                    right_on=["game_pk", team_col],
                    how="left",
                )
                if team_col in mart_df.columns:
                    mart_df = mart_df.drop(columns=[team_col])

            mart_df = _merge_moneyline_targets(mart_df, dirs["processed_dir"])

            pa_path = dirs["processed_dir"] / "events_pa.parquet"
            if pa_path.exists():
                pa = pd.read_parquet(
                    pa_path,
                    columns=["game_pk", "game_date", "inning_topbot", "events", "home_team", "away_team"],
                )
                half = pa["inning_topbot"].astype(str).str.lower().str.strip()
                is_top = half.str.startswith("top")
                is_bot = half.str.startswith("bot") | half.str.startswith("bottom")
                pa["batting_team"] = pd.NA
                pa.loc[is_bot, "batting_team"] = pa.loc[is_bot, "home_team"]
                pa.loc[is_top, "batting_team"] = pa.loc[is_top, "away_team"]
                logging.info(
                    "moneyline inning_topbot sample counts=%s batting_team_null_pct=%.4f",
                    pa["inning_topbot"].astype(str).value_counts(dropna=False).head(5).to_dict(),
                    float(pa["batting_team"].isna().mean()) if len(pa) else 0.0,
                )

                pa["k"] = (pa["events"] == "strikeout").astype(int)
                pa["bb"] = pa["events"].isin(["walk", "intent_walk"]).astype(int)
                pa["hr"] = (pa["events"] == "home_run").astype(int)

                off = (
                    pa.groupby(["game_pk", "batting_team"], dropna=False)
                    .agg(pa=("events", "size"), k=("k", "sum"), bb=("bb", "sum"), hr=("hr", "sum"))
                    .reset_index()
                )
                game_dates = pa[["game_pk", "game_date"]].drop_duplicates(subset=["game_pk"])
                off = off.merge(game_dates, on="game_pk", how="left")
                off["game_date"] = pd.to_datetime(off["game_date"], errors="coerce")

                off["k_rate"] = off["k"] / off["pa"].replace(0, pd.NA)
                off["bb_rate"] = off["bb"] / off["pa"].replace(0, pd.NA)
                off["hr_rate"] = off["hr"] / off["pa"].replace(0, pd.NA)

                off = off.sort_values(["batting_team", "game_date"])
                for col in ["k_rate", "bb_rate", "hr_rate"]:
                    for window in [3, 7, 15, 30]:
                        off[f"{col}_roll{window}"] = (
                            off.groupby("batting_team")[col]
                            .transform(lambda s: s.shift(1).rolling(window).mean())
                        )

                roll_cols = [
                    c for c in off.columns if c.endswith("_roll3") or c.endswith("_roll7") or c.endswith("_roll15") or c.endswith("_roll30")
                ]

                home_off = off[["game_pk", "batting_team"] + roll_cols].rename(
                    columns={c: f"home_off_{c}" for c in roll_cols}
                )
                mart_df = mart_df.merge(
                    home_off,
                    left_on=["game_pk", "home_team"],
                    right_on=["game_pk", "batting_team"],
                    how="left",
                )
                if "batting_team" in mart_df.columns:
                    mart_df = mart_df.drop(columns=["batting_team"])

                away_off = off[["game_pk", "batting_team"] + roll_cols].rename(
                    columns={c: f"away_off_{c}" for c in roll_cols}
                )
                mart_df = mart_df.merge(
                    away_off,
                    left_on=["game_pk", "away_team"],
                    right_on=["game_pk", "batting_team"],
                    how="left",
                )
                if "batting_team" in mart_df.columns:
                    mart_df = mart_df.drop(columns=["batting_team"])

                home_cov = float(mart_df["home_off_k_rate_roll7"].notna().mean()) if "home_off_k_rate_roll7" in mart_df.columns and len(mart_df) else 0.0
                away_cov = float(mart_df["away_off_k_rate_roll7"].notna().mean()) if "away_off_k_rate_roll7" in mart_df.columns and len(mart_df) else 0.0
                logging.info("moneyline home_off_k_rate_roll7 non-null pct: %.4f", home_cov)
                logging.info("moneyline away_off_k_rate_roll7 non-null pct: %.4f", away_cov)

            diff_created = 0
            for home_col in [c for c in mart_df.columns if c.startswith("home_off_")]:
                suffix = home_col[len("home_off_"):]
                away_col = f"away_off_{suffix}"
                diff_col = f"diff_off_{suffix}"
                if away_col in mart_df.columns and diff_col not in mart_df.columns:
                    mart_df[diff_col] = mart_df[home_col] - mart_df[away_col]
                    diff_created += 1
            logging.info("moneyline diff_off_ columns created: %s", diff_created)

            bat_cols = [c for c in mart_df.columns if c.startswith("bat_") or c in {"batter_id", "bat_batter_id"}]
            if bat_cols:
                mart_df = mart_df.drop(columns=bat_cols)

            home_off_n = sum(1 for c in mart_df.columns if c.startswith("home_off_"))
            away_off_n = sum(1 for c in mart_df.columns if c.startswith("away_off_"))
            logging.info(
                "moneyline mart columns=%s offense_agg_skipped=%s bat_batter_id_absent=%s home_off_cols=%s away_off_cols=%s",
                len(mart_df.columns),
                offense_agg_skipped,
                "bat_batter_id" not in mart_df.columns,
                home_off_n,
                away_off_n,
            )

        print_rowcount(filename.replace('.parquet', ''), mart_df)
        out_path = dirs["marts_dir"] / filename
        print(f"Writing to: {out_path.resolve()}")
        write_parquet(mart_df, out_path)
        outputs[filename] = out_path

    if season is not None:
        outputs["hitter_batter_features.parquet"] = build_hitter_batter_features(dirs, season)
        outputs["pitcher_game_features.parquet"] = build_pitcher_game_features(dirs, season)

    return outputs
