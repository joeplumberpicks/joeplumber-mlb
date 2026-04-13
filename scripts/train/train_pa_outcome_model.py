%%bash
cd /content/joeplumber-mlb

mkdir -p scripts/train

cat > scripts/train/train_pa_outcome_model.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, top_k_accuracy_score

from src.models.pa_outcome_model import (
    PA_OUTCOME_CLASSES,
    build_pa_target,
    fit_pa_outcome_model,
    predict_pa_outcome_class,
    predict_pa_outcome_proba,
    save_pa_outcome_artifact,
)
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PA outcome model.")
    parser.add_argument("--train-seasons", type=str, default="2019,2020,2021,2022,2023,2024")
    parser.add_argument("--test-seasons", type=str, default="2025")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _parse_seasons(text: str) -> list[int]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError(f"No seasons parsed from: {text}")
    return vals


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def _load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required parquet: {path}")
    return pd.read_parquet(path)


def _load_base_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pa = _load_parquet_if_exists(processed_dir / "pa.parquet")
    batter_roll = _load_parquet_if_exists(processed_dir / "batter_game_rolling.parquet")
    pitcher_roll = _load_parquet_if_exists(processed_dir / "pitcher_game_rolling.parquet")
    return pa, batter_roll, pitcher_roll


def _prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    for col in ["inning", "outs_before_pa", "outs_after_pa", "rbi", "runs_scored_on_pa"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["event_type", "inning_topbot", "base_state_before", "batting_team", "fielding_team"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    flag_cols = ["is_pa", "is_ab", "is_1b", "is_2b", "is_3b", "is_hr", "is_bb", "is_hbp", "is_so"]
    for col in flag_cols:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
        else:
            out[col] = False

    out = out[out["game_date"].notna()].copy()
    out = out[out["season"].notna()].copy()
    out = out[out["batter_id"].notna()].copy()
    out = out[out["pitcher_id"].notna()].copy()
    out = out[out["is_pa"] == True].copy()

    out["pa_outcome_target"] = build_pa_target(out)

    return out


def _prep_rollings(batter_roll: pd.DataFrame, pitcher_roll: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    br = batter_roll.copy()
    pr = pitcher_roll.copy()

    for df, id_col in [(br, "batter_id"), (pr, "pitcher_id")]:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

    return br, pr


def _join_rollings(
    pa: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    out = pa.copy()

    br_keep = _pick_existing(
        batter_roll,
        [
            "game_pk",
            "batter_id",
            "pa_roll3", "pa_roll7", "pa_roll15", "pa_roll30",
            "ab_roll3", "ab_roll7", "ab_roll15", "ab_roll30",
            "hits_roll3", "hits_roll7", "hits_roll15", "hits_roll30",
            "hr_roll3", "hr_roll7", "hr_roll15", "hr_roll30",
            "tb_roll3", "tb_roll7", "tb_roll15", "tb_roll30",
            "bb_roll3", "bb_roll7", "bb_roll15", "bb_roll30",
            "so_roll3", "so_roll7", "so_roll15", "so_roll30",
            "hit_rate_roll3", "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
            "hr_rate_roll3", "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
            "tb_pa_rate_roll3", "tb_pa_rate_roll7", "tb_pa_rate_roll15", "tb_pa_rate_roll30",
            "bb_rate_roll3", "bb_rate_roll7", "bb_rate_roll15", "bb_rate_roll30",
            "so_rate_roll3", "so_rate_roll7", "so_rate_roll15", "so_rate_roll30",
        ],
    )

    pr_keep = _pick_existing(
        pitcher_roll,
        [
            "game_pk",
            "pitcher_id",
            "batters_faced_roll3", "batters_faced_roll7", "batters_faced_roll15", "batters_faced_roll30",
            "hits_allowed_roll3", "hits_allowed_roll7", "hits_allowed_roll15", "hits_allowed_roll30",
            "hr_allowed_roll3", "hr_allowed_roll7", "hr_allowed_roll15", "hr_allowed_roll30",
            "bb_allowed_roll3", "bb_allowed_roll7", "bb_allowed_roll15", "bb_allowed_roll30",
            "so_roll3", "so_roll7", "so_roll15", "so_roll30",
            "runs_allowed_roll3", "runs_allowed_roll7", "runs_allowed_roll15", "runs_allowed_roll30",
            "tb_allowed_roll3", "tb_allowed_roll7", "tb_allowed_roll15", "tb_allowed_roll30",
            "k_rate_roll3", "k_rate_roll7", "k_rate_roll15", "k_rate_roll30",
            "bb_rate_roll3", "bb_rate_roll7", "bb_rate_roll15", "bb_rate_roll30",
            "hr_rate_roll3", "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
            "hit_rate_roll3", "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
            "runs_rate_roll3", "runs_rate_roll7", "runs_rate_roll15", "runs_rate_roll30",
        ],
    )

    br = batter_roll[br_keep].copy().rename(
        columns={c: f"bat_{c}" for c in br_keep if c not in {"game_pk", "batter_id"}}
    )
    pr = pitcher_roll[pr_keep].copy().rename(
        columns={c: f"pit_{c}" for c in pr_keep if c not in {"game_pk", "pitcher_id"}}
    )

    out = out.merge(br, on=["game_pk", "batter_id"], how="left")
    out = out.merge(pr, on=["game_pk", "pitcher_id"], how="left")

    return out


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "base_state_before" in out.columns:
        bsb = out["base_state_before"].astype("string").fillna("")
        out["on_1b"] = bsb.str.contains("1").astype(int)
        out["on_2b"] = bsb.str.contains("2").astype(int)
        out["on_3b"] = bsb.str.contains("3").astype(int)
        out["base_runner_count"] = out[["on_1b", "on_2b", "on_3b"]].sum(axis=1)

    if "inning_topbot" in out.columns:
        out["is_top_inning"] = out["inning_topbot"].astype("string").str.upper().eq("TOP").astype(int)
        out["is_bot_inning"] = out["inning_topbot"].astype("string").str.upper().eq("BOT").astype(int)

    numeric_to_force = _pick_existing(
        out,
        [
            "inning", "outs_before_pa",
            "bat_pa_roll3", "bat_pa_roll7", "bat_pa_roll15", "bat_pa_roll30",
            "bat_ab_roll3", "bat_ab_roll7", "bat_ab_roll15", "bat_ab_roll30",
            "bat_hits_roll3", "bat_hits_roll7", "bat_hits_roll15", "bat_hits_roll30",
            "bat_hr_roll3", "bat_hr_roll7", "bat_hr_roll15", "bat_hr_roll30",
            "bat_tb_roll3", "bat_tb_roll7", "bat_tb_roll15", "bat_tb_roll30",
            "bat_bb_roll3", "bat_bb_roll7", "bat_bb_roll15", "bat_bb_roll30",
            "bat_so_roll3", "bat_so_roll7", "bat_so_roll15", "bat_so_roll30",
            "bat_hit_rate_roll3", "bat_hit_rate_roll7", "bat_hit_rate_roll15", "bat_hit_rate_roll30",
            "bat_hr_rate_roll3", "bat_hr_rate_roll7", "bat_hr_rate_roll15", "bat_hr_rate_roll30",
            "bat_tb_pa_rate_roll3", "bat_tb_pa_rate_roll7", "bat_tb_pa_rate_roll15", "bat_tb_pa_rate_roll30",
            "bat_bb_rate_roll3", "bat_bb_rate_roll7", "bat_bb_rate_roll15", "bat_bb_rate_roll30",
            "bat_so_rate_roll3", "bat_so_rate_roll7", "bat_so_rate_roll15", "bat_so_rate_roll30",
            "pit_batters_faced_roll3", "pit_batters_faced_roll7", "pit_batters_faced_roll15", "pit_batters_faced_roll30",
            "pit_hits_allowed_roll3", "pit_hits_allowed_roll7", "pit_hits_allowed_roll15", "pit_hits_allowed_roll30",
            "pit_hr_allowed_roll3", "pit_hr_allowed_roll7", "pit_hr_allowed_roll15", "pit_hr_allowed_roll30",
            "pit_bb_allowed_roll3", "pit_bb_allowed_roll7", "pit_bb_allowed_roll15", "pit_bb_allowed_roll30",
            "pit_so_roll3", "pit_so_roll7", "pit_so_roll15", "pit_so_roll30",
            "pit_runs_allowed_roll3", "pit_runs_allowed_roll7", "pit_runs_allowed_roll15", "pit_runs_allowed_roll30",
            "pit_tb_allowed_roll3", "pit_tb_allowed_roll7", "pit_tb_allowed_roll15", "pit_tb_allowed_roll30",
            "pit_k_rate_roll3", "pit_k_rate_roll7", "pit_k_rate_roll15", "pit_k_rate_roll30",
            "pit_bb_rate_roll3", "pit_bb_rate_roll7", "pit_bb_rate_roll15", "pit_bb_rate_roll30",
            "pit_hr_rate_roll3", "pit_hr_rate_roll7", "pit_hr_rate_roll15", "pit_hr_rate_roll30",
            "pit_hit_rate_roll3", "pit_hit_rate_roll7", "pit_hit_rate_roll15", "pit_hit_rate_roll30",
            "pit_runs_rate_roll3", "pit_runs_rate_roll7", "pit_runs_rate_roll15", "pit_runs_rate_roll30",
            "on_1b", "on_2b", "on_3b", "base_runner_count", "is_top_inning", "is_bot_inning",
        ],
    )
    _safe_numeric(out, numeric_to_force)

    return out


def _build_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = _pick_existing(
        df,
        [
            "inning",
            "outs_before_pa",
            "base_state_before",
            "inning_topbot",
            "batting_team",
            "fielding_team",
            "on_1b",
            "on_2b",
            "on_3b",
            "base_runner_count",
            "is_top_inning",
            "is_bot_inning",

            "bat_pa_roll3", "bat_pa_roll7", "bat_pa_roll15", "bat_pa_roll30",
            "bat_ab_roll3", "bat_ab_roll7", "bat_ab_roll15", "bat_ab_roll30",
            "bat_hits_roll3", "bat_hits_roll7", "bat_hits_roll15", "bat_hits_roll30",
            "bat_hr_roll3", "bat_hr_roll7", "bat_hr_roll15", "bat_hr_roll30",
            "bat_tb_roll3", "bat_tb_roll7", "bat_tb_roll15", "bat_tb_roll30",
            "bat_bb_roll3", "bat_bb_roll7", "bat_bb_roll15", "bat_bb_roll30",
            "bat_so_roll3", "bat_so_roll7", "bat_so_roll15", "bat_so_roll30",
            "bat_hit_rate_roll3", "bat_hit_rate_roll7", "bat_hit_rate_roll15", "bat_hit_rate_roll30",
            "bat_hr_rate_roll3", "bat_hr_rate_roll7", "bat_hr_rate_roll15", "bat_hr_rate_roll30",
            "bat_tb_pa_rate_roll3", "bat_tb_pa_rate_roll7", "bat_tb_pa_rate_roll15", "bat_tb_pa_rate_roll30",
            "bat_bb_rate_roll3", "bat_bb_rate_roll7", "bat_bb_rate_roll15", "bat_bb_rate_roll30",
            "bat_so_rate_roll3", "bat_so_rate_roll7", "bat_so_rate_roll15", "bat_so_rate_roll30",

            "pit_batters_faced_roll3", "pit_batters_faced_roll7", "pit_batters_faced_roll15", "pit_batters_faced_roll30",
            "pit_hits_allowed_roll3", "pit_hits_allowed_roll7", "pit_hits_allowed_roll15", "pit_hits_allowed_roll30",
            "pit_hr_allowed_roll3", "pit_hr_allowed_roll7", "pit_hr_allowed_roll15", "pit_hr_allowed_roll30",
            "pit_bb_allowed_roll3", "pit_bb_allowed_roll7", "pit_bb_allowed_roll15", "pit_bb_allowed_roll30",
            "pit_so_roll3", "pit_so_roll7", "pit_so_roll15", "pit_so_roll30",
            "pit_runs_allowed_roll3", "pit_runs_allowed_roll7", "pit_runs_allowed_roll15", "pit_runs_allowed_roll30",
            "pit_tb_allowed_roll3", "pit_tb_allowed_roll7", "pit_tb_allowed_roll15", "pit_tb_allowed_roll30",
            "pit_k_rate_roll3", "pit_k_rate_roll7", "pit_k_rate_roll15", "pit_k_rate_roll30",
            "pit_bb_rate_roll3", "pit_bb_rate_roll7", "pit_bb_rate_roll15", "pit_bb_rate_roll30",
            "pit_hr_rate_roll3", "pit_hr_rate_roll7", "pit_hr_rate_roll15", "pit_hr_rate_roll30",
            "pit_hit_rate_roll3", "pit_hit_rate_roll7", "pit_hit_rate_roll15", "pit_hit_rate_roll30",
            "pit_runs_rate_roll3", "pit_runs_rate_roll7", "pit_runs_rate_roll15", "pit_runs_rate_roll30",
        ],
    )

    if not feature_columns:
        raise ValueError("No feature columns found for PA outcome training.")

    return feature_columns


def _multiclass_metrics(df: pd.DataFrame, proba: pd.DataFrame, pred: pd.Series) -> dict:
    y_true = df["pa_outcome_target"].astype("string")
    y_true_id = y_true.map({name: i for i, name in enumerate(PA_OUTCOME_CLASSES)}).astype(int).to_numpy()

    proba_cols = [f"p_{c}" for c in PA_OUTCOME_CLASSES]
    proba_mat = proba[proba_cols].to_numpy()

    metrics = {
        "n_test": int(len(df)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "top2_accuracy": float(top_k_accuracy_score(y_true_id, proba_mat, k=2, labels=np.arange(len(PA_OUTCOME_CLASSES)))),
        "logloss": float(log_loss(y_true_id, proba_mat, labels=np.arange(len(PA_OUTCOME_CLASSES)))),
        "class_rates_test": {
            c: float((y_true == c).mean()) for c in PA_OUTCOME_CLASSES
        },
        "pred_rates_test": {
            c: float((pred == c).mean()) for c in PA_OUTCOME_CLASSES
        },
    }

    return metrics


def main() -> None:
    args = parse_args()

    train_seasons = _parse_seasons(args.train_seasons)
    test_seasons = _parse_seasons(args.test_seasons)

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    models_dir = Path(dirs["models_dir"]) / "pa_outcome_xgb"
    backtests_dir = Path(dirs["backtests_dir"]) / "pa_outcome_xgb"

    models_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("JOE PLUMBER PA OUTCOME TRAIN")
    print("========================================")
    print(f"train_seasons={train_seasons}")
    print(f"test_seasons={test_seasons}")

    pa, batter_roll, pitcher_roll = _load_base_tables(processed_dir)
    pa = _prep_pa(pa)
    batter_roll, pitcher_roll = _prep_rollings(batter_roll, pitcher_roll)

    df = _join_rollings(pa, batter_roll, pitcher_roll)
    df = _add_context_features(df)

    feature_columns = _build_feature_columns(df)

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    if train_df.empty:
        raise ValueError(f"No train rows for seasons={train_seasons}")
    if test_df.empty:
        raise ValueError(f"No test rows for seasons={test_seasons}")

    artifact = fit_pa_outcome_model(
        train_df=train_df,
        feature_columns=feature_columns,
        random_state=args.random_state,
    )

    test_proba = predict_pa_outcome_proba(artifact, test_df)
    test_pred = predict_pa_outcome_class(artifact, test_df)

    metrics = _multiclass_metrics(test_df, test_proba, test_pred)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_name = f"pa_outcome_xgb_TEST{'-'.join(map(str, test_seasons))}_{stamp}"

    saved = save_pa_outcome_artifact(
        artifact=artifact,
        out_dir=models_dir,
        artifact_name=artifact_name,
    )

    preview = test_df[
        _pick_existing(
            test_df,
            ["game_date", "season", "game_pk", "batter_id", "pitcher_id", "event_type", "pa_outcome_target"]
        )
    ].copy()
    preview["pred_pa_outcome"] = test_pred
    preview = pd.concat([preview.reset_index(drop=True), test_proba.reset_index(drop=True)], axis=1)

    preview_path = backtests_dir / f"{artifact_name}_preview.parquet"
    metrics_path = backtests_dir / f"{artifact_name}_metrics.json"

    preview.to_parquet(preview_path, index=False)

    payload = {
        "artifact_name": artifact_name,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_path": saved["model_path"],
        "meta_path": saved["meta_path"],
        "preview_path": str(preview_path),
        "metrics": metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("")
    print("=== TRAIN SUMMARY ===")
    print(f"train_rows={len(train_df):,}")
    print(f"test_rows={len(test_df):,}")
    print(f"feature_count={len(feature_columns):,}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"top2_accuracy={metrics['top2_accuracy']:.4f}")
    print(f"logloss={metrics['logloss']:.4f}")

    print("")
    print("=== TEST CLASS RATES ===")
    for c in PA_OUTCOME_CLASSES:
        print(f"{c:>10s} | actual={metrics['class_rates_test'][c]:.4f} | pred={metrics['pred_rates_test'][c]:.4f}")

    print("")
    print("model_out:", saved["model_path"])
    print("meta_out:", saved["meta_path"])
    print("metrics_out:", metrics_path)
    print("preview_out:", preview_path)


if __name__ == "__main__":
    main()
PY

chmod +x scripts/train/train_pa_outcome_model.py