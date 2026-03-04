from __future__ import annotations

"""
Usage:
python scripts/live/generate_nrfi_post.py --date 2026-03-31 --include-ledger-snaps
"""

import argparse
from pathlib import Path

import pandas as pd

GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]
A_TIER = {"A+", "A", "A-"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NRFI public markdown post from daily outputs.")
    p.add_argument("--date", required=True)
    p.add_argument("--model-version", default="nrfi_xgb_v1.0")
    p.add_argument("--min-grade", default="A-")
    p.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/joeplumber-mlb"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--include-full-slate", action="store_true")
    p.add_argument("--include-ledger-snaps", action="store_true")
    return p.parse_args()


def _fmt_conf(v: object) -> str:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def _to_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No plays._\n"
    out = df[columns].copy()
    return out.to_markdown(index=False) + "\n"


def _recent_perf_section(ledger: pd.DataFrame, as_of: pd.Timestamp) -> str:
    if ledger.empty:
        return "No ledger rows available.\n"
    req = {"game_date", "grade", "win"}
    if not req.issubset(set(ledger.columns)):
        return "Ledger missing required columns for performance snaps.\n"

    d = ledger.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d.dropna(subset=["game_date"])
    d = d[d["grade"].astype(str).isin(A_TIER)]
    d["win_num"] = pd.to_numeric(d["win"], errors="coerce")
    d = d.dropna(subset=["win_num"])
    if d.empty:
        return "No resolved A-tier results yet.\n"

    rows: list[dict[str, str]] = []
    for days in [7, 14, 30]:
        start = as_of.normalize() - pd.Timedelta(days=days - 1)
        w = d[(d["game_date"] >= start) & (d["game_date"] <= as_of.normalize())]
        n = len(w)
        wins = int(w["win_num"].sum()) if n else 0
        pct = (wins / n * 100.0) if n else 0.0
        rows.append({"Window": f"{days}d", "Record": f"{wins}-{n - wins}", "Win%": f"{pct:.1f}%", "Plays": str(n)})

    perf = pd.DataFrame(rows)
    return perf.to_markdown(index=False) + "\n"


def main() -> None:
    args = parse_args()
    run_date = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    drive_root = args.drive_root

    out_dir = args.out_dir or (drive_root / "data/posts/nrfi_xgb/v1.0")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = drive_root / f"data/outputs/nrfi_xgb/v1.0/daily/{args.date}_predictions.csv"
    a_tier_path = drive_root / f"data/outputs/nrfi_xgb/v1.0/public/{args.date}_A_tier_picks.csv"
    ledger_path = drive_root / "data/public_ledgers/nrfi_xgb/v1.0/ledger.csv"

    if not daily_path.exists():
        raise FileNotFoundError(f"Daily predictions file not found: {daily_path}")

    daily = pd.read_csv(daily_path)
    if "pick_prob" in daily.columns:
        daily["pick_prob_num"] = pd.to_numeric(daily["pick_prob"], errors="coerce")
    else:
        daily["pick_prob_num"] = pd.NA

    # Build A-tier from file if present, else from daily grades.
    if a_tier_path.exists():
        a_tier = pd.read_csv(a_tier_path)
    else:
        a_tier = daily[daily.get("grade", pd.Series([], dtype=str)).astype(str).isin(A_TIER)].copy()

    if "pick_prob" not in a_tier.columns and "pick_prob_num" in daily.columns:
        # if sourced from daily filtered above this will already exist
        a_tier["pick_prob"] = a_tier.get("pick_prob_num", pd.NA)

    for df in [daily, a_tier]:
        if "pick_prob" in df.columns:
            df["pick_prob_num"] = pd.to_numeric(df["pick_prob"], errors="coerce")
        else:
            df["pick_prob_num"] = pd.NA
        df.sort_values("pick_prob_num", ascending=False, inplace=True, kind="mergesort")
        df["Confidence"] = df["pick_prob_num"].map(_fmt_conf)
        rename_map = {"away_team": "Away", "home_team": "Home", "pick": "Pick", "grade": "Grade"}
        df.rename(columns=rename_map, inplace=True)

    lines: list[str] = []
    lines.append(f"# Joe Plumber Picks — NRFI/YRFI ({args.date})\n")
    lines.append("**Disclaimer:**")
    lines.append(f"- Model: {args.model_version.replace('_', ' ')}")
    lines.append("- Grades reflect confidence only. Bet only if odds are favorable.\n")

    lines.append("## A-Tier Plays\n")
    lines.append(_to_markdown_table(a_tier, ["Away", "Home", "Pick", "Grade", "Confidence"]))

    if args.include_full_slate:
        lines.append("## Full Slate\n")
        lines.append(_to_markdown_table(daily, ["Away", "Home", "Pick", "Grade", "Confidence"]))

    if args.include_ledger_snaps:
        lines.append("## Recent A-tier performance\n")
        if ledger_path.exists():
            ledger = pd.read_csv(ledger_path)
            lines.append(_recent_perf_section(ledger, run_date))
        else:
            lines.append("Ledger file not found.\n")

    lines.append("---")
    lines.append(f"Public ledger path: `{ledger_path}`")
    lines.append(f"Daily file path: `{daily_path}`")

    out_path = out_dir / f"{args.date}_nrfi_post.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(f"post_markdown={out_path}")


if __name__ == "__main__":
    main()
