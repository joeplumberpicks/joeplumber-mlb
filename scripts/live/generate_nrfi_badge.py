from __future__ import annotations

"""
Usage:
python scripts/live/generate_nrfi_badge.py --window 7d
python scripts/live/generate_nrfi_badge.py --window mtd --date 2026-04-15
"""

import argparse
from pathlib import Path

import pandas as pd

GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NRFI performance badge (SVG + optional PNG).")
    p.add_argument("--model-version", default="nrfi_xgb_v1.0")
    p.add_argument("--grade", default="A-")
    p.add_argument("--window", default="ytd", choices=["7d", "14d", "30d", "mtd", "ytd"])
    p.add_argument("--date", default=None)
    p.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/joeplumber-mlb"))
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _included_grades(grade: str) -> set[str]:
    if grade not in GRADE_ORDER:
        raise ValueError(f"Unknown grade={grade}. Supported: {GRADE_ORDER}")
    idx = GRADE_ORDER.index(grade)
    return set(GRADE_ORDER[: idx + 1])


def _window_mask(df: pd.DataFrame, window: str, as_of: pd.Timestamp) -> pd.Series:
    d = pd.to_datetime(df["game_date"], errors="coerce")
    if window in {"7d", "14d", "30d"}:
        days = int(window[:-1])
        start = as_of.normalize() - pd.Timedelta(days=days - 1)
        return (d >= start) & (d <= as_of.normalize())
    if window == "mtd":
        return (d.dt.year == as_of.year) & (d.dt.month == as_of.month)
    return d.dt.year == as_of.year


def _build_svg(title: str, subline: str, footer: str, plays: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200" viewBox="0 0 600 200">
  <rect x="0" y="0" width="600" height="200" rx="18" ry="18" fill="#f7f7fb" stroke="#222" stroke-width="2"/>
  <text x="300" y="62" text-anchor="middle" font-size="42" font-family="sans-serif" font-weight="700" fill="#111">{title}</text>
  <text x="300" y="112" text-anchor="middle" font-size="28" font-family="sans-serif" font-weight="600" fill="#111">{subline}</text>
  <text x="300" y="150" text-anchor="middle" font-size="20" font-family="sans-serif" fill="#222">Plays: {plays}</text>
  <text x="300" y="178" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#555">{footer}</text>
</svg>
'''


def main() -> None:
    args = parse_args()
    drive_root = args.drive_root
    out_dir = args.out_dir or (drive_root / "data/badges/nrfi_xgb/v1.0")
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = drive_root / "data/public_ledgers/nrfi_xgb/v1.0/ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")

    ledger = pd.read_csv(ledger_path)
    required = {"game_date", "grade", "win"}
    missing = [c for c in required if c not in ledger.columns]
    if missing:
        raise ValueError(f"Ledger missing required columns: {missing}")

    ledger = ledger.copy()
    ledger["game_date"] = pd.to_datetime(ledger["game_date"], errors="coerce")
    ledger["win_num"] = pd.to_numeric(ledger["win"], errors="coerce")
    ledger = ledger.dropna(subset=["game_date"])

    if args.date:
        as_of = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    else:
        if ledger.empty:
            raise ValueError("Ledger has no valid game_date rows.")
        as_of = pd.to_datetime(ledger["game_date"].max())

    inc_grades = _included_grades(args.grade)
    sel = ledger[ledger["grade"].astype(str).isin(inc_grades)].copy()
    sel = sel[_window_mask(sel, args.window, as_of)]
    resolved = sel.dropna(subset=["win_num"]).copy()

    if resolved.empty:
        wins = 0
        losses = 0
        plays = 0
        subline = "No graded results yet"
    else:
        plays = int(len(resolved))
        wins = int(resolved["win_num"].sum())
        losses = plays - wins
        win_pct = (wins / plays) * 100.0
        subline = f"{args.window.upper()} A-tier: {wins}-{losses} ({win_pct:.1f}%)"

    as_of_date = as_of.strftime("%Y-%m-%d")
    svg_txt = _build_svg("NRFI v1.0", subline, f"Updated: {as_of_date}", plays)

    stem = f"badge_{args.model_version}_{args.window}_{as_of_date}"
    svg_path = out_dir / f"{stem}.svg"
    svg_path.write_text(svg_txt, encoding="utf-8")
    print(f"badge_svg={svg_path}")

    try:
        import cairosvg  # type: ignore

        png_path = out_dir / f"{stem}.png"
        cairosvg.svg2png(bytestring=svg_txt.encode("utf-8"), write_to=str(png_path))
        print(f"badge_png={png_path}")
    except Exception:  # noqa: BLE001
        print("badge_png=skipped (cairosvg unavailable)")


if __name__ == "__main__":
    main()
