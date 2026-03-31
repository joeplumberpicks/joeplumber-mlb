from __future__ import annotations

import argparse
import logging
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# Colab upload support
try:
    from google.colab import files
    COLAB_AVAILABLE = True
except:
    COLAB_AVAILABLE = False


# ---------------- CONFIG ---------------- #

TEAM_MAP = {
    "Angels": "LOS ANGELES ANGELS",
    "Astros": "HOUSTON ASTROS",
    "Athletics": "ATHLETICS",
    "Blue Jays": "TORONTO BLUE JAYS",
    "Braves": "ATLANTA BRAVES",
    "Brewers": "MILWAUKEE BREWERS",
    "Cardinals": "ST. LOUIS CARDINALS",
    "Cubs": "CHICAGO CUBS",
    "Diamondbacks": "ARIZONA DIAMONDBACKS",
    "Dodgers": "LOS ANGELES DODGERS",
    "Giants": "SAN FRANCISCO GIANTS",
    "Guardians": "CLEVELAND GUARDIANS",
    "Mariners": "SEATTLE MARINERS",
    "Marlins": "MIAMI MARLINS",
    "Mets": "NEW YORK METS",
    "Nationals": "WASHINGTON NATIONALS",
    "Orioles": "BALTIMORE ORIOLES",
    "Padres": "SAN DIEGO PADRES",
    "Phillies": "PHILADELPHIA PHILLIES",
    "Pirates": "PITTSBURGH PIRATES",
    "Rangers": "TEXAS RANGERS",
    "Rays": "TAMPA BAY RAYS",
    "Red Sox": "BOSTON RED SOX",
    "Reds": "CINCINNATI REDS",
    "Rockies": "COLORADO ROCKIES",
    "Royals": "KANSAS CITY ROYALS",
    "Tigers": "DETROIT TIGERS",
    "Twins": "MINNESOTA TWINS",
    "White Sox": "CHICAGO WHITE SOX",
    "Yankees": "NEW YORK YANKEES",
}

POSITION_REGEX = re.compile(r"(C|1B|2B|3B|SS|LF|CF|RF|DH|OF)\s*\((\d)\)")


# ---------------- ARGUMENTS ---------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--html-path", type=str, default=None)
    p.add_argument("--upload-html", action="store_true")
    return p.parse_args()


# ---------------- HTML LOAD ---------------- #

def load_html(args):
    if args.upload_html:
        if not COLAB_AVAILABLE:
            raise RuntimeError("Upload only works in Colab")

        print("⬆️ Upload your FanGraphs HTML file")
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        return uploaded[file_name].decode("utf-8")

    if args.html_path:
        return Path(args.html_path).read_text(encoding="utf-8")

    raise RuntimeError("Must provide --html-path or --upload-html")


# ---------------- PARSER ---------------- #

def parse_lineups(html, game_date):
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    current_team = None

    for tag in soup.find_all(["th", "td"]):
        text = tag.get_text(strip=True)

        # Detect team headers
        if "Lineups" in text and "2026" in text:
            parts = text.replace("2026", "").replace("Lineups", "").strip()
            current_team = TEAM_MAP.get(parts, parts.upper())
            continue

        if not current_team:
            continue

        match = POSITION_REGEX.search(text)
        if match:
            position = match.group(1)
            slot = int(match.group(2))

            name_tag = tag.find_previous("a")
            if not name_tag:
                continue

            player = name_tag.get_text(strip=True)

            rows.append({
                "game_date": game_date,
                "batter_team": current_team,
                "canonical_team": current_team,
                "player_name": player,
                "lineup_slot": slot,
                "position": position,
                "lineup_status": "projected",
                "lineup_source": "fangraphs",
                "source_timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return pd.DataFrame(rows)


# ---------------- MAIN ---------------- #

def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Fangraphs lineup build")

    html = load_html(args)

    df = parse_lineups(html, args.date)

    if df.empty:
        raise RuntimeError("No lineup rows parsed")

    print("\n=== SAMPLE ===")
    print(df.head(15))

    out_path = f"/content/drive/MyDrive/joeplumber-mlb/data/processed/live/projected_lineups_{args.season}_{args.date}.parquet"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n✅ Saved to {out_path}")
    print(f"Rows: {len(df)} | Teams: {df['batter_team'].nunique()}")


if __name__ == "__main__":
    main()
