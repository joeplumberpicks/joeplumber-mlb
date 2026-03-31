import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup

ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

POS_SET = {"C","1B","2B","3B","SS","LF","CF","RF","DH"}
HAND_SET = {"L","R","S"}

LINEUP_START = {"Confirmed Lineup", "Expected Lineup"}
STOP_TOKENS = {
    "Home Run Odds",
    "Starting Pitcher Intel",
    "Umpire:",
    "LINE",
    "O/U",
}

def _clean_player_name(text: str) -> str:
    text = text.replace("NONE", "").strip()
    parts = text.split()
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if parts and parts[0] in POS_SET:
        parts = parts[1:]
    return " ".join(parts).strip()

def _is_bad_name(name: str) -> bool:
    if not name:
        return True
    n = name.lower()
    bad = ["lineup", "pitcher", "intel", "era"]
    return any(b in n for b in bad)

def _parse_team(card, team):
    tokens = list(card.stripped_strings)

    # find lineup start
    start_idx = None
    for i, t in enumerate(tokens):
        if t in LINEUP_START:
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    rows = []
    slot = 1
    i = start_idx

    while i < len(tokens):
        tok = tokens[i]

        if tok in STOP_TOKENS:
            break

        if tok in POS_SET:
            pos = tok

            if i + 1 >= len(tokens):
                break

            name = _clean_player_name(tokens[i + 1])

            bats = None
            if i + 2 < len(tokens) and tokens[i + 2] in HAND_SET:
                bats = tokens[i + 2]

            if not _is_bad_name(name):
                rows.append({
                    "batter_team": team,
                    "player_name": name,
                    "lineup_slot": slot,
                    "position": pos,
                    "bats": bats,
                })
                slot += 1

            i += 3
            continue

        i += 1

    return rows[:9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    print("Fetching RotoWire lineups...")
    r = requests.get(ROTOWIRE_URL)
    soup = BeautifulSoup(r.text, "html.parser")

    cards = soup.find_all("div", class_="lineup")

    all_rows = []

    for card in cards:
        teams = card.find_all("div", class_="lineup__team")
        if len(teams) < 2:
            continue

        away = teams[0].text.strip()
        home = teams[1].text.strip()

        all_rows.extend(_parse_team(card, away))
        all_rows.extend(_parse_team(card, home))

    df = pd.DataFrame(all_rows)

    print("Parsed rows:", len(df))
    print(df.head(20))

    out_path = f"/content/drive/MyDrive/joeplumber-mlb/data/raw/live/projected_lineups_{args.season}_{args.date}.parquet"
    df.to_parquet(out_path, index=False)

    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
