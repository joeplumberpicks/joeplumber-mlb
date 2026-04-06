import pandas as pd

def build_rbi_features(lineups, batter_roll, pitcher_roll):

    df = lineups.copy()

    df = df.merge(batter_roll, on="batter_id", how="left")

    df = df.merge(
        pitcher_roll,
        left_on="opp_pitcher_id",
        right_on="pitcher_id",
        how="left",
        suffixes=("", "_opp")
    )

    return df
