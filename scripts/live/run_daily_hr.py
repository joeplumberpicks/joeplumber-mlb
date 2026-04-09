import numpy as np
import pandas as pd

# =========================================================
# STEP 1 — AUDIT REQUIRED HR FEATURES
# =========================================================

required_cols = [
    "barrel_rate",
    "iso",
    "hr_per_pa",
    "hard_hit_rate",
    "lineup_spot"
]

print("\n=== HR FEATURE AUDIT ===")
for col in required_cols:
    if col not in df.columns:
        print(f"❌ MISSING COLUMN: {col}")
        df[col] = np.nan
    else:
        pct_null = df[col].isna().mean() * 100
        print(f"{col}: null_pct={pct_null:.2f}%")

# fill missing safely (do NOT zero out signal)
df["barrel_rate"] = df["barrel_rate"].fillna(df["barrel_rate"].median())
df["iso"] = df["iso"].fillna(df["iso"].median())
df["hr_per_pa"] = df["hr_per_pa"].fillna(df["hr_per_pa"].median())
df["hard_hit_rate"] = df["hard_hit_rate"].fillna(df["hard_hit_rate"].median())
df["lineup_spot"] = df["lineup_spot"].fillna(6)

# =========================================================
# STEP 2 — SOFT POWER FILTER (NOT HARD CUT)
# =========================================================

df["power_flag"] = (
    (df["barrel_rate"] >= 0.06) |
    (df["iso"] >= 0.160) |
    (df["hr_per_pa"] >= 0.025)
).astype(int)

# Instead of removing players → penalize weak profiles
df["power_penalty"] = np.where(df["power_flag"] == 0, 0.75, 1.0)

# =========================================================
# STEP 3 — BUILD HR POWER SCORE (CORE SIGNAL)
# =========================================================

df["hr_power_score"] = (
    df["barrel_rate"] * 0.45 +
    df["iso"] * 0.30 +
    df["hard_hit_rate"] * 0.25
)

# =========================================================
# STEP 4 — BASE SCORE (LOOSEN PENALTIES)
# =========================================================

# Start from your existing score if it exists
if "hr_score_raw" in df.columns:
    df["base_score"] = df["hr_score_raw"]
else:
    df["base_score"] = 0

# Add power boost (IMPORTANT: not too aggressive)
df["score"] = df["base_score"] + (df["hr_power_score"] * 1.25)

# Apply SOFT penalty instead of nuking players
df["score"] *= df["power_penalty"]

# =========================================================
# STEP 5 — LINEUP ADJUSTMENT (LESS AGGRESSIVE)
# =========================================================

df.loc[df["lineup_spot"] >= 7, "score"] *= 0.85
df.loc[df["lineup_spot"] >= 8, "score"] *= 0.70

# =========================================================
# STEP 6 — Z-SCORE NORMALIZATION (PER SLATE)
# =========================================================

mean = df["score"].mean()
std = df["score"].std()

df["z_score"] = (df["score"] - mean) / (std + 1e-6)
df["z_score"] = df["z_score"].clip(-2.5, 2.5)

# =========================================================
# STEP 7 — PROBABILITY CALIBRATION (FIXED)
# =========================================================

# DO NOT use raw score → use z_score
df["p_hr"] = 1 / (1 + np.exp(-(
    -2.2 + (df["z_score"] * 1.15)
)))

# =========================================================
# STEP 8 — CONFIDENCE TIERS (SPREAD FIX)
# =========================================================

df["confidence"] = pd.cut(
    df["z_score"],
    bins=[-10, -1.0, -0.25, 0.5, 1.25, 2.0, 10],
    labels=["F", "D", "C", "B", "A", "A+"]
)

# =========================================================
# STEP 9 — FINAL SORT (FIX YOUR ERROR)
# =========================================================

df["sort_score"] = df["z_score"]

board = df.sort_values(
    ["sort_score", "p_hr"],
    ascending=[False, False]
).reset_index(drop=True)

# =========================================================
# STEP 10 — FINAL OUTPUT CHECK
# =========================================================

print("\n=== HR BOARD DIAGNOSTIC ===")
print(board[[
    "player_name",
    "barrel_rate",
    "iso",
    "hr_per_pa",
    "lineup_spot",
    "z_score",
    "p_hr",
    "confidence"
]].head(15))