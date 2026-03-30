import pandas as pd

def _safe_notna(x):
    if isinstance(x, pd.Series):
        return x.notna()
    # scalar → convert to Series aligned with out
    return pd.Series([pd.notna(x)] * len(out), index=out.index)

out["has_projected_lineups"] = (
    _safe_notna(away_comp) | _safe_notna(home_comp)
)
