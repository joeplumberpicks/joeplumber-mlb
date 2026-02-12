from __future__ import annotations

import numpy as np
import pandas as pd


def air_density_proxy(temp_f: pd.Series, pressure_mb: pd.Series, humidity: pd.Series) -> pd.Series:
    """Simple pressure/temperature/humidity proxy for aerodynamic carry context."""
    temp_k = (pd.to_numeric(temp_f, errors="coerce") - 32.0) * (5.0 / 9.0) + 273.15
    pressure = pd.to_numeric(pressure_mb, errors="coerce")
    rh = pd.to_numeric(humidity, errors="coerce") / 100.0
    denom = temp_k * (1.0 + 0.61 * rh)
    return pressure / denom


def wind_out_to_cf_mph(wind_speed_mph: pd.Series, wind_dir_deg: pd.Series, park_cf_bearing_deg: pd.Series) -> pd.Series:
    """Positive means wind blowing out to CF; negative means blowing in."""
    ws = pd.to_numeric(wind_speed_mph, errors="coerce")
    wd = pd.to_numeric(wind_dir_deg, errors="coerce")
    cf = pd.to_numeric(park_cf_bearing_deg, errors="coerce")
    theta_rad = np.deg2rad(wd - cf)
    return ws * np.cos(theta_rad)


def infer_roof_closed_flag(conditions: pd.Series) -> pd.Series:
    cond = conditions.astype("string").str.lower()
    return cond.str.contains("dome|roof closed|indoors", na=False).astype("Int64")


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def add_weather_transforms(df: pd.DataFrame, park_cf_bearing_default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    if "park_cf_bearing_deg" not in out.columns:
        out["park_cf_bearing_deg"] = float(park_cf_bearing_default)

    out["air_density_proxy"] = air_density_proxy(out.get("temp_f"), out.get("pressure_mb"), out.get("humidity"))
    out["wind_out_to_cf_mph"] = wind_out_to_cf_mph(
        out.get("wind_speed_mph"), out.get("wind_dir_deg"), out.get("park_cf_bearing_deg")
    )
    out["roof_closed_flag"] = infer_roof_closed_flag(out.get("conditions", pd.Series(index=out.index, dtype="string")))
    out["temp_f_z"] = _zscore(out.get("temp_f", pd.Series(index=out.index, dtype=float)))
    out["humidity_z"] = _zscore(out.get("humidity", pd.Series(index=out.index, dtype=float)))
    return out
