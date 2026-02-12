from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests


@dataclass
class RetryConfig:
    max_retries: int = 4
    base_backoff_seconds: float = 1.0
    backoff_cap_seconds: float = 30.0
    timeout_seconds: int = 25


class VisualCrossingClient:
    """Simple Visual Crossing hourly weather client."""

    def __init__(
        self,
        api_key: str | None,
        *,
        retry: RetryConfig | None = None,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key
        self.retry = retry or RetryConfig()
        self.session = session or requests.Session()
        self.log = logger or logging.getLogger(__name__)

    @classmethod
    def from_env(cls, api_key_env: str = "VISUAL_CROSSING_API_KEY", **kwargs: object) -> "VisualCrossingClient":
        return cls(api_key=os.environ.get(api_key_env), **kwargs)

    def fetch_hourly(self, lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch hourly weather rows inclusive between UTC datetimes."""
        if not self.api_key:
            raise RuntimeError(
                "Missing Visual Crossing API key. Set VISUAL_CROSSING_API_KEY or pass key via config/environment."
            )

        start_s = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_s = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{lat},{lon}/{start_s}/{end_s}"
        )
        params = {
            "unitGroup": "us",
            "include": "hours",
            "key": self.api_key,
            "contentType": "json",
        }

        for attempt in range(1, self.retry.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.session.get(url, params=params, timeout=self.retry.timeout_seconds)
                resp.raise_for_status()
                payload = resp.json()
                rows: list[dict[str, object]] = []
                for day in payload.get("days", []):
                    for hour in day.get("hours", []):
                        rows.append(
                            {
                                "obs_time": pd.to_datetime(hour.get("datetimeEpoch"), unit="s", utc=True),
                                "temp_f": hour.get("temp"),
                                "humidity": hour.get("humidity"),
                                "dewpoint_f": hour.get("dew"),
                                "pressure_mb": hour.get("pressure"),
                                "wind_speed_mph": hour.get("windspeed"),
                                "wind_dir_deg": hour.get("winddir"),
                                "precip_in": hour.get("precip"),
                                "conditions": hour.get("conditions"),
                                "provider": "visualcrossing",
                            }
                        )
                out = pd.DataFrame(rows)
                self.log.info(
                    "weather_request provider=visualcrossing lat=%.4f lon=%.4f attempt=%d rows=%d duration_s=%.2f",
                    lat,
                    lon,
                    attempt,
                    len(out),
                    time.time() - t0,
                )
                return out
            except Exception as exc:
                if attempt >= self.retry.max_retries:
                    raise RuntimeError(f"Visual Crossing request failed after {attempt} attempts: {exc}") from exc
                backoff = min(self.retry.backoff_cap_seconds, self.retry.base_backoff_seconds * (2 ** (attempt - 1)))
                backoff += random.uniform(0.0, 0.35 * backoff)
                self.log.warning(
                    "weather_request_retry provider=visualcrossing lat=%.4f lon=%.4f attempt=%d wait_s=%.2f error=%s",
                    lat,
                    lon,
                    attempt,
                    backoff,
                    exc,
                )
                time.sleep(backoff)


class NWSClient:
    """Minimal optional NWS client placeholder; only supports point metadata."""

    def __init__(self, *, session: requests.Session | None = None, logger: logging.Logger | None = None) -> None:
        self.session = session or requests.Session()
        self.log = logger or logging.getLogger(__name__)

    def fetch_hourly(self, lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        raise NotImplementedError("NWS hourly support is not enabled in this build; use provider=visualcrossing.")
