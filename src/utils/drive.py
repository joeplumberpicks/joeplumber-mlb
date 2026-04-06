from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_data_dirs(config: dict[str, Any], prefer_drive: bool = True) -> dict[str, str]:
    """
    Resolve Joe Plumber data lake directories.

    Priority:
    1) /content/drive/MyDrive/<drive_data_root> when prefer_drive=True and mounted
    2) repo-local ./data fallback
    """
    drive_data_root = str(config.get("drive_data_root", "joeplumber-mlb/data")).strip()

    drive_root = Path("/content/drive/MyDrive") / drive_data_root
    local_root = Path("data")

    use_drive = bool(prefer_drive and Path("/content/drive/MyDrive").exists())

    data_root = drive_root if use_drive else local_root

    dirs = {
        "data_root": data_root,
        "raw_dir": data_root / "raw",
        "processed_dir": data_root / "processed",
        "reference_dir": data_root / "reference",
        "marts_dir": data_root / "marts",
        "models_dir": data_root / "models",
        "outputs_dir": data_root / "outputs",
        "backtests_dir": data_root / "backtests",
        "logs_dir": data_root / "logs",
    }

    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return {k: str(v) for k, v in dirs.items()}
