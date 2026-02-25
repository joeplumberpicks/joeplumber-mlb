from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import get_repo_root
from src.utils.io import safe_mkdir


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def ensure_drive_mounted(mount_point: str = "/content/drive") -> bool:
    mount = Path(mount_point)
    if mount.exists() and (mount / "MyDrive").exists():
        return True

    if not _in_colab():
        return False

    from google.colab import drive  # type: ignore

    drive.mount(mount_point, force_remount=False)
    return mount.exists() and (mount / "MyDrive").exists()


def resolve_data_dirs(config: dict[str, Any] | None = None, prefer_drive: bool = True) -> dict[str, Path]:
    cfg = config or {}
    repo_root = get_repo_root()

    drive_mounted = ensure_drive_mounted() if prefer_drive else False
    drive_data_root = cfg.get("drive_data_root", "joeplumber-mlb/data")

    if drive_mounted and prefer_drive:
        data_root = Path("/content/drive/MyDrive") / drive_data_root
    else:
        data_root = repo_root / "data"

    dirs = {
        "data_root": data_root.resolve(),
        "raw_dir": (data_root / "raw").resolve(),
        "processed_dir": (data_root / "processed").resolve(),
        "reference_dir": (data_root / "reference").resolve(),
        "marts_dir": (data_root / "marts").resolve(),
        "models_dir": (data_root / "models").resolve(),
        "outputs_dir": (data_root / "outputs").resolve(),
        "backtests_dir": (data_root / "backtests").resolve(),
        "logs_dir": (data_root / "logs").resolve(),
    }

    for path in dirs.values():
        safe_mkdir(path)

    return dirs
