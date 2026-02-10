from __future__ import annotations

import os
from pathlib import Path


COLAB_DRIVE_MOUNT = Path("/content/drive/MyDrive")


def in_colab() -> bool:
    """Return True when running in a Google Colab runtime."""
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in os.environ.get("PYTHONPATH", "")


def ensure_drive_mounted() -> Path:
    """Ensure Google Drive is mounted by checking the standard MyDrive path.

    Raises:
        RuntimeError: If /content/drive/MyDrive does not exist.
    """
    if not COLAB_DRIVE_MOUNT.exists():
        raise RuntimeError(
            "Google Drive is not mounted. In a Colab notebook cell, run:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "Then re-run this script."
        )
    return COLAB_DRIVE_MOUNT


def resolve_data_dirs(config: dict) -> dict[str, Path]:
    """Resolve and create project data directories rooted at config.drive_root."""
    ensure_drive_mounted()

    drive_root = Path(config["drive_root"]).expanduser()
    path_config = config.get("paths", {})

    resolved: dict[str, Path] = {}
    for key in ("raw", "processed", "marts"):
        rel_path = path_config.get(key)
        if rel_path is None:
            raise KeyError(f"Missing paths.{key} in config")

        relative_parts = Path(rel_path).parts
        if relative_parts and relative_parts[0] == "data":
            relative_parts = relative_parts[1:]

        full_path = drive_root.joinpath("data", *relative_parts)
        full_path.mkdir(parents=True, exist_ok=True)
        resolved[key] = full_path

    return resolved
