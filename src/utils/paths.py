from __future__ import annotations

import os
from pathlib import Path


def get_data_root() -> Path:
    """Resolve repository data root with Drive-friendly defaults."""
    root = os.getenv("JOEPLUMBER_DATA_ROOT")
    if root:
        return Path(root)
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/joeplumber-mlb/data")
    return Path("data")


def raw_dir() -> Path:
    return get_data_root() / "raw"


def processed_dir() -> Path:
    return get_data_root() / "processed"


def outputs_dir() -> Path:
    return get_data_root() / "outputs"


def models_dir() -> Path:
    return get_data_root() / "models"


def reference_dir() -> Path:
    return get_data_root() / "reference"


def _test_get_data_root_env() -> None:
    prev = os.environ.get("JOEPLUMBER_DATA_ROOT")
    try:
        os.environ["JOEPLUMBER_DATA_ROOT"] = "/tmp/joeplumber-data"
        assert get_data_root() == Path("/tmp/joeplumber-data")
    finally:
        if prev is None:
            os.environ.pop("JOEPLUMBER_DATA_ROOT", None)
        else:
            os.environ["JOEPLUMBER_DATA_ROOT"] = prev


def _test_get_data_root_drive_mock() -> None:
    prev = os.environ.get("JOEPLUMBER_DATA_ROOT")
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/content/drive/MyDrive":
            return True
        return original_exists(self)

    try:
        os.environ.pop("JOEPLUMBER_DATA_ROOT", None)
        Path.exists = fake_exists  # type: ignore[assignment]
        assert get_data_root() == Path("/content/drive/MyDrive/joeplumber-mlb/data")
    finally:
        Path.exists = original_exists  # type: ignore[assignment]
        if prev is not None:
            os.environ["JOEPLUMBER_DATA_ROOT"] = prev


def _test_get_data_root_local_default() -> None:
    prev = os.environ.get("JOEPLUMBER_DATA_ROOT")
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/content/drive/MyDrive":
            return False
        return original_exists(self)

    try:
        os.environ.pop("JOEPLUMBER_DATA_ROOT", None)
        Path.exists = fake_exists  # type: ignore[assignment]
        assert get_data_root() == Path("data")
    finally:
        Path.exists = original_exists  # type: ignore[assignment]
        if prev is not None:
            os.environ["JOEPLUMBER_DATA_ROOT"] = prev


def run_path_resolver_self_tests() -> None:
    _test_get_data_root_env()
    _test_get_data_root_drive_mock()
    _test_get_data_root_local_default()


if __name__ == "__main__":
    run_path_resolver_self_tests()
    print("paths.py self-tests passed")
