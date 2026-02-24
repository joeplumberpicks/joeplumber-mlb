from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import safe_mkdir


def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        safe_mkdir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def log_header(script_name: str, repo_root: Path, config_path: Path, dirs_dict: dict[str, Path]) -> None:
    logging.info("========== %s =========", script_name)
    logging.info("repo_root: %s", repo_root.resolve())
    logging.info("config_path: %s", config_path.resolve())
    for key, value in dirs_dict.items():
        logging.info("%s: %s", key, value)
