#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.id_resolution import build_and_save_lookups
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player ID lookup parquets from historical data.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    build_and_save_lookups(processed_dir)


if __name__ == "__main__":
    main()