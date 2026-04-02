"""
Joe Plumber MLB Engine — Layer 2: Spine

This package builds unified pregame spine tables from Layer 1 ingest outputs.
"""

from .build_model_spine_game import build_model_spine_game

__all__ = ["build_model_spine_game"]
