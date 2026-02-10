# AGENTS.md

## Milestone 1 Definition of Done

- Scripts run in Google Colab with standard CLI invocation (`python scripts/build_spine.py ...`).
- Outputs are created under the Drive-rooted processed directory specified in config.
- Scripts must not perform interactive Drive mounting (`drive.mount`) internally.
- `src` package imports work without notebook-specific `sys.path` hacks.
- `games` data is deterministically deduplicated on `game_pk` before validation.
