# AGENTS.md

## Milestone 1 Definition of Done

- Scripts run in Google Colab with standard CLI invocation (`python scripts/build_spine.py ...`).
- Outputs are created under the Drive-rooted processed directory specified in config.
- Scripts must not perform interactive Drive mounting (`drive.mount`) internally.
- `src` package imports work without notebook-specific `sys.path` hacks.
- `games` data is deterministically deduplicated on `game_pk` before validation.

## Milestone 2A Definition of Done

- All scripts run in Colab after user-mounted Drive.
- All Milestone 2A outputs exist in Drive-rooted `data/processed`.
- No import errors when running scripts from repo root with `python scripts/<script>.py ...`.
- PK uniqueness is deterministically deduplicated and validated before writes.
