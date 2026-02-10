# joeplumber-mlb

Milestone 1 foundation for a Colab-first MLB modeling repo that writes spine tables to Google Drive.

## Colab quickstart

1. **Mount Drive in a notebook cell (required before scripts):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Clone the repo:**
   ```bash
   !git clone <your-repo-url>
   %cd joeplumber-mlb
   ```
3. **Install dependencies:**
   ```bash
   !pip install -r requirements.txt
   ```
4. **Build the spine:**
   ```bash
   !python scripts/build_spine.py --season 2024 --start 2024-03-20 --end 2024-09-30
   ```

## Output location

Spine outputs are written to the processed folder under Drive root configured in `config/project.yaml`:

- `/content/drive/MyDrive/joeplumber-mlb/data/processed/games.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/pa.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/weather_game.parquet`
- `/content/drive/MyDrive/joeplumber-mlb/data/processed/parks.parquet`
