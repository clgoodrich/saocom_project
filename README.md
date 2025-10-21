# SAOCOM Validation Project

This repository validates SAOCOM-derived elevation products against multiple reference datasets and documents the reproducible pipeline.

## Quickstart
1. Clone the repository and navigate to the project root.
2. Create the environment:
   ```bash
   mamba env create -f environment.yml
   conda activate saocom
   ```
3. Install pre-commit hooks and linting tools:
   ```bash
   make setup
   ```
4. Run the full analysis pipeline:
   ```bash
   make run
   ```
5. (Optional) Open the numbered notebooks inside `notebooks/` for interactive exploration.

## Repository Layout
```
├─ README.md
├─ environment.yml
├─ Makefile
├─ data/
│  ├─ raw/
│  │  ├─ full/           # full-resolution datasets (not committed)
│  │  └─ sample/         # lightweight clips for demos/tests
│  ├─ interim/
│  └─ processed/
├─ notebooks/
│  ├─ 00_data_intake.ipynb
│  ├─ 01_cleaning.ipynb
│  ├─ 02_analysis.ipynb
│  └─ 03_figures_tables.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ io_utils.py
│  ├─ cleaning.py
│  ├─ analysis.py
│  └─ viz.py
├─ docs/
│  ├─ DATA_DICTIONARY.md
│  └─ METHODS.md
├─ outputs/
│  └─ ...
└─ .pre-commit-config.yaml
```

## Data Availability
Large upstream rasters and CSVs are stored in `data/raw/full/` and are not intended to be versioned. Add small representative samples to `data/raw/sample/` so new contributors can run the pipeline quickly. Document each field in `docs/DATA_DICTIONARY.md` and update provenance notes in `docs/METHODS.md` whenever datasets change.

## Development Notes
- Keep notebooks lean by delegating heavy lifting to functions under `src/`.
- Commit notebooks with outputs stripped (handled automatically by pre-commit).
- Prefer Parquet/Feather for intermediate tables stored in `data/interim/` or `data/processed/` when feasible.
- Record reproducibility metadata (random seeds, CRS definitions, etc.) within the notebooks and module docstrings.
