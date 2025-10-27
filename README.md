# SAOCOM DEM Validation Project

A comprehensive validation analysis of SAOCOM-derived Digital Elevation Models (DEMs) against reference DEMs (Copernicus DEM and TINItaly), incorporating terrain characteristics and land cover analysis.

## Project Overview

This project validates SAOCOM radar-derived elevation data against high-quality reference DEMs, analyzing residuals stratified by:
- Terrain characteristics (slope, aspect, elevation, curvature)
- Land cover types (CORINE classification)
- Spatial patterns

### Key Metrics
- **Bias**: Mean vertical error
- **NMAD**: Normalized Median Absolute Deviation (robust accuracy measure)
- **RMSE**: Root Mean Square Error

## Project Structure

```
saocom_project/
├── README.md                          # This file
├── CLAUDE.md                          # Claude Code project instructions
├── pyproject.toml                     # Project configuration and dependencies
├── environment.yaml                   # Conda environment specification
├── requirements.txt                   # Python package requirements
├── .gitignore                         # Git ignore rules
│
├── data/                              # Input data (not in version control)
│   ├── saocom_csv/                    # SAOCOM point data (10m spacing)
│   ├── copernicus.tif                 # Copernicus DEM (30m)
│   ├── demCOP30.tif                   # Copernicus DEM
│   ├── tinitaly/                      # TINItaly DEM (10m)
│   ├── corine/                        # CORINE land cover
│   ├── ground_cover/                  # Ground cover rasters
│   └── sentinel_data/                 # Sentinel-2 RGB for visualization
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py               # Data loading and geometric prep
│   ├── calibration.py                 # Height calibration routines
│   ├── outlier_detection.py           # Outlier filtering methods
│   ├── landcover.py                   # Land cover processing
│   ├── statistics_prog.py             # Statistical analysis
│   ├── visualization.py               # Plotting functions
│   └── utils.py                       # Utility functions
│
├── notebooks/                         # Jupyter notebooks
│   └── (exploration notebooks)
│
├── tests/                             # Unit tests
│   └── __init__.py
│
├── results/                           # Processed data and analysis results
│   └── (generated tables, grids, caches)
│
├── images/                            # Output figures
│   └── (residual maps, plots, 3D terrain)
│
├── docs/                              # Documentation and slides
│
├── topography_outputs/                # Terrain derivatives
│   └── (slope, aspect, curvature)
│
├── _archive/                          # Archived development files
│   ├── scripts/                       # Old fix/test scripts
│   ├── reports/                       # Development reports
│   ├── old_notebooks/                 # Previous notebook versions
│   ├── temp_files/                    # Temporary analysis files
│   └── ide_config/                    # IDE configuration
│
├── saocom_analysis_clean.ipynb        # Main analysis notebook (clean)
└── saocom_v3.ipynb                    # Primary workflow notebook

```

## Setup

### Option 1: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yaml

# Activate environment
conda activate saocom-dem-validation
```

### Option 2: Using pip + virtualenv

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows Git Bash)
source .venv/Scripts/activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Development Install

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Running the Main Analysis

```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - saocom_analysis_clean.ipynb (cleaned analysis)
# - saocom_v3.ipynb (primary workflow)
```

### Using Source Modules

```python
from src.preprocessing import load_saocom_data, resample_to_common_grid
from src.calibration import calibrate_heights
from src.outlier_detection import detect_outliers_isolation_forest
from src.statistics_prog import compute_residual_stats
from src.visualization import plot_residual_map
```

## Data Requirements

The project expects data in the following structure:
- SAOCOM data: CSV files with point coordinates and relative heights (10m spacing)
- Reference DEMs: GeoTIFF format, EPSG:4326 (WGS84)
- Land cover: CORINE raster with DBF attribute table

All inputs must share a common spatial extent.

## Workflow Overview

1. **Load & QC**: Load data, check coordinate validity and datum
2. **Geometric Prep**: Resample to 10m common grid, create masks
3. **Calibrate**: Apply median offset correction to SAOCOM heights
4. **Outlier Handling**: Filter outliers using Isolation Forest/IQR/KNN
5. **Land Cover Sampling**: Extract CORINE classifications at point locations
6. **Residual Analysis**: Compute Bias/NMAD/RMSE stratified by terrain and land cover
7. **Visualization**: Generate maps, plots, Bland-Altman, 3D terrain views

## Output Artifacts

- `results/`: Processed tables, grids, analysis caches
- `images/`: Residual maps, histograms, violin plots, Bland-Altman, 3D terrain
- `docs/`: Documentation and presentation materials
- `topography_outputs/`: Slope, aspect, curvature derivatives

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

## Citation

If you use this code or methodology, please cite:
[Add citation information here]

## License

[Specify license - MIT suggested in pyproject.toml]

## Contact

[Add contact information]

## Acknowledgments

- Copernicus DEM: EU Copernicus Programme
- TINItaly DEM: INGV (Istituto Nazionale di Geofisica e Vulcanologia)
- CORINE Land Cover: European Environment Agency
- SAOCOM: CONAE (Comisión Nacional de Actividades Espaciales)
