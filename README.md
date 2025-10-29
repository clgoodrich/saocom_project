# SAOCOM DEM Validation Project

A comprehensive validation analysis of SAOCOM-derived Digital Elevation Models (DEMs) against reference DEMs (Copernicus DEM and TINItaly), incorporating terrain characteristics and land cover analysis.

## Project Overview

This project validates SAOCOM radar-derived elevation data against high-quality reference DEMs, analyzing residuals stratified by:
- Terrain characteristics (slope, aspect, elevation, curvature)
- Land cover types (CORINE classification)
- Radar geometry effects (shadow, layover, foreshortening)
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
│   ├── radar_geometry.py              # Radar shadow/geometry analysis
│   ├── control_points.py              # Control point identification
│   └── utils.py                       # Utility functions
│
├── notebooks/                         # Jupyter notebooks
│   ├── radar_shadow_analysis_cells.py # Shadow analysis notebook cells
│   ├── control_points_identification_cells.py # Control points cells
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
│   ├── RADAR_SHADOW_ANALYSIS.md       # Shadow analysis documentation
│   ├── QUICK_START_SHADOW_ANALYSIS.md # Shadow analysis quick start
│   ├── CONTROL_POINTS_GUIDE.md        # Control points guide
│   └── (presentations and papers)
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
from src.radar_geometry import calculate_local_incidence_angle, identify_shadow_areas
from src.control_points import identify_control_points, export_control_points
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
6. **Radar Geometry Analysis**: Calculate local incidence angles, identify shadow/layover areas
7. **Residual Analysis**: Compute Bias/NMAD/RMSE stratified by terrain, land cover, and geometry
8. **Visualization**: Generate maps, plots, Bland-Altman, 3D terrain views

## Output Artifacts

- `results/`: Processed tables, grids, analysis caches
- `images/`: Residual maps, histograms, violin plots, Bland-Altman, 3D terrain, radar geometry
- `docs/`: Documentation and presentation materials
- `topography_outputs/`: Slope, aspect, curvature derivatives, radar geometry layers

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

## 🆕 Radar Shadow Analysis

New feature for analyzing radar geometry effects on DEM accuracy!

### Quick Start

1. Open your Jupyter notebook
2. Copy cells from `notebooks/radar_shadow_analysis_cells.py`
3. Adjust SAOCOM geometry parameters (incidence angle, azimuth)
4. Run cells to generate shadow analysis

### What It Does

- **Calculates local incidence angles** accounting for terrain orientation
- **Identifies shadow areas** where radar cannot reach (>90° local incidence)
- **Detects layover zones** with severe geometric distortion (<20° local incidence)
- **Stratifies accuracy** by geometric quality (Optimal, Acceptable, Foreshortening, Shadow, Layover)
- **Visualizes results** with maps and accuracy plots

### Documentation

- **Full guide**: `docs/RADAR_SHADOW_ANALYSIS.md`
- **Quick start**: `docs/QUICK_START_SHADOW_ANALYSIS.md`
- **Module reference**: See docstrings in `src/radar_geometry.py`

### Example Output

```
Accuracy Statistics by Illumination Category:
Category         Count   Bias (m)   RMSE (m)   NMAD (m)
------------------------------------------------------------
optimal         12456       0.23       2.45       1.82
acceptable       8934       0.41       3.67       2.34
steep            2103       1.12       5.48       3.91
shadow            456       2.34       8.92       6.45
layover           789       1.67       6.73       4.21
```

Shadow areas typically show 2-3× higher RMSE compared to well-illuminated areas.


## 🎯 Control Points Identification

New feature for identifying high-quality control points where all DEMs agree!

### Quick Start

1. Open your Jupyter notebook
2. Copy cells from `notebooks/control_points_identification_cells.py`
3. Adjust tolerance parameter (default: ±2 meters)
4. Run cells to identify and visualize control points

### What It Does

- **Identifies consensus points** where SAOCOM, Copernicus, and TINItaly agree within ±2m
- **Analyzes spatial distribution** across terrain and land cover types
- **Calculates accuracy metrics** specifically at control points
- **Visualizes on Sentinel-2** showing control point locations
- **Recommends calibration points** - spatially distributed subset for validation
- **Exports to multiple formats** (GeoJSON, Shapefile, CSV)

### Why Control Points Matter

Control points represent high-confidence locations where:
- ✅ All three DEMs agree (within tolerance)
- ✅ Measurement quality is highest
- ✅ Terrain is stable and well-measured
- ✅ Ideal for calibration validation
- ✅ Suitable for ground truth collection

### Typical Results

```
Control Points Identified: 2,347 / 10,523 (22.3%)
Mean DEM agreement: 1.12 m
SAOCOM accuracy at control points:
  Bias: +0.18 m
  RMSE: 2.31 m
  NMAD: 1.67 m
```

### Outputs

**Visualizations:**
- `control_points_sentinel_overlay.png` - Control points on Sentinel-2 RGB
- `control_points_analysis_dashboard.png` - 6-panel analysis figure
- `recommended_calibration_points.png` - Distributed calibration points

**Data Files:**
- `results/control_points/*.geojson` - GeoJSON format
- `results/control_points/*.csv` - CSV with coordinates
- `results/control_points/*.shp` - Shapefile

### Documentation

- **Full guide**: `docs/CONTROL_POINTS_GUIDE.md`
- **Module reference**: See docstrings in `src/control_points.py`
- **Cell examples**: `notebooks/control_points_identification_cells.py`

### Tolerance Guidelines

- **±1.0m**: Strict quality (5-15% of points)
- **±2.0m**: Standard (15-30% of points) ← **Recommended**
- **±3.0m**: Moderate (30-50% of points)
- **±5.0m**: Loose (50-70% of points)

Choose based on your accuracy requirements and terrain complexity.

