<<<<<<< HEAD
# SAOCOM DEM Validation Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Comprehensive validation analysis of SAOCOM-derived Digital Elevation Models (DEMs) against reference DEMs with advanced terrain, land cover, and radar geometry analysis.

**Author:** Colton Goodrich
**Version:** 1.1.0
**Status:** Active Development

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Advanced Features](#advanced-features)
- [Output Artifacts](#output-artifacts)
- [Data Requirements](#data-requirements)
- [Development](#development)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project validates SAOCOM L-band radar-derived elevation data against high-quality reference Digital Elevation Models (DEMs), providing comprehensive accuracy assessment stratified by:

- **Terrain characteristics**: slope, aspect, elevation, curvature
- **Land cover types**: CORINE classification hierarchy
- **Radar geometry effects**: shadow, layover, foreshortening
- **Spatial patterns**: control points and void zone analysis

### Key Metrics

- **Bias**: Mean vertical error
- **NMAD**: Normalized Median Absolute Deviation (robust accuracy measure)
- **RMSE**: Root Mean Square Error

All metrics are computed globally and stratified by terrain, land cover, and geometric quality categories.

---

## Features

✨ **Core Capabilities:**
- 📊 Multi-DEM validation (SAOCOM vs Copernicus vs TINItaly)
- 🗺️ Terrain-stratified accuracy assessment
- 🌳 Land cover classification integration (CORINE)
- 📡 Radar geometry quality analysis
- 🎯 Control point identification for calibration
- 🔬 PCA-based void zone factor analysis
- 📈 Comprehensive visualization suite

🆕 **Recent Additions:**
- Principal Component Analysis of data quality factors
- Automated control point identification
- Radar shadow and layover detection
- Sentinel-2 overlay visualizations

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended) or pip + virtualenv
- GDAL-compatible system libraries

### Quick Install

**Option 1: Conda (Recommended)**
```bash
conda env create -f environment.yaml && conda activate saocom-dem-validation
```

**Option 2: pip + virtualenv**
```bash
# Windows (Git Bash/PowerShell)
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt

# Linux/Mac
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Detailed Installation

<details>
<summary>Click to expand detailed installation instructions</summary>

#### Using Conda

```bash
# Create environment from file
conda env create -f environment.yaml

# Activate environment
conda activate saocom-dem-validation

# Verify installation
python -c "import numpy, pandas, geopandas, rasterio, sklearn; print('✓ All packages installed successfully!')"
```

#### Using pip + virtualenv

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows Git Bash/PowerShell)
source .venv/Scripts/activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, geopandas, rasterio, sklearn; print('✓ All packages installed successfully!')"
```

#### Development Install

```bash
# Activate your environment first (conda or venv)
# Then install in editable mode with dev dependencies
pip install -e ".[dev]"
```

#### Troubleshooting

**If conda is slow:**
```bash
# Use mamba for faster dependency resolution
conda install mamba -c conda-forge
mamba env create -f environment.yaml
```

**If GDAL installation fails with pip:**
```bash
# Install GDAL via conda even in a venv environment
conda install -c conda-forge gdal
# Then continue with pip for other packages
```

**To update an existing environment:**
```bash
# Conda
conda env update -f environment.yaml --prune

# Pip
pip install -r requirements.txt --upgrade
```

</details>

---

## Quick Start

1. **Install dependencies** (see [Installation](#installation))

2. **Prepare your data** (see [Data Requirements](#data-requirements))

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Run the main analysis:**
   - Open `saocom_analysis_clean.ipynb`
   - Run all cells sequentially
   - Results will be saved to `images/` and `results/`

5. **Review outputs:**
   - Accuracy statistics in console
   - Visualizations in `images/`
   - Exported data in `results/`

---

## Usage

### Running the Main Analysis

```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - saocom_analysis_clean.ipynb (main analysis notebook)
# - saocom_v3.ipynb (alternative workflow)
```

### Using Source Modules Programmatically

```python
from src.preprocessing import load_saocom_data, resample_to_common_grid
from src.calibration import calibrate_heights
from src.outlier_detection import detect_outliers_isolation_forest
from src.statistics_prog import compute_residual_stats
from src.visualization import plot_residual_map
from src.radar_geometry import calculate_local_incidence_angle, identify_shadow_areas
from src.control_points import identify_control_points, export_control_points

# Example workflow
saocom_data = load_saocom_data('data/saocom_csv/')
calibrated = calibrate_heights(saocom_data, reference_dem)
stats = compute_residual_stats(calibrated)
```

---

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
│   ├── tinitaly/                      # TINItaly DEM (10m)
│   ├── corine/                        # CORINE land cover
│   └── sentinel_data/                 # Sentinel-2 RGB for visualization
│
├── src/                               # Source code modules
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
├── notebooks/                         # Additional Jupyter notebooks
├── tests/                             # Unit tests
├── results/                           # Processed data and analysis results
├── images/                            # Output figures
├── docs/                              # Documentation
├── topography_outputs/                # Terrain derivatives
│
├── saocom_analysis_clean.ipynb        # Main analysis notebook ⭐
└── saocom_v3.ipynb                    # Alternative workflow notebook
```

---

## Workflow

The analysis follows a standardized 9-step pipeline:

1. **Load & QC**: Load SAOCOM points and reference DEMs, check coordinate validity
2. **Geometric Prep**: Resample to common 10m grid, create convex hull masks
3. **Calibrate**: Apply median offset correction to SAOCOM relative heights
4. **Outlier Handling**: Filter outliers using Isolation Forest/IQR/KNN methods
5. **Land Cover Sampling**: Extract CORINE classifications at point locations
6. **Radar Geometry Analysis**: Calculate local incidence angles, identify shadow/layover
7. **Residual Analysis**: Compute Bias/NMAD/RMSE stratified by terrain/land cover/geometry
8. **Visualization**: Generate maps, plots, Bland-Altman diagrams, 3D terrain views
9. **PCA Void Zone Analysis**: Identify factors contributing to poor data quality

Each step is documented in detail within the Jupyter notebooks.

---

## Advanced Features

### 🔬 PCA Void Zone Analysis

Principal Component Analysis reveals which factors contribute most to data quality issues.

**Void zones defined as areas with:**
- Low coherence (< 0.5)
- Shadow areas
- High residuals (top 10%)
- High outlier scores (top 10%)

**Features analyzed:**
- Temporal coherence, terrain metrics, measurement quality
- Geometry: local incidence angles
- Land cover: one-hot encoded categories

**Output:** 5×2 grid visualization with scree plots, loadings, scatter plots, and distributions

📍 **Location:** Section 13 in `saocom_analysis_clean.ipynb` (cells 97-100)

---

### 🎯 Control Points Identification

Identifies high-quality control points where SAOCOM, Copernicus, and TINItaly all agree within ±2m.

**Features:**
- Automated consensus point identification
- Spatial distribution analysis
- Sentinel-2 overlay visualization
- Export to GeoJSON, Shapefile, CSV

**Typical Results:**
```
Control Points: 2,347 / 10,523 (22.3%)
Mean DEM agreement: 1.12 m
SAOCOM RMSE at control points: 2.31 m
```

**Tolerance Guidelines:**
- **±2.0m**: Standard (15-30% of points) ← Recommended
- **±1.0m**: Strict quality (5-15% of points)
- **±3.0m**: Moderate (30-50% of points)

📖 **Documentation:** `docs/CONTROL_POINTS_GUIDE.md`

---

### 📡 Radar Shadow Analysis

Analyzes radar geometry effects on DEM accuracy, including shadow and layover detection.

**Capabilities:**
- Local incidence angle calculation
- Shadow area identification (>90° local incidence)
- Layover detection (<20° local incidence)
- Geometric quality stratification

**Example Output:**
```
Accuracy Statistics by Illumination Category:
Category         Count   Bias (m)   RMSE (m)   NMAD (m)
------------------------------------------------------------
optimal         12,456      0.23       2.45       1.82
shadow             456      2.34       8.92       6.45
layover            789      1.67       6.73       4.21
```

📖 **Documentation:** `docs/RADAR_SHADOW_ANALYSIS.md`

---

## Output Artifacts

### Generated Files

- **`results/`** - Processed tables, grids, analysis caches
- **`images/`** - Residual maps, histograms, violin plots, Bland-Altman, 3D terrain
- **`docs/`** - Documentation and presentation materials
- **`topography_outputs/`** - Slope, aspect, curvature derivatives

### Key Visualizations

- Residual maps overlaid on Sentinel-2
- Accuracy stratified by slope, land cover, geometry
- Control points on RGB backdrop
- PCA feature contribution analysis
- Radar shadow and layover maps

---

## Data Requirements

### Input Data Structure

The project expects data in the following formats:

- **SAOCOM data**: CSV files with columns: `LAT2`, `LON2`, `HEIGHT`, `COHER`, `SIGMA HEIGHT`
  - Point spacing: ~10m
  - Coordinate system: EPSG:4326 (WGS84)

- **Reference DEMs**: GeoTIFF format, EPSG:4326
  - TINItaly: 10m resolution
  - Copernicus: 30m resolution (auto-resampled to 10m)

- **Land cover**: CORINE raster with DBF attribute table
  - Resolution: ~30m
  - Includes Level 1, 2, 3 classification codes

- **Sentinel-2**: RGB composite GeoTIFF for visualization backdrop

**Important:** All inputs must share a common spatial extent.

### Data Directory Structure

```
data/
├── saocom_csv/
│   └── *.csv                  # SAOCOM point measurements
├── copernicus.tif             # Copernicus DEM
├── tinitaly/
│   └── tinitaly_crop.tif      # TINItaly DEM
├── corine/
│   ├── *.tif                  # CORINE land cover raster
│   └── *.dbf                  # CORINE lookup table
└── sentinel_data/
    └── Sentinel2Views_Clip.tif  # RGB composite
```

---


---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{goodrich2025saocom,
  author = {Goodrich, Colton},
  title = {SAOCOM DEM Validation Project},
  year = {2025},
  version = {1.1.0},
  url = {https://github.com/clgoodrich/saocom_project}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

**Colton Goodrich**
- GitHub: [@clgoodrich](https://github.com/clgoodrich)
- Email: coltonlg@gmail.com

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

## Acknowledgments

This project utilizes data and tools from:

- **SAOCOM Mission**: CONAE (Comisión Nacional de Actividades Espaciales), Argentina
- **Copernicus DEM**: European Union Copernicus Programme
- **TINItaly DEM**: INGV (Istituto Nazionale di Geofisica e Vulcanologia), Italy
- **CORINE Land Cover**: European Environment Agency (EEA)
- **Sentinel-2**: ESA (European Space Agency) Copernicus Programme

Special thanks to the open-source geospatial community for developing essential tools: GDAL, Rasterio, GeoPandas, scikit-learn, and Matplotlib.

---

**Last Updated:** 2025-01-10
**Maintained by:** Colton Goodrich
=======
# SAOCOM_DEMs
Applied Geospatial Computation Lab Project: Accuracy assessment of SAOCOM-derived InSAR time-series DEMs over Italy

Presently use:
https://github.com/clgoodrich/saocom_project/
>>>>>>> e13920335db9c4d6efe2f99fc7548c75bc13f86e
