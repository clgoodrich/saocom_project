# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAOCOM DEM Validation Study - A geospatial analysis project that validates SAOCOM satellite InSAR-derived height measurements against reference Digital Elevation Models (DEMs). The project performs statistical comparison, outlier detection, land cover analysis, and comprehensive visualization of height discrepancies across different terrain types.

## Data Architecture

### Input Data Structure

All data files are stored in the `data/` directory and are clipped to the same spatial extent (EPSG:4326):

- **SAOCOM InSAR Heights** (`data/saocom_csv/`): CSV format, 10m resolution
  - Contains extracted heights from SAOCOM satellite imagery
  - Heights are relative and require reference point calibration
  - Negative values may exist before calibration

- **Reference DEMs**:
  - **Copernicus DEM** (`data/copernicus.tif`, `data/demCOP30.tif`): 30m resolution
  - **TINItaly DEM** (`data/tinitaly/tinitaly_crop.tif`): 10m resolution, high-accuracy reference

- **Land Cover Data** (`data/corine/`, `data/ground_cover/`):
  - CORINE Land Cover classification (30m resolution)
  - DBF lookup table for land cover class definitions

- **Sentinel-2 Imagery** (`data/sentinel_data/`): RGB orthoimagery for visualization

### Output Directories

- `results/`: Analysis outputs and processed datasets
- `images/`: Visualization outputs and figures
- `docs/`: Project documentation and presentations
- `topography_outputs/`: Terrain derivative analyses

## Main Analysis Workflow

The primary analysis is contained in `saocom_v3.ipynb`. The workflow follows these steps:

1. **Setup & Data Loading**
   - Load SAOCOM CSV data and filter spatial outliers
   - Load reference DEMs (Copernicus, TINItaly)
   - Verify horizontal datum consistency

2. **Geometric Processing**
   - Resample reference DEMs to 10m resolution for consistency
   - Create rasterized mask from SAOCOM point convex hull
   - Sample reference DEM values at SAOCOM point locations

3. **Height Calibration**
   - SAOCOM heights are relative - must be calibrated to reference DEM
   - Calibration typically uses median offset correction

4. **Outlier Detection**
   - Isolation Forest algorithm for anomaly detection
   - IQR-based filtering of residuals
   - Spatial clustering analysis (KNN-based isolated point removal)

5. **Land Cover Analysis**
   - Sample CORINE land cover at SAOCOM point locations
   - Classify into hierarchical levels (Level 1, 2, 3)
   - Calculate statistics per land cover class

6. **Statistical Comparison**
   - Compute residuals: SAOCOM - Reference DEM
   - Calculate NMAD (Normalized Median Absolute Deviation)
   - Generate summary statistics per terrain type

7. **Visualization**
   - Spatial maps: residuals, overlays, classification
   - Statistical plots: violin plots, histograms, scatter plots
   - Bland-Altman plots for systematic bias analysis
   - 3D terrain models

## Running the Analysis

### Environment Setup

The project uses a Python virtual environment (`.venv/`). Key dependencies:

**Geospatial:**
- `rasterio` - Raster I/O and processing
- `geopandas` - Vector data manipulation
- `shapely` - Geometric operations
- `pyproj` - Coordinate reference system transformations

**Data Analysis:**
- `numpy`, `pandas` - Array and dataframe operations
- `scipy` - Statistical analysis and interpolation
- `scikit-learn` - Machine learning (outlier detection)

**Visualization:**
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualizations
- `plotly` - Interactive 3D visualizations
- `matplotlib-scalebar` - Map scale bars

**Other:**
- `dbfread` - DBF file parsing for land cover lookup

### Running the Notebook

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Launch Jupyter
jupyter notebook saocom_v3.ipynb
```

Execute cells sequentially. The notebook is designed to run top-to-bottom without cell reordering.

## Key Utility Functions

The notebook defines reusable functions for common operations:

### Raster Operations
- `_read_raster_meta(path)` - Extract raster metadata
- `_resample_to_10m(src_path, out_name)` - Resample DEM to 10m
- `_mask_and_write(arr, out_name)` - Apply mask and write raster
- `_sample(arr)` - Sample raster at SAOCOM point locations

### Statistical Functions
- `nmad(x)` - Calculate Normalized Median Absolute Deviation
- `calculate_height_stats(data, name)` - Comprehensive height statistics
- `generate_height_statistics_summary(gdf, gdf_name)` - Summary statistics table

### Outlier Detection
- `score_outliers_isolation_forest(gdf, residual_col, **kwargs)` - Isolation Forest scoring
- `filter_by_score_iqr(gdf_scored, iqr_multiplier)` - IQR-based outlier filtering
- `remove_isolated_knn(gdf, k, distance_threshold)` - Remove spatially isolated points

### Calibration
- `_calibrate(ref_col, out_col)` - Calibrate SAOCOM heights to reference DEM

### Visualization
- `visualize_outlier_results(gdf_original, gdf_cleaned, outliers, residual_col)` - Outlier detection visualization
- `create_difference_grid(gdf, height_col, ref_col, grid_shape, transform, hull_mask)` - Gridded difference maps
- `plot_bland_altman(ax, x_data, y_data, x_label, y_label, title)` - Bland-Altman plot
- `calculate_terrain_derivatives(dem, cellsize)` - Slope, aspect, curvature

### Land Cover
- `get_clc_level1(code)` - Extract Level 1 CORINE class from code

## Important Notes

### SAOCOM Height Calibration
InSAR heights from SAOCOM are **relative, not absolute**. Before any analysis:
1. Select a stable reference point or region
2. Calculate offset between SAOCOM and reference DEM
3. Apply calibration offset to all SAOCOM points

The notebook typically uses median offset correction for calibration.

### Coordinate Reference Systems
- All input data uses **EPSG:4326** (WGS84 lat/lon)
- Vertical datum may differ between datasets - verify before analysis
- The notebook includes horizontal datum verification steps

### Data Processing Order
The notebook cells must be run sequentially. Key dependencies:
- Cell 4 loads reference DEMs (used by later cells)
- Calibration must occur before residual analysis
- Outlier detection should precede final statistical analysis

### Memory Considerations
- Large rasters (Sentinel-2, CORINE) may consume significant memory
- The notebook includes optimizations like data clipping and resampling
- Consider clearing outputs if encountering memory issues

### Missing Dependencies
If encountering import errors, check that all required packages are installed in `.venv`. The notebook does not include a `requirements.txt` - dependencies must be inferred from imports.
