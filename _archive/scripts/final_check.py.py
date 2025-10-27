#!/usr/bin/env python
# coding: utf-8

# # SAOCOM InSAR Height Validation Against Reference DEMs
# 
# **Author:** SAOCOM Analysis Team  
# **Purpose:** Validate SAOCOM satellite InSAR-derived heights against high-quality reference DEMs
# 
# ## Overview
# 
# This notebook demonstrates a complete workflow for validating SAOCOM InSAR height measurements against two reference Digital Elevation Models (DEMs):
# - **TINItaly DEM** (10m resolution, high accuracy)
# - **Copernicus DEM** (30m resolution, global coverage)
# 
# ### Analysis Steps:
# 1. Load and preprocess SAOCOM point cloud data
# 2. Resample reference DEMs to common resolution (10m)
# 3. Sample reference DEM heights at SAOCOM point locations
# 4. Calibrate SAOCOM relative heights to absolute heights
# 5. Detect and remove outliers using machine learning
# 6. Perform statistical analysis of height differences
# 7. Analyze performance by land cover type
# 8. Generate comprehensive visualizations
# 
# ### Key Concepts:
# - **InSAR Heights**: SAOCOM provides *relative* heights that require calibration to a reference
# - **Coherence**: Quality metric for InSAR measurements (0-1, higher is better)
# - **NMAD**: Normalized Median Absolute Deviation, a robust accuracy metric
# - **Outliers**: Anomalous measurements detected using Isolation Forest algorithm

# ---
# ## 1. Setup & Imports

# In[1]:


# Standard library
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd
import geopandas as gpd

# Geospatial
import rasterio
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import Point

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom modules from src/
import sys
sys.path.append('./src')

from utils import read_raster_meta, load_dem_array
from preprocessing import (
    resample_to_10m,
    mask_and_write,
    sample_raster_at_points,
    create_difference_grid,
    calculate_terrain_derivatives
)
from calibration import calibrate_heights
from outlier_detection import (
    remove_isolated_knn,
    score_outliers_isolation_forest,
    filter_by_score_iqr,
    visualize_outlier_results
)
from statistics_prog import (
    nmad,
    calculate_height_stats,
    generate_height_statistics_summary
)
from landcover import get_clc_level1
from visualization import (
    plot_raster_with_stats,
    plot_distribution_histogram,
    plot_scatter_comparison,
    plot_bland_altman
)

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('colorblind')

print("✓ All modules imported successfully")


# ### Define Paths

# In[2]:


# Base directories
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./results')
IMAGES_DIR = Path('./images')

# Create output directories if needed
RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Input data paths
SAOCOM_CSV = DATA_DIR / 'verona_fullGraph_weighted_Tcoh07_edited.csv'
TINITALY_DEM = DATA_DIR / 'tinitaly' / 'tinitaly_crop.tif'
COPERNICUS_DEM = DATA_DIR / 'copernicus.tif'
CORINE_LC = DATA_DIR / 'corine_clip.tif'

# Output paths
TINITALY_10M = RESULTS_DIR / 'tinitaly_10m.tif'
COPERNICUS_10M = RESULTS_DIR / 'copernicus_10m.tif'
SAOCOM_CLEANED_SHP = RESULTS_DIR / 'saocom_cleaned.shp'

print(f"Data directory: {DATA_DIR.absolute()}")
print(f"Results directory: {RESULTS_DIR.absolute()}")
print(f"SAOCOM CSV: {SAOCOM_CSV.name}")


# ---
# ## 2. Load SAOCOM Data
# 
# SAOCOM data comes as a CSV with point coordinates and InSAR-derived heights. Key columns:
# - `HEIGHT_RELATIVE`: Relative height from InSAR (requires calibration)
# - `COHER`: Temporal coherence (quality metric, 0-1)
# - `EASTING`, `NORTHING`: UTM coordinates (EPSG:32632 for Italy)

# In[3]:


# Load SAOCOM CSV
saocom_df = pd.read_csv(SAOCOM_CSV)
print(f"Loaded {len(saocom_df):,} SAOCOM points")
print(f"Columns: {list(saocom_df.columns)}")
print(f"First few rows:")
print(saocom_df.head())
# Use LAT2/LON2 preferentially, fall back to LAT/LON
# Convert from geographic (lat/lon) to UTM Zone 32N
lat_col = 'LAT2' if 'LAT2' in saocom_df.columns else 'LAT'
lon_col = 'LON2' if 'LON2' in saocom_df.columns else 'LON'
print(f"Using coordinate columns: {lat_col}, {lon_col}")
# Create geometry from lat/lon (EPSG:4326)
geometry = [Point(xy) for xy in zip(saocom_df[lon_col], saocom_df[lat_col])]
saocom_gdf = gpd.GeoDataFrame(saocom_df, geometry=geometry, crs='EPSG:4326')
# Convert to UTM Zone 32N for Italy
saocom_gdf = saocom_gdf.to_crs('EPSG:32632')
# Rename HEIGHT column to HEIGHT_RELATIVE for consistency
if 'HEIGHT' in saocom_gdf.columns and 'HEIGHT_RELATIVE' not in saocom_gdf.columns:
    saocom_gdf['HEIGHT_RELATIVE'] = saocom_gdf['HEIGHT']
print(f"GeoDataFrame created")
print(f"  Original CRS: EPSG:4326 (WGS84)")
print(f"  Converted to: {saocom_gdf.crs}")
print(f"  Bounds: {saocom_gdf.total_bounds}")


# ### Remove Spatially Isolated Points
# 
# Isolated points far from other measurements may be erroneous. We use k-nearest neighbors to identify and remove them.

# In[4]:


# Remove isolated points using KNN
print("Removing spatially isolated points...")
saocom_gdf = remove_isolated_knn(saocom_gdf, k=100, distance_threshold=1000)

print(f"\nAfter spatial filtering: {len(saocom_gdf):,} points")

# Quick visualization
fig, ax = plt.subplots(figsize=(10, 8))
saocom_gdf.plot(ax=ax, markersize=1, color='blue', alpha=0.5)
ax.set_title('SAOCOM Point Cloud (after spatial filtering)', fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()


# ---
# ## 3. Load and Resample Reference DEMs
# 
# We need to resample both reference DEMs to a common 10m resolution to match SAOCOM's spatial resolution.

# In[5]:


# Define target grid parameters (10m resolution)
bounds = saocom_gdf.total_bounds  # [minx, miny, maxx, maxy]
RESOLUTION = 10.0  # meters

# Calculate grid dimensions
grid_width = int((bounds[2] - bounds[0]) / RESOLUTION)
grid_height = int((bounds[3] - bounds[1]) / RESOLUTION)

# Create affine transform for 10m grid
target_transform = from_bounds(
    bounds[0], bounds[1], bounds[2], bounds[3],
    grid_width, grid_height
)
target_crs = saocom_gdf.crs

print(f"Target grid: {grid_width} x {grid_height} pixels at {RESOLUTION}m resolution")
print(f"Grid bounds: {bounds}")


# ### Resample TINItaly DEM (10m → 10m)

# In[6]:


# TINItaly is already 10m, but we resample to align grids
print("Resampling TINItaly DEM...")
tinitaly_10m, _ = resample_to_10m(
    src_path=TINITALY_DEM,
    output_path=TINITALY_10M,
    target_transform=target_transform,
    target_crs=target_crs,
    grid_height=grid_height,
    grid_width=grid_width
)

print(f"TINItaly resampled: {tinitaly_10m.shape}")
print(f"Saved to: {TINITALY_10M}")


# ### Resample Copernicus DEM (30m → 10m)

# In[7]:


# Copernicus needs upsampling from 30m to 10m
print("Resampling Copernicus DEM...")
copernicus_10m, _ = resample_to_10m(
    src_path=COPERNICUS_DEM,
    output_path=COPERNICUS_10M,
    target_transform=target_transform,
    target_crs=target_crs,
    grid_height=grid_height,
    grid_width=grid_width
)

print(f"Copernicus resampled: {copernicus_10m.shape}")
print(f"Saved to: {COPERNICUS_10M}")


# ---
# ## 4. Sample DEMs at SAOCOM Point Locations
# 
# Extract reference DEM heights at each SAOCOM measurement point for comparison.

# In[8]:


# Convert point coordinates to raster row/col indices
rows, cols = rowcol(
    target_transform,
    saocom_gdf.geometry.x,
    saocom_gdf.geometry.y
)
rows = np.array(rows, dtype=int)
cols = np.array(cols, dtype=int)

# Check which points are within grid bounds
inbounds = (
    (rows >= 0) & (rows < grid_height) &
    (cols >= 0) & (cols < grid_width)
)

print(f"Points within grid bounds: {inbounds.sum():,} / {len(saocom_gdf):,}")


# In[9]:


# Sample TINItaly at SAOCOM points
saocom_gdf['tinitaly_height'] = sample_raster_at_points(
    tinitaly_10m, rows, cols, inbounds, nodata=-9999
)

# Sample Copernicus at SAOCOM points
saocom_gdf['copernicus_height'] = sample_raster_at_points(
    copernicus_10m, rows, cols, inbounds, nodata=-9999
)

# Check sampling success
n_tinitaly = saocom_gdf['tinitaly_height'].notna().sum()
n_copernicus = saocom_gdf['copernicus_height'].notna().sum()

print(f"\nSuccessfully sampled:")
print(f"  TINItaly: {n_tinitaly:,} points")
print(f"  Copernicus: {n_copernicus:,} points")

# Preview sampled data
print("\nSample data:")
print(saocom_gdf[['HEIGHT_RELATIVE', 'tinitaly_height', 'copernicus_height', 'COHER']].head(10))


# ---
# ## 5. Calibrate SAOCOM Heights
# 
# SAOCOM InSAR provides **relative** heights, not absolute elevations. We calibrate to reference DEMs using high-coherence points to estimate the vertical offset.

# ### Calibrate to TINItaly

# In[10]:


# Calibrate using high-coherence points (COHER >= 0.8)
print("Calibrating SAOCOM heights to TINItaly...")
offset_tin, rmse_tin, n_tin = calibrate_heights(
    saocom_gdf,
    ref_col='tinitaly_height',
    out_col='HEIGHT_ABSOLUTE_TIN',
    coherence_threshold=0.8
)

print(f"\nCalibration Results (TINItaly):")
print(f"  Offset applied: {offset_tin:.2f} m")
print(f"  RMSE: {rmse_tin:.2f} m")
print(f"  Calibration points: {n_tin:,}")

# Calculate residuals (difference after calibration)
saocom_gdf['diff_tinitaly'] = saocom_gdf['HEIGHT_ABSOLUTE_TIN'] - saocom_gdf['tinitaly_height']


# ### Calibrate to Copernicus

# In[11]:


print("Calibrating SAOCOM heights to Copernicus...")
offset_cop, rmse_cop, n_cop = calibrate_heights(
    saocom_gdf,
    ref_col='copernicus_height',
    out_col='HEIGHT_ABSOLUTE_COP',
    coherence_threshold=0.8
)

print(f"\nCalibration Results (Copernicus):")
print(f"  Offset applied: {offset_cop:.2f} m")
print(f"  RMSE: {rmse_cop:.2f} m")
print(f"  Calibration points: {n_cop:,}")

# Calculate residuals
saocom_gdf['diff_copernicus'] = saocom_gdf['HEIGHT_ABSOLUTE_COP'] - saocom_gdf['copernicus_height']


# ---
# ## 6. Outlier Detection
# 
# Use **Isolation Forest** machine learning algorithm to detect spatial and statistical anomalies in the residuals.

# In[12]:


# Score outliers using TINItaly residuals (more accurate reference)
print("Detecting outliers using Isolation Forest...")
saocom_scored = score_outliers_isolation_forest(
    saocom_gdf,
    residual_col='diff_tinitaly',
    contamination=0.05,  # Expect ~5% outliers
    n_estimators=100,
    random_state=42
)

print(f"Outlier scores computed for {len(saocom_scored):,} points")
print(f"Score range: [{saocom_scored['outlier_score'].min():.3f}, {saocom_scored['outlier_score'].max():.3f}]")


# In[13]:


# Filter outliers using IQR method
saocom_cleaned, outliers = filter_by_score_iqr(
    saocom_scored,
    iqr_multiplier=1.5  # More permissive than default (1.0)
)

print(f"\nOutlier Detection Results:")
print(f"  Original points: {len(saocom_gdf):,}")
print(f"  Outliers detected: {len(outliers):,} ({100*len(outliers)/len(saocom_gdf):.1f}%)")
print(f"  Cleaned dataset: {len(saocom_cleaned):,} points")


# ### Visualize Outlier Detection Results

# In[14]:


# Generate outlier visualization
visualize_outlier_results(
    gdf_original=saocom_gdf,
    gdf_cleaned=saocom_cleaned,
    outliers=outliers,
    residual_col='diff_tinitaly',
    results_dir=RESULTS_DIR
)


# ---
# ## 7. Statistical Analysis
# 
# Compute comprehensive statistics comparing SAOCOM to both reference DEMs.

# In[15]:


# Generate complete statistical summary
generate_height_statistics_summary(saocom_cleaned, gdf_name="SAOCOM (Cleaned)")


# ### Calculate NMAD (Robust Accuracy Metric)

# In[16]:


# NMAD for TINItaly comparison
residuals_tin = saocom_cleaned['diff_tinitaly'].dropna()
nmad_tin = nmad(residuals_tin)

# NMAD for Copernicus comparison
residuals_cop = saocom_cleaned['diff_copernicus'].dropna()
nmad_cop = nmad(residuals_cop)

print("\n" + "="*60)
print("ROBUST ACCURACY METRICS (NMAD)")
print("="*60)
print(f"SAOCOM vs TINItaly:    NMAD = {nmad_tin:.2f} m  (n={len(residuals_tin):,})")
print(f"SAOCOM vs Copernicus:  NMAD = {nmad_cop:.2f} m  (n={len(residuals_cop):,})")
print("="*60)

# NMAD is preferred over RMSE for height accuracy as it's less sensitive to outliers


# ### Distribution Analysis

# In[17]:


# Create distribution comparison plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# TINItaly residuals
metrics_tin = {
    'n_points': len(residuals_tin),
    'mean_diff': residuals_tin.mean(),
    'rmse': np.sqrt((residuals_tin**2).mean()),
    'nmad': nmad_tin,
    'std_diff': residuals_tin.std()
}
plot_distribution_histogram(axes[0], residuals_tin, 'SAOCOM - TINItaly', metrics_tin)

# Copernicus residuals
metrics_cop = {
    'n_points': len(residuals_cop),
    'mean_diff': residuals_cop.mean(),
    'rmse': np.sqrt((residuals_cop**2).mean()),
    'nmad': nmad_cop,
    'std_diff': residuals_cop.std()
}
plot_distribution_histogram(axes[1], residuals_cop, 'SAOCOM - Copernicus', metrics_cop)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'residual_distributions.png', dpi=300, bbox_inches='tight')
plt.show()


# ---
# ## 8. Terrain Analysis
# 
# Calculate slope and aspect from reference DEMs to understand how terrain affects InSAR accuracy.

# In[18]:


# Calculate slope and aspect from TINItaly
print("Calculating terrain derivatives from TINItaly...")
slope_tin, aspect_tin = calculate_terrain_derivatives(
    tinitaly_10m,
    cellsize=10,
    nodata=-9999
)

print(f"Slope range: [{np.nanmin(slope_tin):.1f}°, {np.nanmax(slope_tin):.1f}°]")
print(f"Mean slope: {np.nanmean(slope_tin):.1f}°")


# In[19]:


# Sample slope and aspect at SAOCOM cleaned points
# IMPORTANT: Must recalculate row/col indices for cleaned dataset
rows_clean, cols_clean = rowcol(    target_transform,    saocom_cleaned.geometry.x,    saocom_cleaned.geometry.y)
rows_clean = np.array(rows_clean, dtype=int)
cols_clean = np.array(cols_clean, dtype=int)
# Check bounds for cleaned dataset
inbounds_clean = (    (rows_clean >= 0) & (rows_clean < grid_height) &    (cols_clean >= 0) & (cols_clean < grid_width))
# Sample terrain derivatives
saocom_cleaned['slope_tin'] = sample_raster_at_points(    slope_tin, rows_clean, cols_clean, inbounds_clean, nodata=-9999)
saocom_cleaned['aspect_tin'] = sample_raster_at_points(    aspect_tin, rows_clean, cols_clean, inbounds_clean, nodata=-9999)
print(f"Sampled terrain derivatives for {saocom_cleaned['slope_tin'].notna().sum():,} points")


# ### Analyze Accuracy vs Slope

# In[20]:


# Bin residuals by slope categories
slope_bins = [0, 5, 15, 30, 90]
slope_labels = ['Flat (0-5°)', 'Gentle (5-15°)', 'Moderate (15-30°)', 'Steep (>30°)']

saocom_cleaned['slope_category'] = pd.cut(
    saocom_cleaned['slope_tin'],
    bins=slope_bins,
    labels=slope_labels
)

# Calculate NMAD by slope category
slope_stats = saocom_cleaned.groupby('slope_category')['diff_tinitaly'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('nmad', lambda x: nmad(x.dropna()))
]).round(2)

print("\nAccuracy by Slope Category:")
print(slope_stats)


# ---
# ## 9. Land Cover Analysis
# 
# Sample CORINE Land Cover to understand how different surface types affect InSAR accuracy.

# In[21]:


# Load and sample land cover
with rasterio.open(CORINE_LC) as src:
    corine_data = src.read(1)
    corine_transform = src.transform
    corine_crs = src.crs

# Reproject SAOCOM points if needed
if saocom_cleaned.crs != corine_crs:
    saocom_lc = saocom_cleaned.to_crs(corine_crs)
else:
    saocom_lc = saocom_cleaned

# Sample land cover codes
lc_rows, lc_cols = rowcol(
    corine_transform,
    saocom_lc.geometry.x,
    saocom_lc.geometry.y
)
lc_rows = np.array(lc_rows, dtype=int)
lc_cols = np.array(lc_cols, dtype=int)

# Check bounds
lc_inbounds = (
    (lc_rows >= 0) & (lc_rows < corine_data.shape[0]) &
    (lc_cols >= 0) & (lc_cols < corine_data.shape[1])
)

# Extract codes
lc_codes = np.full(len(saocom_lc), np.nan)
lc_codes[lc_inbounds] = corine_data[lc_rows[lc_inbounds], lc_cols[lc_inbounds]]

saocom_cleaned['corine_code'] = lc_codes
saocom_cleaned['land_cover'] = saocom_cleaned['corine_code'].apply(
    lambda x: get_clc_level1(int(x)) if pd.notna(x) else 'Unknown'
)

print(f"Land cover sampled for {saocom_cleaned['land_cover'].notna().sum():,} points")
print("\nLand cover distribution:")
print(saocom_cleaned['land_cover'].value_counts())


# ### Accuracy by Land Cover Type

# In[22]:


# Calculate statistics by land cover
lc_stats = saocom_cleaned.groupby('land_cover')['diff_tinitaly'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('nmad', lambda x: nmad(x.dropna()))
]).round(2)

print("\nAccuracy by Land Cover Type:")
print(lc_stats)

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
lc_stats['nmad'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('SAOCOM Accuracy (NMAD) by Land Cover Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Land Cover Category', fontsize=12)
ax.set_ylabel('NMAD (m)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(IMAGES_DIR / 'accuracy_by_landcover.png', dpi=300, bbox_inches='tight')
plt.show()


# ---
# ## 10. Advanced Visualizations

# ### Scatter Plots: SAOCOM vs Reference DEMs

# In[23]:


# Prepare data for scatter plots
valid_tin = saocom_cleaned[['HEIGHT_ABSOLUTE_TIN', 'tinitaly_height']].dropna()
valid_cop = saocom_cleaned[['HEIGHT_ABSOLUTE_COP', 'copernicus_height']].dropna()

# Calculate statistics
stats_tin_scatter = {
    'n_points': len(valid_tin),
    'mean_diff': (valid_tin['HEIGHT_ABSOLUTE_TIN'] - valid_tin['tinitaly_height']).mean(),
    'rmse': np.sqrt(((valid_tin['HEIGHT_ABSOLUTE_TIN'] - valid_tin['tinitaly_height'])**2).mean()),
    'correlation': np.corrcoef(valid_tin['HEIGHT_ABSOLUTE_TIN'], valid_tin['tinitaly_height'])[0, 1]
}

stats_cop_scatter = {
    'n_points': len(valid_cop),
    'mean_diff': (valid_cop['HEIGHT_ABSOLUTE_COP'] - valid_cop['copernicus_height']).mean(),
    'rmse': np.sqrt(((valid_cop['HEIGHT_ABSOLUTE_COP'] - valid_cop['copernicus_height'])**2).mean()),
    'correlation': np.corrcoef(valid_cop['HEIGHT_ABSOLUTE_COP'], valid_cop['copernicus_height'])[0, 1]
}

# Create scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_scatter_comparison(
    axes[0],
    valid_tin['tinitaly_height'].values,
    valid_tin['HEIGHT_ABSOLUTE_TIN'].values,
    'TINItaly Height (m)',
    'SAOCOM Height (m)',
    'SAOCOM vs TINItaly',
    stats_tin_scatter
)

plot_scatter_comparison(
    axes[1],
    valid_cop['copernicus_height'].values,
    valid_cop['HEIGHT_ABSOLUTE_COP'].values,
    'Copernicus Height (m)',
    'SAOCOM Height (m)',
    'SAOCOM vs Copernicus',
    stats_cop_scatter
)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'scatter_comparisons.png', dpi=300, bbox_inches='tight')
plt.show()


# ### Bland-Altman Plots

# In[24]:


# Bland-Altman analysis shows agreement between measurement methods
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_bland_altman(
    axes[0],
    valid_tin['tinitaly_height'].values,
    valid_tin['HEIGHT_ABSOLUTE_TIN'].values,
    'TINItaly',
    'SAOCOM',
    'Bland-Altman: SAOCOM vs TINItaly',
    fig=fig
)

plot_bland_altman(
    axes[1],
    valid_cop['copernicus_height'].values,
    valid_cop['HEIGHT_ABSOLUTE_COP'].values,
    'Copernicus',
    'SAOCOM',
    'Bland-Altman: SAOCOM vs Copernicus',
    fig=fig
)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'bland_altman.png', dpi=300, bbox_inches='tight')
plt.show()


# ### Spatial Distribution of Residuals

# In[25]:


# Create spatial map of residuals
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# TINItaly residuals
valid_pts_tin = saocom_cleaned[saocom_cleaned['diff_tinitaly'].notna()]
vmin, vmax = np.percentile(valid_pts_tin['diff_tinitaly'], [2, 98])

sc1 = axes[0].scatter(
    valid_pts_tin.geometry.x,
    valid_pts_tin.geometry.y,
    c=valid_pts_tin['diff_tinitaly'],
    cmap='RdBu_r',
    s=3,
    vmin=vmin,
    vmax=vmax,
    alpha=0.7
)
plt.colorbar(sc1, ax=axes[0], label='Residual (m)')
axes[0].set_title('SAOCOM - TINItaly Residuals', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('Northing (m)')
axes[0].set_aspect('equal')
axes[0].grid(alpha=0.3)

# Copernicus residuals
valid_pts_cop = saocom_cleaned[saocom_cleaned['diff_copernicus'].notna()]
vmin2, vmax2 = np.percentile(valid_pts_cop['diff_copernicus'], [2, 98])

sc2 = axes[1].scatter(
    valid_pts_cop.geometry.x,
    valid_pts_cop.geometry.y,
    c=valid_pts_cop['diff_copernicus'],
    cmap='RdBu_r',
    s=3,
    vmin=vmin2,
    vmax=vmax2,
    alpha=0.7
)
plt.colorbar(sc2, ax=axes[1], label='Residual (m)')
axes[1].set_title('SAOCOM - Copernicus Residuals', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Northing (m)')
axes[1].set_aspect('equal')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'spatial_residuals.png', dpi=300, bbox_inches='tight')
plt.show()


# ---
# ## 11. Export Results

# In[26]:


# Save cleaned dataset to shapefile
saocom_cleaned.to_file(SAOCOM_CLEANED_SHP)
print(f"Cleaned SAOCOM data saved to: {SAOCOM_CLEANED_SHP}")

# Export summary statistics to CSV
summary_stats = {
    'Reference_DEM': ['TINItaly', 'Copernicus'],
    'N_Points': [len(residuals_tin), len(residuals_cop)],
    'Mean_Residual_m': [residuals_tin.mean(), residuals_cop.mean()],
    'Std_Dev_m': [residuals_tin.std(), residuals_cop.std()],
    'RMSE_m': [np.sqrt((residuals_tin**2).mean()), np.sqrt((residuals_cop**2).mean())],
    'NMAD_m': [nmad_tin, nmad_cop],
    'Min_m': [residuals_tin.min(), residuals_cop.min()],
    'Max_m': [residuals_tin.max(), residuals_cop.max()]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(RESULTS_DIR / 'validation_summary.csv', index=False)
print(f"Summary statistics saved to: {RESULTS_DIR / 'validation_summary.csv'}")

print("\n" + summary_df.to_string(index=False))


# ---
# ## Summary & Conclusions
# 
# This notebook demonstrated a complete workflow for validating SAOCOM InSAR heights against reference DEMs:
# 
# ### Key Findings:
# 1. **SAOCOM requires calibration**: InSAR heights are relative and need reference DEM calibration
# 2. **Accuracy varies by terrain**: Flat terrain shows better agreement than steep slopes
# 3. **Land cover matters**: Accuracy differs across vegetation types and surface characteristics
# 4. **Outlier detection improves results**: Machine learning helps identify anomalous measurements
# 
# ### Best Practices:
# - Always use high-coherence points (COHER >= 0.8) for calibration
# - Apply spatial filtering to remove isolated points
# - Use NMAD instead of RMSE for robust accuracy assessment
# - Consider terrain and land cover when interpreting results
# 
# ### Next Steps:
# - Temporal analysis: Compare multiple acquisition dates
# - Physical modeling: Incorporate atmospheric corrections
# - Machine learning: Predict accuracy from terrain/land cover features
# - Integration: Combine SAOCOM with other SAR sensors (Sentinel-1, etc.)

# ---
# ## 12. Additional Visualizations
# 
# Comprehensive visualization suite from the original analysis.
# 

# ### 12.1 Spatial Coverage Map
# 
# Verify that SAOCOM points fall within the reference DEM extent.
# 

# In[ ]:


# Spatial overlap visualization
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(12, 10))

# TINITALY extent box
with rasterio.open(TINITALY_DEM) as src:
    dem_bounds = src.bounds
    # Reproject bounds to target CRS if needed
    import rasterio.warp
    dem_bounds_utm = rasterio.warp.transform_bounds(src.crs, TARGET_CRS, *dem_bounds)

    ax.add_patch(Rectangle(
        (dem_bounds_utm[0], dem_bounds_utm[1]),
        dem_bounds_utm[2] - dem_bounds_utm[0],
        dem_bounds_utm[3] - dem_bounds_utm[1],
        linewidth=3, edgecolor='blue', facecolor='none', label='TINItaly Extent'
    ))

# SAOCOM points
saocom_cleaned.plot(ax=ax, markersize=1, color='red', alpha=0.5, label='SAOCOM Points')

# Study area hull
from shapely.geometry import box
hull = saocom_cleaned.geometry.unary_union.convex_hull
hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)
hull_gdf.boundary.plot(ax=ax, color='green', linewidth=2, linestyle='--', label='Study Area Hull')

ax.set_xlabel('UTM Easting (m)', fontsize=12)
ax.set_ylabel('UTM Northing (m)', fontsize=12)
ax.set_title('Spatial Coverage: SAOCOM vs TINItaly DEM', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'spatial_coverage.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.2 Gridded Comparison Analysis
# 
# Create gridded difference maps to show spatial patterns of height differences.
# 

# In[ ]:


# Create gridded difference maps
print("Creating gridded difference maps...")

# Create difference grids
diff_grid_tin = create_difference_grid(
    saocom_cleaned,
    height_col='HEIGHT_ABSOLUTE_TIN',
    ref_col='tinitaly_height',
    grid_shape=(grid_height, grid_width),
    transform=target_transform,
    hull_mask=None
)

diff_grid_cop = create_difference_grid(
    saocom_cleaned,
    height_col='HEIGHT_ABSOLUTE_COP',
    ref_col='copernicus_height',
    grid_shape=(grid_height, grid_width),
    transform=target_transform,
    hull_mask=None
)

# Visualize gridded differences
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# TINItaly grid
valid_mask_tin = ~np.isnan(diff_grid_tin)
if valid_mask_tin.any():
    vmin, vmax = np.percentile(diff_grid_tin[valid_mask_tin], [2, 98])
    im1 = axes[0].imshow(diff_grid_tin, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[0], label='Difference (m)')
    axes[0].set_title('SAOCOM - TINItaly (Gridded)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

# Copernicus grid
valid_mask_cop = ~np.isnan(diff_grid_cop)
if valid_mask_cop.any():
    vmin2, vmax2 = np.percentile(diff_grid_cop[valid_mask_cop], [2, 98])
    im2 = axes[1].imshow(diff_grid_cop, cmap='RdBu_r', vmin=vmin2, vmax=vmax2)
    plt.colorbar(im2, ax=axes[1], label='Difference (m)')
    axes[1].set_title('SAOCOM - Copernicus (Gridded)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'gridded_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.3 Density Plots (Hexbin)
# 
# Hexbin plots show the density of measurements, useful for identifying data clustering.
# 

# In[ ]:


# Hexbin density plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# TINItaly hexbin
valid_tin = saocom_cleaned[['HEIGHT_ABSOLUTE_TIN', 'tinitaly_height']].dropna()
hb1 = axes[0].hexbin(
    valid_tin['tinitaly_height'],
    valid_tin['HEIGHT_ABSOLUTE_TIN'],
    gridsize=50,
    cmap='YlOrRd',
    mincnt=1,
    edgecolors='none'
)
plt.colorbar(hb1, ax=axes[0], label='Count')

# 1:1 line
lims = [min(valid_tin['tinitaly_height'].min(), valid_tin['HEIGHT_ABSOLUTE_TIN'].min()),
        max(valid_tin['tinitaly_height'].max(), valid_tin['HEIGHT_ABSOLUTE_TIN'].max())]
axes[0].plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='1:1 Line')

axes[0].set_xlabel('TINItaly Height (m)', fontsize=12)
axes[0].set_ylabel('SAOCOM Height (m)', fontsize=12)
axes[0].set_title('Density: SAOCOM vs TINItaly', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Copernicus hexbin
valid_cop = saocom_cleaned[['HEIGHT_ABSOLUTE_COP', 'copernicus_height']].dropna()
hb2 = axes[1].hexbin(
    valid_cop['copernicus_height'],
    valid_cop['HEIGHT_ABSOLUTE_COP'],
    gridsize=50,
    cmap='YlOrRd',
    mincnt=1,
    edgecolors='none'
)
plt.colorbar(hb2, ax=axes[1], label='Count')

# 1:1 line
lims2 = [min(valid_cop['copernicus_height'].min(), valid_cop['HEIGHT_ABSOLUTE_COP'].min()),
         max(valid_cop['copernicus_height'].max(), valid_cop['HEIGHT_ABSOLUTE_COP'].max())]
axes[1].plot(lims2, lims2, 'k--', alpha=0.5, linewidth=2, label='1:1 Line')

axes[1].set_xlabel('Copernicus Height (m)', fontsize=12)
axes[1].set_ylabel('SAOCOM Height (m)', fontsize=12)
axes[1].set_title('Density: SAOCOM vs Copernicus', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'hexbin_density.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.4 2D Histograms
# 
# Alternative visualization of measurement density.
# 

# In[ ]:


# 2D histogram plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# TINItaly 2D histogram
h1 = axes[0].hist2d(
    valid_tin['tinitaly_height'],
    valid_tin['HEIGHT_ABSOLUTE_TIN'],
    bins=100,
    cmap='viridis',
    cmin=1
)
plt.colorbar(h1[3], ax=axes[0], label='Count')

# 1:1 line
axes[0].plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='1:1 Line')
axes[0].set_xlabel('TINItaly Height (m)', fontsize=12)
axes[0].set_ylabel('SAOCOM Height (m)', fontsize=12)
axes[0].set_title('2D Histogram: SAOCOM vs TINItaly', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Copernicus 2D histogram
h2 = axes[1].hist2d(
    valid_cop['copernicus_height'],
    valid_cop['HEIGHT_ABSOLUTE_COP'],
    bins=100,
    cmap='viridis',
    cmin=1
)
plt.colorbar(h2[3], ax=axes[1], label='Count')

# 1:1 line
axes[1].plot(lims2, lims2, 'r--', alpha=0.7, linewidth=2, label='1:1 Line')
axes[1].set_xlabel('Copernicus Height (m)', fontsize=12)
axes[1].set_ylabel('SAOCOM Height (m)', fontsize=12)
axes[1].set_title('2D Histogram: SAOCOM vs Copernicus', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'hist2d_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.5 Violin Plots - Accuracy by Slope Category
# 
# Detailed performance breakdown showing full distribution of residuals for each terrain type.
# 

# In[ ]:


# Violin plot of residuals by slope category
fig, ax = plt.subplots(figsize=(12, 7))

# Prepare data for violin plot
slope_data = saocom_cleaned[['slope_category', 'diff_tinitaly']].dropna()

# Create violin plot
parts = ax.violinplot(
    [slope_data[slope_data['slope_category'] == cat]['diff_tinitaly'].values 
     for cat in slope_labels],
    positions=range(len(slope_labels)),
    showmeans=True,
    showmedians=True,
    widths=0.7
)

# Customize colors
for pc in parts['bodies']:
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.7)

ax.set_xticks(range(len(slope_labels)))
ax.set_xticklabels(slope_labels, rotation=0)
ax.set_xlabel('Slope Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (SAOCOM - TINItaly) [m]', fontsize=12, fontweight='bold')
ax.set_title('Residual Distribution by Slope Category', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Zero Error')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'violin_plot_slope.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nStatistics by slope category:")
print(slope_stats)


# ### 12.6 Residuals vs Coherence
# 
# Investigate the relationship between measurement quality (coherence) and accuracy.
# 

# In[ ]:


# Scatter plot of residuals vs coherence
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# TINItaly residuals vs coherence
valid_data_tin = saocom_cleaned[['COHER', 'diff_tinitaly']].dropna()
axes[0].scatter(
    valid_data_tin['COHER'],
    valid_data_tin['diff_tinitaly'],
    c=valid_data_tin['diff_tinitaly'],
    cmap='RdBu_r',
    s=5,
    alpha=0.3,
    vmin=-10,
    vmax=10
)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Coherence', fontsize=12)
axes[0].set_ylabel('Residual (m)', fontsize=12)
axes[0].set_title('Residuals vs Coherence (TINItaly)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Copernicus residuals vs coherence
valid_data_cop = saocom_cleaned[['COHER', 'diff_copernicus']].dropna()
axes[1].scatter(
    valid_data_cop['COHER'],
    valid_data_cop['diff_copernicus'],
    c=valid_data_cop['diff_copernicus'],
    cmap='RdBu_r',
    s=5,
    alpha=0.3,
    vmin=-10,
    vmax=10
)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Coherence', fontsize=12)
axes[1].set_ylabel('Residual (m)', fontsize=12)
axes[1].set_title('Residuals vs Coherence (Copernicus)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'residuals_vs_coherence.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.7 Terrain Slope Map
# 
# Visualize the terrain slope across the study area.
# 

# In[ ]:


# Display slope raster
fig, ax = plt.subplots(figsize=(12, 10))

# Plot slope
slope_plot = ax.imshow(slope_tin, cmap='terrain', vmin=0, vmax=45)
cbar = plt.colorbar(slope_plot, ax=ax, label='Slope (degrees)')
cbar.ax.tick_params(labelsize=10)

ax.set_title('Terrain Slope from TINItaly DEM', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'terrain_slope.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Slope statistics:")
print(f"  Mean: {np.nanmean(slope_tin):.1f}°")
print(f"  Median: {np.nanmedian(slope_tin):.1f}°")
print(f"  Max: {np.nanmax(slope_tin):.1f}°")


# ### 12.8 Reference DEM Comparison
# 
# Direct comparison of TINItaly and Copernicus DEMs.
# 

# In[ ]:


# Reference DEM comparison
print("Creating reference DEM comparison...")

# Calculate difference between reference DEMs
dem_diff = tinitaly_10m - copernicus_10m
dem_diff[tinitaly_10m == -9999] = np.nan
dem_diff[copernicus_10m == -9999] = np.nan

# Create multi-panel comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# TINItaly DEM
tin_plot = tinitaly_10m.copy()
tin_plot[tin_plot == -9999] = np.nan
im1 = axes[0, 0].imshow(tin_plot, cmap='terrain')
plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
axes[0, 0].set_title('TINItaly DEM (10m)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Copernicus DEM
cop_plot = copernicus_10m.copy()
cop_plot[cop_plot == -9999] = np.nan
im2 = axes[0, 1].imshow(cop_plot, cmap='terrain')
plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)')
axes[0, 1].set_title('Copernicus DEM (10m)', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# Difference map
if not np.all(np.isnan(dem_diff)):
    vmin, vmax = np.nanpercentile(dem_diff, [2, 98])
    im3 = axes[1, 0].imshow(dem_diff, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im3, ax=axes[1, 0], label='Difference (m)')
    axes[1, 0].set_title('TINItaly - Copernicus', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

# Statistics panel
axes[1, 1].axis('off')
stats_text = f"""Reference DEM Comparison Statistics

TINItaly:
  Resolution: 10m (native)
  Range: [{np.nanmin(tin_plot):.1f}, {np.nanmax(tin_plot):.1f}] m
  Mean: {np.nanmean(tin_plot):.1f} m

Copernicus:
  Resolution: 30m → 10m (resampled)
  Range: [{np.nanmin(cop_plot):.1f}, {np.nanmax(cop_plot):.1f}] m
  Mean: {np.nanmean(cop_plot):.1f} m

Difference (TINItaly - Copernicus):
  Mean: {np.nanmean(dem_diff):.2f} m
  Std: {np.nanstd(dem_diff):.2f} m
  NMAD: {1.4826 * np.nanmedian(np.abs(dem_diff - np.nanmedian(dem_diff))):.2f} m
  Range: [{np.nanmin(dem_diff):.2f}, {np.nanmax(dem_diff):.2f}] m
"""

axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                fontsize=12, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'reference_dem_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.9 Coverage Grid and Void Zones
# 
# Analyze spatial coverage and identify void zones (areas without measurements).
# 

# In[ ]:


# Create SAOCOM coverage grid
print("Creating coverage grid...")

# Initialize coverage grid
coverage_grid = np.zeros((grid_height, grid_width), dtype=bool)

# Mark cells with SAOCOM data
for idx, row in saocom_cleaned.iterrows():
    r, c = rowcol(target_transform, row.geometry.x, row.geometry.y)
    r, c = int(r), int(c)
    if 0 <= r < grid_height and 0 <= c < grid_width:
        coverage_grid[r, c] = True

# Calculate void zones
total_cells = grid_height * grid_width
covered_cells = coverage_grid.sum()
void_cells = total_cells - covered_cells
coverage_pct = 100 * covered_cells / total_cells

print(f"Coverage statistics:")
print(f"  Total grid cells: {total_cells:,}")
print(f"  Covered cells: {covered_cells:,}")
print(f"  Void cells: {void_cells:,}")
print(f"  Coverage: {coverage_pct:.1f}%")

# Visualize coverage
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Coverage map
axes[0].imshow(coverage_grid, cmap='binary', interpolation='nearest')
axes[0].set_title(f'SAOCOM Coverage Grid ({coverage_pct:.1f}% covered)', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Void zones overlay on slope
void_mask = ~coverage_grid
slope_with_voids = slope_tin.copy()
slope_with_voids[void_mask] = np.nan

im2 = axes[1].imshow(slope_tin, cmap='terrain', alpha=0.7)
axes[1].imshow(void_mask, cmap='Reds', alpha=0.3)
plt.colorbar(im2, ax=axes[1], label='Slope (degrees)')
axes[1].set_title('Void Zones (red) over Terrain Slope', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'coverage_and_voids.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.10 Residuals by Elevation Bins
# 
# Investigate if accuracy varies with elevation.
# 

# In[ ]:


# Bin residuals by elevation
height_bins = [0, 200, 400, 600, 800, 1000]
height_labels = ['0-200m', '200-400m', '400-600m', '600-800m', '800-1000m']

saocom_cleaned['height_category'] = pd.cut(
    saocom_cleaned['tinitaly_height'],
    bins=height_bins,
    labels=height_labels
)

# Calculate statistics by height
height_stats = saocom_cleaned.groupby('height_category')['diff_tinitaly'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('nmad', lambda x: nmad(x.dropna()))
]).round(2)

print("\nAccuracy by elevation:")
print(height_stats)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot of NMAD by elevation
height_stats['nmad'].plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
axes[0].set_title('NMAD by Elevation Range', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Elevation Range', fontsize=12)
axes[0].set_ylabel('NMAD (m)', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Sample counts by elevation
height_stats['count'].plot(kind='bar', ax=axes[1], color='skyblue', edgecolor='black')
axes[1].set_title('Sample Count by Elevation Range', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Elevation Range', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'accuracy_by_elevation.png', dpi=300, bbox_inches='tight')
plt.show()


# ### 12.11 Summary Dashboard
# 
# Comprehensive summary of all validation metrics in one figure.
# 

# In[ ]:


# Create summary dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Spatial distribution
ax1 = fig.add_subplot(gs[0, 0])
saocom_cleaned.plot(ax=ax1, markersize=0.5, color='blue', alpha=0.3)
ax1.set_title('SAOCOM Point Distribution', fontweight='bold')
ax1.set_xlabel('Easting (m)')
ax1.set_ylabel('Northing (m)')
ax1.set_aspect('equal')

# 2. Residual histogram (TINItaly)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals_tin, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_title(f'Residuals (NMAD={nmad_tin:.2f}m)', fontweight='bold')
ax2.set_xlabel('SAOCOM - TINItaly (m)')
ax2.set_ylabel('Frequency')
ax2.grid(alpha=0.3)

# 3. Accuracy by slope
ax3 = fig.add_subplot(gs[0, 2])
slope_stats['nmad'].plot(kind='bar', ax=ax3, color='coral', edgecolor='black')
ax3.set_title('NMAD by Slope Category', fontweight='bold')
ax3.set_ylabel('NMAD (m)')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# 4. Scatter plot
ax4 = fig.add_subplot(gs[1, 0])
sample_size = min(10000, len(valid_tin))
sample_indices = np.random.choice(len(valid_tin), sample_size, replace=False)
ax4.scatter(
    valid_tin['tinitaly_height'].iloc[sample_indices],
    valid_tin['HEIGHT_ABSOLUTE_TIN'].iloc[sample_indices],
    s=1, alpha=0.3, color='blue'
)
lims = [valid_tin['tinitaly_height'].min(), valid_tin['tinitaly_height'].max()]
ax4.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
ax4.set_title('SAOCOM vs TINItaly', fontweight='bold')
ax4.set_xlabel('TINItaly Height (m)')
ax4.set_ylabel('SAOCOM Height (m)')
ax4.grid(alpha=0.3)

# 5. Slope map
ax5 = fig.add_subplot(gs[1, 1])
slope_plot = ax5.imshow(slope_tin, cmap='terrain', vmin=0, vmax=45)
plt.colorbar(slope_plot, ax=ax5, label='Slope (°)', fraction=0.046)
ax5.set_title('Terrain Slope', fontweight='bold')
ax5.axis('off')

# 6. Residuals spatial map
ax6 = fig.add_subplot(gs[1, 2])
valid_pts = saocom_cleaned[saocom_cleaned['diff_tinitaly'].notna()]
sample_pts = valid_pts.sample(min(10000, len(valid_pts)))
vmin, vmax = np.percentile(sample_pts['diff_tinitaly'], [2, 98])
sc = ax6.scatter(
    sample_pts.geometry.x,
    sample_pts.geometry.y,
    c=sample_pts['diff_tinitaly'],
    cmap='RdBu_r',
    s=1,
    vmin=vmin,
    vmax=vmax,
    alpha=0.5
)
plt.colorbar(sc, ax=ax6, label='Residual (m)', fraction=0.046)
ax6.set_title('Spatial Residuals', fontweight='bold')
ax6.set_aspect('equal')
ax6.axis('off')

# 7. Statistics text
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

summary_text = f"""\nSAOCOM INSAR VALIDATION SUMMARY
{"="*80}

Dataset Statistics:
  Total points: {len(saocom_gdf):,}
  Outliers removed: {len(outliers):,} ({100*len(outliers)/len(saocom_gdf):.1f}%)
  Clean dataset: {len(saocom_cleaned):,}

Validation against TINItaly (10m resolution):
  NMAD: {nmad_tin:.2f} m
  RMSE: {np.sqrt((residuals_tin**2).mean()):.2f} m
  Mean error: {residuals_tin.mean():.2f} m
  Correlation: {np.corrcoef(valid_tin["HEIGHT_ABSOLUTE_TIN"], valid_tin["tinitaly_height"])[0,1]:.4f}

Validation against Copernicus (30m resampled to 10m):
  NMAD: {nmad_cop:.2f} m
  RMSE: {np.sqrt((residuals_cop**2).mean()):.2f} m
  Mean error: {residuals_cop.mean():.2f} m
  Correlation: {np.corrcoef(valid_cop["HEIGHT_ABSOLUTE_COP"], valid_cop["copernicus_height"])[0,1]:.4f}

Performance by Terrain:
  Flat (0-5°):        NMAD = {slope_stats.loc["Flat (0-5°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Flat (0-5°)", "count"]):,})
  Gentle (5-15°):     NMAD = {slope_stats.loc["Gentle (5-15°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Gentle (5-15°)", "count"]):,})
  Moderate (15-30°):  NMAD = {slope_stats.loc["Moderate (15-30°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Moderate (15-30°)", "count"]):,})
  Steep (>30°):       NMAD = {slope_stats.loc["Steep (>30°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Steep (>30°)", "count"]):,})
"""

ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes,
         fontsize=11, verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

fig.suptitle('SAOCOM InSAR Height Validation - Complete Summary',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(IMAGES_DIR / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

