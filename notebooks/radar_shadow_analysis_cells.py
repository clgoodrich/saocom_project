"""
Radar Shadow Analysis - Notebook Cells
=======================================

These cells can be added to your SAOCOM analysis notebook to analyze
the effects of radar geometry and topographic shadows on DEM accuracy.

Copy and paste these cells into your Jupyter notebook in sequence.
"""

# =============================================================================
# CELL 1: Import radar geometry module
# =============================================================================
print("### Radar Shadow Analysis ###\n")
print("Analyzing radar geometry effects on SAOCOM DEM accuracy...")

from src.radar_geometry import (
    calculate_local_incidence_angle,
    identify_shadow_areas,
    identify_layover_areas,
    classify_geometric_quality,
    calculate_radar_brightness,
    analyze_shadow_statistics
)

# =============================================================================
# CELL 2: Set SAOCOM radar geometry parameters
# =============================================================================
# SAOCOM radar geometry parameters
# Adjust these based on your specific acquisition geometry

# Incidence angle: typical SAOCOM range is 20-50°
RADAR_INCIDENCE = 35.0  # degrees from vertical

# Radar look azimuth: direction radar is looking
# Descending orbit: ~192° (looking south-southwest)
# Ascending orbit: ~12° (looking north-northeast)
RADAR_AZIMUTH = 192.0  # degrees, 0=North, clockwise

print(f"SAOCOM Geometry Configuration:")
print(f"  Incidence Angle: {RADAR_INCIDENCE}°")
print(f"  Look Azimuth: {RADAR_AZIMUTH}° ({'Descending' if RADAR_AZIMUTH > 90 else 'Ascending'})")

# =============================================================================
# CELL 3: Load or calculate terrain derivatives
# =============================================================================
# Calculate slope and aspect from reference DEM if not already done
from src.preprocessing import calculate_terrain_derivatives
from src.utils import load_dem_array

# Load TINItaly DEM (highest resolution reference)
tinitaly_path = 'data/tinitaly/tinitaly_crop.tif'
dem_array, dem_transform = load_dem_array(tinitaly_path)

# Calculate terrain derivatives
slope, aspect = calculate_terrain_derivatives(dem_array, cellsize=10, nodata=-9999)

print(f"\nTerrain Statistics:")
print(f"  Slope range: {np.nanmin(slope):.1f}° to {np.nanmax(slope):.1f}°")
print(f"  Mean slope: {np.nanmean(slope):.1f}°")
print(f"  Aspect range: {np.nanmin(aspect):.1f}° to {np.nanmax(aspect):.1f}°")

# =============================================================================
# CELL 4: Calculate local incidence angle
# =============================================================================
# Calculate local incidence angle accounting for terrain
local_incidence = calculate_local_incidence_angle(
    slope,
    aspect,
    radar_incidence=RADAR_INCIDENCE,
    radar_azimuth=RADAR_AZIMUTH
)

print(f"\nLocal Incidence Angle Statistics:")
print(f"  Range: {np.nanmin(local_incidence):.1f}° to {np.nanmax(local_incidence):.1f}°")
print(f"  Mean: {np.nanmean(local_incidence):.1f}°")
print(f"  Median: {np.nanmedian(local_incidence):.1f}°")

# =============================================================================
# CELL 5: Identify shadow and layover areas
# =============================================================================
# Identify problematic geometric areas
shadow_mask = identify_shadow_areas(local_incidence, shadow_threshold=90.0)
layover_mask = identify_layover_areas(local_incidence, layover_threshold=20.0)

# Calculate area fractions
total_valid = np.sum(~np.isnan(local_incidence))
shadow_fraction = np.sum(shadow_mask) / total_valid * 100 if total_valid > 0 else 0
layover_fraction = np.sum(layover_mask) / total_valid * 100 if total_valid > 0 else 0

print(f"\nGeometric Distortion Areas:")
print(f"  Shadow: {shadow_fraction:.2f}% of area")
print(f"  Layover: {layover_fraction:.2f}% of area")
print(f"  Well-illuminated: {100 - shadow_fraction - layover_fraction:.2f}% of area")

# =============================================================================
# CELL 6: Classify geometric quality
# =============================================================================
# Classify areas by radar geometric quality
geometric_quality = classify_geometric_quality(
    local_incidence,
    slope,
    shadow_thresh=90.0,
    layover_thresh=20.0,
    steep_slope_thresh=30.0
)

# Quality class names
quality_names = {
    0: 'Optimal',
    1: 'Acceptable',
    2: 'Foreshortening',
    3: 'Shadow',
    4: 'Layover'
}

# Calculate area distribution
print(f"\nGeometric Quality Classification:")
for class_id, class_name in quality_names.items():
    class_fraction = np.sum(geometric_quality == class_id) / total_valid * 100
    print(f"  {class_name}: {class_fraction:.2f}% of area")

# =============================================================================
# CELL 7: Add geometric data to SAOCOM point data
# =============================================================================
# Sample geometric parameters at SAOCOM point locations
# Assuming you have saocom_gdf with geometry already

from rasterio.transform import rowcol
import rasterio

# Load raster metadata for sampling
with rasterio.open(tinitaly_path) as src:
    transform = src.transform

# Sample at point locations
def sample_raster_at_points(gdf, raster_array, transform):
    """Sample raster values at point locations."""
    values = []
    for geom in gdf.geometry:
        row, col = rowcol(transform, geom.x, geom.y)
        if (0 <= row < raster_array.shape[0] and
            0 <= col < raster_array.shape[1]):
            values.append(raster_array[row, col])
        else:
            values.append(np.nan)
    return np.array(values)

# Add geometric columns to SAOCOM data
saocom_gdf['local_incidence'] = sample_raster_at_points(saocom_gdf, local_incidence, transform)
saocom_gdf['is_shadow'] = sample_raster_at_points(saocom_gdf, shadow_mask.astype(float), transform).astype(bool)
saocom_gdf['is_layover'] = sample_raster_at_points(saocom_gdf, layover_mask.astype(float), transform).astype(bool)
saocom_gdf['geometric_quality'] = sample_raster_at_points(saocom_gdf, geometric_quality, transform).astype(int)

print(f"\nAdded geometric columns to SAOCOM data:")
print(f"  Points in shadow: {saocom_gdf['is_shadow'].sum()} ({saocom_gdf['is_shadow'].sum()/len(saocom_gdf)*100:.1f}%)")
print(f"  Points in layover: {saocom_gdf['is_layover'].sum()} ({saocom_gdf['is_layover'].sum()/len(saocom_gdf)*100:.1f}%)")

# =============================================================================
# CELL 8: Analyze accuracy by geometric quality
# =============================================================================
# Compute accuracy statistics stratified by shadow conditions
shadow_stats = analyze_shadow_statistics(
    saocom_gdf,
    local_incidence_col='local_incidence',
    residual_col='diff_tinitaly'  # or 'diff_copernicus'
)

print(f"\nAccuracy Statistics by Illumination Category:")
print(f"{'Category':<15} {'Count':>8} {'Bias (m)':>10} {'RMSE (m)':>10} {'NMAD (m)':>10}")
print("-" * 60)
for category, stats in shadow_stats.items():
    if stats['count'] > 0:
        print(f"{category:<15} {stats['count']:>8} "
              f"{stats['bias']:>10.2f} {stats['rmse']:>10.2f} {stats['nmad']:>10.2f}")

# =============================================================================
# CELL 9: Visualize local incidence angle map
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Local incidence angle
im1 = axes[0].imshow(local_incidence, cmap='RdYlGn_r', vmin=0, vmax=90)
axes[0].set_title('Local Incidence Angle', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')
cbar1 = plt.colorbar(im1, ax=axes[0], label='Angle (degrees)')

# Mark shadow areas
shadow_overlay = np.where(shadow_mask, 1, np.nan)
axes[0].imshow(shadow_overlay, cmap='binary', alpha=0.6)

# Plot 2: Geometric quality classification
quality_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#34495e', '#9b59b6']
quality_cmap = LinearSegmentedColormap.from_list('quality', quality_colors, N=5)

im2 = axes[1].imshow(geometric_quality, cmap=quality_cmap, vmin=0, vmax=4)
axes[1].set_title('Radar Geometric Quality', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')

# Create legend
patches = [mpatches.Patch(color=quality_colors[i], label=quality_names[i])
           for i in range(5)]
axes[1].legend(handles=patches, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('images/radar_geometry_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: images/radar_geometry_analysis.png")
plt.show()

# =============================================================================
# CELL 10: Visualize residuals by shadow condition
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Filter valid data
valid_data = saocom_gdf[saocom_gdf['diff_tinitaly'].notna()].copy()

# Plot 1: Residuals vs Local Incidence Angle
axes[0, 0].hexbin(valid_data['local_incidence'], valid_data['diff_tinitaly'],
                  gridsize=50, cmap='YlOrRd', mincnt=1)
axes[0, 0].axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].axvline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Shadow Threshold')
axes[0, 0].set_xlabel('Local Incidence Angle (degrees)', fontsize=11)
axes[0, 0].set_ylabel('Residual (m)', fontsize=11)
axes[0, 0].set_title('Residuals vs Local Incidence Angle', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution by shadow condition
shadow_data = valid_data[valid_data['is_shadow']]
illuminated_data = valid_data[~valid_data['is_shadow']]

axes[0, 1].hist([illuminated_data['diff_tinitaly'], shadow_data['diff_tinitaly']],
                bins=50, label=['Illuminated', 'Shadow'], alpha=0.7, color=['green', 'red'])
axes[0, 1].axvline(0, color='blue', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Residual (m)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Residual Distribution: Illuminated vs Shadow', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Box plot by geometric quality
quality_data = [valid_data[valid_data['geometric_quality'] == i]['diff_tinitaly'].dropna()
                for i in range(5)]
bp = axes[1, 0].boxplot(quality_data, labels=[quality_names[i] for i in range(5)],
                        patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], quality_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].axhline(0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Geometric Quality Category', fontsize=11)
axes[1, 0].set_ylabel('Residual (m)', fontsize=11)
axes[1, 0].set_title('Residuals by Geometric Quality', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=15, ha='right')

# Plot 4: RMSE vs Local Incidence Angle (binned)
bins = np.arange(0, 100, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_rmse = []
bin_counts = []

for i in range(len(bins)-1):
    mask = ((valid_data['local_incidence'] >= bins[i]) &
            (valid_data['local_incidence'] < bins[i+1]))
    if mask.sum() > 10:
        bin_rmse.append(np.sqrt(np.mean(valid_data.loc[mask, 'diff_tinitaly']**2)))
        bin_counts.append(mask.sum())
    else:
        bin_rmse.append(np.nan)
        bin_counts.append(0)

axes[1, 1].plot(bin_centers, bin_rmse, 'o-', linewidth=2, markersize=8, color='darkred')
axes[1, 1].axvline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Shadow')
axes[1, 1].set_xlabel('Local Incidence Angle (degrees)', fontsize=11)
axes[1, 1].set_ylabel('RMSE (m)', fontsize=11)
axes[1, 1].set_title('RMSE vs Local Incidence Angle', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/shadow_effect_on_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/shadow_effect_on_accuracy.png")
plt.show()

# =============================================================================
# CELL 11: Create summary statistics table
# =============================================================================
import pandas as pd

# Create summary DataFrame
summary_data = []
for category, stats in shadow_stats.items():
    if stats['count'] > 0:
        summary_data.append({
            'Category': category.capitalize(),
            'Points': stats['count'],
            'Fraction (%)': f"{stats['count']/len(valid_data)*100:.1f}",
            'Bias (m)': f"{stats['bias']:.2f}",
            'RMSE (m)': f"{stats['rmse']:.2f}",
            'NMAD (m)': f"{stats['nmad']:.2f}",
            'Std Dev (m)': f"{stats['std']:.2f}",
            'Mean Inc. (°)': f"{stats['mean_incidence']:.1f}"
        })

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*80)
print("SHADOW EFFECT SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

# =============================================================================
# CELL 12: Save geometric quality rasters
# =============================================================================
import rasterio
from rasterio.transform import from_bounds

# Save local incidence angle raster
output_folder = Path('topography_outputs/radar_geometry')
output_folder.mkdir(parents=True, exist_ok=True)

# Get metadata from reference DEM
with rasterio.open(tinitaly_path) as src:
    profile = src.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

# Save local incidence
with rasterio.open(output_folder / 'local_incidence_angle.tif', 'w', **profile) as dst:
    dst.write(local_incidence.astype(np.float32), 1)

# Save geometric quality
profile.update(dtype=rasterio.uint8)
with rasterio.open(output_folder / 'geometric_quality.tif', 'w', **profile) as dst:
    dst.write(geometric_quality.astype(np.uint8), 1)

# Save shadow mask
with rasterio.open(output_folder / 'shadow_mask.tif', 'w', **profile) as dst:
    dst.write(shadow_mask.astype(np.uint8), 1)

print(f"\n✓ Saved raster outputs to: {output_folder}")
print(f"  - local_incidence_angle.tif")
print(f"  - geometric_quality.tif")
print(f"  - shadow_mask.tif")

print("\n" + "="*80)
print("RADAR SHADOW ANALYSIS COMPLETE")
print("="*80)
