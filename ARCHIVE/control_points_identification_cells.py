"""
Control Points Identification and Visualization - Notebook Cells
=================================================================

These cells identify high-quality control points where SAOCOM, Copernicus,
and TINItaly all agree within ±2 meters, then visualize them on Sentinel-2 imagery.

Copy and paste these cells into your Jupyter notebook after DEM sampling and calibration.
"""

# =============================================================================
# CELL 1: Import control points module and set parameters
# =============================================================================
print("### Control Points Identification ###\n")
print("Finding locations where all three DEMs agree...\n")

from src.control_points import (
    identify_control_points,
    analyze_control_point_distribution,
    calculate_control_point_bias,
    spatial_clustering_control_points,
    recommend_calibration_points,
    export_control_points
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from rasterio.plot import show
from pathlib import Path

# Configuration
TOLERANCE = 2.0  # meters - maximum difference for DEMs to be considered "identical"
USE_CALIBRATED_SAOCOM = True  # Use calibrated SAOCOM heights

print(f"Configuration:")
print(f"  Tolerance: ±{TOLERANCE} meters")
print(f"  Using calibrated SAOCOM: {USE_CALIBRATED_SAOCOM}")

# =============================================================================
# CELL 2: Identify control points
# =============================================================================
# Find control points where SAOCOM, Copernicus, and TINItaly all agree
control_points = identify_control_points(
    saocom_gdf,
    tolerance=TOLERANCE,
    saocom_col='HEIGHT_RELATIVE',
    copernicus_col='copernicus_height',
    tinitaly_col='tinitaly_height',
    calibrated=USE_CALIBRATED_SAOCOM
)

print(f"\n{'='*70}")
print(f"CONTROL POINTS IDENTIFIED")
print(f"{'='*70}")
print(f"Total SAOCOM points: {len(saocom_gdf):,}")
print(f"Control points found: {len(control_points):,}")
print(f"Percentage: {len(control_points)/len(saocom_gdf)*100:.2f}%")
print(f"{'='*70}\n")

if len(control_points) == 0:
    print("⚠️  No control points found! Consider:")
    print("   - Increasing tolerance (e.g., ±3 or ±5 meters)")
    print("   - Checking if all three DEMs have been sampled")
    print("   - Verifying calibration was applied")
else:
    print(f"✓ Found {len(control_points):,} high-quality control points")
    print(f"  These locations have elevation agreement within ±{TOLERANCE}m across all DEMs")

# =============================================================================
# CELL 3: Analyze control point distribution
# =============================================================================
# Get detailed statistics on control point distribution
stats = analyze_control_point_distribution(control_points, saocom_gdf)

print(f"\n{'='*70}")
print(f"CONTROL POINT DISTRIBUTION ANALYSIS")
print(f"{'='*70}")
print(f"Spatial Coverage:")
print(f"  Density: {stats.get('spatial_density', 0):.2f} points/km²")
print(f"\nElevation Statistics:")
print(f"  Mean elevation: {stats.get('mean_elevation', np.nan):.2f} m")
if 'elevation_range' in stats:
    print(f"  Elevation range: {stats['elevation_range'][0]:.2f} - {stats['elevation_range'][1]:.2f} m")
print(f"  Std deviation: {stats.get('elevation_std', np.nan):.2f} m")
print(f"\nAgreement Quality:")
print(f"  Mean DEM disagreement: {stats.get('mean_agreement', np.nan):.3f} m")
print(f"  Max DEM disagreement: {stats.get('max_disagreement', np.nan):.3f} m")

if 'mean_slope' in stats:
    print(f"\nTerrain Characteristics:")
    print(f"  Mean slope: {stats['mean_slope']:.2f}°")
    if 'slope_range' in stats:
        print(f"  Slope range: {stats['slope_range'][0]:.2f}° - {stats['slope_range'][1]:.2f}°")

if 'by_slope' in stats:
    print(f"\nDistribution by Slope Class:")
    for slope_class, count in stats['by_slope'].items():
        pct = count / len(control_points) * 100
        print(f"  {slope_class}: {count:,} points ({pct:.1f}%)")

print(f"{'='*70}\n")

# =============================================================================
# CELL 4: Calculate bias at control points
# =============================================================================
# Assess SAOCOM accuracy at control points
bias_stats = calculate_control_point_bias(
    control_points,
    saocom_col='HEIGHT_RELATIVE',
    reference_col='mean_elevation',
    calibrated=USE_CALIBRATED_SAOCOM
)

print(f"\n{'='*70}")
print(f"SAOCOM ACCURACY AT CONTROL POINTS")
print(f"{'='*70}")
print(f"Bias Statistics:")
print(f"  Mean bias: {bias_stats['mean_bias']:+.3f} m")
print(f"  Median bias: {bias_stats['median_bias']:+.3f} m")
print(f"  Std deviation: {bias_stats['std_bias']:.3f} m")
print(f"  RMSE: {bias_stats['rmse']:.3f} m")
print(f"  NMAD: {bias_stats['nmad']:.3f} m")
print(f"  Residual range: {bias_stats['min_residual']:+.3f} to {bias_stats['max_residual']:+.3f} m")
print(f"{'='*70}\n")

if abs(bias_stats['mean_bias']) < 0.5:
    print("✓ Excellent: Mean bias < 0.5m at control points")
elif abs(bias_stats['mean_bias']) < 1.0:
    print("✓ Good: Mean bias < 1.0m at control points")
else:
    print("⚠️  Note: Mean bias > 1.0m - may need recalibration")

# =============================================================================
# CELL 5: Load Sentinel-2 imagery for visualization
# =============================================================================
# Load Sentinel-2 RGB composite
sentinel_path = 'data/sentinel_data/Sentinel2Views_Clip.tif'

print(f"\nLoading Sentinel-2 imagery from: {sentinel_path}")

try:
    with rasterio.open(sentinel_path) as src:
        # Read RGB bands
        sentinel_rgb = src.read([1, 2, 3])  # Assuming bands 1,2,3 are RGB
        sentinel_transform = src.transform
        sentinel_bounds = src.bounds
        sentinel_crs = src.crs

        # Transpose to (height, width, channels) for matplotlib
        sentinel_rgb = np.transpose(sentinel_rgb, (1, 2, 0))

        # Normalize to 0-1 for display (adjust if needed)
        sentinel_rgb = sentinel_rgb.astype(float)

        # Simple contrast stretch (2-98 percentile)
        p2, p98 = np.percentile(sentinel_rgb, (2, 98))
        sentinel_rgb = np.clip((sentinel_rgb - p2) / (p98 - p2), 0, 1)

        print(f"✓ Loaded Sentinel-2 imagery")
        print(f"  Shape: {sentinel_rgb.shape}")
        print(f"  Bounds: {sentinel_bounds}")
        print(f"  CRS: {sentinel_crs}")

except FileNotFoundError:
    print(f"⚠️  Sentinel-2 file not found at: {sentinel_path}")
    print("   Visualization will use simple background")
    sentinel_rgb = None
    sentinel_bounds = None

# =============================================================================
# CELL 6: Visualize control points on Sentinel-2 imagery - Main Map
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 14))

# Plot Sentinel-2 imagery as background
if sentinel_rgb is not None:
    extent = [sentinel_bounds.left, sentinel_bounds.right,
              sentinel_bounds.bottom, sentinel_bounds.top]
    ax.imshow(sentinel_rgb, extent=extent, interpolation='bilinear', alpha=0.9)
    ax.set_xlim(sentinel_bounds.left, sentinel_bounds.right)
    ax.set_ylim(sentinel_bounds.bottom, sentinel_bounds.top)

# Plot all SAOCOM points in light gray
all_pts_plot = ax.scatter(
    saocom_gdf.geometry.x,
    saocom_gdf.geometry.y,
    c='lightgray',
    s=1,
    alpha=0.3,
    label=f'All SAOCOM points (n={len(saocom_gdf):,})',
    zorder=2
)

# Plot control points colored by elevation agreement
if len(control_points) > 0:
    scatter = ax.scatter(
        control_points.geometry.x,
        control_points.geometry.y,
        c=control_points['elevation_range'],
        cmap='RdYlGn_r',
        s=30,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        vmin=0,
        vmax=TOLERANCE,
        label=f'Control points (n={len(control_points):,})',
        zorder=3
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='DEM Disagreement (m)',
                       orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Max - Min Elevation Across 3 DEMs (m)', fontsize=12)

# Title and labels
ax.set_title(f'Control Points: SAOCOM/Copernicus/TINItaly Agreement ≤ ±{TOLERANCE}m\n'
             f'Overlaid on Sentinel-2 True Color Composite',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add statistics text box
if len(control_points) > 0:
    stats_text = (
        f"Control Point Statistics:\n"
        f"  Found: {len(control_points):,} / {len(saocom_gdf):,} ({len(control_points)/len(saocom_gdf)*100:.1f}%)\n"
        f"  Mean agreement: {stats.get('mean_agreement', 0):.3f} m\n"
        f"  SAOCOM RMSE: {bias_stats['rmse']:.3f} m\n"
        f"  SAOCOM Bias: {bias_stats['mean_bias']:+.3f} m"
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('images/control_points_sentinel_overlay.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: images/control_points_sentinel_overlay.png")
plt.show()

# =============================================================================
# CELL 7: Detailed multi-panel visualization
# =============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel 1: Sentinel with control points (repeat of main view)
ax1 = fig.add_subplot(gs[0, :2])
if sentinel_rgb is not None:
    extent = [sentinel_bounds.left, sentinel_bounds.right,
              sentinel_bounds.bottom, sentinel_bounds.top]
    ax1.imshow(sentinel_rgb, extent=extent, interpolation='bilinear', alpha=0.9)

if len(control_points) > 0:
    scatter1 = ax1.scatter(control_points.geometry.x, control_points.geometry.y,
                          c=control_points['elevation_range'], cmap='RdYlGn_r',
                          s=20, alpha=0.8, edgecolors='black', linewidths=0.3,
                          vmin=0, vmax=TOLERANCE)
    plt.colorbar(scatter1, ax=ax1, label='Disagreement (m)', shrink=0.7)

ax1.set_title('Control Points on Sentinel-2', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)

# Panel 2: Elevation agreement histogram
ax2 = fig.add_subplot(gs[0, 2])
if len(control_points) > 0:
    ax2.hist(control_points['elevation_range'], bins=30, color='steelblue',
             edgecolor='black', alpha=0.7)
    ax2.axvline(control_points['elevation_range'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Mean: {control_points["elevation_range"].mean():.3f}m')
    ax2.set_xlabel('Max - Min Elevation (m)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('DEM Agreement Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: SAOCOM residuals at control points
ax3 = fig.add_subplot(gs[1, 0])
if len(control_points) > 0:
    # Calculate SAOCOM residuals
    if USE_CALIBRATED_SAOCOM and 'HEIGHT_CALIBRATED' in control_points.columns:
        saocom_h = control_points['HEIGHT_CALIBRATED']
    else:
        saocom_h = control_points['HEIGHT_RELATIVE']

    residuals = saocom_h - control_points['mean_elevation']

    ax3.hist(residuals, bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='blue', linestyle='--', linewidth=2, label='Zero')
    ax3.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {residuals.mean():+.3f}m')
    ax3.set_xlabel('SAOCOM - Consensus (m)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('SAOCOM Residuals at Control Points', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Control points by slope
ax4 = fig.add_subplot(gs[1, 1])
if len(control_points) > 0 and 'slope' in control_points.columns:
    slope_bins = [0, 5, 10, 20, 30, 90]
    slope_labels = ['0-5°', '5-10°', '10-20°', '20-30°', '>30°']
    control_points_copy = control_points.copy()
    control_points_copy['slope_class'] = pd.cut(
        control_points_copy['slope'],
        bins=slope_bins,
        labels=slope_labels,
        include_lowest=True
    )

    slope_counts = control_points_copy.groupby('slope_class', observed=True).size()
    colors_slope = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

    bars = ax4.bar(range(len(slope_counts)), slope_counts.values, color=colors_slope, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(slope_counts)))
    ax4.set_xticklabels(slope_counts.index, rotation=15, ha='right')
    ax4.set_xlabel('Slope Class', fontsize=10)
    ax4.set_ylabel('Number of Control Points', fontsize=10)
    ax4.set_title('Control Points by Terrain Slope', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# Panel 5: Elevation vs. Agreement scatter
ax5 = fig.add_subplot(gs[1, 2])
if len(control_points) > 0:
    ax5.scatter(control_points['mean_elevation'], control_points['elevation_range'],
                c=control_points['slope'] if 'slope' in control_points.columns else 'steelblue',
                cmap='viridis', s=10, alpha=0.6, edgecolors='none')
    ax5.set_xlabel('Mean Elevation (m)', fontsize=10)
    ax5.set_ylabel('DEM Disagreement (m)', fontsize=10)
    ax5.set_title('Elevation vs. Agreement', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(TOLERANCE, color='red', linestyle='--', linewidth=1,
                label=f'Tolerance: {TOLERANCE}m')
    ax5.legend(fontsize=9)

plt.suptitle(f'Control Points Analysis Dashboard - Agreement Tolerance: ±{TOLERANCE}m',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('images/control_points_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/control_points_analysis_dashboard.png")
plt.show()

# =============================================================================
# CELL 8: Spatial clustering and recommended calibration points
# =============================================================================
if len(control_points) > 10:
    print("\n### Recommended Calibration Points ###\n")

    # Get spatially distributed control points for calibration
    n_calibration_points = min(20, len(control_points))

    calibration_pts = recommend_calibration_points(
        control_points,
        n_points=n_calibration_points,
        method='distributed'
    )

    print(f"Selected {len(calibration_pts)} spatially distributed calibration points")
    print(f"These represent high-quality, well-distributed locations for:")
    print(f"  - Calibration validation")
    print(f"  - Ground truth collection")
    print(f"  - Independent accuracy assessment\n")

    # Visualize calibration points
    fig, ax = plt.subplots(figsize=(14, 12))

    if sentinel_rgb is not None:
        extent = [sentinel_bounds.left, sentinel_bounds.right,
                  sentinel_bounds.bottom, sentinel_bounds.top]
        ax.imshow(sentinel_rgb, extent=extent, interpolation='bilinear', alpha=0.9)

    # All control points in light blue
    ax.scatter(control_points.geometry.x, control_points.geometry.y,
               c='lightblue', s=15, alpha=0.5, edgecolors='blue',
               linewidths=0.5, label=f'All control points (n={len(control_points)})')

    # Recommended calibration points in red
    ax.scatter(calibration_pts.geometry.x, calibration_pts.geometry.y,
               c='red', s=100, alpha=0.9, edgecolors='darkred',
               linewidths=2, marker='*',
               label=f'Recommended calibration points (n={len(calibration_pts)})',
               zorder=5)

    # Number the calibration points
    for idx, (x, y) in enumerate(zip(calibration_pts.geometry.x, calibration_pts.geometry.y)):
        ax.text(x, y, str(idx+1), fontsize=8, ha='center', va='center',
                color='white', fontweight='bold', zorder=6)

    ax.set_title(f'Recommended Calibration Points (Spatially Distributed)\n'
                 f'Based on {len(control_points)} Control Points with ±{TOLERANCE}m Agreement',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/recommended_calibration_points.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: images/recommended_calibration_points.png")
    plt.show()
else:
    print(f"\n⚠️  Too few control points ({len(control_points)}) for calibration recommendation")

# =============================================================================
# CELL 9: Export control points to files
# =============================================================================
print("\n### Exporting Control Points ###\n")

# Create output directory
output_dir = Path('results/control_points')
output_dir.mkdir(parents=True, exist_ok=True)

if len(control_points) > 0:
    # Export as GeoJSON
    export_control_points(
        control_points,
        output_dir / 'control_points.geojson',
        format='GeoJSON'
    )

    # Export as CSV
    export_control_points(
        control_points,
        output_dir / 'control_points.csv',
        format='CSV'
    )

    # Export as Shapefile
    export_control_points(
        control_points,
        output_dir / 'control_points.shp',
        format='Shapefile'
    )

    # Export recommended calibration points if available
    if len(control_points) > 10:
        export_control_points(
            calibration_pts,
            output_dir / 'calibration_points_recommended.geojson',
            format='GeoJSON'
        )
        export_control_points(
            calibration_pts,
            output_dir / 'calibration_points_recommended.csv',
            format='CSV'
        )

    print(f"\n✓ All control points exported to: {output_dir}")
else:
    print("⚠️  No control points to export")

# =============================================================================
# CELL 10: Summary statistics table
# =============================================================================
print("\n" + "="*80)
print("CONTROL POINTS IDENTIFICATION - SUMMARY")
print("="*80)
print(f"\nInput Data:")
print(f"  Total SAOCOM points: {len(saocom_gdf):,}")
print(f"  Agreement tolerance: ±{TOLERANCE} meters")
print(f"  Using calibrated SAOCOM: {USE_CALIBRATED_SAOCOM}")

print(f"\nControl Points:")
print(f"  Identified: {len(control_points):,} ({len(control_points)/len(saocom_gdf)*100:.2f}%)")
if len(control_points) > 0:
    print(f"  Mean DEM agreement: {stats.get('mean_agreement', 0):.3f} m")
    print(f"  Max DEM disagreement: {stats.get('max_disagreement', 0):.3f} m")
    print(f"  Spatial density: {stats.get('spatial_density', 0):.2f} points/km²")

print(f"\nSAOCOM Accuracy at Control Points:")
if len(control_points) > 0:
    print(f"  Bias: {bias_stats['mean_bias']:+.3f} ± {bias_stats['std_bias']:.3f} m")
    print(f"  RMSE: {bias_stats['rmse']:.3f} m")
    print(f"  NMAD: {bias_stats['nmad']:.3f} m")

print(f"\nOutputs Generated:")
print(f"  ✓ images/control_points_sentinel_overlay.png")
print(f"  ✓ images/control_points_analysis_dashboard.png")
if len(control_points) > 10:
    print(f"  ✓ images/recommended_calibration_points.png")
if len(control_points) > 0:
    print(f"  ✓ results/control_points/*.geojson")
    print(f"  ✓ results/control_points/*.csv")
    print(f"  ✓ results/control_points/*.shp")

print("="*80)
print("\n✓ CONTROL POINTS ANALYSIS COMPLETE\n")
