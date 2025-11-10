"""
Script to add Radar Shadow Analysis and Control Points cells to saocom_analysis_clean.ipynb

This script:
1. Loads saocom_analysis_clean.ipynb
2. Inserts Control Points analysis after outlier filtering (after cell ~26)
3. Inserts Radar Shadow analysis after slope analysis starts (after cell ~39)
4. Saves the updated notebook
"""

import json
from pathlib import Path

# Load the notebook
notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading notebook: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# =============================================================================
# CONTROL POINTS CELLS (Insert after outlier filtering, around cell 27)
# =============================================================================

control_points_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 6. Control Points Identification\n",
            "\n",
            "Identify high-quality control points where SAOCOM, Copernicus, and TINItaly all agree within ±2 meters.\n",
            "\n",
            "**Purpose:**\n",
            "- Validate calibration quality\n",
            "- Identify stable, high-confidence locations\n",
            "- Plan ground truth collection\n",
            "- Independent accuracy assessment"
        ]
    },
    # Code cell 1: Import and configuration
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Control Points Identification\n",
            "print(\"### Control Points Identification ###\\n\")\n",
            "\n",
            "from src.control_points import (\n",
            "    identify_control_points,\n",
            "    analyze_control_point_distribution,\n",
            "    calculate_control_point_bias,\n",
            "    recommend_calibration_points,\n",
            "    export_control_points\n",
            ")\n",
            "\n",
            "# Configuration\n",
            "TOLERANCE = 2.0  # meters\n",
            "USE_CALIBRATED_SAOCOM = True\n",
            "\n",
            "print(f\"Configuration:\")\n",
            "print(f\"  Tolerance: ±{TOLERANCE} meters\")\n",
            "print(f\"  Using calibrated SAOCOM: {USE_CALIBRATED_SAOCOM}\")"
        ]
    },
    # Code cell 2: Identify control points
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Identify control points where all three DEMs agree\n",
            "control_points = identify_control_points(\n",
            "    saocom_cleaned,\n",
            "    tolerance=TOLERANCE,\n",
            "    saocom_col='HEIGHT_RELATIVE',\n",
            "    copernicus_col='copernicus_height',\n",
            "    tinitaly_col='tinitaly_height',\n",
            "    calibrated=USE_CALIBRATED_SAOCOM\n",
            ")\n",
            "\n",
            "print(f\"\\n{'='*70}\")\n",
            "print(f\"CONTROL POINTS IDENTIFIED\")\n",
            "print(f\"{'='*70}\")\n",
            "print(f\"Total points: {len(saocom_cleaned):,}\")\n",
            "print(f\"Control points: {len(control_points):,}\")\n",
            "print(f\"Percentage: {len(control_points)/len(saocom_cleaned)*100:.2f}%\")\n",
            "print(f\"{'='*70}\\n\")\n",
            "\n",
            "if len(control_points) > 0:\n",
            "    # Analyze distribution\n",
            "    stats = analyze_control_point_distribution(control_points, saocom_cleaned)\n",
            "    \n",
            "    print(f\"Distribution Analysis:\")\n",
            "    print(f\"  Mean DEM agreement: {stats.get('mean_agreement', 0):.3f} m\")\n",
            "    print(f\"  Spatial density: {stats.get('spatial_density', 0):.2f} points/km²\")\n",
            "    \n",
            "    # Calculate bias at control points\n",
            "    bias_stats = calculate_control_point_bias(control_points)\n",
            "    \n",
            "    print(f\"\\nSAOCOM Accuracy at Control Points:\")\n",
            "    print(f\"  Bias: {bias_stats['mean_bias']:+.3f} m\")\n",
            "    print(f\"  RMSE: {bias_stats['rmse']:.3f} m\")\n",
            "    print(f\"  NMAD: {bias_stats['nmad']:.3f} m\")\n",
            "else:\n",
            "    print(\"⚠️  No control points found with current tolerance\")"
        ]
    },
    # Code cell 3: Visualization
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize control points on Sentinel-2\n",
            "if len(control_points) > 0:\n",
            "    # Load Sentinel-2\n",
            "    sentinel_path = 'data/sentinel_data/Sentinel2Views_Clip.tif'\n",
            "    \n",
            "    try:\n",
            "        with rasterio.open(sentinel_path) as src:\n",
            "            sentinel_rgb = src.read([1, 2, 3])\n",
            "            sentinel_rgb = np.transpose(sentinel_rgb, (1, 2, 0))\n",
            "            sentinel_bounds = src.bounds\n",
            "            \n",
            "            # Normalize\n",
            "            p2, p98 = np.percentile(sentinel_rgb, (2, 98))\n",
            "            sentinel_rgb = np.clip((sentinel_rgb - p2) / (p98 - p2), 0, 1)\n",
            "            \n",
            "        # Create visualization\n",
            "        fig, ax = plt.subplots(figsize=(16, 14))\n",
            "        \n",
            "        # Sentinel background\n",
            "        extent = [sentinel_bounds.left, sentinel_bounds.right,\n",
            "                  sentinel_bounds.bottom, sentinel_bounds.top]\n",
            "        ax.imshow(sentinel_rgb, extent=extent, interpolation='bilinear', alpha=0.9)\n",
            "        \n",
            "        # All points\n",
            "        ax.scatter(saocom_cleaned.geometry.x, saocom_cleaned.geometry.y,\n",
            "                   c='lightgray', s=1, alpha=0.3, \n",
            "                   label=f'All points (n={len(saocom_cleaned):,})')\n",
            "        \n",
            "        # Control points\n",
            "        scatter = ax.scatter(control_points.geometry.x, control_points.geometry.y,\n",
            "                            c=control_points['elevation_range'], cmap='RdYlGn_r',\n",
            "                            s=30, alpha=0.8, edgecolors='black', linewidths=0.5,\n",
            "                            vmin=0, vmax=TOLERANCE,\n",
            "                            label=f'Control points (n={len(control_points):,})')\n",
            "        \n",
            "        plt.colorbar(scatter, ax=ax, label='DEM Disagreement (m)')\n",
            "        \n",
            "        ax.set_title(f'Control Points: All DEMs Agree ≤ ±{TOLERANCE}m\\n'\n",
            "                     f'Overlaid on Sentinel-2', fontsize=16, fontweight='bold')\n",
            "        ax.set_xlabel('Longitude')\n",
            "        ax.set_ylabel('Latitude')\n",
            "        ax.legend(loc='upper right')\n",
            "        ax.grid(True, alpha=0.3)\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.savefig('images/control_points_sentinel_overlay.png', dpi=300, bbox_inches='tight')\n",
            "        print(\"✓ Saved: images/control_points_sentinel_overlay.png\")\n",
            "        plt.show()\n",
            "        \n",
            "    except FileNotFoundError:\n",
            "        print(f\"⚠️  Sentinel-2 file not found, skipping visualization\")\n",
            "    \n",
            "    # Export control points\n",
            "    output_dir = Path('results/control_points')\n",
            "    output_dir.mkdir(parents=True, exist_ok=True)\n",
            "    \n",
            "    export_control_points(control_points, output_dir / 'control_points.geojson')\n",
            "    export_control_points(control_points, output_dir / 'control_points.csv', format='CSV')\n",
            "    \n",
            "    print(f\"✓ Control points exported to: {output_dir}\")"
        ]
    }
]

# =============================================================================
# RADAR SHADOW ANALYSIS CELLS (Insert after slope analysis, around cell 40)
# =============================================================================

radar_shadow_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 8. Radar Shadow and Geometry Analysis\n",
            "\n",
            "Analyze radar geometry effects including shadow, layover, and foreshortening.\n",
            "\n",
            "**Purpose:**\n",
            "- Identify areas affected by poor radar geometry\n",
            "- Stratify accuracy by geometric quality\n",
            "- Understand spatial patterns in errors\n",
            "- Mask unreliable shadow/layover areas"
        ]
    },
    # Code cell 1: Import and configuration
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Radar Shadow Analysis\n",
            "print(\"### Radar Shadow Analysis ###\\n\")\n",
            "\n",
            "from src.radar_geometry import (\n",
            "    calculate_local_incidence_angle,\n",
            "    identify_shadow_areas,\n",
            "    identify_layover_areas,\n",
            "    classify_geometric_quality,\n",
            "    analyze_shadow_statistics\n",
            ")\n",
            "\n",
            "# SAOCOM geometry parameters\n",
            "RADAR_INCIDENCE = 35.0  # degrees from vertical\n",
            "RADAR_AZIMUTH = 192.0   # degrees (192° = descending, 12° = ascending)\n",
            "\n",
            "print(f\"SAOCOM Geometry:\")\n",
            "print(f\"  Incidence angle: {RADAR_INCIDENCE}°\")\n",
            "print(f\"  Look azimuth: {RADAR_AZIMUTH}° ({'Descending' if RADAR_AZIMUTH > 90 else 'Ascending'})\")"
        ]
    },
    # Code cell 2: Calculate local incidence
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate local incidence angle from slope and aspect\n",
            "print(\"\\nCalculating local incidence angles...\")\n",
            "\n",
            "local_incidence = calculate_local_incidence_angle(\n",
            "    slope,\n",
            "    aspect,\n",
            "    radar_incidence=RADAR_INCIDENCE,\n",
            "    radar_azimuth=RADAR_AZIMUTH\n",
            ")\n",
            "\n",
            "# Identify shadow and layover\n",
            "shadow_mask = identify_shadow_areas(local_incidence)\n",
            "layover_mask = identify_layover_areas(local_incidence)\n",
            "\n",
            "# Classify geometric quality\n",
            "geometric_quality = classify_geometric_quality(local_incidence, slope)\n",
            "\n",
            "print(f\"\\nGeometric Quality Distribution:\")\n",
            "total_pixels = np.sum(~np.isnan(local_incidence))\n",
            "print(f\"  Shadow: {np.sum(shadow_mask)/total_pixels*100:.2f}% of area\")\n",
            "print(f\"  Layover: {np.sum(layover_mask)/total_pixels*100:.2f}% of area\")\n",
            "\n",
            "quality_names = ['Optimal', 'Acceptable', 'Foreshortening', 'Shadow', 'Layover']\n",
            "for i, name in enumerate(quality_names):\n",
            "    pct = np.sum(geometric_quality == i) / total_pixels * 100\n",
            "    print(f\"  {name}: {pct:.2f}% of area\")"
        ]
    },
    # Code cell 3: Sample at SAOCOM points
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sample geometric data at SAOCOM point locations\n",
            "print(\"\\nSampling geometric data at SAOCOM points...\")\n",
            "\n",
            "from rasterio.transform import rowcol\n",
            "\n",
            "def sample_raster_at_points(gdf, raster_array, transform):\n",
            "    \"\"\"Sample raster values at point locations.\"\"\"\n",
            "    values = []\n",
            "    for geom in gdf.geometry:\n",
            "        row, col = rowcol(transform, geom.x, geom.y)\n",
            "        if (0 <= row < raster_array.shape[0] and 0 <= col < raster_array.shape[1]):\n",
            "            values.append(raster_array[row, col])\n",
            "        else:\n",
            "            values.append(np.nan)\n",
            "    return np.array(values)\n",
            "\n",
            "# Sample at cleaned points\n",
            "saocom_cleaned['local_incidence'] = sample_raster_at_points(\n",
            "    saocom_cleaned, local_incidence, dem_transform\n",
            ")\n",
            "saocom_cleaned['is_shadow'] = sample_raster_at_points(\n",
            "    saocom_cleaned, shadow_mask.astype(float), dem_transform\n",
            ").astype(bool)\n",
            "saocom_cleaned['geometric_quality'] = sample_raster_at_points(\n",
            "    saocom_cleaned, geometric_quality, dem_transform\n",
            ").astype(int)\n",
            "\n",
            "print(f\"Points in shadow: {saocom_cleaned['is_shadow'].sum()} \"\n",
            "      f\"({saocom_cleaned['is_shadow'].sum()/len(saocom_cleaned)*100:.1f}%)\")"
        ]
    },
    # Code cell 4: Analyze accuracy by geometry
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze accuracy stratified by geometric quality\n",
            "shadow_stats = analyze_shadow_statistics(\n",
            "    saocom_cleaned,\n",
            "    local_incidence_col='local_incidence',\n",
            "    residual_col='diff_tinitaly'\n",
            ")\n",
            "\n",
            "print(f\"\\n{'='*70}\")\n",
            "print(f\"ACCURACY BY RADAR GEOMETRY\")\n",
            "print(f\"{'='*70}\")\n",
            "print(f\"{'Category':<15} {'Count':>8} {'Bias (m)':>10} {'RMSE (m)':>10} {'NMAD (m)':>10}\")\n",
            "print(\"-\" * 70)\n",
            "\n",
            "for category, stats in shadow_stats.items():\n",
            "    if stats['count'] > 0:\n",
            "        print(f\"{category:<15} {stats['count']:>8} \"\n",
            "              f\"{stats['bias']:>10.2f} {stats['rmse']:>10.2f} {stats['nmad']:>10.2f}\")\n",
            "\n",
            "print(f\"{'='*70}\\n\")"
        ]
    },
    # Code cell 5: Visualization
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize radar geometry\n",
            "from matplotlib.colors import LinearSegmentedColormap\n",
            "import matplotlib.patches as mpatches\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "# Plot 1: Local incidence angle\n",
            "im1 = axes[0].imshow(local_incidence, cmap='RdYlGn_r', vmin=0, vmax=90)\n",
            "axes[0].set_title('Local Incidence Angle', fontsize=14, fontweight='bold')\n",
            "axes[0].set_xlabel('Column')\n",
            "axes[0].set_ylabel('Row')\n",
            "plt.colorbar(im1, ax=axes[0], label='Angle (degrees)')\n",
            "\n",
            "# Shadow overlay\n",
            "shadow_overlay = np.where(shadow_mask, 1, np.nan)\n",
            "axes[0].imshow(shadow_overlay, cmap='binary', alpha=0.6)\n",
            "\n",
            "# Plot 2: Geometric quality\n",
            "quality_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#34495e', '#9b59b6']\n",
            "quality_cmap = LinearSegmentedColormap.from_list('quality', quality_colors, N=5)\n",
            "\n",
            "im2 = axes[1].imshow(geometric_quality, cmap=quality_cmap, vmin=0, vmax=4)\n",
            "axes[1].set_title('Radar Geometric Quality', fontsize=14, fontweight='bold')\n",
            "axes[1].set_xlabel('Column')\n",
            "axes[1].set_ylabel('Row')\n",
            "\n",
            "patches = [mpatches.Patch(color=quality_colors[i], label=quality_names[i])\n",
            "           for i in range(5)]\n",
            "axes[1].legend(handles=patches, loc='upper right', fontsize=10)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('images/radar_geometry_analysis.png', dpi=300, bbox_inches='tight')\n",
            "print(\"✓ Saved: images/radar_geometry_analysis.png\")\n",
            "plt.show()"
        ]
    }
]

# =============================================================================
# INSERT CELLS INTO NOTEBOOK
# =============================================================================

# Find insertion points
# Control points: after outlier filtering (around cell 27)
control_points_insert_index = 27

# Radar shadow: after slope analysis starts (around cell 40)
radar_shadow_insert_index = 40

print(f"\\nInserting cells:")
print(f"  Control Points: after cell {control_points_insert_index} ({len(control_points_cells)} cells)")
print(f"  Radar Shadow: after cell {radar_shadow_insert_index} ({len(radar_shadow_cells)} cells)")

# Insert radar shadow cells first (higher index) to avoid shifting
# Insert in reverse order to maintain indices
new_cells = nb['cells'].copy()

# Insert radar shadow cells
for i, cell in enumerate(radar_shadow_cells):
    new_cells.insert(radar_shadow_insert_index + 1 + i, cell)

# Insert control points cells (index stays same since we inserted after)
for i, cell in enumerate(control_points_cells):
    new_cells.insert(control_points_insert_index + 1 + i, cell)

nb['cells'] = new_cells

print(f"\\nUpdated notebook now has {len(nb['cells'])} cells")
print(f"  Added {len(control_points_cells)} control points cells")
print(f"  Added {len(radar_shadow_cells)} radar shadow cells")
print(f"  Total added: {len(control_points_cells) + len(radar_shadow_cells)} cells")

# =============================================================================
# SAVE UPDATED NOTEBOOK
# =============================================================================

# Backup original
backup_path = notebook_path.with_suffix('.ipynb.backup')
print(f"\\nCreating backup: {backup_path}")
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Save updated notebook
print(f"Saving updated notebook: {notebook_path}")
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\\n{'='*70}")
print(f"SUCCESS: Notebook updated with both analysis features!")
print(f"{'='*70}")
print(f"\\nNew cells added:")
print(f"  Section 6: Control Points Identification (4 cells)")
print(f"  Section 8: Radar Shadow Analysis (5 cells)")
print(f"\\nBackup saved to: {backup_path}")
print(f"\\nYou can now run saocom_analysis_clean.ipynb to execute both analyses!")
print(f"{'='*70}")
