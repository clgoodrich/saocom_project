"""
Script to add Radar Shadow Analysis to saocom_analysis_clean.ipynb

This script inserts radar shadow/geometry analysis cells after slope calculation.
"""

import json
from pathlib import Path

# Load the notebook
notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading notebook: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# Find the insertion point - after slope/aspect calculation
# Look for cells containing "slope" and "aspect" calculation
insertion_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', '')).lower()
        if 'calculate_terrain_derivatives' in source or ('slope' in source and 'aspect' in source and 'calculating' in source):
            insertion_index = i + 2  # Insert after this cell and the next sampling cell
            print(f"Found slope calculation at cell {i}")
            break

if insertion_index is None:
    # Default to around cell 40 if not found
    insertion_index = 40
    print(f"Using default insertion point: cell {insertion_index}")

# =============================================================================
# RADAR SHADOW ANALYSIS CELLS
# =============================================================================

radar_shadow_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Radar Shadow and Geometry Analysis\n",
            "\n",
            "Analyze radar geometry effects including shadow, layover, and foreshortening.\n",
            "\n",
            "**Purpose:**\n",
            "- Identify areas affected by poor radar geometry\n",
            "- Stratify accuracy by geometric quality\n",
            "- Understand spatial patterns in errors\n",
            "- Mask unreliable shadow/layover areas\n",
            "\n",
            "**Key Concepts:**\n",
            "- **Shadow**: Areas where radar beam is blocked by terrain (local incidence >90°)\n",
            "- **Layover**: Steep slopes facing radar causing geometric distortion (<20°)\n",
            "- **Optimal**: Well-illuminated areas with good geometry (30-60°)"
        ]
    },

    # Code cell 1: Import and configuration
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Radar Shadow Analysis - Configuration\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"RADAR SHADOW AND GEOMETRY ANALYSIS\")\n",
            "print(\"=\"*70 + \"\\n\")\n",
            "\n",
            "from src.radar_geometry import (\n",
            "    calculate_local_incidence_angle,\n",
            "    identify_shadow_areas,\n",
            "    identify_layover_areas,\n",
            "    classify_geometric_quality,\n",
            "    analyze_shadow_statistics\n",
            ")\n",
            "from matplotlib.colors import LinearSegmentedColormap\n",
            "import matplotlib.patches as mpatches\n",
            "\n",
            "# SAOCOM geometry parameters\n",
            "# Adjust these based on your SAOCOM acquisition metadata\n",
            "RADAR_INCIDENCE = 35.0  # degrees from vertical (typical SAOCOM: 20-50°)\n",
            "RADAR_AZIMUTH = 192.0   # degrees (192° = descending, 12° = ascending)\n",
            "\n",
            "print(f\"SAOCOM Geometry Configuration:\")\n",
            "print(f\"  Incidence angle: {RADAR_INCIDENCE}°\")\n",
            "print(f\"  Look azimuth: {RADAR_AZIMUTH}° ({'Descending' if RADAR_AZIMUTH > 90 else 'Ascending'} orbit)\")\n",
            "print(f\"\\nNote: Adjust these parameters based on your SAOCOM metadata\")"
        ]
    },

    # Code cell 2: Calculate local incidence
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate local incidence angle accounting for terrain orientation\n",
            "print(\"\\nCalculating local incidence angles...\")\n",
            "\n",
            "# Ensure slope and aspect exist from previous cells\n",
            "if 'slope' not in locals() or 'aspect' not in locals():\n",
            "    print(\"⚠️  Warning: slope and aspect not found. Calculating now...\")\n",
            "    from src.preprocessing import calculate_terrain_derivatives\n",
            "    from src.utils import load_dem_array\n",
            "    \n",
            "    # Load TINItaly DEM\n",
            "    tinitaly_path = 'data/tinitaly/tinitaly_crop.tif'\n",
            "    dem_array, dem_transform = load_dem_array(tinitaly_path)\n",
            "    \n",
            "    # Calculate terrain derivatives\n",
            "    slope, aspect = calculate_terrain_derivatives(dem_array, cellsize=10, nodata=-9999)\n",
            "    print(\"✓ Calculated slope and aspect from TINItaly DEM\")\n",
            "\n",
            "# Calculate local incidence angle\n",
            "local_incidence = calculate_local_incidence_angle(\n",
            "    slope,\n",
            "    aspect,\n",
            "    radar_incidence=RADAR_INCIDENCE,\n",
            "    radar_azimuth=RADAR_AZIMUTH\n",
            ")\n",
            "\n",
            "print(f\"✓ Calculated local incidence angles\")\n",
            "print(f\"  Range: {np.nanmin(local_incidence):.1f}° to {np.nanmax(local_incidence):.1f}°\")\n",
            "print(f\"  Mean: {np.nanmean(local_incidence):.1f}°\")\n",
            "\n",
            "# Identify problematic geometric areas\n",
            "shadow_mask = identify_shadow_areas(local_incidence, shadow_threshold=90.0)\n",
            "layover_mask = identify_layover_areas(local_incidence, layover_threshold=20.0)\n",
            "\n",
            "# Classify geometric quality\n",
            "geometric_quality = classify_geometric_quality(\n",
            "    local_incidence,\n",
            "    slope,\n",
            "    shadow_thresh=90.0,\n",
            "    layover_thresh=20.0,\n",
            "    steep_slope_thresh=30.0\n",
            ")\n",
            "\n",
            "# Calculate area statistics\n",
            "total_pixels = np.sum(~np.isnan(local_incidence))\n",
            "shadow_pct = np.sum(shadow_mask) / total_pixels * 100\n",
            "layover_pct = np.sum(layover_mask) / total_pixels * 100\n",
            "\n",
            "print(f\"\\nGeometric Distortion Areas:\")\n",
            "print(f\"  Shadow (>90°): {shadow_pct:.2f}% of area\")\n",
            "print(f\"  Layover (<20°): {layover_pct:.2f}% of area\")\n",
            "print(f\"  Well-illuminated: {100 - shadow_pct - layover_pct:.2f}% of area\")\n",
            "\n",
            "quality_names = ['Optimal', 'Acceptable', 'Foreshortening', 'Shadow', 'Layover']\n",
            "print(f\"\\nGeometric Quality Distribution:\")\n",
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
            "print(\"\\nSampling radar geometry at SAOCOM points...\")\n",
            "\n",
            "from rasterio.transform import rowcol\n",
            "\n",
            "def sample_raster_at_points(gdf, raster_array, transform):\n",
            "    \"\"\"Sample raster values at point locations.\"\"\"\n",
            "    values = []\n",
            "    for geom in gdf.geometry:\n",
            "        try:\n",
            "            row, col = rowcol(transform, geom.x, geom.y)\n",
            "            if (0 <= row < raster_array.shape[0] and 0 <= col < raster_array.shape[1]):\n",
            "                values.append(raster_array[row, col])\n",
            "            else:\n",
            "                values.append(np.nan)\n",
            "        except:\n",
            "            values.append(np.nan)\n",
            "    return np.array(values)\n",
            "\n",
            "# Check which GeoDataFrame to use\n",
            "if 'saocom_cleaned' in locals():\n",
            "    target_gdf = saocom_cleaned\n",
            "    gdf_name = 'saocom_cleaned'\n",
            "elif 'saocom_gdf' in locals():\n",
            "    target_gdf = saocom_gdf\n",
            "    gdf_name = 'saocom_gdf'\n",
            "else:\n",
            "    raise ValueError(\"No SAOCOM GeoDataFrame found! Expected 'saocom_cleaned' or 'saocom_gdf'\")\n",
            "\n",
            "# Sample geometric parameters\n",
            "target_gdf['local_incidence'] = sample_raster_at_points(\n",
            "    target_gdf, local_incidence, dem_transform\n",
            ")\n",
            "target_gdf['is_shadow'] = sample_raster_at_points(\n",
            "    target_gdf, shadow_mask.astype(float), dem_transform\n",
            ").astype(bool)\n",
            "target_gdf['is_layover'] = sample_raster_at_points(\n",
            "    target_gdf, layover_mask.astype(float), dem_transform\n",
            ").astype(bool)\n",
            "target_gdf['geometric_quality'] = sample_raster_at_points(\n",
            "    target_gdf, geometric_quality, dem_transform\n",
            ").astype(int)\n",
            "\n",
            "# Report statistics\n",
            "n_shadow = target_gdf['is_shadow'].sum()\n",
            "n_layover = target_gdf['is_layover'].sum()\n",
            "n_total = len(target_gdf)\n",
            "\n",
            "print(f\"✓ Sampled geometry at {n_total:,} SAOCOM points\")\n",
            "print(f\"\\nPoints by Geometric Condition:\")\n",
            "print(f\"  Shadow: {n_shadow:,} ({n_shadow/n_total*100:.1f}%)\")\n",
            "print(f\"  Layover: {n_layover:,} ({n_layover/n_total*100:.1f}%)\")\n",
            "print(f\"  Well-illuminated: {n_total-n_shadow-n_layover:,} ({(1-n_shadow/n_total-n_layover/n_total)*100:.1f}%)\")"
        ]
    },

    # Code cell 4: Analyze accuracy by geometry
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze SAOCOM accuracy stratified by radar geometry\n",
            "print(\"\\nAnalyzing accuracy by geometric quality...\\n\")\n",
            "\n",
            "# Determine which residual column to use\n",
            "if 'diff_tinitaly' in target_gdf.columns:\n",
            "    residual_col = 'diff_tinitaly'\n",
            "    ref_name = 'TINItaly'\n",
            "elif 'diff_copernicus' in target_gdf.columns:\n",
            "    residual_col = 'diff_copernicus'\n",
            "    ref_name = 'Copernicus'\n",
            "else:\n",
            "    raise ValueError(\"No residual column found! Expected 'diff_tinitaly' or 'diff_copernicus'\")\n",
            "\n",
            "# Calculate statistics by illumination category\n",
            "shadow_stats = analyze_shadow_statistics(\n",
            "    target_gdf,\n",
            "    local_incidence_col='local_incidence',\n",
            "    residual_col=residual_col\n",
            ")\n",
            "\n",
            "print(f\"={'='*80}\")\n",
            "print(f\"SAOCOM ACCURACY BY RADAR GEOMETRY (vs {ref_name})\")\n",
            "print(f\"={'='*80}\")\n",
            "print(f\"{'Category':<15} {'Count':>8} {'Incid(°)':>9} {'Bias(m)':>9} {'RMSE(m)':>9} {'NMAD(m)':>9}\")\n",
            "print(\"-\" * 80)\n",
            "\n",
            "for category, stats in shadow_stats.items():\n",
            "    if stats['count'] > 0:\n",
            "        print(f\"{category.capitalize():<15} {stats['count']:>8} \"\n",
            "              f\"{stats['mean_incidence']:>9.1f} \"\n",
            "              f\"{stats['bias']:>9.2f} \"\n",
            "              f\"{stats['rmse']:>9.2f} \"\n",
            "              f\"{stats['nmad']:>9.2f}\")\n",
            "\n",
            "print(f\"={'='*80}\\n\")\n",
            "\n",
            "# Highlight key findings\n",
            "if shadow_stats['shadow']['count'] > 0 and shadow_stats['optimal']['count'] > 0:\n",
            "    rmse_ratio = shadow_stats['shadow']['rmse'] / shadow_stats['optimal']['rmse']\n",
            "    print(f\"Key Finding: Shadow areas show {rmse_ratio:.1f}× higher RMSE than optimal geometry\")\n",
            "\n",
            "if shadow_stats['layover']['count'] > 0 and shadow_stats['optimal']['count'] > 0:\n",
            "    bias_diff = abs(shadow_stats['layover']['bias']) - abs(shadow_stats['optimal']['bias'])\n",
            "    print(f\"Key Finding: Layover areas have {bias_diff:+.2f}m additional bias\")"
        ]
    },

    # Code cell 5: Visualization
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize radar geometry and shadow effects\n",
            "print(\"\\nCreating radar geometry visualizations...\\n\")\n",
            "\n",
            "fig, axes = plt.subplots(2, 2, figsize=(16, 14))\n",
            "\n",
            "# Plot 1: Local incidence angle map\n",
            "ax1 = axes[0, 0]\n",
            "im1 = ax1.imshow(local_incidence, cmap='RdYlGn_r', vmin=0, vmax=90)\n",
            "ax1.set_title('Local Incidence Angle\\n(accounting for terrain orientation)', \n",
            "              fontsize=13, fontweight='bold')\n",
            "ax1.set_xlabel('Column')\n",
            "ax1.set_ylabel('Row')\n",
            "cbar1 = plt.colorbar(im1, ax=ax1, label='Angle (degrees)')\n",
            "\n",
            "# Overlay shadow areas\n",
            "shadow_overlay = np.where(shadow_mask, 1, np.nan)\n",
            "ax1.imshow(shadow_overlay, cmap='binary', alpha=0.6)\n",
            "ax1.text(0.02, 0.98, 'Black = Shadow', transform=ax1.transAxes,\n",
            "         fontsize=10, verticalalignment='top',\n",
            "         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
            "\n",
            "# Plot 2: Geometric quality classification\n",
            "ax2 = axes[0, 1]\n",
            "quality_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#34495e', '#9b59b6']\n",
            "quality_cmap = LinearSegmentedColormap.from_list('quality', quality_colors, N=5)\n",
            "im2 = ax2.imshow(geometric_quality, cmap=quality_cmap, vmin=0, vmax=4)\n",
            "ax2.set_title('Radar Geometric Quality Classification', fontsize=13, fontweight='bold')\n",
            "ax2.set_xlabel('Column')\n",
            "ax2.set_ylabel('Row')\n",
            "\n",
            "patches = [mpatches.Patch(color=quality_colors[i], label=quality_names[i])\n",
            "           for i in range(5)]\n",
            "ax2.legend(handles=patches, loc='upper right', fontsize=10)\n",
            "\n",
            "# Plot 3: Residuals vs Local Incidence Angle\n",
            "ax3 = axes[1, 0]\n",
            "valid_data = target_gdf[target_gdf[residual_col].notna() & target_gdf['local_incidence'].notna()]\n",
            "if len(valid_data) > 0:\n",
            "    ax3.hexbin(valid_data['local_incidence'], valid_data[residual_col],\n",
            "               gridsize=50, cmap='YlOrRd', mincnt=1)\n",
            "    ax3.axhline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero error')\n",
            "    ax3.axvline(90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Shadow threshold')\n",
            "    ax3.set_xlabel('Local Incidence Angle (degrees)', fontsize=11)\n",
            "    ax3.set_ylabel(f'Residual (m) vs {ref_name}', fontsize=11)\n",
            "    ax3.set_title('Residuals vs Local Incidence Angle', fontsize=13, fontweight='bold')\n",
            "    ax3.legend(fontsize=10)\n",
            "    ax3.grid(True, alpha=0.3)\n",
            "\n",
            "# Plot 4: RMSE by geometric quality\n",
            "ax4 = axes[1, 1]\n",
            "categories = []\n",
            "rmse_values = []\n",
            "counts = []\n",
            "colors_bar = []\n",
            "\n",
            "for i, (cat, stats) in enumerate(shadow_stats.items()):\n",
            "    if stats['count'] > 0:\n",
            "        categories.append(cat.capitalize())\n",
            "        rmse_values.append(stats['rmse'])\n",
            "        counts.append(stats['count'])\n",
            "        # Map category to color\n",
            "        color_map = {'optimal': quality_colors[0], 'acceptable': quality_colors[1],\n",
            "                    'steep': quality_colors[2], 'shadow': quality_colors[3],\n",
            "                    'layover': quality_colors[4]}\n",
            "        colors_bar.append(color_map.get(cat, 'gray'))\n",
            "\n",
            "bars = ax4.bar(range(len(categories)), rmse_values, color=colors_bar, alpha=0.7, edgecolor='black')\n",
            "ax4.set_xticks(range(len(categories)))\n",
            "ax4.set_xticklabels(categories, rotation=30, ha='right')\n",
            "ax4.set_ylabel('RMSE (m)', fontsize=11)\n",
            "ax4.set_title(f'RMSE by Geometric Quality Category\\nvs {ref_name}', fontsize=13, fontweight='bold')\n",
            "ax4.grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Add count labels on bars\n",
            "for bar, count in zip(bars, counts):\n",
            "    height = bar.get_height()\n",
            "    ax4.text(bar.get_x() + bar.get_width()/2., height,\n",
            "            f'n={count}', ha='center', va='bottom', fontsize=9)\n",
            "\n",
            "plt.suptitle(f'Radar Shadow and Geometry Analysis\\n'\n",
            "             f'SAOCOM Incidence: {RADAR_INCIDENCE}°, Azimuth: {RADAR_AZIMUTH}°',\n",
            "             fontsize=15, fontweight='bold', y=0.995)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('images/radar_geometry_analysis.png', dpi=300, bbox_inches='tight')\n",
            "print(\"✓ Saved: images/radar_geometry_analysis.png\")\n",
            "plt.show()\n",
            "\n",
            "# Save geometric quality rasters\n",
            "output_folder = Path('topography_outputs/radar_geometry')\n",
            "output_folder.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "import rasterio\n",
            "with rasterio.open('data/tinitaly/tinitaly_crop.tif') as src:\n",
            "    profile = src.profile.copy()\n",
            "    profile.update(dtype=rasterio.float32, count=1, compress='lzw')\n",
            "\n",
            "with rasterio.open(output_folder / 'local_incidence_angle.tif', 'w', **profile) as dst:\n",
            "    dst.write(local_incidence.astype(np.float32), 1)\n",
            "\n",
            "profile.update(dtype=rasterio.uint8)\n",
            "with rasterio.open(output_folder / 'geometric_quality.tif', 'w', **profile) as dst:\n",
            "    dst.write(geometric_quality.astype(np.uint8), 1)\n",
            "\n",
            "with rasterio.open(output_folder / 'shadow_mask.tif', 'w', **profile) as dst:\n",
            "    dst.write(shadow_mask.astype(np.uint8), 1)\n",
            "\n",
            "print(f\"✓ Saved raster outputs to: {output_folder}\")\n",
            "print(f\"  - local_incidence_angle.tif\")\n",
            "print(f\"  - geometric_quality.tif\")\n",
            "print(f\"  - shadow_mask.tif\")\n",
            "\n",
            "print(f\"\\n{'='*80}\")\n",
            "print(f\"RADAR SHADOW ANALYSIS COMPLETE\")\n",
            "print(f\"{'='*80}\")"
        ]
    }
]

# =============================================================================
# INSERT CELLS INTO NOTEBOOK
# =============================================================================

print(f"\\nInserting {len(radar_shadow_cells)} cells at position {insertion_index}")

# Insert cells
new_cells = nb['cells'][:insertion_index] + radar_shadow_cells + nb['cells'][insertion_index:]
nb['cells'] = new_cells

print(f"Updated notebook now has {len(nb['cells'])} cells (was {len(nb['cells']) - len(radar_shadow_cells)})")

# =============================================================================
# SAVE UPDATED NOTEBOOK
# =============================================================================

# Backup original
backup_path = notebook_path.with_suffix('.ipynb.backup')
print(f"\\nCreating backup: {backup_path}")

import shutil
shutil.copy2(notebook_path, backup_path)

# Save updated notebook
print(f"Saving updated notebook: {notebook_path}")
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\\n{'='*70}")
print(f"SUCCESS: Radar Shadow Analysis added to notebook!")
print(f"{'='*70}")
print(f"\\nAdded {len(radar_shadow_cells)} cells:")
print(f"  1 markdown header")
print(f"  5 code cells (configuration, calculation, sampling, analysis, visualization)")
print(f"\\nBackup saved to: {backup_path}")
print(f"\\nYou can now run the updated notebook!")
print(f"{'='*70}")
