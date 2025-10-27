"""
Add missing visualizations from original notebook to saocom_analysis_clean.ipynb
Maintains the clean format while adding all graphics
"""

import json
from pathlib import Path

def create_cell(cell_type, source):
    """Create a notebook cell"""
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': source if isinstance(source, list) else [source]
    }
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell

def add_missing_visualizations():
    """Add all missing visualizations to clean notebook"""

    # Load clean notebook
    clean_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(clean_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded clean notebook: {len(nb['cells'])} cells")

    # New cells to add after current content
    new_cells = []

    # 1. Spatial Overlap Visualization
    new_cells.append(create_cell('markdown', [
        '---\n',
        '## 12. Additional Visualizations\n',
        '\n',
        'Comprehensive visualization suite from the original analysis.\n'
    ]))

    new_cells.append(create_cell('markdown', [
        '### 12.1 Spatial Coverage Map\n',
        '\n',
        'Verify that SAOCOM points fall within the reference DEM extent.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Spatial overlap visualization\n',
        'from matplotlib.patches import Rectangle\n',
        '\n',
        'fig, ax = plt.subplots(figsize=(12, 10))\n',
        '\n',
        '# TINITALY extent box\n',
        'with rasterio.open(TINITALY_DEM) as src:\n',
        '    dem_bounds = src.bounds\n',
        '    # Reproject bounds to target CRS if needed\n',
        '    import rasterio.warp\n',
        '    dem_bounds_utm = rasterio.warp.transform_bounds(src.crs, TARGET_CRS, *dem_bounds)\n',
        '    \n',
        '    ax.add_patch(Rectangle(\n',
        '        (dem_bounds_utm[0], dem_bounds_utm[1]),\n',
        '        dem_bounds_utm[2] - dem_bounds_utm[0],\n',
        '        dem_bounds_utm[3] - dem_bounds_utm[1],\n',
        '        linewidth=3, edgecolor=\'blue\', facecolor=\'none\', label=\'TINItaly Extent\'\n',
        '    ))\n',
        '\n',
        '# SAOCOM points\n',
        'saocom_cleaned.plot(ax=ax, markersize=1, color=\'red\', alpha=0.5, label=\'SAOCOM Points\')\n',
        '\n',
        '# Study area hull\n',
        'from shapely.geometry import box\n',
        'hull = saocom_cleaned.geometry.unary_union.convex_hull\n',
        'hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)\n',
        'hull_gdf.boundary.plot(ax=ax, color=\'green\', linewidth=2, linestyle=\'--\', label=\'Study Area Hull\')\n',
        '\n',
        'ax.set_xlabel(\'UTM Easting (m)\', fontsize=12)\n',
        'ax.set_ylabel(\'UTM Northing (m)\', fontsize=12)\n',
        'ax.set_title(\'Spatial Coverage: SAOCOM vs TINItaly DEM\', fontsize=14, fontweight=\'bold\')\n',
        'ax.legend(loc=\'best\', fontsize=10)\n',
        'ax.grid(True, alpha=0.3)\n',
        'ax.set_aspect(\'equal\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'spatial_coverage.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 2. Gridded Comparison Analysis
    new_cells.append(create_cell('markdown', [
        '### 12.2 Gridded Comparison Analysis\n',
        '\n',
        'Create gridded difference maps to show spatial patterns of height differences.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Create gridded difference maps\n',
        'print("Creating gridded difference maps...")\n',
        '\n',
        '# Create difference grids\n',
        'diff_grid_tin = create_difference_grid(\n',
        '    saocom_cleaned,\n',
        '    height_col=\'HEIGHT_ABSOLUTE_TIN\',\n',
        '    ref_col=\'tinitaly_height\',\n',
        '    grid_shape=(grid_height, grid_width),\n',
        '    transform=target_transform,\n',
        '    hull_mask=None\n',
        ')\n',
        '\n',
        'diff_grid_cop = create_difference_grid(\n',
        '    saocom_cleaned,\n',
        '    height_col=\'HEIGHT_ABSOLUTE_COP\',\n',
        '    ref_col=\'copernicus_height\',\n',
        '    grid_shape=(grid_height, grid_width),\n',
        '    transform=target_transform,\n',
        '    hull_mask=None\n',
        ')\n',
        '\n',
        '# Visualize gridded differences\n',
        'fig, axes = plt.subplots(1, 2, figsize=(18, 8))\n',
        '\n',
        '# TINItaly grid\n',
        'valid_mask_tin = ~np.isnan(diff_grid_tin)\n',
        'if valid_mask_tin.any():\n',
        '    vmin, vmax = np.percentile(diff_grid_tin[valid_mask_tin], [2, 98])\n',
        '    im1 = axes[0].imshow(diff_grid_tin, cmap=\'RdBu_r\', vmin=vmin, vmax=vmax)\n',
        '    plt.colorbar(im1, ax=axes[0], label=\'Difference (m)\')\n',
        '    axes[0].set_title(\'SAOCOM - TINItaly (Gridded)\', fontsize=14, fontweight=\'bold\')\n',
        '    axes[0].axis(\'off\')\n',
        '\n',
        '# Copernicus grid\n',
        'valid_mask_cop = ~np.isnan(diff_grid_cop)\n',
        'if valid_mask_cop.any():\n',
        '    vmin2, vmax2 = np.percentile(diff_grid_cop[valid_mask_cop], [2, 98])\n',
        '    im2 = axes[1].imshow(diff_grid_cop, cmap=\'RdBu_r\', vmin=vmin2, vmax=vmax2)\n',
        '    plt.colorbar(im2, ax=axes[1], label=\'Difference (m)\')\n',
        '    axes[1].set_title(\'SAOCOM - Copernicus (Gridded)\', fontsize=14, fontweight=\'bold\')\n',
        '    axes[1].axis(\'off\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'gridded_comparison.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 3. Hexbin Density Plots
    new_cells.append(create_cell('markdown', [
        '### 12.3 Density Plots (Hexbin)\n',
        '\n',
        'Hexbin plots show the density of measurements, useful for identifying data clustering.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Hexbin density plots\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n',
        '\n',
        '# TINItaly hexbin\n',
        'valid_tin = saocom_cleaned[[\'HEIGHT_ABSOLUTE_TIN\', \'tinitaly_height\']].dropna()\n',
        'hb1 = axes[0].hexbin(\n',
        '    valid_tin[\'tinitaly_height\'],\n',
        '    valid_tin[\'HEIGHT_ABSOLUTE_TIN\'],\n',
        '    gridsize=50,\n',
        '    cmap=\'YlOrRd\',\n',
        '    mincnt=1,\n',
        '    edgecolors=\'none\'\n',
        ')\n',
        'plt.colorbar(hb1, ax=axes[0], label=\'Count\')\n',
        '\n',
        '# 1:1 line\n',
        'lims = [min(valid_tin[\'tinitaly_height\'].min(), valid_tin[\'HEIGHT_ABSOLUTE_TIN\'].min()),\n',
        '        max(valid_tin[\'tinitaly_height\'].max(), valid_tin[\'HEIGHT_ABSOLUTE_TIN\'].max())]\n',
        'axes[0].plot(lims, lims, \'k--\', alpha=0.5, linewidth=2, label=\'1:1 Line\')\n',
        '\n',
        'axes[0].set_xlabel(\'TINItaly Height (m)\', fontsize=12)\n',
        'axes[0].set_ylabel(\'SAOCOM Height (m)\', fontsize=12)\n',
        'axes[0].set_title(\'Density: SAOCOM vs TINItaly\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0].legend()\n',
        'axes[0].grid(alpha=0.3)\n',
        '\n',
        '# Copernicus hexbin\n',
        'valid_cop = saocom_cleaned[[\'HEIGHT_ABSOLUTE_COP\', \'copernicus_height\']].dropna()\n',
        'hb2 = axes[1].hexbin(\n',
        '    valid_cop[\'copernicus_height\'],\n',
        '    valid_cop[\'HEIGHT_ABSOLUTE_COP\'],\n',
        '    gridsize=50,\n',
        '    cmap=\'YlOrRd\',\n',
        '    mincnt=1,\n',
        '    edgecolors=\'none\'\n',
        ')\n',
        'plt.colorbar(hb2, ax=axes[1], label=\'Count\')\n',
        '\n',
        '# 1:1 line\n',
        'lims2 = [min(valid_cop[\'copernicus_height\'].min(), valid_cop[\'HEIGHT_ABSOLUTE_COP\'].min()),\n',
        '         max(valid_cop[\'copernicus_height\'].max(), valid_cop[\'HEIGHT_ABSOLUTE_COP\'].max())]\n',
        'axes[1].plot(lims2, lims2, \'k--\', alpha=0.5, linewidth=2, label=\'1:1 Line\')\n',
        '\n',
        'axes[1].set_xlabel(\'Copernicus Height (m)\', fontsize=12)\n',
        'axes[1].set_ylabel(\'SAOCOM Height (m)\', fontsize=12)\n',
        'axes[1].set_title(\'Density: SAOCOM vs Copernicus\', fontsize=14, fontweight=\'bold\')\n',
        'axes[1].legend()\n',
        'axes[1].grid(alpha=0.3)\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'hexbin_density.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 4. 2D Histograms
    new_cells.append(create_cell('markdown', [
        '### 12.4 2D Histograms\n',
        '\n',
        'Alternative visualization of measurement density.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# 2D histogram plots\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n',
        '\n',
        '# TINItaly 2D histogram\n',
        'h1 = axes[0].hist2d(\n',
        '    valid_tin[\'tinitaly_height\'],\n',
        '    valid_tin[\'HEIGHT_ABSOLUTE_TIN\'],\n',
        '    bins=100,\n',
        '    cmap=\'viridis\',\n',
        '    cmin=1\n',
        ')\n',
        'plt.colorbar(h1[3], ax=axes[0], label=\'Count\')\n',
        '\n',
        '# 1:1 line\n',
        'axes[0].plot(lims, lims, \'r--\', alpha=0.7, linewidth=2, label=\'1:1 Line\')\n',
        'axes[0].set_xlabel(\'TINItaly Height (m)\', fontsize=12)\n',
        'axes[0].set_ylabel(\'SAOCOM Height (m)\', fontsize=12)\n',
        'axes[0].set_title(\'2D Histogram: SAOCOM vs TINItaly\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0].legend()\n',
        'axes[0].grid(alpha=0.3)\n',
        '\n',
        '# Copernicus 2D histogram\n',
        'h2 = axes[1].hist2d(\n',
        '    valid_cop[\'copernicus_height\'],\n',
        '    valid_cop[\'HEIGHT_ABSOLUTE_COP\'],\n',
        '    bins=100,\n',
        '    cmap=\'viridis\',\n',
        '    cmin=1\n',
        ')\n',
        'plt.colorbar(h2[3], ax=axes[1], label=\'Count\')\n',
        '\n',
        '# 1:1 line\n',
        'axes[1].plot(lims2, lims2, \'r--\', alpha=0.7, linewidth=2, label=\'1:1 Line\')\n',
        'axes[1].set_xlabel(\'Copernicus Height (m)\', fontsize=12)\n',
        'axes[1].set_ylabel(\'SAOCOM Height (m)\', fontsize=12)\n',
        'axes[1].set_title(\'2D Histogram: SAOCOM vs Copernicus\', fontsize=14, fontweight=\'bold\')\n',
        'axes[1].legend()\n',
        'axes[1].grid(alpha=0.3)\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'hist2d_comparison.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 5. Violin Plots by Slope
    new_cells.append(create_cell('markdown', [
        '### 12.5 Violin Plots - Accuracy by Slope Category\n',
        '\n',
        'Detailed performance breakdown showing full distribution of residuals for each terrain type.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Violin plot of residuals by slope category\n',
        'fig, ax = plt.subplots(figsize=(12, 7))\n',
        '\n',
        '# Prepare data for violin plot\n',
        'slope_data = saocom_cleaned[[\'slope_category\', \'diff_tinitaly\']].dropna()\n',
        '\n',
        '# Create violin plot\n',
        'parts = ax.violinplot(\n',
        '    [slope_data[slope_data[\'slope_category\'] == cat][\'diff_tinitaly\'].values \n',
        '     for cat in slope_labels],\n',
        '    positions=range(len(slope_labels)),\n',
        '    showmeans=True,\n',
        '    showmedians=True,\n',
        '    widths=0.7\n',
        ')\n',
        '\n',
        '# Customize colors\n',
        'for pc in parts[\'bodies\']:\n',
        '    pc.set_facecolor(\'steelblue\')\n',
        '    pc.set_alpha(0.7)\n',
        '\n',
        'ax.set_xticks(range(len(slope_labels)))\n',
        'ax.set_xticklabels(slope_labels, rotation=0)\n',
        'ax.set_xlabel(\'Slope Category\', fontsize=12, fontweight=\'bold\')\n',
        'ax.set_ylabel(\'Residual (SAOCOM - TINItaly) [m]\', fontsize=12, fontweight=\'bold\')\n',
        'ax.set_title(\'Residual Distribution by Slope Category\', fontsize=14, fontweight=\'bold\')\n',
        'ax.axhline(y=0, color=\'red\', linestyle=\'--\', alpha=0.5, linewidth=2, label=\'Zero Error\')\n',
        'ax.grid(True, alpha=0.3, axis=\'y\')\n',
        'ax.legend()\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'violin_plot_slope.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n',
        '\n',
        'print("\\nStatistics by slope category:")\n',
        'print(slope_stats)\n'
    ]))

    # 6. Residuals vs Coherence
    new_cells.append(create_cell('markdown', [
        '### 12.6 Residuals vs Coherence\n',
        '\n',
        'Investigate the relationship between measurement quality (coherence) and accuracy.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Scatter plot of residuals vs coherence\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n',
        '\n',
        '# TINItaly residuals vs coherence\n',
        'valid_data_tin = saocom_cleaned[[\'COHER\', \'diff_tinitaly\']].dropna()\n',
        'axes[0].scatter(\n',
        '    valid_data_tin[\'COHER\'],\n',
        '    valid_data_tin[\'diff_tinitaly\'],\n',
        '    c=valid_data_tin[\'diff_tinitaly\'],\n',
        '    cmap=\'RdBu_r\',\n',
        '    s=5,\n',
        '    alpha=0.3,\n',
        '    vmin=-10,\n',
        '    vmax=10\n',
        ')\n',
        'axes[0].axhline(y=0, color=\'black\', linestyle=\'--\', alpha=0.5)\n',
        'axes[0].set_xlabel(\'Coherence\', fontsize=12)\n',
        'axes[0].set_ylabel(\'Residual (m)\', fontsize=12)\n',
        'axes[0].set_title(\'Residuals vs Coherence (TINItaly)\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0].grid(True, alpha=0.3)\n',
        '\n',
        '# Copernicus residuals vs coherence\n',
        'valid_data_cop = saocom_cleaned[[\'COHER\', \'diff_copernicus\']].dropna()\n',
        'axes[1].scatter(\n',
        '    valid_data_cop[\'COHER\'],\n',
        '    valid_data_cop[\'diff_copernicus\'],\n',
        '    c=valid_data_cop[\'diff_copernicus\'],\n',
        '    cmap=\'RdBu_r\',\n',
        '    s=5,\n',
        '    alpha=0.3,\n',
        '    vmin=-10,\n',
        '    vmax=10\n',
        ')\n',
        'axes[1].axhline(y=0, color=\'black\', linestyle=\'--\', alpha=0.5)\n',
        'axes[1].set_xlabel(\'Coherence\', fontsize=12)\n',
        'axes[1].set_ylabel(\'Residual (m)\', fontsize=12)\n',
        'axes[1].set_title(\'Residuals vs Coherence (Copernicus)\', fontsize=14, fontweight=\'bold\')\n',
        'axes[1].grid(True, alpha=0.3)\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'residuals_vs_coherence.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 7. Slope Raster Visualization
    new_cells.append(create_cell('markdown', [
        '### 12.7 Terrain Slope Map\n',
        '\n',
        'Visualize the terrain slope across the study area.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Display slope raster\n',
        'fig, ax = plt.subplots(figsize=(12, 10))\n',
        '\n',
        '# Plot slope\n',
        'slope_plot = ax.imshow(slope_tin, cmap=\'terrain\', vmin=0, vmax=45)\n',
        'cbar = plt.colorbar(slope_plot, ax=ax, label=\'Slope (degrees)\')\n',
        'cbar.ax.tick_params(labelsize=10)\n',
        '\n',
        'ax.set_title(\'Terrain Slope from TINItaly DEM\', fontsize=14, fontweight=\'bold\')\n',
        'ax.axis(\'off\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'terrain_slope.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n',
        '\n',
        'print(f"Slope statistics:")\n',
        'print(f"  Mean: {np.nanmean(slope_tin):.1f}°")\n',
        'print(f"  Median: {np.nanmedian(slope_tin):.1f}°")\n',
        'print(f"  Max: {np.nanmax(slope_tin):.1f}°")\n'
    ]))

    # Add all new cells to notebook
    nb['cells'].extend(new_cells)

    print(f"\\nAdded {len(new_cells)} new cells")
    print(f"Total cells now: {len(nb['cells'])}")

    # Save updated notebook
    output_path = clean_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\\nSaved updated notebook to: {output_path.name}")
    print("\\nNew visualizations added:")
    print("  1. Spatial coverage map")
    print("  2. Gridded comparison analysis")
    print("  3. Hexbin density plots")
    print("  4. 2D histograms")
    print("  5. Violin plots by slope")
    print("  6. Residuals vs coherence scatter")
    print("  7. Terrain slope map")

if __name__ == '__main__':
    add_missing_visualizations()
