"""
Add remaining visualizations from original to clean notebook
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

def add_more_visualizations():
    """Add remaining visualizations"""

    # Load notebook
    clean_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(clean_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook: {len(nb['cells'])} cells")

    new_cells = []

    # 1. Reference DEM Comparison
    new_cells.append(create_cell('markdown', [
        '### 12.8 Reference DEM Comparison\n',
        '\n',
        'Direct comparison of TINItaly and Copernicus DEMs.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Reference DEM comparison\n',
        'print("Creating reference DEM comparison...")\n',
        '\n',
        '# Calculate difference between reference DEMs\n',
        'dem_diff = tinitaly_10m - copernicus_10m\n',
        'dem_diff[tinitaly_10m == -9999] = np.nan\n',
        'dem_diff[copernicus_10m == -9999] = np.nan\n',
        '\n',
        '# Create multi-panel comparison\n',
        'fig, axes = plt.subplots(2, 2, figsize=(18, 16))\n',
        '\n',
        '# TINItaly DEM\n',
        'tin_plot = tinitaly_10m.copy()\n',
        'tin_plot[tin_plot == -9999] = np.nan\n',
        'im1 = axes[0, 0].imshow(tin_plot, cmap=\'terrain\')\n',
        'plt.colorbar(im1, ax=axes[0, 0], label=\'Elevation (m)\')\n',
        'axes[0, 0].set_title(\'TINItaly DEM (10m)\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0, 0].axis(\'off\')\n',
        '\n',
        '# Copernicus DEM\n',
        'cop_plot = copernicus_10m.copy()\n',
        'cop_plot[cop_plot == -9999] = np.nan\n',
        'im2 = axes[0, 1].imshow(cop_plot, cmap=\'terrain\')\n',
        'plt.colorbar(im2, ax=axes[0, 1], label=\'Elevation (m)\')\n',
        'axes[0, 1].set_title(\'Copernicus DEM (10m)\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0, 1].axis(\'off\')\n',
        '\n',
        '# Difference map\n',
        'if not np.all(np.isnan(dem_diff)):\n',
        '    vmin, vmax = np.nanpercentile(dem_diff, [2, 98])\n',
        '    im3 = axes[1, 0].imshow(dem_diff, cmap=\'RdBu_r\', vmin=vmin, vmax=vmax)\n',
        '    plt.colorbar(im3, ax=axes[1, 0], label=\'Difference (m)\')\n',
        '    axes[1, 0].set_title(\'TINItaly - Copernicus\', fontsize=14, fontweight=\'bold\')\n',
        '    axes[1, 0].axis(\'off\')\n',
        '\n',
        '# Statistics panel\n',
        'axes[1, 1].axis(\'off\')\n',
        'stats_text = f"""Reference DEM Comparison Statistics\n',
        '\n',
        'TINItaly:\n',
        '  Resolution: 10m (native)\n',
        '  Range: [{np.nanmin(tin_plot):.1f}, {np.nanmax(tin_plot):.1f}] m\n',
        '  Mean: {np.nanmean(tin_plot):.1f} m\n',
        '\n',
        'Copernicus:\n',
        '  Resolution: 30m → 10m (resampled)\n',
        '  Range: [{np.nanmin(cop_plot):.1f}, {np.nanmax(cop_plot):.1f}] m\n',
        '  Mean: {np.nanmean(cop_plot):.1f} m\n',
        '\n',
        'Difference (TINItaly - Copernicus):\n',
        '  Mean: {np.nanmean(dem_diff):.2f} m\n',
        '  Std: {np.nanstd(dem_diff):.2f} m\n',
        '  NMAD: {1.4826 * np.nanmedian(np.abs(dem_diff - np.nanmedian(dem_diff))):.2f} m\n',
        '  Range: [{np.nanmin(dem_diff):.2f}, {np.nanmax(dem_diff):.2f}] m\n',
        '"""\n',
        '\n',
        'axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,\n',
        '                fontsize=12, verticalalignment=\'center\', family=\'monospace\',\n',
        '                bbox=dict(boxstyle=\'round\', facecolor=\'wheat\', alpha=0.5))\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'reference_dem_comparison.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 2. Coverage Grid and Void Zones
    new_cells.append(create_cell('markdown', [
        '### 12.9 Coverage Grid and Void Zones\n',
        '\n',
        'Analyze spatial coverage and identify void zones (areas without measurements).\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Create SAOCOM coverage grid\n',
        'print("Creating coverage grid...")\n',
        '\n',
        '# Initialize coverage grid\n',
        'coverage_grid = np.zeros((grid_height, grid_width), dtype=bool)\n',
        '\n',
        '# Mark cells with SAOCOM data\n',
        'for idx, row in saocom_cleaned.iterrows():\n',
        '    r, c = rowcol(target_transform, row.geometry.x, row.geometry.y)\n',
        '    r, c = int(r), int(c)\n',
        '    if 0 <= r < grid_height and 0 <= c < grid_width:\n',
        '        coverage_grid[r, c] = True\n',
        '\n',
        '# Calculate void zones\n',
        'total_cells = grid_height * grid_width\n',
        'covered_cells = coverage_grid.sum()\n',
        'void_cells = total_cells - covered_cells\n',
        'coverage_pct = 100 * covered_cells / total_cells\n',
        '\n',
        'print(f"Coverage statistics:")\n',
        'print(f"  Total grid cells: {total_cells:,}")\n',
        'print(f"  Covered cells: {covered_cells:,}")\n',
        'print(f"  Void cells: {void_cells:,}")\n',
        'print(f"  Coverage: {coverage_pct:.1f}%")\n',
        '\n',
        '# Visualize coverage\n',
        'fig, axes = plt.subplots(1, 2, figsize=(18, 8))\n',
        '\n',
        '# Coverage map\n',
        'axes[0].imshow(coverage_grid, cmap=\'binary\', interpolation=\'nearest\')\n',
        'axes[0].set_title(f\'SAOCOM Coverage Grid ({coverage_pct:.1f}% covered)\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0].axis(\'off\')\n',
        '\n',
        '# Void zones overlay on slope\n',
        'void_mask = ~coverage_grid\n',
        'slope_with_voids = slope_tin.copy()\n',
        'slope_with_voids[void_mask] = np.nan\n',
        '\n',
        'im2 = axes[1].imshow(slope_tin, cmap=\'terrain\', alpha=0.7)\n',
        'axes[1].imshow(void_mask, cmap=\'Reds\', alpha=0.3)\n',
        'plt.colorbar(im2, ax=axes[1], label=\'Slope (degrees)\')\n',
        'axes[1].set_title(\'Void Zones (red) over Terrain Slope\', fontsize=14, fontweight=\'bold\')\n',
        'axes[1].axis(\'off\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'coverage_and_voids.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 3. Residuals by Height Bins
    new_cells.append(create_cell('markdown', [
        '### 12.10 Residuals by Elevation Bins\n',
        '\n',
        'Investigate if accuracy varies with elevation.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Bin residuals by elevation\n',
        'height_bins = [0, 200, 400, 600, 800, 1000]\n',
        'height_labels = [\'0-200m\', \'200-400m\', \'400-600m\', \'600-800m\', \'800-1000m\']\n',
        '\n',
        'saocom_cleaned[\'height_category\'] = pd.cut(\n',
        '    saocom_cleaned[\'tinitaly_height\'],\n',
        '    bins=height_bins,\n',
        '    labels=height_labels\n',
        ')\n',
        '\n',
        '# Calculate statistics by height\n',
        'height_stats = saocom_cleaned.groupby(\'height_category\')[\'diff_tinitaly\'].agg([\n',
        '    (\'count\', \'count\'),\n',
        '    (\'mean\', \'mean\'),\n',
        '    (\'std\', \'std\'),\n',
        '    (\'nmad\', lambda x: nmad(x.dropna()))\n',
        ']).round(2)\n',
        '\n',
        'print("\\nAccuracy by elevation:")\n',
        'print(height_stats)\n',
        '\n',
        '# Visualize\n',
        'fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n',
        '\n',
        '# Bar plot of NMAD by elevation\n',
        'height_stats[\'nmad\'].plot(kind=\'bar\', ax=axes[0], color=\'coral\', edgecolor=\'black\')\n',
        'axes[0].set_title(\'NMAD by Elevation Range\', fontsize=14, fontweight=\'bold\')\n',
        'axes[0].set_xlabel(\'Elevation Range\', fontsize=12)\n',
        'axes[0].set_ylabel(\'NMAD (m)\', fontsize=12)\n',
        'axes[0].grid(axis=\'y\', alpha=0.3)\n',
        'axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha=\'right\')\n',
        '\n',
        '# Sample counts by elevation\n',
        'height_stats[\'count\'].plot(kind=\'bar\', ax=axes[1], color=\'skyblue\', edgecolor=\'black\')\n',
        'axes[1].set_title(\'Sample Count by Elevation Range\', fontsize=14, fontweight=\'bold\')\n',
        'axes[1].set_xlabel(\'Elevation Range\', fontsize=12)\n',
        'axes[1].set_ylabel(\'Count\', fontsize=12)\n',
        'axes[1].grid(axis=\'y\', alpha=0.3)\n',
        'axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha=\'right\')\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig(IMAGES_DIR / \'accuracy_by_elevation.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # 4. Final Summary Figure
    new_cells.append(create_cell('markdown', [
        '### 12.11 Summary Dashboard\n',
        '\n',
        'Comprehensive summary of all validation metrics in one figure.\n'
    ]))

    new_cells.append(create_cell('code', [
        '# Create summary dashboard\n',
        'fig = plt.figure(figsize=(20, 12))\n',
        'gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)\n',
        '\n',
        '# 1. Spatial distribution\n',
        'ax1 = fig.add_subplot(gs[0, 0])\n',
        'saocom_cleaned.plot(ax=ax1, markersize=0.5, color=\'blue\', alpha=0.3)\n',
        'ax1.set_title(\'SAOCOM Point Distribution\', fontweight=\'bold\')\n',
        'ax1.set_xlabel(\'Easting (m)\')\n',
        'ax1.set_ylabel(\'Northing (m)\')\n',
        'ax1.set_aspect(\'equal\')\n',
        '\n',
        '# 2. Residual histogram (TINItaly)\n',
        'ax2 = fig.add_subplot(gs[0, 1])\n',
        'ax2.hist(residuals_tin, bins=100, color=\'steelblue\', edgecolor=\'black\', alpha=0.7)\n',
        'ax2.axvline(x=0, color=\'red\', linestyle=\'--\', linewidth=2)\n',
        'ax2.set_title(f\'Residuals (NMAD={nmad_tin:.2f}m)\', fontweight=\'bold\')\n',
        'ax2.set_xlabel(\'SAOCOM - TINItaly (m)\')\n',
        'ax2.set_ylabel(\'Frequency\')\n',
        'ax2.grid(alpha=0.3)\n',
        '\n',
        '# 3. Accuracy by slope\n',
        'ax3 = fig.add_subplot(gs[0, 2])\n',
        'slope_stats[\'nmad\'].plot(kind=\'bar\', ax=ax3, color=\'coral\', edgecolor=\'black\')\n',
        'ax3.set_title(\'NMAD by Slope Category\', fontweight=\'bold\')\n',
        'ax3.set_ylabel(\'NMAD (m)\')\n',
        'ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha=\'right\')\n',
        'ax3.grid(axis=\'y\', alpha=0.3)\n',
        '\n',
        '# 4. Scatter plot\n',
        'ax4 = fig.add_subplot(gs[1, 0])\n',
        'sample_size = min(10000, len(valid_tin))\n',
        'sample_indices = np.random.choice(len(valid_tin), sample_size, replace=False)\n',
        'ax4.scatter(\n',
        '    valid_tin[\'tinitaly_height\'].iloc[sample_indices],\n',
        '    valid_tin[\'HEIGHT_ABSOLUTE_TIN\'].iloc[sample_indices],\n',
        '    s=1, alpha=0.3, color=\'blue\'\n',
        ')\n',
        'lims = [valid_tin[\'tinitaly_height\'].min(), valid_tin[\'tinitaly_height\'].max()]\n',
        'ax4.plot(lims, lims, \'r--\', alpha=0.5, linewidth=2)\n',
        'ax4.set_title(\'SAOCOM vs TINItaly\', fontweight=\'bold\')\n',
        'ax4.set_xlabel(\'TINItaly Height (m)\')\n',
        'ax4.set_ylabel(\'SAOCOM Height (m)\')\n',
        'ax4.grid(alpha=0.3)\n',
        '\n',
        '# 5. Slope map\n',
        'ax5 = fig.add_subplot(gs[1, 1])\n',
        'slope_plot = ax5.imshow(slope_tin, cmap=\'terrain\', vmin=0, vmax=45)\n',
        'plt.colorbar(slope_plot, ax=ax5, label=\'Slope (°)\', fraction=0.046)\n',
        'ax5.set_title(\'Terrain Slope\', fontweight=\'bold\')\n',
        'ax5.axis(\'off\')\n',
        '\n',
        '# 6. Residuals spatial map\n',
        'ax6 = fig.add_subplot(gs[1, 2])\n',
        'valid_pts = saocom_cleaned[saocom_cleaned[\'diff_tinitaly\'].notna()]\n',
        'sample_pts = valid_pts.sample(min(10000, len(valid_pts)))\n',
        'vmin, vmax = np.percentile(sample_pts[\'diff_tinitaly\'], [2, 98])\n',
        'sc = ax6.scatter(\n',
        '    sample_pts.geometry.x,\n',
        '    sample_pts.geometry.y,\n',
        '    c=sample_pts[\'diff_tinitaly\'],\n',
        '    cmap=\'RdBu_r\',\n',
        '    s=1,\n',
        '    vmin=vmin,\n',
        '    vmax=vmax,\n',
        '    alpha=0.5\n',
        ')\n',
        'plt.colorbar(sc, ax=ax6, label=\'Residual (m)\', fraction=0.046)\n',
        'ax6.set_title(\'Spatial Residuals\', fontweight=\'bold\')\n',
        'ax6.set_aspect(\'equal\')\n',
        'ax6.axis(\'off\')\n',
        '\n',
        '# 7. Statistics text\n',
        'ax7 = fig.add_subplot(gs[2, :])\n',
        'ax7.axis(\'off\')\n',
        '\n',
        'summary_text = f"""\\n',
        'SAOCOM INSAR VALIDATION SUMMARY\n',
        '{"="*80}\n',
        '\n',
        'Dataset Statistics:\n',
        '  Total points: {len(saocom_gdf):,}\n',
        '  Outliers removed: {len(outliers):,} ({100*len(outliers)/len(saocom_gdf):.1f}%)\n',
        '  Clean dataset: {len(saocom_cleaned):,}\n',
        '\n',
        'Validation against TINItaly (10m resolution):\n',
        '  NMAD: {nmad_tin:.2f} m\n',
        '  RMSE: {np.sqrt((residuals_tin**2).mean()):.2f} m\n',
        '  Mean error: {residuals_tin.mean():.2f} m\n',
        '  Correlation: {np.corrcoef(valid_tin["HEIGHT_ABSOLUTE_TIN"], valid_tin["tinitaly_height"])[0,1]:.4f}\n',
        '\n',
        'Validation against Copernicus (30m resampled to 10m):\n',
        '  NMAD: {nmad_cop:.2f} m\n',
        '  RMSE: {np.sqrt((residuals_cop**2).mean()):.2f} m\n',
        '  Mean error: {residuals_cop.mean():.2f} m\n',
        '  Correlation: {np.corrcoef(valid_cop["HEIGHT_ABSOLUTE_COP"], valid_cop["copernicus_height"])[0,1]:.4f}\n',
        '\n',
        'Performance by Terrain:\n',
        '  Flat (0-5°):        NMAD = {slope_stats.loc["Flat (0-5°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Flat (0-5°)", "count"]):,})\n',
        '  Gentle (5-15°):     NMAD = {slope_stats.loc["Gentle (5-15°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Gentle (5-15°)", "count"]):,})\n',
        '  Moderate (15-30°):  NMAD = {slope_stats.loc["Moderate (15-30°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Moderate (15-30°)", "count"]):,})\n',
        '  Steep (>30°):       NMAD = {slope_stats.loc["Steep (>30°)", "nmad"]:.2f} m  (n={int(slope_stats.loc["Steep (>30°)", "count"]):,})\n',
        '"""\n',
        '\n',
        'ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes,\n',
        '         fontsize=11, verticalalignment=\'center\', family=\'monospace\',\n',
        '         bbox=dict(boxstyle=\'round\', facecolor=\'lightblue\', alpha=0.3))\n',
        '\n',
        'fig.suptitle(\'SAOCOM InSAR Height Validation - Complete Summary\',\n',
        '             fontsize=16, fontweight=\'bold\', y=0.98)\n',
        '\n',
        'plt.savefig(IMAGES_DIR / \'summary_dashboard.png\', dpi=300, bbox_inches=\'tight\')\n',
        'plt.show()\n'
    ]))

    # Add all new cells
    nb['cells'].extend(new_cells)

    print(f"\\nAdded {len(new_cells)} new cells")
    print(f"Total cells now: {len(nb['cells'])}")

    # Save
    with open(clean_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\\nSaved updated notebook to: {clean_path.name}")
    print("\\nAdditional visualizations added:")
    print("  8. Reference DEM comparison (multi-panel)")
    print("  9. Coverage grid and void zones")
    print("  10. Residuals by elevation bins")
    print("  11. Summary dashboard (comprehensive)")

if __name__ == '__main__':
    add_more_visualizations()
