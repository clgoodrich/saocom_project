"""
Comprehensive fixes for all map visualizations:
1. Add grids and hull bounding boxes to ALL maps
2. Fix zoom/clipping issues with proper bounds
3. Fix residuals vs coherence to use bins (width 0.05)
4. Fix before/after histogram colors and empty bins
"""

import json
from pathlib import Path
import re

def fix_all_maps():
    """Apply all map visualization fixes"""

    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells\n")

    # ==================== FIX 1: ADD GRIDS AND HULL BOUNDING BOXES ====================
    print("=== FIX 1: Adding grids and hull bounding boxes to ALL maps ===\n")

    maps_fixed = 0

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])

        # Skip if no visualization
        if 'savefig' not in source or '.png' not in source:
            continue

        # Skip if already has hull and grid
        if 'convex_hull' in source and 'ax.grid(' in source:
            continue

        # Get the filename for context
        filename = 'unknown'
        for line in source.split('\n'):
            if 'savefig' in line and '.png' in line:
                filename = line.split('/')[-1].split('.png')[0].split("'")[0]
                break

        original_source = source
        modified = False

        # Add hull bounding box if it's a spatial map (has ax.scatter or plot with geometry.x)
        if ('geometry.x' in source or 'geometry.y' in source) and 'convex_hull' not in source:
            # Determine the main axes variable
            ax_var = 'ax'
            if 'axes[0]' in source:
                ax_var = 'axes[0]'
            elif 'axes[1]' in source:
                ax_var = 'axes[1]'
            elif 'ax1' in source:
                ax_var = 'ax1'

            # Find where to insert hull code (before set_title or set_xlabel)
            insert_patterns = [
                (f"{ax_var}.set_title(", f"\n# Add hull bounding box\nhull = saocom_cleaned.geometry.unary_union.convex_hull\nhull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)\nhull_gdf.boundary.plot(ax={ax_var}, color='red', linewidth=2, linestyle='--', label='Study Area Hull')\n\n{ax_var}.set_title("),
                (f"{ax_var}.set_xlabel(", f"\n# Add hull bounding box\nhull = saocom_cleaned.geometry.unary_union.convex_hull\nhull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)\nhull_gdf.boundary.plot(ax={ax_var}, color='red', linewidth=2, linestyle='--', label='Study Area Hull')\n\n{ax_var}.set_xlabel("),
            ]

            for pattern, replacement in insert_patterns:
                if pattern in source:
                    source = source.replace(pattern, replacement)
                    modified = True
                    break

        # Add grid if missing
        if 'ax.grid(' not in source and '.grid(' not in source:
            # Find appropriate place to add grid (before tight_layout or savefig)
            if 'plt.tight_layout()' in source:
                source = source.replace('plt.tight_layout()', '# Add grid to all axes\nif isinstance(ax, np.ndarray):\n    for a in ax.flat:\n        a.grid(True, alpha=0.3, linestyle=\"--\", color=\"gray\")\nelse:\n    ax.grid(True, alpha=0.3, linestyle=\"--\", color=\"gray\")\n\nplt.tight_layout()')
                modified = True
            elif 'plt.savefig' in source:
                savefig_line = [line for line in source.split('\n') if 'plt.savefig' in line][0]
                source = source.replace(savefig_line, '# Add grid\nif isinstance(ax, np.ndarray):\n    for a in ax.flat:\n        a.grid(True, alpha=0.3, linestyle=\"--\", color=\"gray\")\nelse:\n    ax.grid(True, alpha=0.3, linestyle=\"--\", color=\"gray\")\n\n' + savefig_line)
                modified = True

        # Fix bounds/zoom for spatial maps
        if 'geometry.x' in source and 'set_xlim' not in source:
            # Add proper bounds with margin
            if 'ax.set_aspect(' in source:
                source = source.replace('ax.set_aspect(', '# Set proper bounds with margin\nbounds = saocom_cleaned.total_bounds\nmargin_x = (bounds[2] - bounds[0]) * 0.05\nmargin_y = (bounds[3] - bounds[1]) * 0.05\nax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)\nax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)\n\nax.set_aspect(')
                modified = True

        if modified and source != original_source:
            cell['source'] = [source]
            maps_fixed += 1
            print(f"Cell {i}: Fixed {filename}.png")

    print(f"\n[OK] Fixed {maps_fixed} map visualizations\n")

    # ==================== FIX 2: RESIDUALS VS COHERENCE (BINNED) ====================
    print("=== FIX 2: Fixing residuals vs coherence to use bins ===\n")

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])

        # Find the residuals vs coherence cell
        if 'residuals_vs_coherence.png' in source and 'scatter' in source:
            print(f"Found residuals vs coherence in cell {i}")

            new_code = """# Binned analysis of residuals vs coherence
print("Creating binned coherence analysis...")

from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# TINItaly residuals vs coherence (binned)
valid_data_tin = saocom_cleaned[['COHER', 'diff_tinitaly']].dropna()

# Create bins of width 0.05
coherence_bins = np.arange(0, 1.05, 0.05)
bin_centers = (coherence_bins[:-1] + coherence_bins[1:]) / 2

# Bin the data
valid_data_tin['coher_bin'] = pd.cut(valid_data_tin['COHER'], bins=coherence_bins, labels=bin_centers)

# Calculate statistics per bin
bin_stats_tin = valid_data_tin.groupby('coher_bin', observed=True)['diff_tinitaly'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count'),
    ('median', 'median')
]).reset_index()

# Filter bins with at least 10 points
bin_stats_tin = bin_stats_tin[bin_stats_tin['count'] >= 10]

# Plot
axes[0].errorbar(bin_stats_tin['coher_bin'], bin_stats_tin['mean'],
                 yerr=bin_stats_tin['std'], fmt='o-', capsize=5,
                 markersize=8, linewidth=2, color='steelblue', label='Mean ± Std')
axes[0].plot(bin_stats_tin['coher_bin'], bin_stats_tin['median'],
             's--', markersize=6, linewidth=1.5, color='coral', label='Median')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)

axes[0].set_xlabel('Coherence (binned, width=0.05)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Height Residual (m)', fontsize=12, fontweight='bold')
axes[0].set_title('SAOCOM - TINItaly: Residuals vs Coherence (Binned)',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3, linestyle='--')

# Add sample size text
for idx, row in bin_stats_tin.iterrows():
    if idx % 3 == 0:  # Show every 3rd label to avoid crowding
        axes[0].text(row['coher_bin'], axes[0].get_ylim()[1] * 0.9,
                     f"n={int(row['count'])}", fontsize=8, ha='center', alpha=0.7)

# Copernicus residuals vs coherence (binned)
valid_data_cop = saocom_cleaned[['COHER', 'diff_copernicus']].dropna()
valid_data_cop['coher_bin'] = pd.cut(valid_data_cop['COHER'], bins=coherence_bins, labels=bin_centers)

bin_stats_cop = valid_data_cop.groupby('coher_bin', observed=True)['diff_copernicus'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count'),
    ('median', 'median')
]).reset_index()

bin_stats_cop = bin_stats_cop[bin_stats_cop['count'] >= 10]

axes[1].errorbar(bin_stats_cop['coher_bin'], bin_stats_cop['mean'],
                 yerr=bin_stats_cop['std'], fmt='o-', capsize=5,
                 markersize=8, linewidth=2, color='steelblue', label='Mean ± Std')
axes[1].plot(bin_stats_cop['coher_bin'], bin_stats_cop['median'],
             's--', markersize=6, linewidth=1.5, color='coral', label='Median')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)

axes[1].set_xlabel('Coherence (binned, width=0.05)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Height Residual (m)', fontsize=12, fontweight='bold')
axes[1].set_title('SAOCOM - Copernicus: Residuals vs Coherence (Binned)',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'residuals_vs_coherence.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"[OK] Saved binned coherence analysis")
"""

            cell['source'] = [new_code]
            print("[OK] Updated to use bins (width 0.05)\n")
            break

    # ==================== FIX 3: BEFORE/AFTER HISTOGRAM ====================
    print("=== FIX 3: Fixing before/after histogram colors and bins ===\n")

    # This fix needs to be applied to src/outlier_detection.py
    outlier_py_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/src/outlier_detection.py')
    with open(outlier_py_path, 'r', encoding='utf-8') as f:
        outlier_py = f.read()

    # Find and replace the histogram section
    old_histogram = """    ax2.hist(orig, bins=100, alpha=0.5, label=f'Before (n={orig.size:,})',
             color='gray')
    ax2.hist(cln, bins=50, alpha=1.0, label=f'After (n={cln.size:,})',
             color='#2E86AB')"""

    new_histogram = """    # Calculate appropriate bins to avoid empty bins
    # Use Freedman-Diaconis rule for bin width
    def calculate_bins(data, max_bins=100):
        if len(data) < 2:
            return 10
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return 10
        bin_width = 2 * iqr / (len(data) ** (1/3))
        data_range = data.max() - data.min()
        n_bins = int(np.ceil(data_range / bin_width))
        return min(n_bins, max_bins)

    bins_orig = calculate_bins(orig, max_bins=80)
    bins_cln = calculate_bins(cln, max_bins=60)

    # Plot with proper colors and alpha
    ax2.hist(orig, bins=bins_orig, alpha=0.6, label=f'Before (n={orig.size:,})',
             color='gray', edgecolor='black', linewidth=0.5)
    ax2.hist(cln, bins=bins_cln, alpha=0.8, label=f'After (n={cln.size:,})',
             color='#2E86AB', edgecolor='darkblue', linewidth=0.8)"""

    if old_histogram in outlier_py:
        outlier_py = outlier_py.replace(old_histogram, new_histogram)

        with open(outlier_py_path, 'w', encoding='utf-8') as f:
            f.write(outlier_py)

        print("[OK] Fixed before/after histogram in src/outlier_detection.py")
        print("    - Fixed 'After' color to be visible (alpha=0.8 with edge color)")
        print("    - Added smart bin calculation to avoid empty bins\n")

    # ==================== SAVE NOTEBOOK ====================
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("="*80)
    print("[SUCCESS] All fixes applied!")
    print(f"  - Fixed {maps_fixed} map visualizations with grids and hull boxes")
    print("  - Fixed residuals vs coherence to use bins (width 0.05)")
    print("  - Fixed before/after histogram colors and bins")
    print(f"  - Total cells: {len(nb['cells'])}")
    print("="*80)

if __name__ == '__main__':
    fix_all_maps()
