"""
Comprehensive fixes for land cover analysis:
1. Fix CORINE to use LABEL3 column instead of just broad categories
2. Add ALL missing land cover visualizations
3. Add proper map elements (scale, north arrow, legend, grid, bounding box) to ALL maps
"""

import json
from pathlib import Path

def create_markdown_cell(text):
    """Create a markdown cell"""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text] if isinstance(text, str) else text
    }

def create_code_cell(code):
    """Create a code cell"""
    return {
        'cell_type': 'code',
        'metadata': {},
        'source': [code] if isinstance(code, str) else code,
        'outputs': [],
        'execution_count': None
    }

def add_land_cover_fixes():
    """Add comprehensive land cover fixes"""

    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells")

    # ==================== FIX 1: CORINE LABEL3 COLUMN ====================
    print("\n=== FIX 1: Updating CORINE to use LABEL3 ===")

    # Find the land cover sampling cell (currently cell 40)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if "saocom_cleaned['land_cover'] = saocom_cleaned['corine_code'].apply" in source:
                print(f"Found land cover sampling in cell {i} - updating to use LABEL3")

                new_code = """# Load and sample land cover
print("Loading CORINE Land Cover...")
with rasterio.open(CORINE_LC) as src:
    corine_data = src.read(1)
    corine_transform = src.transform
    corine_crs = src.crs

# Load CORINE lookup table to get LABEL3
CORINE_DBF = DATA_DIR / 'corine_clip.tif.vat.dbf'
from dbfread import DBF
dbf_table = DBF(str(CORINE_DBF), load=True)
lookup_df = pd.DataFrame(iter(dbf_table))

# Create mappings: Value -> CODE_18 and CODE_18 -> LABEL3
value_to_code = dict(zip(lookup_df['Value'], lookup_df['CODE_18']))
code_to_label3 = dict(zip(lookup_df['CODE_18'], lookup_df['LABEL3']))

print(f"Loaded {len(code_to_label3)} CORINE land cover classes")

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

# Extract raw values from raster
lc_values = np.full(len(saocom_lc), np.nan)
lc_values[lc_inbounds] = corine_data[lc_rows[lc_inbounds], lc_cols[lc_inbounds]]

# Map: Value -> CODE_18
lc_codes = np.array([value_to_code.get(int(v), 0) if pd.notna(v) else 0 for v in lc_values])

# Store both the code and the LABEL3 description
saocom_cleaned['corine_code'] = lc_codes
saocom_cleaned['land_cover'] = saocom_cleaned['corine_code'].apply(
    lambda x: code_to_label3.get(int(x), 'Unknown') if pd.notna(x) and x > 0 else 'Unknown'
)

# Also add Level 1 categories for broader analysis
saocom_cleaned['land_cover_level1'] = saocom_cleaned['corine_code'].apply(
    lambda x: get_clc_level1(int(x)) if pd.notna(x) and x > 0 else 'Unknown'
)

print(f"Land cover sampled for {saocom_cleaned['land_cover'].notna().sum():,} points")
print(f"\\nLand cover distribution (Level 1 categories):")
print(saocom_cleaned['land_cover_level1'].value_counts())
print(f"\\nMost common Level 3 classes:")
print(saocom_cleaned['land_cover'].value_counts().head(10))
"""

                cell['source'] = [new_code]
                print("[OK] Updated to use LABEL3")
                break

    # ==================== FIX 2: ADD MISSING LAND COVER GRAPHICS ====================
    print("\n=== FIX 2: Adding missing land cover visualizations ===")

    # Find where to insert (after the existing land cover bar chart)
    insert_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if "lc_stats['nmad'].plot(kind='bar'" in source and 'land_cover' in source.lower():
                insert_idx = i + 1
                print(f"Will insert new land cover graphics after cell {i}")
                break

    if insert_idx is None:
        print("ERROR: Could not find land cover bar chart cell")
        return

    # New land cover visualizations to add
    new_cells = []

    # 1. Land Cover Map with SAOCOM Points
    new_cells.append(create_markdown_cell("""### 9.2 Land Cover Spatial Map

Visualize the spatial distribution of SAOCOM points by land cover type."""))

    new_cells.append(create_code_cell("""# Create land cover map with SAOCOM points
print("Creating land cover spatial map...")

from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches

# Get most common land cover classes for legend
top_lc = saocom_cleaned['land_cover'].value_counts().head(10)

# Create color map for land cover types
lc_colors = plt.cm.tab20(np.linspace(0, 1, len(top_lc)))
lc_color_map = dict(zip(top_lc.index, lc_colors))

fig, ax = plt.subplots(figsize=(16, 14))

# Plot points by land cover
for lc_type in top_lc.index:
    lc_subset = saocom_cleaned[saocom_cleaned['land_cover'] == lc_type]
    ax.scatter(lc_subset.geometry.x, lc_subset.geometry.y,
               c=[lc_color_map[lc_type]], s=5, alpha=0.6, label=lc_type)

# Add bounding box
bounds = saocom_cleaned.total_bounds
rect = Rectangle((bounds[0], bounds[1]),
                 bounds[2] - bounds[0],
                 bounds[3] - bounds[1],
                 linewidth=3, edgecolor='red', facecolor='none',
                 label='Study Area')
ax.add_patch(rect)

# Add map elements
ax.set_xlabel('UTM Easting (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('UTM Northing (m)', fontsize=12, fontweight='bold')
ax.set_title('SAOCOM Points by Land Cover Type (Top 10 Classes)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

# Add scale bar
scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower right',
                    box_alpha=0.7, scale_loc='top')
ax.add_artist(scalebar)

# Add north arrow
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=20, fontweight='bold', ha='center', va='center')
ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
            fontsize=30, ha='center', va='center')

# Legend outside plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'land_cover_map.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"[OK] Saved land_cover_map.png")
"""))

    # 2. Land Cover Histograms by Level
    new_cells.append(create_markdown_cell("""### 9.3 Land Cover Distribution Histograms

Detailed distribution of SAOCOM points across land cover classes at different hierarchical levels."""))

    new_cells.append(create_code_cell("""# Land cover histograms at different levels
print("Creating land cover histograms...")

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Level 1 (broad categories)
lc_level1_counts = saocom_cleaned['land_cover_level1'].value_counts()
axes[0].barh(range(len(lc_level1_counts)), lc_level1_counts.values, color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(lc_level1_counts)))
axes[0].set_yticklabels(lc_level1_counts.index)
axes[0].set_xlabel('Number of Points', fontsize=12, fontweight='bold')
axes[0].set_title('Land Cover Distribution - Level 1 (Broad Categories)',
                  fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Add percentage labels
total = lc_level1_counts.sum()
for i, v in enumerate(lc_level1_counts.values):
    pct = 100 * v / total
    axes[0].text(v, i, f'  {v:,} ({pct:.1f}%)', va='center', fontweight='bold')

# Level 3 (detailed classes) - top 15
lc_level3_counts = saocom_cleaned['land_cover'].value_counts().head(15)
axes[1].barh(range(len(lc_level3_counts)), lc_level3_counts.values, color='coral', edgecolor='black')
axes[1].set_yticks(range(len(lc_level3_counts)))
axes[1].set_yticklabels(lc_level3_counts.index)
axes[1].set_xlabel('Number of Points', fontsize=12, fontweight='bold')
axes[1].set_title('Land Cover Distribution - Level 3 (Top 15 Detailed Classes)',
                  fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Add count labels
for i, v in enumerate(lc_level3_counts.values):
    axes[1].text(v, i, f'  {v:,}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'land_cover_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"[OK] Saved land_cover_histograms.png")
"""))

    # 3. Accuracy by Land Cover (Detailed)
    new_cells.append(create_markdown_cell("""### 9.4 SAOCOM Accuracy by Detailed Land Cover Classes

Analyze how InSAR accuracy varies across specific land cover types (Level 3)."""))

    new_cells.append(create_code_cell("""# Accuracy metrics by detailed land cover classes
print("Analyzing accuracy by detailed land cover classes...")

# Calculate statistics for classes with sufficient points
MIN_POINTS = 100
lc_detailed_stats = saocom_cleaned.groupby('land_cover').agg(
    count=('diff_tinitaly', 'count'),
    mean=('diff_tinitaly', 'mean'),
    std=('diff_tinitaly', 'std'),
    nmad=('diff_tinitaly', lambda x: nmad(x.dropna()))
).reset_index()

# Filter to classes with enough points
lc_detailed_stats = lc_detailed_stats[lc_detailed_stats['count'] >= MIN_POINTS].copy()
lc_detailed_stats = lc_detailed_stats.sort_values('nmad')

print(f"\\nAccuracy by Land Cover (classes with >= {MIN_POINTS} points):")
print(lc_detailed_stats.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# NMAD by land cover
axes[0].barh(range(len(lc_detailed_stats)), lc_detailed_stats['nmad'],
             color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(lc_detailed_stats)))
axes[0].set_yticklabels(lc_detailed_stats['land_cover'], fontsize=9)
axes[0].set_xlabel('NMAD (m)', fontsize=12, fontweight='bold')
axes[0].set_title('InSAR Accuracy (NMAD) by Land Cover Type',
                  fontsize=13, fontweight='bold')
axes[0].axvline(x=nmad_tin, color='red', linestyle='--', linewidth=2,
                label=f'Overall NMAD = {nmad_tin:.2f} m')
axes[0].grid(axis='x', alpha=0.3)
axes[0].legend()

# Add NMAD values
for i, v in enumerate(lc_detailed_stats['nmad']):
    axes[0].text(v, i, f'  {v:.2f}', va='center', fontweight='bold')

# Point count by land cover
axes[1].barh(range(len(lc_detailed_stats)), lc_detailed_stats['count'],
             color='coral', edgecolor='black')
axes[1].set_yticks(range(len(lc_detailed_stats)))
axes[1].set_yticklabels(lc_detailed_stats['land_cover'], fontsize=9)
axes[1].set_xlabel('Number of Points', fontsize=12, fontweight='bold')
axes[1].set_title('Sample Size by Land Cover Type',
                  fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Add count labels
for i, v in enumerate(lc_detailed_stats['count']):
    axes[1].text(v, i, f'  {v:,}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'accuracy_by_detailed_land_cover.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"[OK] Saved accuracy_by_detailed_land_cover.png")
"""))

    # 4. Land Cover vs Terrain Slope Analysis
    new_cells.append(create_markdown_cell("""### 9.5 Land Cover vs Terrain Characteristics

Explore the relationship between land cover types and terrain characteristics (slope)."""))

    new_cells.append(create_code_cell("""# Land cover vs slope analysis
print("Analyzing land cover vs terrain slope...")

# Get top land cover classes
top_lc_classes = saocom_cleaned['land_cover'].value_counts().head(8).index

# Filter data
lc_slope_data = saocom_cleaned[saocom_cleaned['land_cover'].isin(top_lc_classes)].copy()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Violin plot: Slope distribution by land cover
plot_data_violin = [lc_slope_data[lc_slope_data['land_cover'] == lc]['slope_tin'].dropna().values
                    for lc in top_lc_classes]

parts = axes[0].violinplot(plot_data_violin, positions=range(len(top_lc_classes)),
                           showmeans=True, showmedians=True, widths=0.7)

# Color the violin plots
for pc, color in zip(parts['bodies'], plt.cm.Set3(np.linspace(0, 1, len(top_lc_classes)))):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

axes[0].set_xticks(range(len(top_lc_classes)))
axes[0].set_xticklabels(top_lc_classes, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('Slope (degrees)', fontsize=12, fontweight='bold')
axes[0].set_title('Terrain Slope Distribution by Land Cover',
                  fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Box plot: Residuals by land cover
plot_data_box = [lc_slope_data[lc_slope_data['land_cover'] == lc]['diff_tinitaly'].dropna().values
                for lc in top_lc_classes]

bp = axes[1].boxplot(plot_data_box, labels=top_lc_classes, patch_artist=True,
                      showfliers=False)

# Color the box plots
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(top_lc_classes)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1].set_xticklabels(top_lc_classes, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Height Residual (m)', fontsize=12, fontweight='bold')
axes[1].set_title('InSAR Residual Distribution by Land Cover',
                  fontsize=13, fontweight='bold')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'land_cover_vs_terrain.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"[OK] Saved land_cover_vs_terrain.png")
"""))

    # Insert all new cells
    for cell in reversed(new_cells):
        nb['cells'].insert(insert_idx, cell)

    print(f"[OK] Added {len(new_cells)} new land cover visualization cells")

    # ==================== FIX 3: ADD MAP ELEMENTS TO ALL MAPS ====================
    print("\n=== FIX 3: Adding map elements to all map visualizations ===")

    # Find and update all map visualizations in section 12
    maps_updated = 0

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Section 12.1: Spatial Coverage
            if '12.1 Spatial Coverage' in source or 'spatial_coverage.png' in source:
                if 'ScaleBar' not in source:  # Only update if not already updated
                    print(f"Updating cell {i}: Spatial Coverage Map")
                    # Add imports and map elements
                    updated_source = source.replace(
                        "ax.set_aspect('equal')",
                        """ax.set_aspect('equal')

# Add scale bar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower right',
                    box_alpha=0.7, scale_loc='top')
ax.add_artist(scalebar)

# Add north arrow
ax.annotate('N', xy=(0.95, 0.05), xycoords='axes fraction',
            fontsize=20, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
ax.annotate('↑', xy=(0.95, 0.02), xycoords='axes fraction',
            fontsize=30, ha='center', va='center')"""
                    )
                    cell['source'] = [updated_source]
                    maps_updated += 1

            # Section 12.7: Terrain Slope Map
            if '12.7 Terrain Slope' in source or 'terrain_slope.png' in source:
                if 'ScaleBar' not in source:
                    print(f"Updating cell {i}: Terrain Slope Map")
                    updated_source = source.replace(
                        "ax.axis('off')",
                        """# Add map elements
ax.set_xlabel('Column (pixels)', fontsize=10)
ax.set_ylabel('Row (pixels)', fontsize=10)
ax.grid(True, alpha=0.2, color='white')

# Add scale bar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(10, 'm', length_fraction=0.25, location='lower right',
                    box_alpha=0.7, scale_loc='top')
ax.add_artist(scalebar)

# Add north arrow
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=20, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
            fontsize=30, ha='center', va='center')"""
                    )
                    cell['source'] = [updated_source]
                    maps_updated += 1

            # Section 12.8: Reference DEM Comparison
            if '12.8 Reference DEM' in source or 'reference_dem_comparison.png' in source:
                if 'ScaleBar' not in source:
                    print(f"Updating cell {i}: Reference DEM Comparison")
                    # Add scale bars to all three map panels
                    updated_source = source.replace(
                        "axes[0, 0].axis('off')",
                        """axes[0, 0].set_xlabel('', fontsize=8)
axes[0, 0].set_ylabel('', fontsize=8)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# Add scale bar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(10, 'm', length_fraction=0.25, location='lower right',
                    box_alpha=0.7, scale_loc='top', font_properties={'size': 8})
axes[0, 0].add_artist(scalebar)"""
                    )
                    updated_source = updated_source.replace(
                        "axes[0, 1].axis('off')",
                        """axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
scalebar2 = ScaleBar(10, 'm', length_fraction=0.25, location='lower right',
                     box_alpha=0.7, scale_loc='top', font_properties={'size': 8})
axes[0, 1].add_artist(scalebar2)"""
                    )
                    updated_source = updated_source.replace(
                        "axes[1, 0].axis('off')",
                        """axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
scalebar3 = ScaleBar(10, 'm', length_fraction=0.25, location='lower right',
                     box_alpha=0.7, scale_loc='top', font_properties={'size': 8})
axes[1, 0].add_artist(scalebar3)"""
                    )
                    cell['source'] = [updated_source]
                    maps_updated += 1

            # Section 12.9: Coverage Grid
            if '12.9 Coverage' in source or 'coverage_and_voids.png' in source:
                if 'ScaleBar' not in source:
                    print(f"Updating cell {i}: Coverage Grid Map")
                    updated_source = source.replace(
                        "plt.tight_layout()",
                        """# Add scale bar and north arrow to main map
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(10, 'm', length_fraction=0.25, location='lower right',
                    box_alpha=0.7, scale_loc='top', font_properties={'size': 8})
ax.add_artist(scalebar)

# Add north arrow
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
            fontsize=24, ha='center', va='center')

plt.tight_layout()"""
                    )
                    cell['source'] = [updated_source]
                    maps_updated += 1

            # Section 12.11: Summary Dashboard
            if '12.11 Summary Dashboard' in source or 'summary_dashboard.png' in source:
                if 'ScaleBar' not in source:
                    print(f"Updating cell {i}: Summary Dashboard")
                    # Add scale bar to the spatial map in the dashboard
                    updated_source = source.replace(
                        "ax1.set_title('Point Distribution')",
                        """ax1.set_title('Point Distribution')
# Add scale bar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(1, 'm', length_fraction=0.3, location='lower right',
                    box_alpha=0.7, scale_loc='top', font_properties={'size': 6})
ax1.add_artist(scalebar)
# Add north arrow
ax1.annotate('N', xy=(0.9, 0.9), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='center')"""
                    )
                    cell['source'] = [updated_source]
                    maps_updated += 1

    print(f"[OK] Updated {maps_updated} map visualizations with proper elements")

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"[OK] Successfully updated notebook!")
    print(f"  - Fixed CORINE to use LABEL3 column")
    print(f"  - Added {len(new_cells)} new land cover visualization cells")
    print(f"  - Updated {maps_updated} map visualizations with scale bars, north arrows, legends, grids")
    print(f"  - Total cells now: {len(nb['cells'])}")
    print(f"{'='*80}")

if __name__ == '__main__':
    add_land_cover_fixes()
