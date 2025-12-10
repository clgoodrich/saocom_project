"""
NEW PCA SECTION: Grid Cell Coverage Analysis
Add this to your notebook to replace/supplement the existing PCA section.

This analyzes the TRUE void zones - the 95% of grid cells with NO SAOCOM coverage.
"""

# ============================================================================
# Section X: PCA Analysis of Grid Cell Coverage vs Void Zones
# ============================================================================
#
# Goal: Identify terrain/land cover factors that predict whether a grid cell
#       has SAOCOM coverage or remains a void zone.
#
# Approach:
#   1. Sample ALL grid cells (both covered and void)
#   2. Extract terrain/land cover features for each cell
#   3. Run PCA to identify factors correlated with void zones
# ============================================================================

from src.void_coverage_pca import (
    create_grid_cell_dataframe,
    add_terrain_derivatives,
    add_land_cover_features,
    run_void_coverage_pca,
    plot_void_coverage_pca,
    export_void_coverage_summary
)

print("="*80)
print("GRID CELL COVERAGE ANALYSIS - PCA of Void Zone Factors")
print("="*80)

# ----------------------------------------------------------------------------
# Step 1: Create dataframe of ALL grid cells with their features
# ----------------------------------------------------------------------------
#
# We need:
# - coverage_grid (already computed earlier): boolean grid of cells with coverage
# - slope_grid: slope at each cell (from TINItaly or Copernicus)
# - aspect_grid: aspect at each cell
# - elevation_grid: elevation at each cell (TINItaly or Copernicus)
# - land_cover_grid: CORINE land cover at each cell
#
# IMPORTANT: These grids should all be aligned to the same 10m grid!
# ----------------------------------------------------------------------------

print("\n1. Preparing terrain and land cover grids...")

# Assuming you have these from earlier in the notebook:
# - tinitaly_10m (or copernicus_10m) - elevation DEM
# - slope_tin_grid, aspect_tin_grid - computed terrain derivatives
# - land_cover_grid - CORINE land cover resampled to 10m grid
# - coverage_grid - boolean grid from Section 12.9

# If slope/aspect grids don't exist yet, compute them:
if 'slope_tin_grid' not in locals():
    print("  Computing slope and aspect from TINItaly DEM...")
    from src.utils import compute_slope_aspect

    slope_tin_grid, aspect_tin_grid = compute_slope_aspect(
        tinitaly_10m,
        transform_10m,
        cell_size=10.0
    )
    print(f"    Slope range: {np.nanmin(slope_tin_grid):.1f}° to {np.nanmax(slope_tin_grid):.1f}°")
    print(f"    Aspect range: {np.nanmin(aspect_tin_grid):.1f}° to {np.nanmax(aspect_tin_grid):.1f}°")

# If land cover grid doesn't exist, load it:
if 'land_cover_grid' not in locals():
    print("  Loading land cover grid...")
    import rasterio
    with rasterio.open('data/ground_cover/land_cover_clipped.tif') as src:
        # Resample to match 10m grid if needed
        from rasterio.warp import reproject, Resampling

        land_cover_grid = np.zeros((grid_height, grid_width), dtype=np.int16)
        reproject(
            source=rasterio.band(src, 1),
            destination=land_cover_grid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_10m,
            dst_crs=crs,
            resampling=Resampling.nearest
        )
    print(f"    Land cover codes: {np.unique(land_cover_grid[land_cover_grid > 0])[:10]}...")

# ----------------------------------------------------------------------------
# Step 2: Create grid cell dataframe
# ----------------------------------------------------------------------------

print("\n2. Creating grid cell dataframe...")

# For large grids (>1M cells), use sampling to reduce memory
total_cells = grid_height * grid_width
sample_rate = 1.0 if total_cells < 1_000_000 else 0.2  # Sample 20% if >1M cells

df_grid = create_grid_cell_dataframe(
    coverage_grid=coverage_grid,
    slope_grid=slope_tin_grid,
    aspect_grid=aspect_tin_grid,
    elevation_grid=tinitaly_10m,  # or copernicus_10m
    land_cover_grid=land_cover_grid,
    transform=transform_10m,
    crs=crs,
    sample_rate=sample_rate,
    random_state=42
)

# Add derived features
print("\n3. Adding derived terrain features...")
df_grid = add_terrain_derivatives(df_grid, slope_tin_grid, aspect_tin_grid, tinitaly_10m)

# Add land cover one-hot encoding
print("\n4. Adding land cover features...")
# Optionally load CORINE lookup table for better names
try:
    import dbfread
    dbf_path = 'data/ground_cover/CLC2018_CLC2018_V2018_20.tif.vat.dbf'
    dbf = dbfread.DBF(dbf_path, encoding='latin1')
    land_cover_lookup = {rec['VALUE']: rec['LABEL3'] for rec in dbf}
except:
    print("  (CORINE lookup table not found, using codes)")
    land_cover_lookup = None

df_grid = add_land_cover_features(df_grid, land_cover_lookup)

print(f"\nDataframe shape: {df_grid.shape}")
print(f"Columns: {df_grid.columns.tolist()}")

# ----------------------------------------------------------------------------
# Step 3: Select features for PCA
# ----------------------------------------------------------------------------

print("\n5. Selecting PCA features...")

# Continuous features
continuous_features = ['slope', 'aspect', 'elevation']

# Land cover one-hot features (columns starting with 'lc_')
lc_features = [col for col in df_grid.columns if col.startswith('lc_')]

# Combine
feature_columns = continuous_features + lc_features

print(f"  Total features: {len(feature_columns)}")
print(f"    Continuous: {continuous_features}")
print(f"    Land cover: {lc_features}")

# ----------------------------------------------------------------------------
# Step 4: Run PCA
# ----------------------------------------------------------------------------

print("\n6. Running PCA...")

pca_model, X_pca, loadings, comparison_df = run_void_coverage_pca(
    df=df_grid,
    feature_columns=feature_columns,
    n_components=5,
    random_state=42
)

# ----------------------------------------------------------------------------
# Step 5: Visualize results
# ----------------------------------------------------------------------------

print("\n7. Creating visualizations...")

# First, create coverage vs slope analysis
print("\n7a. Coverage vs Slope Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

covered = df_grid[df_grid['has_coverage'] == True]
void = df_grid[df_grid['has_coverage'] == False]

# Panel 1: Coverage rate vs slope bins
ax = axes[0, 0]
slope_bins = np.arange(0, df_grid['slope'].max() + 5, 5)
df_grid['slope_bin'] = pd.cut(df_grid['slope'], bins=slope_bins)

coverage_by_slope = df_grid.groupby('slope_bin', observed=True).agg({
    'has_coverage': ['sum', 'count', 'mean']
})
coverage_by_slope.columns = ['covered', 'total', 'coverage_rate']
coverage_by_slope['coverage_pct'] = coverage_by_slope['coverage_rate'] * 100

bin_centers = [interval.mid for interval in coverage_by_slope.index]

ax.plot(bin_centers, coverage_by_slope['coverage_pct'],
        marker='o', linewidth=2, markersize=8, color='#3498DB')
ax.fill_between(bin_centers, 0, coverage_by_slope['coverage_pct'],
                 alpha=0.3, color='#3498DB')
ax.set_xlabel('Slope (degrees)', fontsize=13, fontweight='bold')
ax.set_ylabel('Coverage Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('SAOCOM Coverage Rate vs Terrain Slope', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

# Panel 2: Histogram - slope distribution
ax = axes[0, 1]
bins = np.arange(0, min(df_grid['slope'].max(), 60) + 2, 2)
ax.hist(covered['slope'], bins=bins, alpha=0.6, color='blue',
        label=f'Covered (n={len(covered):,})', density=True, edgecolor='black', linewidth=0.5)
ax.hist(void['slope'], bins=bins, alpha=0.6, color='red',
        label=f'Void (n={len(void):,})', density=True, edgecolor='black', linewidth=0.5)

ax.axvline(covered['slope'].mean(), color='blue', linestyle='--',
           linewidth=2, label=f'Covered mean: {covered["slope"].mean():.1f}°')
ax.axvline(void['slope'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Void mean: {void["slope"].mean():.1f}°')

ax.set_xlabel('Slope (degrees)', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Slope Distribution: Covered vs Void Cells', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# Panel 3: Cumulative coverage by slope
ax = axes[1, 0]
df_sorted = df_grid.sort_values('slope')
cumulative_covered = df_sorted['has_coverage'].cumsum()
cumulative_total = np.arange(1, len(df_sorted) + 1)
cumulative_pct = 100 * cumulative_covered / cumulative_total

sample_step = max(1, len(df_sorted) // 1000)

ax.plot(df_sorted['slope'].iloc[::sample_step],
        cumulative_pct.iloc[::sample_step],
        linewidth=2, color='#E67E22')
ax.axhline(df_grid['has_coverage'].mean() * 100, color='gray',
           linestyle='--', linewidth=2, label=f'Overall: {df_grid["has_coverage"].mean()*100:.1f}%')

ax.set_xlabel('Slope (degrees)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Coverage Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Cumulative Coverage Rate by Slope', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

# Panel 4: Box plot
ax = axes[1, 1]
data_to_plot = [covered['slope'].values, void['slope'].values]
bp = ax.boxplot(data_to_plot, labels=['Covered', 'Void'],
                patch_artist=True, widths=0.6,
                boxprops=dict(linewidth=2),
                medianprops=dict(color='darkred', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

colors = ['#3498DB', '#E74C3C']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Slope (degrees)', fontsize=13, fontweight='bold')
ax.set_title('Slope Distribution by Coverage Status', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

means = [covered['slope'].mean(), void['slope'].mean()]
ax.plot([1, 2], means, 'D', color='green', markersize=10,
        label='Mean', zorder=3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('images/coverage_vs_slope_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: images/coverage_vs_slope_analysis.png")
plt.show()

# Print statistics
print(f"\n{'='*60}")
print("COVERAGE vs SLOPE STATISTICS")
print(f"{'='*60}")
print(f"\nCovered cells:")
print(f"  Mean slope: {covered['slope'].mean():.2f}°")
print(f"  Median slope: {covered['slope'].median():.2f}°")
print(f"  Std dev: {covered['slope'].std():.2f}°")
print(f"\nVoid cells:")
print(f"  Mean slope: {void['slope'].mean():.2f}°")
print(f"  Median slope: {void['slope'].median():.2f}°")
print(f"  Std dev: {void['slope'].std():.2f}°")
print(f"\nDifference:")
diff_mean = void['slope'].mean() - covered['slope'].mean()
diff_pct = 100 * diff_mean / covered['slope'].mean()
print(f"  Mean difference: {diff_mean:+.2f}° ({diff_pct:+.1f}%)")

from scipy import stats as sp_stats
t_stat, p_val = sp_stats.ttest_ind(void['slope'], covered['slope'])
print(f"  T-statistic: {t_stat:.3f}")
print(f"  P-value: {p_val:.2e}")

print(f"\nCoverage rate by slope category:")
slope_categories = [
    (0, 5, 'Flat'),
    (5, 15, 'Gentle'),
    (15, 30, 'Moderate'),
    (30, 90, 'Steep')
]

for min_slope, max_slope, label in slope_categories:
    mask = (df_grid['slope'] >= min_slope) & (df_grid['slope'] < max_slope)
    if mask.sum() > 0:
        coverage = df_grid[mask]['has_coverage'].mean() * 100
        n_cells = mask.sum()
        n_covered = df_grid[mask]['has_coverage'].sum()
        print(f"  {label:12s} ({min_slope:2d}-{max_slope:2d}°): {coverage:5.2f}% " +
              f"({n_covered:,}/{n_cells:,} cells)")
print(f"{'='*60}\n")

# Now create PCA visualizations
print("\n7b. PCA Visualizations...")

plot_void_coverage_pca(
    df=df_grid,
    loadings=loadings,
    save_dir='images'
)

# ----------------------------------------------------------------------------
# Step 6: Export summary
# ----------------------------------------------------------------------------

print("\n8. Exporting results...")

export_void_coverage_summary(
    df=df_grid,
    comp_df=comparison_df,
    save_path='results/void_coverage_pca_summary.csv'
)

# ----------------------------------------------------------------------------
# Step 7: Key Findings Summary
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("KEY FINDINGS: VOID ZONE FACTORS")
print("="*80)

# Coverage statistics
coverage_pct = 100 * df_grid['has_coverage'].mean()
void_pct = 100 - coverage_pct

print(f"\nCoverage Statistics:")
print(f"  Grid cells analyzed: {len(df_grid):,}")
print(f"  Covered cells: {df_grid['has_coverage'].sum():,} ({coverage_pct:.1f}%)")
print(f"  Void cells: {(~df_grid['has_coverage']).sum():,} ({void_pct:.1f}%)")

# Top discriminating factors
print(f"\nTop 5 Factors Predicting Void Zones:")
for i, row in comparison_df.head(5).iterrows():
    direction = "higher" if row['pct_difference'] > 0 else "lower"
    print(f"  {i+1}. {row['feature']:25s}: {abs(row['pct_difference']):6.1f}% {direction} in voids (p={row['p_value']:.2e})")

# Principal component interpretation
print(f"\nPrincipal Component Interpretation:")
print(f"  PC1 explains {pca_model.explained_variance_ratio_[0]*100:.1f}% of variance")
top_pc1_features = loadings['PC1'].abs().nlargest(3)
print(f"  Top PC1 features: {', '.join(top_pc1_features.index.tolist())}")

covered = df_grid[df_grid['has_coverage'] == True]
void = df_grid[df_grid['has_coverage'] == False]
pc1_diff = void['PC1'].mean() - covered['PC1'].mean()
print(f"  PC1 difference (void - covered): {pc1_diff:+.3f}")

print("\n" + "="*80)
print("Analysis complete! Check images/ and results/ folders for outputs.")
print("="*80)
