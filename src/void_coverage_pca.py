"""
PCA Analysis of Grid Cell Coverage vs Void Zones

This module analyzes factors that predict whether a grid cell has SAOCOM coverage or not.
Unlike quality-based void analysis, this examines the ~95% of cells with NO coverage.

Approach:
1. Sample terrain/land cover factors at ALL grid cells (covered + void)
2. Label: coverage_grid (True = has coverage, False = void)
3. Run PCA to identify which factors predict void zones

Key factors analyzed:
- Slope, aspect, curvature (terrain complexity)
- Elevation
- Land cover type
- Radar geometry (incidence angle, shadow/layover potential)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import rasterio
from rasterio.transform import rowcol


def create_grid_cell_dataframe(
    coverage_grid,
    slope_grid,
    aspect_grid,
    elevation_grid,
    land_cover_grid,
    transform,
    crs,
    sample_rate=1.0,
    random_state=42
):
    """
    Create a DataFrame with one row per grid cell (or sampled cells).

    Parameters
    ----------
    coverage_grid : np.ndarray (bool)
        Boolean grid: True = has SAOCOM coverage, False = void
    slope_grid : np.ndarray (float)
        Slope in degrees
    aspect_grid : np.ndarray (float)
        Aspect in degrees
    elevation_grid : np.ndarray (float)
        Elevation in meters
    land_cover_grid : np.ndarray (int)
        Land cover classification codes
    transform : affine.Affine
        Geotransform for the grid
    crs : CRS
        Coordinate reference system
    sample_rate : float, default=1.0
        Fraction of cells to sample (1.0 = all cells, 0.1 = 10% sample)
        Use <1.0 for large grids to reduce memory
    random_state : int, default=42
        Random seed for sampling

    Returns
    -------
    pd.DataFrame
        Columns: has_coverage, slope, aspect, elevation, land_cover, row, col
    """
    grid_height, grid_width = coverage_grid.shape

    # Create row, col indices for all cells
    rows, cols = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing='ij')

    # Flatten all grids
    data = {
        'row': rows.ravel(),
        'col': cols.ravel(),
        'has_coverage': coverage_grid.ravel(),
        'slope': slope_grid.ravel(),
        'aspect': aspect_grid.ravel(),
        'elevation': elevation_grid.ravel(),
        'land_cover': land_cover_grid.ravel()
    }

    df = pd.DataFrame(data)

    # Remove NaN cells (outside ROI)
    valid_mask = (
        np.isfinite(df['slope']) &
        np.isfinite(df['aspect']) &
        np.isfinite(df['elevation']) &
        (df['land_cover'] > 0)  # Assume 0 = nodata
    )
    df = df[valid_mask].copy()

    # Sample if requested
    if sample_rate < 1.0:
        n_sample = int(len(df) * sample_rate)
        df = df.sample(n=n_sample, random_state=random_state).copy()

    print(f"Grid cell dataframe created:")
    print(f"  Total cells: {len(df):,}")
    print(f"  Covered cells: {df['has_coverage'].sum():,} ({100*df['has_coverage'].mean():.1f}%)")
    print(f"  Void cells: {(~df['has_coverage']).sum():,} ({100*(~df['has_coverage']).mean():.1f}%)")

    return df


def add_terrain_derivatives(df, slope_grid, aspect_grid, elevation_grid):
    """
    Add derived terrain features to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe with 'row', 'col' columns
    slope_grid : np.ndarray
        Slope grid
    aspect_grid : np.ndarray
        Aspect grid
    elevation_grid : np.ndarray
        Elevation grid

    Returns
    -------
    pd.DataFrame
        Input df with additional columns added in-place
    """
    # Slope categories
    df['slope_category'] = pd.cut(
        df['slope'],
        bins=[0, 5, 15, 30, 90],
        labels=['flat', 'gentle', 'moderate', 'steep']
    )

    # Aspect categories (N, E, S, W, flat)
    df['aspect_category'] = pd.cut(
        df['aspect'],
        bins=[-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361],
        labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
    )

    # Elevation bins
    df['elevation_bin'] = pd.qcut(
        df['elevation'],
        q=5,
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )

    return df


def add_land_cover_features(df, land_cover_lookup=None):
    """
    Add land cover one-hot encoding features.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe with 'land_cover' column
    land_cover_lookup : dict, optional
        Mapping from land cover codes to names

    Returns
    -------
    pd.DataFrame
        Input df with land cover one-hot columns added
    """
    # Get top N land cover types
    top_lc = df['land_cover'].value_counts().head(5).index

    for lc_code in top_lc:
        lc_name = land_cover_lookup.get(lc_code, f"LC_{lc_code}") if land_cover_lookup else f"LC_{lc_code}"
        col_name = f"lc_{lc_name.replace(' ', '_')}"
        df[col_name] = (df['land_cover'] == lc_code).astype(int)

    return df


def run_void_coverage_pca(
    df,
    feature_columns,
    n_components=5,
    random_state=42
):
    """
    Run PCA on grid cells to identify factors predicting void zones.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe with features and 'has_coverage' label
    feature_columns : list of str
        Column names to use as PCA features
    n_components : int, default=5
        Number of principal components
    random_state : int, default=42
        Random seed

    Returns
    -------
    tuple
        (pca_model, X_pca, feature_importance_df)
    """
    print(f"\n{'='*60}")
    print("VOID COVERAGE PCA ANALYSIS")
    print(f"{'='*60}")
    print(f"Analyzing {len(df):,} grid cells")
    print(f"Features: {len(feature_columns)}")
    print(f"Components: {n_components}\n")

    # Prepare feature matrix
    X = df[feature_columns].copy()

    # Handle any remaining NaNs
    X = X.fillna(X.median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Add PC scores to dataframe
    for i in range(n_components):
        df[f'PC{i+1}'] = X_pca[:, i]

    # Analyze component importance
    print("Variance Explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    print(f"  Cumulative: {pca.explained_variance_ratio_.sum()*100:.2f}%\n")

    # Feature loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_columns
    )

    print("Top Features per Component:")
    for i in range(min(3, n_components)):
        pc_col = f'PC{i+1}'
        top_pos = loadings[pc_col].nlargest(3)
        top_neg = loadings[pc_col].nsmallest(3)
        print(f"\n{pc_col}:")
        print("  Positive:")
        for feat, val in top_pos.items():
            print(f"    {feat:30s}: +{val:.3f}")
        print("  Negative:")
        for feat, val in top_neg.items():
            print(f"    {feat:30s}: {val:.3f}")

    # Compare void vs covered cells
    print(f"\n{'='*60}")
    print("VOID vs COVERED COMPARISON (PC Scores)")
    print(f"{'='*60}")

    covered_df = df[df['has_coverage'] == True]
    void_df = df[df['has_coverage'] == False]

    for i in range(min(3, n_components)):
        pc_col = f'PC{i+1}'
        covered_mean = covered_df[pc_col].mean()
        void_mean = void_df[pc_col].mean()
        diff = void_mean - covered_mean

        # T-test
        t_stat, p_val = stats.ttest_ind(void_df[pc_col], covered_df[pc_col])

        print(f"\n{pc_col}:")
        print(f"  Covered cells:  {covered_mean:+.3f}")
        print(f"  Void cells:     {void_mean:+.3f}")
        print(f"  Difference:     {diff:+.3f}")
        print(f"  T-statistic:    {t_stat:.3f}")
        print(f"  P-value:        {p_val:.2e}")

    # Feature-level comparison
    print(f"\n{'='*60}")
    print("VOID vs COVERED COMPARISON (Raw Features)")
    print(f"{'='*60}")

    feature_comparison = []
    for feat in feature_columns:
        covered_mean = covered_df[feat].mean()
        void_mean = void_df[feat].mean()
        diff = void_mean - covered_mean
        pct_diff = 100 * diff / (covered_mean + 1e-10)

        t_stat, p_val = stats.ttest_ind(void_df[feat], covered_df[feat])

        feature_comparison.append({
            'feature': feat,
            'covered_mean': covered_mean,
            'void_mean': void_mean,
            'difference': diff,
            'pct_difference': pct_diff,
            't_statistic': t_stat,
            'p_value': p_val
        })

    comp_df = pd.DataFrame(feature_comparison)
    comp_df = comp_df.sort_values('pct_difference', key=abs, ascending=False)

    print("\nTop 10 Discriminating Features (by % difference):")
    for idx, row in comp_df.head(10).iterrows():
        print(f"\n{row['feature']:30s}")
        print(f"  Covered: {row['covered_mean']:10.3f}")
        print(f"  Void:    {row['void_mean']:10.3f}")
        print(f"  Diff:    {row['pct_difference']:+10.1f}% (p={row['p_value']:.2e})")

    return pca, X_pca, loadings, comp_df


def plot_coverage_vs_slope(df, save_dir='images'):
    """
    Create visualization showing point coverage against slope.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe with 'has_coverage' and 'slope' columns
    save_dir : str, default='images'
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coverage rate vs slope bins
    ax = axes[0, 0]
    slope_bins = np.arange(0, df['slope'].max() + 5, 5)
    df['slope_bin'] = pd.cut(df['slope'], bins=slope_bins)

    coverage_by_slope = df.groupby('slope_bin', observed=True).agg({
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

    # 2. Histogram: slope distribution for covered vs void
    ax = axes[0, 1]
    covered = df[df['has_coverage'] == True]
    void = df[df['has_coverage'] == False]

    bins = np.arange(0, min(df['slope'].max(), 60) + 2, 2)
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

    # 3. Cumulative coverage by slope
    ax = axes[1, 0]

    # Sort by slope and calculate cumulative coverage
    df_sorted = df.sort_values('slope')
    cumulative_covered = df_sorted['has_coverage'].cumsum()
    cumulative_total = np.arange(1, len(df_sorted) + 1)
    cumulative_pct = 100 * cumulative_covered / cumulative_total

    # Sample for plotting (every Nth point to avoid overplotting)
    sample_step = max(1, len(df_sorted) // 1000)

    ax.plot(df_sorted['slope'].iloc[::sample_step],
            cumulative_pct.iloc[::sample_step],
            linewidth=2, color='#E67E22')
    ax.axhline(df['has_coverage'].mean() * 100, color='gray',
               linestyle='--', linewidth=2, label=f'Overall: {df["has_coverage"].mean()*100:.1f}%')

    ax.set_xlabel('Slope (degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Coverage Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Coverage Rate by Slope', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # 4. Box plot: slope distribution by coverage status
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

    # Add mean markers
    means = [covered['slope'].mean(), void['slope'].mean()]
    ax.plot([1, 2], means, 'D', color='green', markersize=10,
            label='Mean', zorder=3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'coverage_vs_slope_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

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

    # Statistical test
    from scipy import stats as sp_stats
    t_stat, p_val = sp_stats.ttest_ind(void['slope'], covered['slope'])
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_val:.2e}")

    # Coverage by slope category
    print(f"\nCoverage rate by slope category:")
    slope_categories = [
        (0, 5, 'Flat'),
        (5, 15, 'Gentle'),
        (15, 30, 'Moderate'),
        (30, 90, 'Steep')
    ]

    for min_slope, max_slope, label in slope_categories:
        mask = (df['slope'] >= min_slope) & (df['slope'] < max_slope)
        if mask.sum() > 0:
            coverage = df[mask]['has_coverage'].mean() * 100
            n_cells = mask.sum()
            n_covered = df[mask]['has_coverage'].sum()
            print(f"  {label:12s} ({min_slope:2d}-{max_slope:2d}°): {coverage:5.2f}% " +
                  f"({n_covered:,}/{n_cells:,} cells)")
    print(f"{'='*60}\n")


def plot_void_coverage_pca(df, loadings, save_dir='images', include_slope_analysis=False):
    """
    Create visualizations for void coverage PCA analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe with PC scores and 'has_coverage' column
    loadings : pd.DataFrame
        Feature loadings from PCA
    save_dir : str, default='images'
        Directory to save plots
    include_slope_analysis : bool, default=False
        If True, also create coverage vs slope analysis
        (Set to False if already created in notebook)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Optionally create coverage vs slope analysis
    if include_slope_analysis:
        print("\nGenerating coverage vs slope analysis...")
        plot_coverage_vs_slope(df, save_dir)

    # 1. PC1 vs PC2 scatter (void vs covered)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample for plotting if too many points
    df_plot = df.sample(min(10000, len(df)), random_state=42)

    covered = df_plot[df_plot['has_coverage'] == True]
    void = df_plot[df_plot['has_coverage'] == False]

    ax.scatter(covered['PC1'], covered['PC2'],
               c='blue', alpha=0.3, s=5, label=f'Covered ({len(covered):,})')
    ax.scatter(void['PC1'], void['PC2'],
               c='red', alpha=0.3, s=5, label=f'Void ({len(void):,})')

    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('Void vs Covered Grid Cells (PCA Space)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'void_coverage_pca_scatter.png'), dpi=150)
    print(f"Saved: {save_dir}/void_coverage_pca_scatter.png")
    plt.close()

    # 2. PC score distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(min(4, len([c for c in df.columns if c.startswith('PC')]))):
        pc_col = f'PC{i+1}'
        ax = axes[i]

        covered[pc_col].hist(ax=ax, bins=50, alpha=0.6, color='blue',
                              label='Covered', density=True)
        void[pc_col].hist(ax=ax, bins=50, alpha=0.6, color='red',
                           label='Void', density=True)

        ax.set_xlabel(pc_col, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{pc_col} Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'void_coverage_pca_distributions.png'), dpi=150)
    print(f"Saved: {save_dir}/void_coverage_pca_distributions.png")
    plt.close()

    # 3. Feature loadings heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(loadings)*0.3)))

    im = ax.imshow(loadings.iloc[:, :5].values, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)

    ax.set_xticks(np.arange(min(5, loadings.shape[1])))
    ax.set_xticklabels([f'PC{i+1}' for i in range(min(5, loadings.shape[1]))], fontsize=11)
    ax.set_yticks(np.arange(len(loadings)))
    ax.set_yticklabels(loadings.index, fontsize=9)

    # Add text annotations
    for i in range(len(loadings)):
        for j in range(min(5, loadings.shape[1])):
            val = loadings.iloc[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Loading')
    ax.set_title('PCA Feature Loadings (Void Coverage Analysis)',
                fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'void_coverage_pca_loadings.png'), dpi=150)
    print(f"Saved: {save_dir}/void_coverage_pca_loadings.png")
    plt.close()

    print("\nAll visualizations saved!")


def export_void_coverage_summary(df, comp_df, save_path='results/void_coverage_summary.csv'):
    """
    Export summary statistics to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Grid cell dataframe
    comp_df : pd.DataFrame
        Feature comparison dataframe
    save_path : str
        Output file path
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    comp_df.to_csv(save_path, index=False)
    print(f"\nSaved summary: {save_path}")
