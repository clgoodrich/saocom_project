"""
Outlier detection and filtering for SAOCOM height data.
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def remove_isolated_knn(gdf, k=100, distance_threshold=1000):
    """
    Remove spatially isolated points using k-nearest neighbors.

    Points whose k nearest neighbors are on average farther than the threshold
    are considered isolated and removed.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with point geometries
    k : int, optional
        Number of nearest neighbors to consider (default: 100)
    distance_threshold : float, optional
        Average distance threshold in meters (default: 1000)

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame with isolated points removed, index reset

    Examples
    --------
    >>> clean_gdf = remove_isolated_knn(saocom_gdf, k=100, distance_threshold=1000)
    >>> print(f"Removed {len(saocom_gdf) - len(clean_gdf)} isolated points")
    """
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    distances = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        coords, return_distance=True
    )[0]

    print(f"Total points: {len(distances)}")

    # Mean distance to k nearest neighbors (excluding self at index 0)
    mean_distances = distances[:, 1:].mean(axis=1)

    return gdf[mean_distances < distance_threshold].reset_index(drop=True)


def score_outliers_isolation_forest(gdf, residual_col, **kwargs):
    """
    Score outliers using Isolation Forest algorithm.

    Uses spatial coordinates (x, y) and residual values to detect anomalies.
    More negative scores indicate more anomalous points.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with point geometries
    residual_col : str
        Column name containing residual values
    **kwargs : dict
        Additional parameters passed to IsolationForest:
        - n_estimators: Number of trees (default: 100)
        - contamination: Expected proportion of outliers (default: 'auto')
        - random_state: Random seed (default: 42)
        - n_jobs: Parallel jobs (default: -1, use all cores)

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with added 'outlier_score' column

    Notes
    -----
    The function standardizes features before fitting to ensure equal weighting
    of spatial and residual dimensions.

    Examples
    --------
    >>> gdf_scored = score_outliers_isolation_forest(
    ...     saocom_gdf,
    ...     residual_col='diff_tinitaly',
    ...     contamination=0.05
    ... )
    >>> print(f"Min score: {gdf_scored['outlier_score'].min():.3f}")
    """
    if gdf.empty or residual_col not in gdf.columns:
        gdf = gdf.copy()
        gdf['outlier_score'] = np.nan
        return gdf

    # Stack features: x, y, residual
    pts = np.column_stack((
        gdf.geometry.x.values,
        gdf.geometry.y.values,
        gdf[residual_col].fillna(0).values
    ))

    # Standardize features
    X = StandardScaler().fit_transform(pts)

    # Default parameters
    params = {
        'n_estimators': 100,
        'contamination': 'auto',
        'random_state': 42,
        'n_jobs': -1
    }
    params.update(kwargs)

    # Fit Isolation Forest
    model = IsolationForest(**params).fit(X)

    # Add outlier scores
    gdf_scored = gdf.copy()
    gdf_scored['outlier_score'] = model.decision_function(X)

    return gdf_scored


def filter_by_score_iqr(gdf_scored, iqr_multiplier=1):
    """
    Filter outliers using IQR method on outlier scores.

    Points with scores below Q1 - (iqr_multiplier * IQR) are considered outliers.

    Parameters
    ----------
    gdf_scored : gpd.GeoDataFrame
        GeoDataFrame with 'outlier_score' column
    iqr_multiplier : float, optional
        IQR multiplier for threshold (default: 1)
        Higher values = more permissive filtering

    Returns
    -------
    tuple
        (cleaned_gdf, outliers_gdf)
        - cleaned_gdf: Inlier points
        - outliers_gdf: Outlier points

    Raises
    ------
    ValueError
        If input lacks 'outlier_score' column

    Examples
    --------
    >>> cleaned, outliers = filter_by_score_iqr(gdf_scored, iqr_multiplier=1.5)
    >>> print(f"Retained {len(cleaned)} points, removed {len(outliers)}")
    """
    if 'outlier_score' not in gdf_scored.columns:
        raise ValueError("Input GeoDataFrame must have an 'outlier_score' column.")

    s = gdf_scored['outlier_score'].values
    q1, q3 = np.percentile(s, [25, 75])
    threshold = q1 - iqr_multiplier * (q3 - q1)

    mask = s < threshold

    return gdf_scored[~mask].copy(), gdf_scored[mask].copy()


def visualize_outlier_results(gdf_original, gdf_cleaned, outliers, residual_col,
                                results_dir=None):
    """
    Visualize outlier detection results.

    Creates a two-panel figure:
    - Left: Spatial map showing cleaned data and outliers
    - Right: Histogram comparing residual distributions before/after

    Parameters
    ----------
    gdf_original : gpd.GeoDataFrame
        Original dataset before filtering
    gdf_cleaned : gpd.GeoDataFrame
        Cleaned dataset after filtering
    outliers : gpd.GeoDataFrame
        Detected outliers
    residual_col : str
        Column name containing residuals
    results_dir : str or Path, optional
        Directory to save figure (default: None, uses './results')

    Notes
    -----
    The figure is saved as 'difference_by_coherence.png' at 300 DPI.

    Examples
    --------
    >>> visualize_outlier_results(
    ...     saocom_gdf,
    ...     cleaned_gdf,
    ...     outliers,
    ...     'diff_tinitaly',
    ...     results_dir='./outputs'
    ... )
    """
    if results_dir is None:
        results_dir = Path('./results')
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(exist_ok=True)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(20, 9),
        facecolor='white',
        gridspec_kw={'width_ratios': [1.2, 1]}
    )

    # Spatial map
    vmin, vmax = np.nanpercentile(gdf_cleaned[residual_col].values, [2, 98])

    sc = ax1.scatter(
        gdf_cleaned.geometry.x.values,
        gdf_cleaned.geometry.y.values,
        c=gdf_cleaned[residual_col].values,
        cmap='RdBu_r',
        s=5,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        label='Cleaned Data'
    )

    plt.colorbar(sc, ax=ax1, label=f'Residual ({residual_col}) (m)', shrink=0.7)

    if not outliers.empty:
        outliers.plot(
            ax=ax1,
            markersize=25,
            color='yellow',
            edgecolors='black',
            linewidth=0.8,
            label=f'Outliers (n={len(outliers):,})',
            zorder=5
        )

    ax1.set(
        title='Spatial Distribution of Cleaned Data and Outliers',
        xlabel='UTM Easting (m)',
        ylabel='UTM Northing (m)'
    )
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_aspect('equal', adjustable='box')

    # Histogram comparison
    orig = gdf_original[residual_col].dropna().values
    cln = gdf_cleaned[residual_col].dropna().values

    # Calculate appropriate bins to avoid empty bins
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
             color='#2E86AB', edgecolor='darkblue', linewidth=0.8)

    ax2.set(
        title='Residual Distribution Before and After Cleaning',
        xlabel=f'Residual ({residual_col}) (m)',
        ylabel='Frequency (Log Scale)'
    )
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(results_dir / "difference_by_coherence.png", dpi=300,
                bbox_inches='tight')
    plt.show()
