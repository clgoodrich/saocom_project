"""
Control point identification and analysis for multi-DEM validation.

This module identifies high-quality control points where multiple DEMs agree,
useful for:
- Calibration and co-registration
- Quality assessment
- Bias correction
- Reliability analysis
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def identify_control_points(gdf, tolerance=2.0,
                            saocom_col='HEIGHT_RELATIVE',
                            copernicus_col='copernicus_height',
                            tinitaly_col='tinitaly_height',
                            calibrated=True):
    """
    Identify control points where all three DEMs agree within a tolerance.

    Control points are locations where SAOCOM (after calibration), Copernicus,
    and TINItaly elevations all agree within ±tolerance meters. These represent
    high-confidence elevation measurements.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with SAOCOM data and sampled DEM heights
    tolerance : float, optional
        Maximum elevation difference (meters) for agreement (default: 2.0)
    saocom_col : str, optional
        Column name for SAOCOM heights (default: 'HEIGHT_RELATIVE')
    copernicus_col : str, optional
        Column name for Copernicus heights (default: 'copernicus_height')
    tinitaly_col : str, optional
        Column name for TINItaly heights (default: 'tinitaly_height')
    calibrated : bool, optional
        If True, use calibrated SAOCOM heights (default: True)
        If False, use raw SAOCOM heights

    Returns
    -------
    gpd.GeoDataFrame
        Subset of input GeoDataFrame containing only control points

    Notes
    -----
    Agreement criterion: max(elev1, elev2, elev3) - min(elev1, elev2, elev3) <= tolerance

    This ensures all three elevations are within a 'tolerance' meter range.

    Examples
    --------
    >>> control_pts = identify_control_points(saocom_gdf, tolerance=2.0)
    >>> print(f"Found {len(control_pts)} control points ({len(control_pts)/len(saocom_gdf)*100:.1f}%)")

    >>> # Stricter tolerance
    >>> high_quality = identify_control_points(saocom_gdf, tolerance=1.0)
    """
    # Create a copy to avoid modifying original
    data = gdf.copy()

    # Filter for points with all three elevation values
    valid_mask = (
        data[saocom_col].notna() &
        data[copernicus_col].notna() &
        data[tinitaly_col].notna()
    )

    data_valid = data[valid_mask].copy()

    if len(data_valid) == 0:
        print("Warning: No points with all three DEM values available")
        return gpd.GeoDataFrame(columns=data.columns, crs=data.crs)

    # Use calibrated SAOCOM if available and requested
    if calibrated and 'HEIGHT_CALIBRATED' in data_valid.columns:
        saocom_heights = data_valid['HEIGHT_CALIBRATED']
    else:
        saocom_heights = data_valid[saocom_col]

    copernicus_heights = data_valid[copernicus_col]
    tinitaly_heights = data_valid[tinitaly_col]

    # Calculate range (max - min) for each point
    elevation_stack = np.column_stack([
        saocom_heights,
        copernicus_heights,
        tinitaly_heights
    ])

    elevation_range = np.max(elevation_stack, axis=1) - np.min(elevation_stack, axis=1)

    # Identify control points (range <= tolerance)
    control_point_mask = elevation_range <= tolerance

    control_points = data_valid[control_point_mask].copy()

    # Add useful metadata columns
    control_points['elevation_range'] = elevation_range[control_point_mask]
    control_points['mean_elevation'] = np.mean(elevation_stack[control_point_mask], axis=1)
    control_points['std_elevation'] = np.std(elevation_stack[control_point_mask], axis=1)

    return control_points


def analyze_control_point_distribution(control_pts, all_pts):
    """
    Analyze spatial and statistical distribution of control points.

    Parameters
    ----------
    control_pts : gpd.GeoDataFrame
        Control points GeoDataFrame
    all_pts : gpd.GeoDataFrame
        All points GeoDataFrame (for comparison)

    Returns
    -------
    dict
        Dictionary with statistics:
        - count: Number of control points
        - fraction: Fraction of total points
        - spatial_density: Points per km²
        - mean_elevation: Mean elevation of control points
        - elevation_range: Range of elevations covered
        - terrain_stats: Statistics by terrain type

    Examples
    --------
    >>> stats = analyze_control_point_distribution(control_pts, saocom_gdf)
    >>> print(f"Control points: {stats['count']} ({stats['fraction']*100:.1f}%)")
    """
    stats = {}

    # Basic counts
    stats['count'] = len(control_pts)
    stats['fraction'] = len(control_pts) / len(all_pts) if len(all_pts) > 0 else 0

    if len(control_pts) == 0:
        return stats

    # Spatial statistics
    bounds = control_pts.total_bounds  # minx, miny, maxx, maxy
    area_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) * 111 * 111  # rough lat/lon to km
    stats['spatial_density'] = stats['count'] / area_km2 if area_km2 > 0 else 0

    # Elevation statistics
    if 'mean_elevation' in control_pts.columns:
        stats['mean_elevation'] = float(control_pts['mean_elevation'].mean())
        stats['elevation_range'] = (
            float(control_pts['mean_elevation'].min()),
            float(control_pts['mean_elevation'].max())
        )
        stats['elevation_std'] = float(control_pts['mean_elevation'].std())

    # Terrain statistics (if available)
    if 'slope' in control_pts.columns:
        stats['mean_slope'] = float(control_pts['slope'].mean())
        stats['slope_range'] = (
            float(control_pts['slope'].min()),
            float(control_pts['slope'].max())
        )

    # Agreement quality
    if 'elevation_range' in control_pts.columns:
        stats['mean_agreement'] = float(control_pts['elevation_range'].mean())
        stats['max_disagreement'] = float(control_pts['elevation_range'].max())

    # Stratification by slope if available
    if 'slope' in control_pts.columns:
        slope_bins = [0, 5, 10, 20, 30, 90]
        slope_labels = ['0-5°', '5-10°', '10-20°', '20-30°', '>30°']
        control_pts_copy = control_pts.copy()
        control_pts_copy['slope_class'] = pd.cut(
            control_pts_copy['slope'],
            bins=slope_bins,
            labels=slope_labels,
            include_lowest=True
        )
        stats['by_slope'] = control_pts_copy.groupby('slope_class', observed=True).size().to_dict()

    return stats


def calculate_control_point_bias(control_pts,
                                 saocom_col='HEIGHT_RELATIVE',
                                 reference_col='mean_elevation',
                                 calibrated=True):
    """
    Calculate bias of SAOCOM relative to consensus elevation at control points.

    Parameters
    ----------
    control_pts : gpd.GeoDataFrame
        Control points GeoDataFrame
    saocom_col : str, optional
        SAOCOM height column name
    reference_col : str, optional
        Reference elevation column (default: 'mean_elevation' from control points)
    calibrated : bool, optional
        Use calibrated SAOCOM heights if available

    Returns
    -------
    dict
        Bias statistics:
        - mean_bias: Mean difference
        - std_bias: Standard deviation
        - rmse: Root mean square error
        - nmad: Normalized median absolute deviation

    Examples
    --------
    >>> bias_stats = calculate_control_point_bias(control_pts)
    >>> print(f"SAOCOM bias at control points: {bias_stats['mean_bias']:.3f} m")
    """
    from scipy.stats import median_abs_deviation

    if len(control_pts) == 0:
        return {'mean_bias': np.nan, 'std_bias': np.nan, 'rmse': np.nan, 'nmad': np.nan}

    # Use calibrated or raw SAOCOM
    if calibrated and 'HEIGHT_CALIBRATED' in control_pts.columns:
        saocom_heights = control_pts['HEIGHT_CALIBRATED']
    else:
        saocom_heights = control_pts[saocom_col]

    reference = control_pts[reference_col]

    # Calculate residuals
    residuals = saocom_heights - reference

    # Statistics
    bias_stats = {
        'mean_bias': float(np.mean(residuals)),
        'median_bias': float(np.median(residuals)),
        'std_bias': float(np.std(residuals)),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'nmad': float(1.4826 * median_abs_deviation(residuals, nan_policy='omit')),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals))
    }

    return bias_stats


def spatial_clustering_control_points(control_pts, n_clusters=5):
    """
    Cluster control points spatially for representative sampling.

    Uses K-means clustering to identify representative control points
    distributed across the study area.

    Parameters
    ----------
    control_pts : gpd.GeoDataFrame
        Control points GeoDataFrame
    n_clusters : int, optional
        Number of clusters (default: 5)

    Returns
    -------
    gpd.GeoDataFrame
        Control points with added 'cluster' column

    Examples
    --------
    >>> control_pts_clustered = spatial_clustering_control_points(control_pts, n_clusters=10)
    >>> # Select one point from each cluster for ground validation
    >>> representative_pts = control_pts_clustered.groupby('cluster').first()
    """
    from sklearn.cluster import KMeans

    if len(control_pts) < n_clusters:
        print(f"Warning: Fewer control points ({len(control_pts)}) than clusters ({n_clusters})")
        n_clusters = len(control_pts)

    # Extract coordinates
    coords = np.column_stack([
        control_pts.geometry.x,
        control_pts.geometry.y
    ])

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)

    # Add cluster labels
    control_pts_clustered = control_pts.copy()
    control_pts_clustered['cluster'] = clusters

    return control_pts_clustered


def export_control_points(control_pts, output_path, format='GeoJSON'):
    """
    Export control points to file.

    Parameters
    ----------
    control_pts : gpd.GeoDataFrame
        Control points to export
    output_path : str or Path
        Output file path
    format : str, optional
        Output format: 'GeoJSON', 'Shapefile', 'GeoPackage', 'CSV'
        Default: 'GeoJSON'

    Examples
    --------
    >>> export_control_points(control_pts, 'results/control_points.geojson')
    >>> export_control_points(control_pts, 'results/control_points.shp', format='Shapefile')
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'csv':
        # Export as CSV with coordinates
        df = control_pts.copy()
        df['longitude'] = df.geometry.x
        df['latitude'] = df.geometry.y
        df = df.drop(columns=['geometry'])
        df.to_csv(output_path, index=False)
        print(f"✓ Exported {len(control_pts)} control points to {output_path}")

    elif format.lower() == 'geojson':
        control_pts.to_file(output_path, driver='GeoJSON')
        print(f"✓ Exported {len(control_pts)} control points to {output_path}")

    elif format.lower() == 'shapefile':
        control_pts.to_file(output_path, driver='ESRI Shapefile')
        print(f"✓ Exported {len(control_pts)} control points to {output_path}")

    elif format.lower() == 'geopackage' or format.lower() == 'gpkg':
        control_pts.to_file(output_path, driver='GPKG')
        print(f"✓ Exported {len(control_pts)} control points to {output_path}")

    else:
        raise ValueError(f"Unsupported format: {format}")


def recommend_calibration_points(control_pts, n_points=10, method='distributed'):
    """
    Recommend subset of control points for calibration.

    Parameters
    ----------
    control_pts : gpd.GeoDataFrame
        Control points GeoDataFrame
    n_points : int, optional
        Number of points to recommend (default: 10)
    method : str, optional
        Selection method:
        - 'distributed': Spatially distributed (K-means)
        - 'best': Lowest elevation_range (best agreement)
        - 'random': Random sample

    Returns
    -------
    gpd.GeoDataFrame
        Recommended calibration points

    Examples
    --------
    >>> calibration_pts = recommend_calibration_points(control_pts, n_points=10, method='distributed')
    """
    if method == 'distributed':
        control_pts_clustered = spatial_clustering_control_points(control_pts, n_clusters=n_points)
        # Select point with best agreement from each cluster
        recommended = []
        for cluster_id in range(n_points):
            cluster_pts = control_pts_clustered[control_pts_clustered['cluster'] == cluster_id]
            if len(cluster_pts) > 0:
                # Select point with minimum elevation_range (best agreement)
                best_idx = cluster_pts['elevation_range'].idxmin()
                recommended.append(cluster_pts.loc[best_idx])

        return gpd.GeoDataFrame(recommended, crs=control_pts.crs)

    elif method == 'best':
        # Select n_points with lowest elevation_range
        return control_pts.nsmallest(n_points, 'elevation_range')

    elif method == 'random':
        return control_pts.sample(n=min(n_points, len(control_pts)), random_state=42)

    else:
        raise ValueError(f"Unknown method: {method}")
