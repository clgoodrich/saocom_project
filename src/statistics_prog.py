"""
Statistical analysis functions for height validation.
"""

import numpy as np
import pandas as pd


def nmad(x):
    """
    Calculate Normalized Median Absolute Deviation (NMAD).

    NMAD is a robust measure of dispersion that is less sensitive to outliers
    than standard deviation.

    Parameters
    ----------
    x : array-like
        Input data

    Returns
    -------
    float
        NMAD value

    Notes
    -----
    NMAD = 1.4826 * median(|x - median(x)|)

    The factor 1.4826 makes NMAD approximately equal to the standard deviation
    for normally distributed data.

    Examples
    --------
    >>> residuals = np.array([1.2, -0.5, 0.8, -1.0, 0.3])
    >>> nmad(residuals)
    1.06
    """
    x = np.asarray(x)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def calculate_nmad(series):
    """
    Calculate NMAD from a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input data series

    Returns
    -------
    float
        NMAD value

    See Also
    --------
    nmad : Numpy-based NMAD calculation
    """
    return (series - series.median()).abs().median() * 1.4826


def calculate_height_stats(data, name):
    """
    Calculate comprehensive statistics for height data.

    Parameters
    ----------
    data : array-like
        Height data array
    name : str
        Dataset name for labeling

    Returns
    -------
    dict or None
        Dictionary containing statistics:
        - Dataset: Name
        - Count: Number of valid observations
        - Min/Max/Mean/Median: Central tendency measures
        - Std Dev: Standard deviation
        - Range: Max - Min
        - Q25/Q75/IQR: Quartile statistics

        Returns None if no valid data.

    Examples
    --------
    >>> heights = np.array([100.5, 101.2, 99.8, 100.0, 102.1])
    >>> stats = calculate_height_stats(heights, "Test DEM")
    >>> print(f"Mean: {stats['Mean']:.2f}m")
    """
    v = np.asarray(data)
    v = v[~np.isnan(v)]

    if v.size == 0:
        return None

    q25, q75 = np.percentile(v, [25, 75])

    return {
        'Dataset': name,
        'Count': int(v.size),
        'Min': float(v.min()),
        'Max': float(v.max()),
        'Mean': float(v.mean()),
        'Median': float(np.median(v)),
        'Std Dev': float(v.std()),
        'Range': float(v.max() - v.min()),
        'Q25': float(q25),
        'Q75': float(q75),
        'IQR': float(q75 - q25)
    }


def generate_summary_string(values):
    """
    Generate a compact summary string for residuals.

    Parameters
    ----------
    values : array-like
        Residual values

    Returns
    -------
    str
        Formatted summary string

    Examples
    --------
    >>> residuals = np.array([1.2, -0.5, 0.8])
    >>> print(generate_summary_string(residuals))
    mean=+0.500, median=+0.800, sd=0.702, rmse=0.854 m
    """
    v = np.asarray(values)
    return (f"mean={np.mean(v):+.3f}, median={np.median(v):+.3f}, "
            f"sd={np.std(v):.3f}, rmse={np.sqrt(np.mean(v**2)):.3f} m")


def generate_height_statistics_summary(gdf, gdf_name="SAOCOM Data"):
    """
    Print comprehensive height statistics summary for SAOCOM vs reference DEMs.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing height columns
    gdf_name : str, optional
        Name for labeling the dataset

    Notes
    -----
    Required columns in gdf:
    - HEIGHT_RELATIVE: SAOCOM relative heights
    - tinitaly_height: TINItaly reference heights
    - copernicus_height: Copernicus reference heights

    Prints formatted tables to console with:
    - Individual height statistics for each dataset
    - Difference statistics (SAOCOM - Reference)
    """
    need = ['HEIGHT_RELATIVE', 'tinitaly_height', 'copernicus_height']

    if not all(c in gdf.columns for c in need):
        print(f"Error: Input GeoDataFrame '{gdf_name}' is missing required "
              f"height columns: {need}")
        return

    def _series_stats(s, name):
        """Helper to compute statistics for a pandas Series."""
        v = s.dropna().values
        if v.size == 0:
            return None

        q25, q75 = np.percentile(v, [25, 75])

        return {
            'Dataset': name,
            'Count': int(v.size),
            'Min': float(v.min()),
            'Max': float(v.max()),
            'Mean': float(v.mean()),
            'Median': float(np.median(v)),
            'Std Dev': float(v.std()),
            'Q25': float(q25),
            'Q75': float(q75)
        }

    def _diff_stats(a, b, label):
        """Helper to compute difference statistics."""
        d = (a - b).dropna().values
        if d.size == 0:
            print(f"\n{gdf_name} - {label}: No data.")
            return

        print(f"\n{gdf_name} - {label}:")
        print(f"  Mean: {d.mean():+.3f} m | Median: {np.median(d):+.3f} m | "
              f"Std: {d.std():.3f} m | RMSE: {np.sqrt((d**2).mean()):.3f} m")

    # Print header
    print("\n" + "=" * 95)
    print(f" STATISTICAL SUMMARY FOR: {gdf_name.upper()} ({len(gdf)} points)")
    print("=" * 95)

    # Height statistics
    rows = [
        _series_stats(gdf['HEIGHT_RELATIVE'], f'{gdf_name} (Relative)'),
        _series_stats(gdf['tinitaly_height'], f'TINITALY (at {gdf_name} pts)'),
        _series_stats(gdf['copernicus_height'], f'Copernicus (at {gdf_name} pts)')
    ]

    df = pd.DataFrame([r for r in rows if r is not None])
    print("\nHEIGHT STATISTICS SUMMARY (m)\n" + "-" * 95)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("-" * 95)

    # Difference statistics
    print("\nDIFFERENCE STATISTICS (SAOCOM Relative - Reference DEM):\n" + "-" * 95)
    _diff_stats(gdf['HEIGHT_RELATIVE'], gdf['tinitaly_height'], 'TINITALY')
    _diff_stats(gdf['HEIGHT_RELATIVE'], gdf['copernicus_height'], 'Copernicus')
    print("=" * 95)
