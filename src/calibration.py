"""
Height calibration functions for SAOCOM InSAR data.
"""

import numpy as np


def calibrate_heights(saocom_gdf, ref_col, out_col, coherence_threshold=0.8):
    """
    Calibrate SAOCOM relative heights to reference DEM using median offset.

    SAOCOM InSAR heights are relative, not absolute. This function computes
    the median offset between SAOCOM and a reference DEM, then applies this
    calibration to convert relative heights to absolute heights.

    Parameters
    ----------
    saocom_gdf : gpd.GeoDataFrame
        GeoDataFrame containing SAOCOM data with columns:
        - HEIGHT_RELATIVE: Relative SAOCOM heights
        - COHER: Temporal coherence values
        - {ref_col}: Reference DEM heights at SAOCOM locations
    ref_col : str
        Column name containing reference DEM heights
    out_col : str
        Output column name for calibrated absolute heights
    coherence_threshold : float, optional
        Minimum coherence threshold for calibration points (default: 0.8)

    Returns
    -------
    tuple
        (offset, rmse, n_points)
        - offset: Median offset applied (meters)
        - rmse: Root Mean Square Error after calibration (meters)
        - n_points: Number of points used for calibration

    Notes
    -----
    The calibration uses only high-coherence points (COHER >= threshold) within
    a reasonable height range (|HEIGHT_RELATIVE| < 1000m) to avoid outliers.

    The function modifies saocom_gdf in-place by adding the out_col column.

    Examples
    --------
    >>> offset, rmse, n = calibrate_heights(saocom_gdf,
    ...                                     ref_col='tinitaly_height',
    ...                                     out_col='HEIGHT_ABSOLUTE_TIN')
    >>> print(f"Calibration offset: {offset:.2f}m using {n} points")
    >>> print(f"RMSE after calibration: {rmse:.2f}m")
    """
    # Filter to high-quality points for calibration
    mask = (
        (saocom_gdf['COHER'] >= coherence_threshold) &
        saocom_gdf[ref_col].notna() &
        saocom_gdf['HEIGHT_RELATIVE'].notna() &
        (np.abs(saocom_gdf['HEIGHT_RELATIVE']) < 1000)
    )

    subset = saocom_gdf[mask]

    # Compute median offset
    diff = subset[ref_col] - subset['HEIGHT_RELATIVE']
    offset = np.median(diff)

    # Apply calibration to all points
    saocom_gdf[out_col] = saocom_gdf['HEIGHT_RELATIVE'] + offset

    # Calculate RMSE on calibration subset
    calibrated_subset = subset['HEIGHT_RELATIVE'] + offset
    rmse = np.sqrt(np.mean((subset[ref_col] - calibrated_subset) ** 2))

    return offset, rmse, len(subset)
