"""
Raster preprocessing functions: resampling, masking, sampling, and terrain analysis.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.interpolate import griddata
from pathlib import Path


def resample_to_10m(src_path, output_path, target_transform, target_crs,
                     grid_height, grid_width, nodata=-9999, profile=None):
    """
    Resample a raster to 10m resolution using cubic interpolation.

    Parameters
    ----------
    src_path : str or Path
        Path to source raster file
    output_path : str or Path
        Path for output resampled raster
    target_transform : affine.Affine
        Target affine transformation
    target_crs : CRS
        Target coordinate reference system
    grid_height : int
        Output grid height in pixels
    grid_width : int
        Output grid width in pixels
    nodata : float, optional
        Nodata value (default: -9999)
    profile : dict, optional
        Rasterio profile for output file. If None, basic profile is created.

    Returns
    -------
    tuple
        (resampled_array, output_path)
        - resampled_array: Numpy array of resampled data
        - output_path: Path object of output file

    Examples
    --------
    >>> arr, path = resample_to_10m(
    ...     'copernicus_30m.tif',
    ...     'copernicus_10m.tif',
    ...     target_transform,
    ...     target_crs,
    ...     1000, 1000
    ... )
    """
    arr = np.full((grid_height, grid_width), nodata, np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            rasterio.band(src, 1), arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.cubic,
            src_nodata=src.nodata,
            dst_nodata=nodata
        )

    output_path = Path(output_path)

    if profile is None:
        profile = {
            'driver': 'GTiff',
            'height': grid_height,
            'width': grid_width,
            'count': 1,
            'dtype': 'float32',
            'crs': target_crs,
            'transform': target_transform,
            'nodata': nodata
        }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(arr, 1)

    return arr, output_path


def mask_and_write(arr, hull_mask, output_path, nodata=-9999, profile=None):
    """
    Apply a spatial mask to a raster and write to file.

    Parameters
    ----------
    arr : np.ndarray
        Input raster array
    hull_mask : np.ndarray (bool)
        Boolean mask (True = valid, False = mask out)
    output_path : str or Path
        Output file path
    nodata : float, optional
        Nodata value (default: -9999)
    profile : dict, optional
        Rasterio profile for output

    Returns
    -------
    tuple
        (masked_array, output_path)

    Examples
    --------
    >>> masked, path = mask_and_write(dem_array, hull_mask, 'masked_dem.tif')
    """
    masked = arr.copy()
    masked[~hull_mask] = nodata

    output_path = Path(output_path)

    if profile is not None:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(masked, 1)

    return masked, output_path


def sample_raster_at_points(arr, rows, cols, inbounds_mask, nodata=-9999):
    """
    Sample raster values at point locations.

    Parameters
    ----------
    arr : np.ndarray
        Raster array to sample
    rows : np.ndarray
        Row indices
    cols : np.ndarray
        Column indices
    inbounds_mask : np.ndarray (bool)
        Mask indicating which indices are within bounds
    nodata : float, optional
        Nodata value to use for invalid samples

    Returns
    -------
    np.ndarray
        Sampled values (NaN where invalid or out of bounds)

    Examples
    --------
    >>> values = sample_raster_at_points(dem_array, rows, cols, inb_mask)
    """
    out = np.full(len(rows), np.nan, dtype=np.float32)

    # Sample only inbounds points
    inb_rows = rows[inbounds_mask]
    inb_cols = cols[inbounds_mask]

    v = arr[inb_rows, inb_cols]
    v = np.where(v == nodata, np.nan, v)

    out[inbounds_mask] = v

    return out


def create_difference_grid(gdf, height_col, ref_col, grid_shape, transform,
                            hull_mask, coherence_threshold=0.7):
    """
    Create gridded difference map from point data using nearest neighbor interpolation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with point data
    height_col : str
        Column name for SAOCOM heights
    ref_col : str
        Column name for reference heights
    grid_shape : tuple
        (height, width) of output grid
    transform : affine.Affine
        Affine transformation
    hull_mask : np.ndarray (bool)
        Spatial mask
    coherence_threshold : float, optional
        Minimum coherence for points to include (default: 0.7)

    Returns
    -------
    tuple
        (diff_grid, valid_points_gdf)
        - diff_grid: Gridded difference array
        - valid_points_gdf: GeoDataFrame of points used for gridding

    Notes
    -----
    Only points with valid heights and coherence >= threshold are used.
    The grid is masked using hull_mask.

    Examples
    --------
    >>> diff_grid, pts = create_difference_grid(
    ...     saocom_gdf,
    ...     'HEIGHT_ABSOLUTE_TIN',
    ...     'tinitaly_height',
    ...     (1000, 1000),
    ...     transform,
    ...     hull_mask
    ... )
    """
    # Filter to valid points
    query_str = (f"`{height_col}`.notna() & `{ref_col}`.notna() & "
                 f"COHER >= @coherence_threshold")
    valid_points = gdf.query(query_str).copy()
    valid_points['diff'] = valid_points[height_col] - valid_points[ref_col]

    if valid_points.empty:
        return np.full(grid_shape, np.nan), valid_points

    # Create grid coordinates
    grid_height, grid_width = grid_shape
    x_coords = np.linspace(
        transform.c,
        transform.c + transform.a * grid_width,
        grid_width
    )
    y_coords = np.linspace(
        transform.f,
        transform.f + transform.e * grid_height,
        grid_height
    )
    xi_grid, yi_grid = np.meshgrid(x_coords, y_coords)

    # Interpolate
    diff_grid = griddata(
        (valid_points.geometry.x, valid_points.geometry.y),
        valid_points['diff'],
        (xi_grid, yi_grid),
        method='nearest'
    )

    # Apply mask if provided
    if hull_mask is not None:
        diff_grid[~hull_mask] = np.nan

    return diff_grid, valid_points


def calculate_terrain_derivatives(dem, cellsize=10, nodata=-9999):
    """
    Calculate slope and aspect from DEM using Horn's method.

    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model array
    cellsize : float, optional
        Cell size in meters (default: 10)
    nodata : float, optional
        Nodata value (default: -9999)

    Returns
    -------
    tuple
        (slope, aspect)
        - slope: Slope in degrees
        - aspect: Aspect in degrees (0-360, 0=North, clockwise)

    Notes
    -----
    Horn's method uses a 3x3 kernel for calculating derivatives.
    Nodata values are converted to NaN before processing.

    Examples
    --------
    >>> slope, aspect = calculate_terrain_derivatives(dem_array, cellsize=10)
    >>> print(f"Max slope: {np.nanmax(slope):.1f}°")
    """
    # Convert nodata to NaN
    dem_clean = np.where(dem == nodata, np.nan, dem)

    # Horn's method kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0

    # Calculate derivatives
    dz_dx = ndimage.convolve(dem_clean, kernel_x) / cellsize
    dz_dy = ndimage.convolve(dem_clean, kernel_y) / cellsize

    # Calculate slope and aspect
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))

    # Normalize aspect to 0-360
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    return slope, aspect


def prepare_analysis_dataframe(saocom_gdf, dem_name):
    """
    Prepare filtered dataframe for terrain analysis.

    Parameters
    ----------
    saocom_gdf : gpd.GeoDataFrame
        Input GeoDataFrame with SAOCOM data
    dem_name : str
        'TINITALY' or 'COPERNICUS'

    Returns
    -------
    gpd.GeoDataFrame
        Filtered dataframe with standardized columns:
        - slope, aspect, elevation, residual, abs_residual

    Notes
    -----
    Filters out points with:
    - Missing slope/aspect/difference data
    - |residual| >= 50m

    Examples
    --------
    >>> df_tin = prepare_analysis_dataframe(saocom_gdf, 'TINITALY')
    >>> df_cop = prepare_analysis_dataframe(saocom_gdf, 'COPERNICUS')
    """
    if dem_name.upper() == 'TINITALY':
        df = saocom_gdf[
            (saocom_gdf['slope_tin'].notna()) &
            (saocom_gdf['aspect_tin'].notna()) &
            (saocom_gdf['diff_tinitaly'].notna()) &
            (np.abs(saocom_gdf['diff_tinitaly']) < 50)
        ].copy()

        df['slope'] = df['slope_tin']
        df['aspect'] = df['aspect_tin']
        df['elevation'] = df['tinitaly_height']
        df['residual'] = df['diff_tinitaly']

    else:  # COPERNICUS
        df = saocom_gdf[
            (saocom_gdf['slope_cop'].notna()) &
            (saocom_gdf['aspect_cop'].notna()) &
            (saocom_gdf['diff_copernicus'].notna()) &
            (np.abs(saocom_gdf['diff_copernicus']) < 50)
        ].copy()

        df['slope'] = df['slope_cop']
        df['aspect'] = df['aspect_cop']
        df['elevation'] = df['copernicus_height']
        df['residual'] = df['diff_copernicus']

    df['abs_residual'] = np.abs(df['residual'])

    return df


def calculate_suitability_index(slope, aspect, elevation):
    """
    Calculate terrain suitability index for InSAR measurements.

    Parameters
    ----------
    slope : np.ndarray
        Slope in degrees
    aspect : np.ndarray
        Aspect in degrees
    elevation : np.ndarray
        Elevation in meters

    Returns
    -------
    np.ndarray
        Suitability index (0-1, higher = better)

    Notes
    -----
    Weighting:
    - Slope: 60% (lower slope = better)
    - Aspect: 30% (south-facing = better)
    - Elevation: 10% (lower elevation = slightly better)

    Examples
    --------
    >>> suitability = calculate_suitability_index(slope, aspect, elevation)
    """
    # Slope score (lower is better)
    slope_score = np.where(slope <= 5, 1.0,
                   np.where(slope <= 15, 0.8,
                   np.where(slope <= 30, 0.5, 0.2)))

    # Aspect score (south-facing is better)
    aspect_rad = np.radians(aspect)
    aspect_score = 0.7 + 0.3 * np.cos(aspect_rad - np.radians(180))

    # Elevation score (lower is slightly better)
    elev_norm = np.clip((elevation - 100) / 900, 0, 1)
    elev_score = 1.0 - 0.3 * elev_norm

    # Weighted combination
    return 0.6 * slope_score + 0.3 * aspect_score + 0.1 * elev_score


def classify_terrain_suitability(suitability):
    """
    Classify terrain suitability into discrete categories.

    Parameters
    ----------
    suitability : np.ndarray
        Suitability index (0-1)

    Returns
    -------
    np.ndarray
        Class labels (0-4)
        - 0: Unsuitable (< 0.35)
        - 1: Poor (0.35-0.5)
        - 2: Moderate (0.5-0.65)
        - 3: Good (0.65-0.8)
        - 4: Excellent (>= 0.8)

    Examples
    --------
    >>> classes = classify_terrain_suitability(suitability)
    >>> print(f"Excellent terrain: {(classes == 4).sum()} points")
    """
    return np.where(suitability >= 0.8, 4,
           np.where(suitability >= 0.65, 3,
           np.where(suitability >= 0.5, 2,
           np.where(suitability >= 0.35, 1, 0))))
