"""
Utility functions for raster I/O and metadata operations.
"""

import rasterio
import rasterio.windows


def read_raster_meta(path):
    """
    Read metadata from a raster file.

    Parameters
    ----------
    path : str or Path
        Path to the raster file

    Returns
    -------
    tuple
        (crs, resolution, bounds, nodata)
        - crs: Coordinate reference system
        - resolution: Tuple of (x_res, y_res)
        - bounds: Raster bounds
        - nodata: Nodata value

    Examples
    --------
    >>> crs, res, bounds, nodata = read_raster_meta("dem.tif")
    >>> print(f"Resolution: {res[0]}m")
    """
    with rasterio.open(path) as src:
        return src.crs, src.res, src.bounds, src.nodata


def load_dem_array(path, bounds=None):
    """
    Load DEM array, optionally cropped to bounds.

    Parameters
    ----------
    path : str or Path
        Path to DEM raster file
    bounds : tuple or np.ndarray, optional
        Bounding box (minx, miny, maxx, maxy) to crop to.
        If None, loads entire raster.

    Returns
    -------
    array : np.ndarray
        DEM array
    transform : affine.Affine
        Geotransform

    Examples
    --------
    >>> # Load full DEM
    >>> dem, transform = load_dem_array("tinitaly.tif")

    >>> # Load DEM cropped to SAOCOM extent
    >>> bounds = saocom_gdf.total_bounds
    >>> dem, transform = load_dem_array("tinitaly.tif", bounds)
    >>> print(f"DEM shape: {dem.shape}")
    """
    with rasterio.open(path) as src:
        # Check if bounds is provided using "is not None" to avoid
        # ValueError when bounds is a numpy array
        if bounds is not None:
            window = rasterio.windows.from_bounds(*bounds, src.transform)
            array = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            array = src.read(1)
            transform = src.transform

    return array, transform
