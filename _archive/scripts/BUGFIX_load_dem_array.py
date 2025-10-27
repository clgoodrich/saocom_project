"""
Fixed version of load_dem_array function
Replace your current version with this.
"""

import rasterio
import rasterio.windows


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
    """
    with rasterio.open(path) as src:
        # FIX: Use "is not None" instead of just "if bounds:"
        if bounds is not None:
            window = rasterio.windows.from_bounds(*bounds, src.transform)
            array = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            array = src.read(1)
            transform = src.transform

    return array, transform


# Example usage:
if __name__ == "__main__":
    import geopandas as gpd
    from pathlib import Path

    # Load SAOCOM data
    saocom_gdf = gpd.read_file("data/saocom_csv/verona_mstgraph_ASI056_weighted_Tcoh00_Bn0_202307-202507.csv")

    # Get bounds (this is a numpy array!)
    bounds = saocom_gdf.total_bounds  # array([minx, miny, maxx, maxy])

    # Load DEM cropped to bounds
    dem_path = Path("data/tinitaly/tinitaly_crop.tif")
    dem, transform = load_dem_array(dem_path, bounds)

    print(f"✓ DEM loaded: {dem.shape}")
