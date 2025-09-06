import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds


def points_to_grid(csv_file, reference_dem_path, output_path):
    """Convert SAOCOM points to gridded DEM matching reference"""

    # Load SAOCOM points
    df = pd.read_csv(csv_file)
    points = df[['LON', 'LAT']].values
    heights = df['HEIGHT'].values
    coherence = df['COHER'].values

    # Get reference grid from existing DEM
    with rasterio.open(reference_dem_path) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        bounds = ref.bounds

    # Create coordinate grids
    cols, rows = np.meshgrid(
        np.linspace(bounds.left, bounds.right, width),
        np.linspace(bounds.top, bounds.bottom, height)
    )

    # Interpolate heights to grid
    grid_heights = griddata(
        points, heights, (cols, rows),
        method='linear', fill_value=np.nan
    )

    # Save as GeoTIFF
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': crs,
        'transform': transform,
        'nodata': -9999
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(grid_heights.astype(rasterio.float32), 1)