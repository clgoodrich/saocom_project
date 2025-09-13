"""
Convert SAOCOM CSV point data to raster in EPSG:32632
Assumes reference DEMs are already properly projected
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pathlib import Path

def read_saocom_csv(csv_path, coherence_threshold=0.3):
    """
    Read SAOCOM CSV data and apply coherence filtering

    Parameters:
    -----------
    csv_path : str or Path
        Path to SAOCOM CSV file
    coherence_threshold : float
        Minimum coherence value to keep points

    Returns:
    --------
    geopandas.GeoDataFrame
        Projected point data in EPSG:32632
    """
    # Try to read CSV - handle different separators and formats
    with open(csv_path, 'r') as f:
        first_line = f.readline()

    if ',' in first_line:
        separator = ','
    elif '\t' in first_line:
        separator = '\t'
    else:
        separator = r'\s+'
    try:
        # Read the data
        df = pd.read_csv(csv_path, sep=separator)
    except:
        # Fallback: assume standard column names
        columns = ['ID', 'SVET', 'LVET', 'LAT', 'LON', 'HEIGHT',
                  'HEIGHT_WRT_DEM', 'SIGMA_HEIGHT', 'COHER']
        df = pd.read_csv(csv_path, sep=separator, names=columns, skiprows=1)
    # Standardize column names (handle variations)
    column_mapping = {
        'LAT': 'LAT', 'LATITUDE': 'LAT', 'lat': 'LAT',
        'LON': 'LON', 'LONGITUDE': 'LON', 'lon': 'LON', 'LONG': 'LON',
        'HEIGHT': 'HEIGHT', 'ELEVATION': 'HEIGHT', 'ELEV': 'HEIGHT',
        'COHER': 'COHER', 'COHERENCE': 'COHER', 'COH': 'COHER'
    }

    # Clean column names and apply mapping
    df.columns = [col.strip().upper() for col in df.columns]
    df = df.rename(columns=column_mapping)

    # Convert to numeric
    for col in ['LAT', 'LON', 'HEIGHT', 'COHER']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove invalid points
    df = df.dropna(subset=['LAT', 'LON', 'HEIGHT'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    # Apply coherence filter if available
    if 'COHER' in df.columns:
        df = df[df['COHER'] >= coherence_threshold]

    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Reproject to UTM 32N
    gdf_utm = gdf.to_crs('EPSG:32632')

    # Add UTM coordinates as columns
    gdf_utm['X'] = gdf_utm.geometry.x
    gdf_utm['Y'] = gdf_utm.geometry.y

    return gdf_utm

def points_to_raster(gdf, resolution=10, method='linear', buffer=500):
    """
    Convert point data to raster using interpolation

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Point data with X, Y, HEIGHT columns
    resolution : float
        Pixel size in meters
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
    buffer : float
        Buffer around points in meters

    Returns:
    --------
    tuple
        (raster_array, transform, bounds)
    """

    # Get bounds with buffer
    minx = gdf['X'].min() - buffer
    maxx = gdf['X'].max() + buffer
    miny = gdf['Y'].min() - buffer
    maxy = gdf['Y'].max() + buffer

    # Calculate grid dimensions
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Create coordinate grids
    x = np.linspace(minx, maxx, width)
    y = np.linspace(miny, maxy, height)
    grid_x, grid_y = np.meshgrid(x, y)

    # Extract point coordinates and values
    points = np.column_stack((gdf['X'].values, gdf['Y'].values))
    values = gdf['HEIGHT'].values

    # Interpolate
    grid_z = griddata(points, values, (grid_x, grid_y), method=method, fill_value=np.nan)

    # Flip array for correct orientation (rasterio convention)
    grid_z = np.flipud(grid_z)

    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return grid_z, transform, (minx, miny, maxx, maxy)

def save_raster(array, transform, output_path, crs='EPSG:32632', nodata=-9999):
    """
    Save raster array to GeoTIFF

    Parameters:
    -----------
    array : numpy.ndarray
        Raster data
    transform : rasterio.transform.Affine
        Geotransform
    output_path : str or Path
        Output file path
    crs : str
        Coordinate reference system
    nodata : float
        NoData value
    """
    # Replace NaN with nodata value
    array_out = array.copy()
    array_out[np.isnan(array_out)] = nodata

    # Write to file
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(array_out, 1)


def plot_conversion_results(gdf, raster_array, transform, output_dir):
    """
    Create visualization of conversion results
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original points
    ax1 = axes[0]
    scatter = ax1.scatter(gdf['X'], gdf['Y'], c=gdf['HEIGHT'],
                         cmap='terrain', s=1, alpha=0.6)
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_title(f'Original SAOCOM Points\n({len(gdf)} points)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Height (m)')

    # Plot 2: Interpolated raster
    ax2 = axes[1]
    extent = [transform.c, transform.c + transform.a * raster_array.shape[1],
              transform.f + transform.e * raster_array.shape[0], transform.f]

    im = ax2.imshow(raster_array, extent=extent, cmap='terrain',
                   origin='upper', aspect='equal')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.set_title('Interpolated Raster')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax2, label='Height (m)')

    # Plot 3: Data coverage
    ax3 = axes[2]
    coverage = ~np.isnan(raster_array)
    ax3.imshow(coverage, extent=extent, cmap='Greys',
              origin='upper', aspect='equal')
    ax3.scatter(gdf['X'], gdf['Y'], c='red', s=0.5, alpha=0.3)
    ax3.set_xlabel('Easting (m)')
    ax3.set_ylabel('Northing (m)')
    ax3.set_title('Data Coverage\n(Gray=Interpolated, Red=Original)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "saocom_conversion_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Main conversion workflow
    """
    # Configuration
    config = {
        'input_csv': 'data/verona.csv',  # Adjust filename as needed
        'output_dir': Path('results/saocom_raster'),
        'resolution': 10,  # meters
        'coherence_threshold': 0.3,
        'interpolation_method': 'linear',  # 'linear', 'nearest', or 'cubic'
        'buffer': 500  # meters around points
    }

    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Step 1: Read and project CSV data
    gdf = read_saocom_csv(
        config['input_csv'],
        coherence_threshold=config['coherence_threshold']
    )

    # Save projected points for reference
    points_output = config['output_dir'] / "saocom_points_utm32n.gpkg"
    gdf.to_file(points_output, driver="GPKG")

    # Step 2: Convert to raster
    raster_array, transform, bounds = points_to_raster(
        gdf,
        resolution=config['resolution'],
        method=config['interpolation_method'],
        buffer=config['buffer']
    )

    # Step 3: Save raster
    raster_output = config['output_dir'] / f"saocom_dem_utm32n_{config['resolution']}m.tif"
    save_raster(raster_array, transform, raster_output)

    # Step 4: Create visualization
    plot_conversion_results(gdf, raster_array, transform, config['output_dir'])



if __name__ == "__main__":
    main()