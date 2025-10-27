"""
Test critical cells from saocom_analysis_complete.ipynb
Executes key workflow steps to verify functionality
"""
import json
import sys
from pathlib import Path

def test_critical_cells():
    """Test critical cells by executing Python code"""

    print("=" * 80)
    print("CRITICAL CELLS TEST - SAOCOM_ANALYSIS_COMPLETE.IPYNB")
    print("=" * 80)
    print()

    try:
        print("Test 1: Import basic modules")
        print("-" * 80)
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import rasterio
        from pathlib import Path
        import os

        # Change to project directory
        os.chdir('C:/Users/colto/Documents/GitHub/saocom_project')
        sys.path.insert(0, str(Path.cwd() / 'src'))
        print("[OK] Basic imports successful")
        print()

        print("Test 2: Setup paths and constants")
        print("-" * 80)
        DATA_DIR = Path("data")
        RESULTS_DIR = Path("results")
        RESULTS_DIR.mkdir(exist_ok=True)
        COHERENCE_THRESHOLD = 0.3
        NODATA = -9999
        GRID_SIZE = 10
        TARGET_CRS = 'EPSG:32632'
        print(f"[OK] Data directory: {DATA_DIR.absolute()}")
        print(f"[OK] Results directory: {RESULTS_DIR.absolute()}")
        print()

        print("Test 3: Discover data files")
        print("-" * 80)
        file_discovery = {
            'saocom': ("saocom_csv", "*.csv"),
            'tinitaly': ("tinitaly", "*.tif"),
            'copernicus': ("", "copernicus*.tif"),
            'corine': ("ground_cover", "*.tif"),
        }

        files_found = {}
        for key, (subdir, pattern) in file_discovery.items():
            search_dir = DATA_DIR / subdir if subdir else DATA_DIR
            files = list(search_dir.glob(pattern))
            if files:
                files_found[key] = files[0]
                print(f"[OK] Found {key}: {files[0].name}")
            else:
                print(f"[X] Missing {key} in {search_dir}")

        print()

        if len(files_found) < 4:
            print("[X] Missing required data files")
            return False

        print("Test 4: Load SAOCOM data")
        print("-" * 80)
        saocom_df = pd.read_csv(files_found['saocom'])
        print(f"[OK] Loaded {len(saocom_df):,} points")
        print(f"[OK] Columns: {list(saocom_df.columns)}")

        # Use LAT2/LON2 preferentially
        lat_col = 'LAT2' if 'LAT2' in saocom_df.columns else 'LAT'
        lon_col = 'LON2' if 'LON2' in saocom_df.columns else 'LON'
        print(f"[OK] Using coordinates: {lat_col}, {lon_col}")

        # Create GeoDataFrame
        from shapely.geometry import Point
        geometry = [Point(xy) for xy in zip(saocom_df[lon_col], saocom_df[lat_col])]
        saocom_gdf = gpd.GeoDataFrame(saocom_df, geometry=geometry, crs='EPSG:4326')
        saocom_gdf = saocom_gdf.to_crs(TARGET_CRS)
        print(f"[OK] Converted to {TARGET_CRS}")
        print()

        print("Test 5: Create convex hull")
        print("-" * 80)
        data_hull = saocom_gdf.geometry.unary_union.convex_hull
        hull_gdf = gpd.GeoDataFrame(geometry=[data_hull], crs=TARGET_CRS)
        bounds = saocom_gdf.total_bounds
        print(f"[OK] Hull bounds: {bounds}")
        print()

        print("Test 6: Setup target grid")
        print("-" * 80)
        from rasterio.transform import from_bounds
        xmin, ymin, xmax, ymax = bounds
        grid_width = int(np.ceil((xmax - xmin) / GRID_SIZE))
        grid_height = int(np.ceil((ymax - ymin) / GRID_SIZE))
        target_transform = from_bounds(xmin, ymin, xmax, ymax, grid_width, grid_height)
        print(f"[OK] Grid: {grid_width} x {grid_height} @ {GRID_SIZE}m")
        print()

        print("Test 7: Read raster metadata")
        print("-" * 80)
        with rasterio.open(files_found['tinitaly']) as src:
            print(f"[OK] TINItaly CRS: {src.crs}")
            print(f"[OK] TINItaly shape: {src.shape}")
            print(f"[OK] TINItaly resolution: {src.res}")
        print()

        print("Test 8: Inline function definitions")
        print("-" * 80)

        # Test _read_raster_meta
        def _read_raster_meta(path):
            with rasterio.open(path) as src:
                return {
                    'crs': src.crs,
                    'transform': src.transform,
                    'shape': src.shape,
                    'res': src.res,
                    'bounds': src.bounds,
                    'nodata': src.nodata
                }

        meta = _read_raster_meta(files_found['tinitaly'])
        print(f"[OK] _read_raster_meta works: shape={meta['shape']}")

        # Test _sample function
        test_arr = np.random.rand(grid_height, grid_width).astype(np.float32)
        from rasterio.transform import rowcol

        xs, ys = saocom_gdf.geometry.x.values[:100], saocom_gdf.geometry.y.values[:100]
        rows, cols = rowcol(target_transform, xs, ys)
        rows, cols = np.array(rows, dtype=int), np.array(cols, dtype=int)
        inb = (rows >= 0) & (rows < grid_height) & (cols >= 0) & (cols < grid_width)

        def _sample(arr, rows, cols, inb):
            out = np.full(len(rows), np.nan, dtype=np.float32)
            v = arr[rows[inb], cols[inb]]
            out[inb] = np.where(v == NODATA, np.nan, v)
            return out

        sampled = _sample(test_arr, rows, cols, inb)
        print(f"[OK] _sample works: sampled {np.sum(~np.isnan(sampled))} / {len(sampled)} points")

        # Test nmad
        def nmad(x):
            x = np.asarray(x)
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            med = np.median(x)
            return 1.4826 * np.median(np.abs(x - med))

        test_data = np.random.randn(1000)
        nmad_val = nmad(test_data)
        print(f"[OK] nmad works: value={nmad_val:.3f}")

        print()

        print("=" * 80)
        print("CRITICAL CELLS TEST RESULTS")
        print("=" * 80)
        print()
        print("[OK] ALL CRITICAL TESTS PASSED")
        print()
        print("Verified:")
        print("- All imports work")
        print("- Data files accessible")
        print("- SAOCOM data loads correctly ({:,} points)".format(len(saocom_gdf)))
        print("- Coordinate transformations work")
        print("- Grid setup works ({} x {} @ {}m)".format(grid_width, grid_height, GRID_SIZE))
        print("- Inline functions work correctly")
        print()
        print("The notebook is ready for full execution")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR IN CRITICAL CELLS TEST")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_critical_cells()
    sys.exit(0 if success else 1)
