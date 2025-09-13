"""
Week 2: Coordinate System Standardization and Alignment
SAOCOM Point Data vs Reference DEMs
Author: [Your Name]
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths relative to the main project folder
PROJECT_DIR = Path(".")  # Assumes script is run from saocom_project folder
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "week2_alignment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'target_crs': 'EPSG:32632',  # UTM Zone 32N for Verona area
    'target_resolution': 10,  # meters
    'vertical_datum': 'EGM2008',  # or 'WGS84'
    'resampling_method': Resampling.cubic,
    'nodata_value': -9999
}

class SaocomPointProcessor:
    """Process SAOCOM interferometric point data"""

    def __init__(self, config):
        self.config = config
        self.target_crs = CRS.from_string(config['target_crs'])

    def read_saocom_points(self, filepath):
        """
        Read SAOCOM point data from text file
        Expected format: ID,SVET,LVET,LAT,LON,HEIGHT,HEIGHT WRT DEM,SIGMA HEIGHT,COHER
        """
        print(f"Reading SAOCOM points from {filepath}")

        # First, try to detect the file format
        with open(filepath, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        # Check if it's comma-separated or tab/space separated
        if ',' in first_lines[0]:
            separator = ','
            print("  Detected comma-separated format")
        elif '\t' in first_lines[0]:
            separator = '\t'
            print("  Detected tab-separated format")
        else:
            separator = r'\s+'
            print("  Detected space-separated format")

        # Try different reading strategies
        df = None

        # Strategy 1: Read with header from file
        try:
            df = pd.read_csv(filepath, sep=separator)
            print(f"  Read with existing headers: {list(df.columns)}")

            # Standardize column names (handle variations)
            column_mapping = {
                'ID': 'ID',
                'SVET': 'SVET',
                'LVET': 'LVET',
                'LAT': 'LAT',
                'LON': 'LON',
                'HEIGHT': 'HEIGHT',
                'HEIGHT WRT DEM': 'HEIGHT_WRT_DEM',
                'HEIGHT_WRT_DEM': 'HEIGHT_WRT_DEM',
                'SIGMA HEIGHT': 'SIGMA_HEIGHT',
                'SIGMA_HEIGHT': 'SIGMA_HEIGHT',
                'COHER': 'COHER'
            }

            # Rename columns to standard names
            df.columns = [col.strip() for col in df.columns]  # Remove any whitespace
            df = df.rename(columns=column_mapping)

        except Exception as e:
            print(f"  Strategy 1 failed: {e}")

            # Strategy 2: Read without header, assign column names
            try:
                columns = ['ID', 'SVET', 'LVET', 'LAT', 'LON', 'HEIGHT',
                          'HEIGHT_WRT_DEM', 'SIGMA_HEIGHT', 'COHER']
                df = pd.read_csv(filepath, sep=separator, names=columns, skiprows=1)
                print(f"  Read with assigned headers")
            except Exception as e2:
                print(f"  Strategy 2 failed: {e2}")
                raise ValueError(f"Could not parse SAOCOM data file: {filepath}")

        # Convert numeric columns
        numeric_columns = ['SVET', 'LVET', 'LAT', 'LON', 'HEIGHT',
                          'HEIGHT_WRT_DEM', 'SIGMA_HEIGHT', 'COHER']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean the data - remove rows with invalid coordinates
        df = df[df['LAT'].notna() & df['LON'].notna()]
        df = df[df['LAT'] != 0]  # Remove any zero coordinates

        print(f"  Loaded {len(df)} valid points")

        if len(df) > 0:
            print(f"  Lat range: {df['LAT'].min():.4f} to {df['LAT'].max():.4f}")
            print(f"  Lon range: {df['LON'].min():.4f} to {df['LON'].max():.4f}")
            print(f"  Height range: {df['HEIGHT'].min():.1f} to {df['HEIGHT'].max():.1f} m")
            print(f"  Coherence range: {df['COHER'].min():.3f} to {df['COHER'].max():.3f}")

        return df

    def points_to_geodataframe(self, df):
        """Convert point dataframe to GeoDataFrame with proper CRS"""
        print("\nConverting to GeoDataFrame...")

        # Create geometry from lat/lon
        geometry = [Point(lon, lat) for lon, lat in zip(df['LON'], df['LAT'])]

        # Create GeoDataFrame in WGS84
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

        # Reproject to target CRS
        gdf_projected = gdf.to_crs(self.target_crs)

        # Add projected coordinates as columns
        gdf_projected['X'] = gdf_projected.geometry.x
        gdf_projected['Y'] = gdf_projected.geometry.y

        print(f"  Projected to {self.config['target_crs']}")
        print(f"  X range: {gdf_projected['X'].min():.1f} to {gdf_projected['X'].max():.1f} m")
        print(f"  Y range: {gdf_projected['Y'].min():.1f} to {gdf_projected['Y'].max():.1f} m")

        return gdf_projected

    def apply_coherence_mask(self, gdf, threshold=0.3):
        """Apply coherence-based quality mask"""
        print(f"\nApplying coherence mask (threshold = {threshold})...")

        original_count = len(gdf)
        gdf_masked = gdf[gdf['COHER'] >= threshold].copy()
        removed_count = original_count - len(gdf_masked)

        print(f"  Removed {removed_count} points ({removed_count/original_count*100:.1f}%)")
        print(f"  Remaining points: {len(gdf_masked)}")

        return gdf_masked

class ReferenceDataProcessor:
    """Process reference DEM data (TINITALY, Copernicus, LiDAR)"""

    def __init__(self, config):
        self.config = config
        self.target_crs = CRS.from_string(config['target_crs'])

    def inspect_and_fix_dem_crs(self, input_path):
        """Inspect DEM CRS and attempt to fix if needed"""
        print(f"  Inspecting CRS for: {input_path.name}")

        with rasterio.open(input_path) as src:
            print(f"    Original CRS: {src.crs}")
            print(f"    Bounds: {src.bounds}")
            print(f"    Shape: {src.width} x {src.height}")

            # Check if CRS is valid
            if src.crs is None:
                print("    WARNING: No CRS defined!")
                return None
            elif not src.crs.is_valid:
                print("    WARNING: Invalid CRS!")
                return None
            else:
                return src.crs

    def reproject_dem(self, input_path, output_path, bounds=None):
        """Robustly reproject and resample DEM to target CRS and resolution"""
        print(f"\nReprojecting DEM: {input_path.name}")

        try:
            with rasterio.open(input_path) as src:
                print(f"  Input CRS: {src.crs}")
                print(f"  Input bounds: {src.bounds}")
                print(f"  Input shape: {src.width} x {src.height}")

                # Check if we need to assign a CRS
                source_crs = src.crs

                # Common CRS fixes for Italy region
                if source_crs is None or not source_crs.is_valid:
                    print("  No valid CRS found. Attempting to determine from filename and bounds...")

                    # Check bounds to guess CRS
                    if (src.bounds[0] > -180 and src.bounds[0] < 180 and
                            src.bounds[1] > -90 and src.bounds[1] < 90):
                        # Likely geographic coordinates
                        source_crs = CRS.from_epsg(4326)  # WGS84
                        print(f"  Assigned geographic CRS: {source_crs}")
                    elif (src.bounds[0] > 200000 and src.bounds[0] < 900000 and
                          src.bounds[1] > 4000000 and src.bounds[1] < 6000000):
                        # Likely UTM for Italy
                        source_crs = CRS.from_epsg(32632)  # UTM 32N
                        print(f"  Assigned UTM CRS: {source_crs}")
                    else:
                        print(f"  Cannot determine CRS from bounds: {src.bounds}")
                        raise ValueError("Cannot determine appropriate CRS")

                # If source is already in target CRS, just clip and resample
                if source_crs == self.target_crs:
                    print("  Already in target CRS, just resampling...")

                    # Calculate new transform for target resolution
                    pixel_size = self.config['target_resolution']

                    # If bounds provided, use them; otherwise use source bounds
                    if bounds:
                        west, south, east, north = bounds
                    else:
                        west, south, east, north = src.bounds

                    # Calculate new dimensions
                    width = int((east - west) / pixel_size)
                    height = int((north - south) / pixel_size)

                    # Create new transform
                    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

                    # Update metadata
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'nodata': self.config['nodata_value']
                    })

                    # Resample to new resolution
                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=source_crs,
                            dst_transform=transform,
                            dst_crs=self.target_crs,
                            resampling=self.config['resampling_method']
                        )

                else:
                    # Different CRS, need to reproject
                    print(f"  Reprojecting from {source_crs} to {self.target_crs}")

                    # Use bounds from SAOCOM points if provided
                    if bounds:
                        dst_bounds = bounds
                    else:
                        # Transform source bounds to target CRS to get appropriate bounds
                        from rasterio.warp import transform_bounds
                        dst_bounds = transform_bounds(source_crs, self.target_crs, *src.bounds)

                    # Calculate transform for target CRS
                    transform, width, height = calculate_default_transform(
                        source_crs, self.target_crs, src.width, src.height,
                        *dst_bounds,
                        resolution=self.config['target_resolution']
                    )

                    # Update metadata
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': self.target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'nodata': self.config['nodata_value']
                    })

                    # Reproject
                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=source_crs,
                            dst_transform=transform,
                            dst_crs=self.target_crs,
                            resampling=self.config['resampling_method']
                        )

                print(f"    Output: {width} x {height} pixels at {self.config['target_resolution']}m")
                print(f"    Output CRS: {self.target_crs}")

                return output_path

        except Exception as e:
            print(f"    ERROR: {e}")

            # Try alternative approach - read as array and assign CRS manually
            try:
                print("  Attempting alternative processing...")
                return self._process_dem_manual_crs(input_path, output_path, bounds)
            except Exception as e2:
                print(f"    Alternative processing failed: {e2}")
                raise e

    def _process_dem_manual_crs(self, input_path, output_path, bounds=None):
        """Manual CRS assignment based on file analysis"""

        with rasterio.open(input_path) as src:
            # Read the data
            data = src.read(1)

            # Analyze bounds to determine likely CRS
            src_bounds = src.bounds
            print(f"  Analyzing bounds: {src_bounds}")

            # Italy is roughly:
            # Geographic: 6°-19°E, 36°-47°N
            # UTM 32N: 290000-790000E, 4000000-5200000N
            # UTM 33N: 170000-570000E, 4000000-5200000N

            if (6 <= src_bounds[0] <= 19 and 36 <= src_bounds[1] <= 47):
                assumed_crs = CRS.from_epsg(4326)  # Geographic WGS84
                print(f"  Assuming geographic coordinates: {assumed_crs}")
            elif (200000 <= src_bounds[0] <= 900000 and 3900000 <= src_bounds[1] <= 5300000):
                if src_bounds[0] < 500000:
                    assumed_crs = CRS.from_epsg(32632)  # UTM 32N
                else:
                    assumed_crs = CRS.from_epsg(32633)  # UTM 33N
                print(f"  Assuming UTM coordinates: {assumed_crs}")
            else:
                print(f"  Cannot determine CRS from bounds: {src_bounds}")
                # Default to geographic
                assumed_crs = CRS.from_epsg(4326)
                print(f"  Defaulting to: {assumed_crs}")

            # Create new dataset with assumed CRS
            profile = src.profile.copy()
            profile['crs'] = assumed_crs

            # Save temporary file with CRS
            temp_path = output_path.with_suffix('.temp.tif')
            with rasterio.open(temp_path, 'w', **profile) as temp_dst:
                temp_dst.write(data, 1)

            # Now reproject the temporary file
            result = self.reproject_dem(temp_path, output_path, bounds)

            # Clean up
            temp_path.unlink()

            return result

    def extract_values_at_points(self, dem_path, points_gdf):
        """Extract DEM values at SAOCOM point locations"""
        print(f"\nExtracting DEM values at point locations...")

        with rasterio.open(dem_path) as src:
            # Extract values at point locations
            values = []
            for idx, point in points_gdf.iterrows():
                x, y = point.geometry.x, point.geometry.y
                row, col = src.index(x, y)

                try:
                    value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                    if value == src.nodata:
                        value = np.nan
                except:
                    value = np.nan

                values.append(value)

        points_gdf['DEM_VALUE'] = values
        valid_count = points_gdf['DEM_VALUE'].notna().sum()

        print(f"  Extracted values for {valid_count}/{len(points_gdf)} points")

        return points_gdf
    def extract_values_at_points(self, dem_path, points_gdf):
        """Extract DEM values at SAOCOM point locations"""
        print(f"\nExtracting DEM values at point locations...")

        with rasterio.open(dem_path) as src:
            # Extract values at point locations
            values = []
            for idx, point in points_gdf.iterrows():
                x, y = point.geometry.x, point.geometry.y
                row, col = src.index(x, y)

                try:
                    value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    if value == src.nodata:
                        value = np.nan
                except:
                    value = np.nan

                values.append(value)

        points_gdf['DEM_VALUE'] = values
        valid_count = points_gdf['DEM_VALUE'].notna().sum()

        print(f"  Extracted values for {valid_count}/{len(points_gdf)} points")

        return points_gdf

class AlignmentVerification:
    """Verify alignment between datasets"""

    def __init__(self, results_dir):
        self.results_dir = results_dir

    def plot_spatial_distribution(self, gdf, title="SAOCOM Points"):
        """Plot spatial distribution of points"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Geographic distribution
        ax1 = axes[0]
        scatter = ax1.scatter(gdf['X'], gdf['Y'], c=gdf['HEIGHT'],
                              cmap='terrain', s=10, alpha=0.6)
        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.set_title(f'{title} - Spatial Distribution')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Height (m)')

        # Height vs Coherence
        ax2 = axes[1]
        scatter2 = ax2.scatter(gdf['COHER'], gdf['HEIGHT'],
                               c=gdf['SIGMA_HEIGHT'], cmap='viridis',
                               s=10, alpha=0.6)
        ax2.set_xlabel('Coherence')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Height vs Coherence')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Sigma Height (m)')

        plt.tight_layout()

        # Fix filename by removing invalid characters
        safe_filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('>', 'gt').replace(
            '<', 'lt')
        output_path = self.results_dir / f"{safe_filename}_distribution.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"  Saved plot to {output_path}")

    def compare_with_reference(self, gdf, dem_name="Reference"):
        """Compare SAOCOM heights with reference DEM"""
        if 'DEM_VALUE' not in gdf.columns:
            print(f"  No {dem_name} values available for comparison")
            return

        # Remove NaN values
        valid_mask = gdf['DEM_VALUE'].notna()
        gdf_valid = gdf[valid_mask].copy()

        if len(gdf_valid) == 0:
            print(f"  No valid {dem_name} values for comparison")
            return

        # Calculate differences
        gdf_valid['DIFF'] = gdf_valid['HEIGHT'] - gdf_valid['DEM_VALUE']

        # Statistics
        stats = {
            'Mean Difference': gdf_valid['DIFF'].mean(),
            'Std Deviation': gdf_valid['DIFF'].std(),
            'RMSE': np.sqrt((gdf_valid['DIFF']**2).mean()),
            'Median Difference': gdf_valid['DIFF'].median(),
            'MAD': (gdf_valid['DIFF'] - gdf_valid['DIFF'].median()).abs().median()
        }

        print(f"\n{dem_name} Comparison Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f} m")

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(gdf_valid['DEM_VALUE'], gdf_valid['HEIGHT'],
                   c=gdf_valid['COHER'], cmap='viridis', s=5, alpha=0.5)
        ax1.plot([gdf_valid['DEM_VALUE'].min(), gdf_valid['DEM_VALUE'].max()],
                [gdf_valid['DEM_VALUE'].min(), gdf_valid['DEM_VALUE'].max()],
                'r--', alpha=0.5, label='1:1 line')
        ax1.set_xlabel(f'{dem_name} Height (m)')
        ax1.set_ylabel('SAOCOM Height (m)')
        ax1.set_title(f'SAOCOM vs {dem_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of differences
        ax2 = axes[1]
        ax2.hist(gdf_valid['DIFF'], bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Height Difference (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of Differences\nMean: {stats["Mean Difference"]:.2f}m')
        ax2.grid(True, alpha=0.3)

        # Spatial distribution of differences
        ax3 = axes[2]
        scatter = ax3.scatter(gdf_valid['X'], gdf_valid['Y'],
                            c=gdf_valid['DIFF'], cmap='RdBu_r',
                            s=5, alpha=0.6, vmin=-10, vmax=10)
        ax3.set_xlabel('Easting (m)')
        ax3.set_ylabel('Northing (m)')
        ax3.set_title('Spatial Distribution of Differences')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Difference (m)')

        plt.tight_layout()
        output_path = self.results_dir / f"saocom_vs_{dem_name.lower()}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

        return stats

def main():
    """Main processing workflow for Week 2"""

    print("=" * 80)
    print("WEEK 2: COORDINATE SYSTEM STANDARDIZATION")
    print("=" * 80)

    # Initialize processors
    saocom_processor = SaocomPointProcessor(CONFIG)
    dem_processor = ReferenceDataProcessor(CONFIG)
    verifier = AlignmentVerification(RESULTS_DIR)

    # Step 1: Process SAOCOM point data
    print("\n1. PROCESSING SAOCOM POINT DATA")
    print("-" * 40)

    # Look for SAOCOM data files
    saocom_files = list(DATA_DIR.glob("verona*")) + list(DATA_DIR.glob("*.txt"))

    if not saocom_files:
        print("ERROR: No SAOCOM data files found in data directory")
        print(f"Looking in: {DATA_DIR}")
        return

    # Process the first SAOCOM file found
    saocom_file = saocom_files[0]
    print(f"Processing: {saocom_file}")

    # Read and process SAOCOM points
    saocom_df = saocom_processor.read_saocom_points(saocom_file)
    saocom_gdf = saocom_processor.points_to_geodataframe(saocom_df)

    # Apply coherence mask
    saocom_gdf_masked = saocom_processor.apply_coherence_mask(saocom_gdf, threshold=0.3)

    # Save processed points
    output_gpkg = RESULTS_DIR / "saocom_points_projected.gpkg"
    saocom_gdf_masked.to_file(output_gpkg, driver="GPKG")
    print(f"\nSaved processed points to: {output_gpkg}")

    # Visualize spatial distribution
    print("\n2. VISUALIZING SPATIAL DISTRIBUTION")
    print("-" * 40)
    verifier.plot_spatial_distribution(saocom_gdf_masked, "SAOCOM Points (Coherence > 0.3)")

    # Step 2: Process reference DEMs (if available)
    print("\n3. PROCESSING REFERENCE DEMS")
    print("-" * 40)

    # Look for DEM files
    dem_patterns = {
        'TINITALY': ['*tinitaly*.tif', '*TINITALY*.tif'],
        'Copernicus': ['*copernicus*.tif', '*GLO30*.tif', '*cop30*.tif'],
        'LiDAR': ['*lidar*.tif', '*LIDAR*.tif']
    }

    processed_dems = {}

    for dem_name, patterns in dem_patterns.items():
        print('dem', dem_name)
        dem_files = []
        for pattern in patterns:
            dem_files.extend(DATA_DIR.glob(pattern))

        if dem_files:
            print(f"\nProcessing {dem_name} DEM...")
            input_dem = dem_files[0]
            output_dem = RESULTS_DIR / f"{dem_name.lower()}_utm32n_10m.tif"

            # Get bounds from SAOCOM points for cropping
            bounds = (saocom_gdf_masked.total_bounds[0] - 1000,
                     saocom_gdf_masked.total_bounds[1] - 1000,
                     saocom_gdf_masked.total_bounds[2] + 1000,
                     saocom_gdf_masked.total_bounds[3] + 1000)

            try:
                # Reproject DEM
                dem_processor.reproject_dem(input_dem, output_dem, bounds)
                processed_dems[dem_name] = output_dem

                # Extract values at SAOCOM points
                saocom_gdf_with_dem = dem_processor.extract_values_at_points(
                    output_dem, saocom_gdf_masked.copy()
                )

                # Compare with SAOCOM
                stats = verifier.compare_with_reference(saocom_gdf_with_dem, dem_name)

                # Save comparison results
                comparison_file = RESULTS_DIR / f"saocom_vs_{dem_name.lower()}_points.gpkg"
                saocom_gdf_with_dem.to_file(comparison_file, driver="GPKG")

            except Exception as e:
                print(f"  ERROR processing {dem_name}: {e}")
        else:
            print(f"\n{dem_name} DEM not found in data directory")

    # Step 3: Create summary report
    print("\n4. CREATING SUMMARY REPORT")
    print("-" * 40)

    report_path = RESULTS_DIR / "alignment_report.txt"
    with open(report_path, 'w') as f:
        f.write("WEEK 2: COORDINATE SYSTEM STANDARDIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Target CRS: {CONFIG['target_crs']}\n")
        f.write(f"  Target Resolution: {CONFIG['target_resolution']}m\n")
        f.write(f"  Vertical Datum: {CONFIG['vertical_datum']}\n\n")

        f.write("SAOCOM Data Summary:\n")
        f.write(f"  Total Points: {len(saocom_gdf)}\n")
        f.write(f"  Points after coherence mask (>0.3): {len(saocom_gdf_masked)}\n")
        f.write(f"  Coverage area: {(saocom_gdf_masked.total_bounds[2]-saocom_gdf_masked.total_bounds[0])/1000:.1f} x "
                f"{(saocom_gdf_masked.total_bounds[3]-saocom_gdf_masked.total_bounds[1])/1000:.1f} km\n\n")

        f.write("Processed DEMs:\n")
        for dem_name, dem_path in processed_dems.items():
            f.write(f"  - {dem_name}: {dem_path.name}\n")

        f.write("\nAlignment verified and standardization complete.\n")

    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 80)
    print("WEEK 2 PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("\nNext steps for Week 3:")
    print("  1. Implement coherence-based quality control")
    print("  2. Analyze temporal coherence distribution")
    print("  3. Create comprehensive difference maps")

if __name__ == "__main__":
    main()