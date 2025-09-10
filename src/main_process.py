import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

# Get the script's directory and navigate to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from src/
os.chdir(project_root)

# Define relative paths
DATA_DIR = "data"
RESULTS_DIR = "results"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def setup_directories():
    """Create necessary directories"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


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


def check_data_structure():
    """Check what data files you have"""
    print("=== Data Inventory ===")
    print(f"Working from: {os.getcwd()}")

    if os.path.exists(DATA_DIR):
        print(f"\nContents of {DATA_DIR}/:")
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isfile(item_path):
                size_kb = os.path.getsize(item_path) / 1024
                print(f"  File: {item} ({size_kb:.1f} KB)")
            elif os.path.isdir(item_path):
                print(f"  Folder: {item}/")
                # List contents of subfolders
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    - {subitem}")
                except PermissionError:
                    print("    (access denied)")
    else:
        print(f"No '{DATA_DIR}' folder found")


def find_data_files():
    """Automatically find relevant data files"""
    files = {}

    if not os.path.exists(DATA_DIR):
        return files

    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isfile(item_path):
            item_lower = item.lower()
            if item_lower.endswith('.csv') and 'verona' in item_lower:
                files['saocom_points'] = item_path
            elif item_lower.endswith('.tif') and 'cop' in item_lower:
                files['copernicus'] = item_path
            elif item_lower.endswith('.tif') and any(x in item_lower for x in ['tinitaly', 'tini']):
                files['tinitaly'] = item_path

    return files


def load_saocom_csv(csv_file):
    """Load SAOCOM CSV with proper parsing"""

    # First, let's check what the file actually looks like
    with open(csv_file, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(5)]

    print("First few lines of CSV:")
    for i, line in enumerate(first_lines):
        print(f"  {i}: {line}")

    # Try different parsing approaches
    try:
        # Method 1: Standard comma delimiter
        df = pd.read_csv(csv_file, sep=',')
        if len(df.columns) > 1:
            return df
    except:
        pass

    try:
        # Method 2: Tab delimiter
        df = pd.read_csv(csv_file, sep='\t')
        if len(df.columns) > 1:
            return df
    except:
        pass

    try:
        # Method 3: Force comma separation and clean up
        df = pd.read_csv(csv_file, sep=',', engine='python')

        # If still one column, manually split
        if len(df.columns) == 1:
            col_name = df.columns[0]
            # Split the column name by commas to get proper headers
            headers = col_name.split(',')

            # Split each row by commas
            data_rows = []
            for _, row in df.iterrows():
                row_data = str(row.iloc[0]).split(',')
                data_rows.append(row_data)

            # Create new dataframe with proper structure
            df = pd.DataFrame(data_rows, columns=headers)

            # Convert numeric columns
            numeric_cols = ['SVET', 'LVET', 'LAT', 'LON', 'HEIGHT', 'HEIGHT WRT DEM', 'SIGMA HEIGHT', 'COHER']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

    return df


def process_saocom_data():
    """Process your specific dataset"""

    setup_directories()

    # Find data files automatically
    files = find_data_files()

    if 'saocom_points' not in files:
        print("No SAOCOM CSV file found (looking for *verona*.csv)")
        return

    if 'copernicus' not in files:
        print("No Copernicus DEM found (looking for *cop*.tif)")
        return

    saocom_points = files['saocom_points']
    reference_dem = files['copernicus']

    print(f"\nUsing files:")
    print(f"  SAOCOM points: {saocom_points}")
    print(f"  Reference DEM: {reference_dem}")

    # Step 1: Load and analyze SAOCOM points with proper parsing
    df = load_saocom_csv(saocom_points)

    if df is None:
        print("Failed to parse CSV file")
        return

    print(f"\n=== CSV Structure ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    # Verify we have the required columns
    required_cols = ['LAT', 'LON', 'HEIGHT', 'COHER']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return

    print(f"\n=== SAOCOM Data Analysis ===")
    print(f"Total points: {len(df)}")
    print(f"Height range: {df['HEIGHT'].min():.1f} to {df['HEIGHT'].max():.1f} m")
    print(f"Coherence range: {df['COHER'].min():.2f} to {df['COHER'].max():.2f}")
    print(f"Mean coherence: {df['COHER'].mean():.2f}")

    # Continue with rest of the processing...

    # Check for negative elevations
    negative_count = (df['HEIGHT'] < 0).sum()
    if negative_count > 0:
        print(f"WARNING: {negative_count} points have negative elevations")
        print("This suggests potential datum issues that need investigation")

    # Step 2: Convert points to grid
    output_grid = os.path.join(PROCESSED_DIR, "saocom_verona_grid.tif")
    print(f"\nCreating gridded DEM...")
    points_to_grid(saocom_points, reference_dem, output_grid)
    print(f"Created: {output_grid}")

    # Step 3: Quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Point locations colored by height
    scatter = axes[0, 0].scatter(df['LON'], df['LAT'], c=df['HEIGHT'], s=1, cmap='terrain')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('SAOCOM Point Locations')
    plt.colorbar(scatter, ax=axes[0, 0], label='Height (m)')

    # # Height distribution
    # axes[0, 1].hist(df['HEIGHT'], bins=30, alpha=0.7, color='blue')
    # axes[0, 1].set_xlabel('Height (m)')
    # axes[0, 1].set_ylabel('Count')
    # axes[0, 1].set_title('Height Distribution')
    #
    # # Coherence distribution
    # axes[1, 0].hist(df['COHER'], bins=20, alpha=0.7, color='green')
    # axes[1, 0].set_xlabel('Coherence')
    # axes[1, 0].set_ylabel('Count')
    # axes[1, 0].set_title('Coherence Distribution')
    #
    # # Height vs Coherence
    # axes[1, 1].scatter(df['COHER'], df['HEIGHT'], alpha=0.5, s=1)
    # axes[1, 1].set_xlabel('Coherence')
    # axes[1, 1].set_ylabel('Height (m)')
    # axes[1, 1].set_title('Height vs Coherence')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "saocom_overview.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot: {plot_path}")
    plt.show()


if __name__ == "__main__":
    print("=== SAOCOM Week 2 Processing ===")
    check_data_structure()
    process_saocom_data()