"""
Simple script to run Week 2 alignment step by step
Run this from the main saocom_project folder
"""

import os
import sys
from pathlib import Path

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# Set up paths
if current_dir.name != 'saocom_project':
    print("WARNING: Not in saocom_project directory!")
    print("Please navigate to C:\\Users\\colto\\OneDrive\\Music\\Documents\\GitHub\\saocom_project")
    sys.exit(1)

# Create necessary directories
data_dir = Path("data")
results_dir = Path("results") / "week2_alignment"
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {data_dir.absolute()}")
print(f"Results directory: {results_dir.absolute()}")

# List available data files
print("\n" + "=" * 60)
print("AVAILABLE DATA FILES:")
print("=" * 60)

print("\nSAOCOM files:")
saocom_files = list(data_dir.glob("verona*")) + \
               list(data_dir.glob("w*.tif")) + \
               list(data_dir.glob("*.txt"))
for f in saocom_files:
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

print("\nPotential DEM files:")
dem_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.TIF"))
for f in dem_files:
    if 'verona' not in f.name.lower() and 'w' not in f.name[0].lower():
        print(f"  - {f.name}")

# Quick data inspection script
print("\n" + "=" * 60)
print("QUICK DATA INSPECTION")
print("=" * 60)

# Try to read a sample of the SAOCOM data
if saocom_files:
    import pandas as pd

    # Find text file with point data
    txt_files = [f for f in saocom_files if f.suffix == '.txt' or 'verona' in f.name]

    if txt_files:
        sample_file = txt_files[0]
        print(f"\nReading sample from: {sample_file.name}")

        try:
            # Try to read first few lines
            with open(sample_file, 'r') as f:
                lines = f.readlines()[:20]

            print("\nFirst 20 lines of file:")
            for i, line in enumerate(lines):
                print(f"{i + 1:3}: {line.rstrip()}")

            # Try to parse as data
            print("\nAttempting to parse as tabular data...")
            columns = ['ID', 'SVET', 'LVET', 'LAT', 'LON', 'HEIGHT',
                       'HEIGHT_WRT_DEM', 'SIGMA_HEIGHT', 'COHER']

            try:
                # Skip header lines that don't contain data
                skip_rows = 0
                for i, line in enumerate(lines):
                    if line.strip() and not any(c.isdigit() for c in line.split()[0]):
                        skip_rows += 1
                    else:
                        break

                df = pd.read_csv(sample_file, sep=r',+', names=columns,
                                 skiprows=skip_rows, nrows=10)
                print("\nSuccessfully parsed data:")
                print(df)

                print(f"\nData statistics:")
                print(f"  Latitude range: {df['LAT'].min():.4f} - {df['LAT'].max():.4f}")
                print(f"  Longitude range: {df['LON'].min():.4f} - {df['LON'].max():.4f}")
                print(f"  Height range: {df['HEIGHT'].min():.1f} - {df['HEIGHT'].max():.1f} m")
                print(f"  Coherence range: {df['COHER'].min():.2f} - {df['COHER'].max():.2f}")

            except Exception as e:
                print(f"Could not parse as expected format: {e}")

        except Exception as e:
            print(f"Error reading file: {e}")

# Instructions for running main script
print("\n" + "=" * 60)
print("TO RUN FULL WEEK 2 ANALYSIS:")
print("=" * 60)
print("""
1. Make sure you have installed required packages:
   pip install numpy pandas geopandas rasterio matplotlib shapely

2. From the saocom_project directory, run:
   python week2_coordinate_alignment.py

3. The script will:
   - Read SAOCOM point data
   - Convert to UTM Zone 32N projection
   - Apply coherence masking
   - Process any available reference DEMs
   - Create comparison plots and statistics
   - Save all results to results/week2_alignment/

4. If you have reference DEMs (TINITALY, Copernicus, LiDAR):
   - Name them with keywords: 'tinitaly', 'copernicus', 'glo30', 'lidar'
   - Place them in the data/ folder
   - They will be automatically detected and processed

5. Check the results folder for:
   - Processed point data (GeoPackage format)
   - Aligned DEMs (if available)
   - Comparison plots
   - Summary statistics report
""")

print("\nReady to proceed? Save the main script as 'week2_coordinate_alignment.py'")
print("in your saocom_project folder and run it!")