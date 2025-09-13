"""
SAOCOM Datum Analysis Tool
Diagnose elevation reference issues and convert between ellipsoidal/orthometric heights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from scipy.interpolate import griddata


def analyze_saocom_data(csv_path):
    """
    Analyze SAOCOM data to understand the elevation reference
    """
    print("=" * 70)
    print("SAOCOM ELEVATION DATUM ANALYSIS")
    print("=" * 70)

    # Read the data
    print(f"\nReading data from: {csv_path}")

    # Try different reading strategies
    try:
        df = pd.read_csv(csv_path)
        print(f"  Columns found: {list(df.columns)}")
    except:
        # Try with different separators
        for sep in [',', '\t', r'\s+']:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                break
            except:
                continue

    # Clean column names
    df.columns = [col.strip().upper() for col in df.columns]

    # Map column variations
    column_mapping = {
        'LAT': 'LAT', 'LATITUDE': 'LAT',
        'LON': 'LON', 'LONGITUDE': 'LON', 'LONG': 'LON',
        'HEIGHT': 'HEIGHT', 'ELEVATION': 'HEIGHT', 'ELEV': 'HEIGHT',
        'HEIGHT_WRT_DEM': 'HEIGHT_WRT_DEM', 'HEIGHT WRT DEM': 'HEIGHT_WRT_DEM',
        'COHER': 'COHER', 'COHERENCE': 'COHER'
    }
    df = df.rename(columns=column_mapping)

    # Convert to numeric
    for col in ['LAT', 'LON', 'HEIGHT', 'COHER']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean data
    df = df.dropna(subset=['LAT', 'LON', 'HEIGHT'])

    print(f"\nData Summary:")
    print(f"  Valid points: {len(df)}")
    print(f"  Lat range: {df['LAT'].min():.4f} to {df['LAT'].max():.4f}")
    print(f"  Lon range: {df['LON'].min():.4f} to {df['LON'].max():.4f}")
    print(f"  Height range: {df['HEIGHT'].min():.1f} to {df['HEIGHT'].max():.1f} m")

    if 'COHER' in df.columns:
        print(f"  Coherence range: {df['COHER'].min():.3f} to {df['COHER'].max():.3f}")

    if 'HEIGHT_WRT_DEM' in df.columns:
        df['HEIGHT_WRT_DEM'] = pd.to_numeric(df['HEIGHT_WRT_DEM'], errors='coerce')
        print(f"  Height w.r.t DEM range: {df['HEIGHT_WRT_DEM'].min():.1f} to {df['HEIGHT_WRT_DEM'].max():.1f} m")

    return df


def get_geoid_height(lat, lon, model='egm96'):
    """
    Get geoid height at a point using online calculator
    Note: This is a simplified approach - for production use a proper geoid model
    """
    # For Verona area, approximate geoid-ellipsoid separation
    # This is a rough approximation - use proper geoid models for accurate work
    if model == 'egm96':
        # Approximate values for northern Italy
        return 47.0  # meters (EGM96 geoid height above WGS84 ellipsoid)
    elif model == 'egm2008':
        return 47.2  # meters (EGM2008 geoid height above WGS84 ellipsoid)
    else:
        return 47.0


def diagnose_elevation_reference(df):
    """
    Try to determine what elevation reference the SAOCOM data uses
    """
    print("\n" + "=" * 50)
    print("ELEVATION REFERENCE DIAGNOSIS")
    print("=" * 50)

    # Get sample coordinates for geoid calculation
    center_lat = df['LAT'].mean()
    center_lon = df['LON'].mean()

    print(f"\nCenter coordinates: {center_lat:.4f}°N, {center_lon:.4f}°E")

    # Expected elevation for Verona area
    verona_elevation_range = (50, 200)  # meters above sea level
    print(
        f"Expected elevation range for Verona: {verona_elevation_range[0]}-{verona_elevation_range[1]} m above sea level")

    # Get geoid height
    geoid_height = get_geoid_height(center_lat, center_lon)
    print(f"Approximate geoid height (EGM96): {geoid_height:.1f} m")

    # Current height statistics
    height_stats = {
        'min': df['HEIGHT'].min(),
        'max': df['HEIGHT'].max(),
        'mean': df['HEIGHT'].mean(),
        'median': df['HEIGHT'].median(),
        'std': df['HEIGHT'].std()
    }

    print(f"\nCurrent HEIGHT column statistics:")
    for key, value in height_stats.items():
        print(f"  {key}: {value:.1f} m")

    # Test different hypotheses
    print(f"\n" + "-" * 30)
    print("TESTING HYPOTHESES:")
    print("-" * 30)

    # Hypothesis 1: Heights are ellipsoidal and need geoid correction
    corrected_heights_1 = df['HEIGHT'] - geoid_height
    print(f"\n1. If heights are ellipsoidal (subtract geoid height {geoid_height:.1f}m):")
    print(f"   Range: {corrected_heights_1.min():.1f} to {corrected_heights_1.max():.1f} m")
    print(f"   Mean: {corrected_heights_1.mean():.1f} m")
    in_range_1 = ((corrected_heights_1 >= verona_elevation_range[0]) &
                  (corrected_heights_1 <= verona_elevation_range[1] * 2)).sum()
    print(f"   Points in reasonable range: {in_range_1}/{len(df)} ({in_range_1 / len(df) * 100:.1f}%)")

    # Hypothesis 2: Heights are relative to some reference and need offset
    offsets_to_test = [100, 150, 200, 250, 300]
    best_offset = None
    best_score = 0

    print(f"\n2. Testing constant offsets to get realistic elevations:")
    for offset in offsets_to_test:
        corrected_heights = df['HEIGHT'] + offset
        in_range = ((corrected_heights >= verona_elevation_range[0]) &
                    (corrected_heights <= verona_elevation_range[1] * 2)).sum()
        score = in_range / len(df)
        print(f"   Offset +{offset}m: {corrected_heights.min():.1f} to {corrected_heights.max():.1f} m, "
              f"{in_range}/{len(df)} in range ({score * 100:.1f}%)")
        if score > best_score:
            best_score = score
            best_offset = offset

    # Hypothesis 3: Check if HEIGHT_WRT_DEM makes sense
    if 'HEIGHT_WRT_DEM' in df.columns:
        print(f"\n3. Analyzing HEIGHT_WRT_DEM column:")
        wrt_dem_stats = {
            'min': df['HEIGHT_WRT_DEM'].min(),
            'max': df['HEIGHT_WRT_DEM'].max(),
            'mean': df['HEIGHT_WRT_DEM'].mean(),
            'median': df['HEIGHT_WRT_DEM'].median()
        }
        for key, value in wrt_dem_stats.items():
            print(f"   {key}: {value:.1f} m")

        # Check if HEIGHT + some reference gives reasonable values
        if abs(df['HEIGHT_WRT_DEM'].mean()) < 50:  # Reasonable relative heights
            print("   HEIGHT_WRT_DEM appears to be relative differences from a reference DEM")
            print("   Main HEIGHT column might need a reference elevation added")

    # Recommendations
    print(f"\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)

    if best_offset and best_score > 0.8:
        print(f"\n✓ LIKELY SOLUTION: Add constant offset of {best_offset}m to HEIGHT values")
        print(f"  This puts {best_score * 100:.1f}% of points in reasonable elevation range")
        print(f"  Corrected range: {df['HEIGHT'].min() + best_offset:.1f} to {df['HEIGHT'].max() + best_offset:.1f} m")
        return 'offset', best_offset

    elif corrected_heights_1.mean() > 0 and ((corrected_heights_1 >= 0) & (corrected_heights_1 <= 1000)).sum() / len(
            df) > 0.8:
        print(f"\n✓ LIKELY SOLUTION: Heights are ellipsoidal, subtract geoid height ({geoid_height:.1f}m)")
        print(f"  This converts from WGS84 ellipsoid to orthometric heights")
        return 'ellipsoidal', geoid_height

    else:
        print(f"\n⚠ UNCLEAR: Need more investigation")
        print(f"  - Check SAOCOM processing documentation")
        print(f"  - Verify what reference DEM was used in processing")
        print(f"  - Consider that heights might be relative to a local reference")
        return 'unknown', None


def convert_heights(df, conversion_type, value):
    """
    Apply height conversion based on diagnosis
    """
    df_corrected = df.copy()

    if conversion_type == 'offset':
        df_corrected['HEIGHT_CORRECTED'] = df['HEIGHT'] + value
        print(f"\nApplied constant offset: +{value} m")
    elif conversion_type == 'ellipsoidal':
        df_corrected['HEIGHT_CORRECTED'] = df['HEIGHT'] - value
        print(f"\nConverted ellipsoidal to orthometric: -{value} m (geoid correction)")
    else:
        df_corrected['HEIGHT_CORRECTED'] = df['HEIGHT']
        print(f"\nNo correction applied - needs manual investigation")

    return df_corrected


def plot_elevation_analysis(df, df_corrected=None):
    """
    Create diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Spatial distribution of original heights
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['LON'], df['LAT'], c=df['HEIGHT'],
                           cmap='RdBu_r', s=1, alpha=0.6)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Original HEIGHT Values')
    plt.colorbar(scatter1, ax=ax1, label='Height (m)')

    # Plot 2: Height histogram
    ax2 = axes[0, 1]
    ax2.hist(df['HEIGHT'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Height (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Original Height Distribution')
    ax2.grid(True, alpha=0.3)

    if df_corrected is not None and 'HEIGHT_CORRECTED' in df_corrected.columns:
        # Plot 3: Corrected heights spatial
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(df_corrected['LON'], df_corrected['LAT'],
                               c=df_corrected['HEIGHT_CORRECTED'],
                               cmap='terrain', s=1, alpha=0.6)
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('Corrected HEIGHT Values')
        plt.colorbar(scatter3, ax=ax3, label='Height (m)')

        # Plot 4: Corrected height histogram
        ax4 = axes[1, 1]
        ax4.hist(df_corrected['HEIGHT_CORRECTED'], bins=50, alpha=0.7,
                 edgecolor='black', color='green')
        ax4.set_xlabel('Height (m)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Corrected Height Distribution')
        ax4.grid(True, alpha=0.3)
    else:
        # Plot 3: HEIGHT vs HEIGHT_WRT_DEM if available
        if 'HEIGHT_WRT_DEM' in df.columns:
            ax3 = axes[1, 0]
            ax3.scatter(df['HEIGHT'], df['HEIGHT_WRT_DEM'], alpha=0.5, s=1)
            ax3.set_xlabel('HEIGHT (m)')
            ax3.set_ylabel('HEIGHT_WRT_DEM (m)')
            ax3.set_title('HEIGHT vs HEIGHT_WRT_DEM')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Coherence vs Height if available
        if 'COHER' in df.columns:
            ax4 = axes[1, 1]
            ax4.scatter(df['COHER'], df['HEIGHT'], alpha=0.5, s=1)
            ax4.set_xlabel('Coherence')
            ax4.set_ylabel('HEIGHT (m)')
            ax4.set_title('Coherence vs HEIGHT')
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main analysis workflow
    """
    # Look for SAOCOM data files
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: 'data' directory not found!")
        return

    # Find potential SAOCOM files
    saocom_files = []
    for pattern in ['*.txt', '*.csv', '*verona*', '*.dat']:
        saocom_files.extend(data_dir.glob(pattern))

    if not saocom_files:
        print("ERROR: No SAOCOM data files found!")
        return

    # Use first file found
    csv_path = saocom_files[0]
    print(f"Analyzing: {csv_path}")

    # Step 1: Read and analyze data
    df = analyze_saocom_data(csv_path)

    # Step 2: Diagnose elevation reference
    conversion_type, value = diagnose_elevation_reference(df)

    # Step 3: Apply correction
    df_corrected = convert_heights(df, conversion_type, value)

    # Step 4: Create plots
    print(f"\nCreating diagnostic plots...")
    plot_elevation_analysis(df, df_corrected)

    # Step 5: Save results
    output_dir = Path("results/elevation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save corrected data
    output_file = output_dir / "saocom_heights_corrected.csv"
    df_corrected.to_csv(output_file, index=False)
    print(f"\nSaved corrected data to: {output_file}")

    # Create summary report
    report_file = output_dir / "elevation_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("SAOCOM ELEVATION DATUM ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {csv_path}\n")
        f.write(f"Points analyzed: {len(df)}\n\n")
        f.write(f"Original height range: {df['HEIGHT'].min():.1f} to {df['HEIGHT'].max():.1f} m\n")
        if 'HEIGHT_CORRECTED' in df_corrected.columns:
            f.write(
                f"Corrected height range: {df_corrected['HEIGHT_CORRECTED'].min():.1f} to {df_corrected['HEIGHT_CORRECTED'].max():.1f} m\n")
        f.write(f"\nDiagnosis: {conversion_type}\n")
        if value:
            f.write(f"Correction applied: {value:.1f} m\n")
        f.write(f"\nNext steps:\n")
        f.write("1. Verify correction makes sense for your study area\n")
        f.write("2. Use HEIGHT_CORRECTED column for DEM comparison\n")
        f.write("3. Check against reference DEMs to validate\n")

    print(f"Saved analysis report to: {report_file}")

    print(f"\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Check the plots and report to understand your elevation reference!")


if __name__ == "__main__":
    main()