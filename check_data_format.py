"""
Quick diagnostic script to check SAOCOM data format
Run this to understand your data structure before processing
"""

import pandas as pd
from pathlib import Path
import sys


def diagnose_file(filepath):
    """Diagnose the format of a data file"""
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSING: {filepath.name}")
    print('=' * 60)

    # Read first few lines raw
    print("\n1. RAW FILE CONTENT (first 10 lines):")
    print("-" * 40)
    with open(filepath, 'r') as f:
        for i in range(10):
            line = f.readline()
            if not line:
                break
            print(f"Line {i + 1}: {repr(line.rstrip())}")

    # Try different separators
    separators = {
        'comma': ',',
        'tab': '\t',
        'space': ' ',
        'multiple_spaces': r'\s+',
        'semicolon': ';',
        'pipe': '|'
    }

    print("\n2. TESTING DIFFERENT SEPARATORS:")
    print("-" * 40)

    successful_reads = []

    for sep_name, sep in separators.items():
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=5)
            if len(df.columns) > 1:  # Successfully split into multiple columns
                print(f"\n✓ {sep_name}: Successfully read {len(df.columns)} columns")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
                successful_reads.append((sep_name, sep, df))
            else:
                print(f"✗ {sep_name}: Only 1 column detected")
        except Exception as e:
            print(f"✗ {sep_name}: Failed - {str(e)[:50]}")

    # Find best separator (most columns)
    if successful_reads:
        best = max(successful_reads, key=lambda x: len(x[2].columns))
        sep_name, sep, df = best

        print(f"\n3. BEST SEPARATOR: {sep_name}")
        print("-" * 40)
        print(f"Detected {len(df.columns)} columns")
        print("\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. '{col}'")

        print("\nFirst 5 rows:")
        print(df.to_string())

        # Try to identify data types
        print("\n4. DATA TYPE ANALYSIS:")
        print("-" * 40)
        for col in df.columns:
            try:
                # Try to convert to numeric
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                num_valid = numeric_vals.notna().sum()
                if num_valid > 0:
                    print(f"  {col}: NUMERIC (range: {numeric_vals.min():.2f} to {numeric_vals.max():.2f})")
                else:
                    print(f"  {col}: TEXT/OTHER")
            except:
                print(f"  {col}: TEXT/OTHER")

        # Generate reading code
        print("\n5. SUGGESTED READING CODE:")
        print("-" * 40)
        if sep == ',':
            print(f"df = pd.read_csv('{filepath.name}')")
        elif sep == '\t':
            print(f"df = pd.read_csv('{filepath.name}', sep='\\t')")
        elif sep == r'\s+':
            print(f"df = pd.read_csv('{filepath.name}', sep=r'\\s+')")
        else:
            print(f"df = pd.read_csv('{filepath.name}', sep='{sep}')")

        # Check for specific SAOCOM columns
        print("\n6. SAOCOM DATA CHECK:")
        print("-" * 40)
        expected_cols = ['LAT', 'LON', 'HEIGHT', 'COHER']

        found_cols = []
        for expected in expected_cols:
            for col in df.columns:
                if expected.lower() in col.lower():
                    found_cols.append((expected, col))
                    break

        if len(found_cols) >= 3:
            print("✓ Appears to be SAOCOM point data!")
            print("  Found columns:", found_cols)
        else:
            print("? May not be SAOCOM point data or columns need mapping")
            print("  Looking for: LAT, LON, HEIGHT, COHER")
            print("  Found in file:", list(df.columns))

    else:
        print("\n❌ Could not automatically detect separator")
        print("File might be:")
        print("  - Binary format (not text)")
        print("  - Fixed-width format")
        print("  - Custom format requiring special parsing")


def main():
    # Get data directory
    data_dir = Path("data")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir.absolute()}")
        print("Make sure you're running from the saocom_project directory")
        sys.exit(1)

    print("=" * 60)
    print("SAOCOM DATA FORMAT DIAGNOSTIC")
    print("=" * 60)

    # Find potential SAOCOM files
    patterns = ['verona*', '*.txt', '*.csv', '*.dat']
    files_found = []

    for pattern in patterns:
        files_found.extend(data_dir.glob(pattern))

    # Remove duplicates
    files_found = list(set(files_found))

    if not files_found:
        print("\nNo potential SAOCOM data files found in data directory")
        print(f"Searched for: {patterns}")
        return

    print(f"\nFound {len(files_found)} potential data file(s):")
    for i, f in enumerate(files_found, 1):
        print(f"  {i}. {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Analyze each file
    for f in files_found:
        if f.suffix.lower() not in ['.tif', '.tiff', '.jpg', '.png']:  # Skip image files
            diagnose_file(f)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nBased on the analysis above:")
    print("1. Identify the correct separator for your data")
    print("2. Check if column names need mapping")
    print("3. Update the main processing script if needed")


if __name__ == "__main__":
    main()