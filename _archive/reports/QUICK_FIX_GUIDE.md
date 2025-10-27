# Quick Fix Applied ✓

## What Was Fixed

1. ✅ Added `load_dem_array` function to `src/utils.py` with correct bounds checking
2. ✅ Updated `saocom_analysis_clean.ipynb` to import `load_dem_array`
3. ✅ Fixed the ValueError: "truth value of an array is ambiguous"

## How to Use the Fixed Function

### In Your Notebook

The function is now imported automatically in **Cell 2**:

```python
from utils import read_raster_meta, load_dem_array  # ✓ Already added
```

### Usage Example

Replace your current code (Cell 9 or wherever you defined `load_dem_array`) with:

```python
# DELETE any existing definition of load_dem_array

# The function is already imported, just use it:
bounds = saocom_gdf.total_bounds  # This is a numpy array

# Load DEM cropped to bounds
dem, transform = load_dem_array(tinitaly_path, bounds)
print(f"✓ DEM loaded: {dem.shape}")

# Or load full DEM without cropping:
dem_full, transform_full = load_dem_array(tinitaly_path)
```

## What Changed in the Fix

**Before (WRONG):**
```python
def load_dem_array(path, bounds):
    with rasterio.open(path) as src:
        if bounds:  # ❌ ERROR: Can't check numpy array as boolean
            ...
```

**After (CORRECT):**
```python
def load_dem_array(path, bounds=None):
    with rasterio.open(path) as src:
        if bounds is not None:  # ✓ FIXED: Explicit None check
            ...
```

## Test It

Run this in a new cell to verify the fix:

```python
# Test the fixed function
from src.utils import load_dem_array
from pathlib import Path

# Your paths
tinitaly_path = Path("data/tinitaly/tinitaly_crop.tif")

# This should now work without errors:
bounds = saocom_gdf.total_bounds
dem, transform = load_dem_array(tinitaly_path, bounds)

print(f"✓ Success! Loaded DEM with shape: {dem.shape}")
print(f"✓ Transform: {transform}")
```

## If You Still Get Errors

1. **Restart your Jupyter kernel** (Kernel → Restart)
2. **Re-run the imports cell** (Cell 2)
3. **Check your code** - make sure you're not redefining `load_dem_array` anywhere

## Your Custom Code

If you have custom functions like `analyze_dem_suitability`, make sure they use the imported function:

```python
def analyze_dem_suitability(dem_path, dem_name, saocom_gdf, grid_size):
    # Get bounds
    bounds = saocom_gdf.total_bounds

    # Use the imported function (no need to define it)
    dem, transform = load_dem_array(dem_path, bounds)  # ✓ Works now

    # ... rest of your analysis
```

## Summary

✅ **Fixed:** `src/utils.py` has correct `load_dem_array`
✅ **Updated:** Notebook imports the function
✅ **Ready:** You can use it without errors

**Just restart your kernel and re-run from the top!**
