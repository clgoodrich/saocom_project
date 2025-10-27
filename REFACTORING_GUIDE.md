# SAOCOM Project Refactoring Guide

## Overview

Your SAOCOM project has been refactored from a single, large Jupyter notebook (~3,959 lines of code) into a clean, modular structure. This makes the code:
- **Easier to read** - Clear organization by function
- **Easier to maintain** - Changes isolated to specific modules
- **Easier to test** - Each module can be tested independently
- **Easier to reuse** - Functions can be imported in other projects
- **Easier for students** - Clean notebook focuses on the analysis story

---

## What Changed

### Before Refactoring
```
saocom_project/
├── saocom_v3.ipynb           (48 MB, 3,959 lines)
├── saocom_v3_refactored.ipynb (9 MB)
└── data/
```

**Problems:**
- All function definitions mixed with analysis code
- Hard to find specific functions
- Difficult to understand the analysis flow
- Code duplication
- No dependency tracking (requirements.txt)

### After Refactoring
```
saocom_project/
├── src/                      # NEW: Organized modules
│   ├── __init__.py
│   ├── utils.py              # Raster I/O utilities
│   ├── preprocessing.py      # Resampling, masking, terrain analysis
│   ├── calibration.py        # Height calibration
│   ├── outlier_detection.py  # Outlier filtering
│   ├── statistics.py         # NMAD, stats, summaries
│   ├── landcover.py          # CORINE classification
│   └── visualization.py      # All plotting functions
│
├── saocom_analysis_clean.ipynb  # NEW: Clean, narrative notebook
├── requirements.txt             # NEW: Dependency list
├── CLAUDE.md                    # Project documentation for Claude
├── REFACTORING_GUIDE.md         # This file
│
├── saocom_v3.ipynb           # KEEP: Original notebook (backup)
└── data/                     # Same data structure
```

---

## Module Descriptions

### `src/utils.py`
**Purpose:** Basic utility functions for file I/O and metadata

**Functions:**
- `read_raster_meta(path)` - Extract CRS, resolution, bounds from raster

**When to use:** Getting basic information about raster files

---

### `src/preprocessing.py`
**Purpose:** Raster processing and terrain analysis

**Functions:**
- `resample_to_10m(...)` - Resample raster to 10m using cubic interpolation
- `mask_and_write(...)` - Apply spatial mask and save raster
- `sample_raster_at_points(...)` - Extract raster values at point locations
- `create_difference_grid(...)` - Grid point differences using nearest neighbor
- `calculate_terrain_derivatives(...)` - Compute slope and aspect (Horn's method)
- `prepare_analysis_dataframe(...)` - Filter and prepare data for analysis
- `calculate_suitability_index(...)` - InSAR suitability based on terrain
- `classify_terrain_suitability(...)` - Classify suitability into categories

**When to use:** Any raster manipulation, sampling, or terrain analysis

---

### `src/calibration.py`
**Purpose:** Convert SAOCOM relative heights to absolute heights

**Functions:**
- `calibrate_heights(gdf, ref_col, out_col, coherence_threshold=0.8)`
  - Uses median offset method
  - Returns (offset, rmse, n_points)

**When to use:** Always use this BEFORE calculating residuals!

**Important:** SAOCOM heights are relative. Calibration is required.

---

### `src/outlier_detection.py`
**Purpose:** Detect and remove anomalous measurements

**Functions:**
- `remove_isolated_knn(gdf, k=100, distance_threshold=1000)` - Spatial filtering
- `score_outliers_isolation_forest(gdf, residual_col, **kwargs)` - ML-based scoring
- `filter_by_score_iqr(gdf_scored, iqr_multiplier=1)` - IQR-based filtering
- `visualize_outlier_results(...)` - Create outlier visualization

**When to use:** After loading data and before statistical analysis

**Workflow:**
1. `remove_isolated_knn()` - Remove spatially isolated points
2. `score_outliers_isolation_forest()` - Score anomalies
3. `filter_by_score_iqr()` - Split into clean/outliers
4. `visualize_outlier_results()` - Visualize results

---

### `src/statistics.py`
**Purpose:** Calculate validation statistics

**Functions:**
- `nmad(x)` - Normalized Median Absolute Deviation (robust error metric)
- `calculate_nmad(series)` - Pandas version
- `calculate_height_stats(data, name)` - Comprehensive statistics dict
- `generate_summary_string(values)` - Compact summary text
- `generate_height_statistics_summary(gdf, gdf_name)` - Print full report

**When to use:** After outlier removal for final accuracy assessment

**Key metric:** NMAD is preferred over RMSE for InSAR validation (more robust to outliers)

---

### `src/landcover.py`
**Purpose:** CORINE Land Cover processing

**Functions:**
- `get_clc_level1(code)` - Map Level 3 code to Level 1 category

**When to use:** Analyzing accuracy by land cover type

**Categories:**
- 100-199: Artificial Surfaces
- 200-299: Agricultural Areas
- 300-399: Forest & Semi-Natural Areas
- 400-499: Wetlands
- 500-599: Water Bodies

---

### `src/visualization.py`
**Purpose:** Create publication-quality visualizations

**Functions:**
- `plot_raster_with_stats(...)` - Raster maps with statistics overlay
- `plot_gridded_panel(...)` - Gridded difference maps
- `plot_points_panel(...)` - Point-based maps
- `plot_distribution_histogram(...)` - Residual distributions
- `plot_scatter_comparison(...)` - 1:1 scatter plots
- `plot_hexbin_density(...)` - Hexbin density plots
- `plot_hist2d(...)` - 2D histogram plots
- `plot_bland_altman(...)` - Bland-Altman agreement plots

**When to use:** Creating figures for reports and publications

**Styling:** All functions use consistent styling (300 DPI, professional colors)

---

## How to Use the New Structure

### For Students

**Option 1: Run the Clean Notebook**
```bash
cd C:\Users\colto\Documents\GitHub\saocom_project
jupyter notebook saocom_analysis_clean.ipynb
```
Then run cells top-to-bottom. Everything is documented!

**Option 2: Use Modules in Your Own Notebook**

```python
# Your custom analysis
from src.calibration import calibrate_heights
from src.statistics_prog import nmad

# Load your data...
offset, rmse, n = calibrate_heights(my_gdf, 'ref_dem', 'abs_height')
error = nmad(residuals)
```

**Option 3: Extend the Modules**
Add your own functions to the modules:
```python
# In src/preprocessing.py
def your_custom_filter(gdf, threshold):
    """Your custom filtering logic"""
    return filtered_gdf
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
cd C:\Users\colto\Documents\GitHub\saocom_project
pip install -r requirements.txt
```

### 2. Verify Installation

```python
# Test imports
from src.utils import read_raster_meta
from src.calibration import calibrate_heights
from src.statistics_prog import nmad

print("✓ All modules loaded successfully")
```

### 3. Run Clean Notebook
```bash
jupyter notebook saocom_analysis_clean.ipynb
```

---

## Comparison: Old vs New

### Code Organization

**Old Notebook (saocom_v3.ipynb):**
```python
# Cell 1: Imports mixed with setup
import numpy as np
# ... 30+ import statements

# Cell 5: Function definition 1
def _read_raster_meta(path):
    ...

# Cell 7: Function definition 2
def remove_isolated_knn(...):
    ...

# Cell 13: Analysis code
saocom_gdf = pd.read_csv(...)

# Cell 25: More function definitions
def score_outliers_isolation_forest(...):
    ...

# Cell 41: More analysis
# ... hard to follow the flow!
```

**New Notebook (saocom_analysis_clean.ipynb):**

```python
# Cell 1: Markdown explanation
"""
## 1. Setup and Imports
Import all necessary libraries and custom modules.
"""

# Cell 2: Clean imports
from src.preprocessing import resample_to_10m
from src.calibration import calibrate_heights
from src.statistics_prog import nmad

# Cell 3: Markdown explanation
"""
## 2. Load SAOCOM Data
Load SAOCOM InSAR point measurements from CSV.
"""

# Cell 4: Clean analysis code
saocom_gdf = gpd.read_file(SAOCOM_CSV)
saocom_gdf = remove_isolated_knn(saocom_gdf)

# ... easy to follow the story!
```

### Function Documentation

**Old:**
```python
def nmad(x: np.ndarray) -> float:
    x = np.asarray(x)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))
```

**New (in src/statistics.py):**
```python
def nmad(x):
    """
    Calculate Normalized Median Absolute Deviation (NMAD).

    NMAD is a robust measure of dispersion that is less sensitive to outliers
    than standard deviation.

    Parameters
    ----------
    x : array-like
        Input data

    Returns
    -------
    float
        NMAD value

    Notes
    -----
    NMAD = 1.4826 * median(|x - median(x)|)

    The factor 1.4826 makes NMAD approximately equal to the standard deviation
    for normally distributed data.

    Examples
    --------
    >>> residuals = np.array([1.2, -0.5, 0.8, -1.0, 0.3])
    >>> nmad(residuals)
    1.06
    """
    x = np.asarray(x)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))
```

---

## Next Steps

### For You (Project Owner)
1. ✅ Review the clean notebook - make sure workflow matches your needs
2. Test the refactored code with your full dataset
3. Update any custom analysis you had in the original notebook
4. Consider adding unit tests for the modules
5. Share `saocom_analysis_clean.ipynb` with students

### For Students
1. Read `CLAUDE.md` for project overview
2. Run `saocom_analysis_clean.ipynb` top-to-bottom
3. Experiment with parameters (coherence threshold, outlier detection)
4. Create your own analysis notebooks importing from `src/`
5. Extend the modules with your own functions

### For Future Development
1. **Add unit tests:** Create `tests/` directory with pytest
2. **Add data loading module:** Create `src/data_loading.py`
3. **Add configuration file:** Create `config.yaml` for parameters
4. **Add CLI interface:** Create command-line script for batch processing
5. **Package the project:** Create `setup.py` for pip installation

---

## Benefits Summary

✅ **Readability:** Clean separation of concerns
✅ **Maintainability:** Changes isolated to specific modules
✅ **Reusability:** Functions can be imported anywhere
✅ **Testability:** Each module can be unit tested
✅ **Documentation:** Comprehensive docstrings
✅ **Education:** Students can follow the analysis story
✅ **Professional:** Industry-standard project structure
✅ **Collaboration:** Multiple people can work on different modules

---

## Troubleshooting

### Import Errors
```python
# Error: ModuleNotFoundError: No module named 'src'
```
**Solution:** Make sure you're running Jupyter from the project root:
```bash
cd C:\Users\colto\Documents\GitHub\saocom_project
jupyter notebook
```

### Missing Dependencies
```python
# Error: ModuleNotFoundError: No module named 'rasterio'
```
**Solution:** Install requirements:
```bash
pip install -r requirements.txt
```

### Function Not Found
```python
# Error: ImportError: cannot import name 'calibrate_heights'
```
**Solution:** Check the module name and function name:
```python
# Correct:
from src.calibration import calibrate_heights

# Incorrect:
from src.calibrate import calibrate_heights  # Wrong module name
```

---

## Questions?

See `CLAUDE.md` for detailed project documentation, or open an issue in the project repository.

**Happy analyzing!** 🛰️📊
