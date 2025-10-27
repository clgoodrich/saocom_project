# SAOCOM Analysis Project - Complete Edition

## Quick Start

**USE THIS NOTEBOOK:** `saocom_analysis_complete.ipynb`

This notebook contains 100% of the functionality from the original `saocom_v3.ipynb` in a clean, properly formatted package ready for student use.

```bash
jupyter notebook saocom_analysis_complete.ipynb
```

Then: Kernel → Restart & Run All

---

## What You Have

### Main Notebook (USE THIS)

**saocom_analysis_complete.ipynb**
- 64 cells (28 markdown, 36 code)
- 27 major analysis sections
- 100+ visualization elements
- All land cover analysis (all 3 levels)
- All void zone analysis
- All inline function definitions
- Properly formatted
- Size: 224 KB (outputs cleared)
- Status: READY FOR USE

### Backup/Reference Files

**saocom_v3.ipynb** (original)
- Keep for backup
- Size: 49 MB

**saocom_analysis_clean.ipynb** (incomplete refactoring)
- Keep for reference only
- Missing many sections
- Do NOT use for students

---

## Complete Feature List

All 27 sections from the original are included:

**Data Processing:**
- Setup and imports
- Load SAOCOM data (68,512 points)
- Horizontal datum verification
- DEM resampling to 10m
- Study area mask creation
- DEM sampling at points
- Coverage grid creation
- Height calibration
- Outlier detection (Isolation Forest + IQR)

**Land Cover Analysis:**
- CORINE data loading and remapping
- DBF lookup table integration
- Level 1, 2, and 3 classification
- Accuracy by land cover type
- Void zone vs land cover analysis
- Individual land cover maps

**Visualizations (20+ types):**
- Spatial overlap maps
- Reference DEM comparisons
- Outlier detection plots
- Violin plots (detailed performance)
- Class overlays (colorblind-friendly)
- Scatter plots
- Density plots (hexbin, 2D histograms)
- Bland-Altman plots
- Swiss cheese void visualization
- Land cover histograms
- Topographic maps
- 3D terrain models

---

## Documentation Files

| File | Purpose |
|------|---------|
| `QA_COMPLETE_REPORT.md` | Comprehensive QA results (14 KB) |
| `NOTEBOOK_GUIDE.md` | Quick reference guide (4.6 KB) |
| `WORK_SUMMARY.txt` | Work completion summary |
| `CLAUDE.md` | Project overview for Claude |
| `DATA_COLUMNS_REFERENCE.md` | Data column guide |
| `REFACTORING_GUIDE.md` | Refactoring details |

---

## Source Code Modules (For Future Use)

The `src/` directory contains modular code:
- `utils.py` - Raster I/O utilities
- `preprocessing.py` - DEM processing
- `calibration.py` - Height calibration
- `outlier_detection.py` - Outlier detection
- `statistics_prog.py` - Statistical functions
- `landcover.py` - CORINE classification
- `visualization.py` - Plotting functions

Note: The complete notebook uses inline functions from the original, not these modules.

---

## Expected Outputs

After running the notebook, you'll find:

**results/** directory:
- `tinitaly_10m_masked.tif` - Masked TINItaly DEM
- `copernicus_10m_masked.tif` - Masked Copernicus DEM
- `saocom_void_mask.tif` - Void area mask
- `corine_10m_masked.tif` - CORINE land cover at 10m
- Additional processed rasters

**images/** directory:
- 20+ high-resolution figures (300 DPI)
- All visualization outputs

---

## System Requirements

**Python Packages:**
```
geopandas
rasterio
shapely
scipy
scikit-learn
scikit-image
seaborn
matplotlib
matplotlib_scalebar
dbfread
numpy
pandas
```

Install all:
```bash
pip install geopandas rasterio shapely scipy scikit-learn scikit-image seaborn matplotlib matplotlib_scalebar dbfread
```

**Data Files Required:**
- SAOCOM CSV (in `data/saocom_csv/`)
- TINItaly DEM (in `data/tinitaly/`)
- Copernicus DEM (in `data/`)
- CORINE land cover + DBF (in `data/ground_cover/`)
- Sentinel-2 RGB (in `data/sentinel_data/`)

---

## Execution Time

Expected: 5-10 minutes for full notebook execution (68,512 points)

Breakdown:
- Data loading: ~10 sec
- DEM resampling: ~30 sec
- Sampling/calibration: ~20 sec
- Outlier detection: ~30 sec
- Land cover processing: ~60 sec
- Visualizations: ~2-5 min

---

## QA Status

**Testing Completed:** 2025-10-26

**Tests Performed:**
- Structural integrity: PASS
- Cell formatting: PASS
- Critical workflow: PASS
- Function definitions: PASS (all 15 functions)
- Visualizations: PASS (100+ elements)
- Completeness: 100%

**Status:** PRODUCTION READY

See `QA_COMPLETE_REPORT.md` for full details.

---

## For Students

This notebook demonstrates:
- InSAR height validation workflow
- DEM comparison techniques
- Outlier detection methods
- Land cover analysis
- Statistical validation (NMAD, RMSE)
- Publication-quality visualizations

**Ready to use for teaching and learning!**

---

## Support

For questions about:
- **Notebook usage:** See `NOTEBOOK_GUIDE.md`
- **QA details:** See `QA_COMPLETE_REPORT.md`
- **Data columns:** See `DATA_COLUMNS_REFERENCE.md`
- **Project overview:** See `CLAUDE.md`

---

## Version History

- **2025-10-26:** Created `saocom_analysis_complete.ipynb` (complete version)
- **2025-10-26:** Created `saocom_analysis_clean.ipynb` (incomplete, deprecated)
- **Original:** `saocom_v3.ipynb` (backup/archive)

**Current version:** saocom_analysis_complete.ipynb (v1.0)
**Recommendation:** Use the complete version for all work

---

The notebook is ready for distribution to students!
