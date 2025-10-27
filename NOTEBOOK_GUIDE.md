# SAOCOM Analysis Notebook Guide

## Quick Start

You now have **TWO** notebooks in this project:

### 1. saocom_analysis_clean.ipynb (INCOMPLETE - Do NOT use)
- Created earlier, missing major functionality
- Missing: void zones, swiss cheese viz, full land cover, topo maps, etc.
- Keep for reference only

### 2. saocom_analysis_complete.ipynb (COMPLETE - USE THIS ONE)
- Contains ALL 64 cells from original saocom_v3.ipynb
- Contains ALL 27 major sections
- Contains ALL visualizations (100+ elements)
- Contains ALL land cover analysis
- Ready for students

---

## How to Run the Complete Notebook

```bash
cd C:\Users\colto\Documents\GitHub\saocom_project
jupyter notebook saocom_analysis_complete.ipynb
```

Then: Kernel → Restart & Run All

Expected time: 5-10 minutes

---

## What's Inside

The complete notebook includes these major sections:

**Core Processing (Sections 1-9):**
1. Setup and imports
2. Load SAOCOM data
3. Horizontal datum verification
4. Resample DEMs to 10m
5. Create study area mask
6. Sample DEMs at SAOCOM points
7. Create coverage grid
8. Load reference DEM data
9. Load CORINE land cover

**Visualizations (Sections 10-27):**
10. Spatial overlap map
11. Reference DEM comparison
12. Outlier detection
13. CORINE land cover sampling
14. Data preparation for plotting
15. Sentinel-2 RGB
16. Violin plots (Level 3 performance)
17. Class overlays basic
18. Individual class overlays (colorblind-friendly)
19. SAOCOM vs TINItaly comparison
20. Histograms
21. Density plots
22. Gridded comparison
23. Void zones vs land cover
24. Swiss cheese visualization
25. Individual land cover maps with voids
26. Land cover histograms
27. Topographic maps

---

## Expected Outputs

### Results Directory

After running, you'll find in `results/`:
- Masked DEMs (TINItaly and Copernicus at 10m)
- Void mask raster
- CORINE land cover rasters (remapped, resampled, masked)

### Images Directory

You'll find in `images/`:
- 20+ high-resolution figures (300 DPI)
- Spatial maps
- Comparison visualizations
- Statistical plots
- Land cover maps
- Void zone visualizations

---

## Key Statistics Generated

The notebook calculates:
- NMAD (robust error metric) for both reference DEMs
- Height calibration offsets
- Accuracy by land cover type (all levels)
- Accuracy by slope category
- Void zone statistics
- Reference DEM comparison metrics

---

## Data Requirements

Make sure these files exist:

```
data/
├── saocom_csv/
│   └── *.csv (SAOCOM InSAR points)
├── tinitaly/
│   └── tinitaly_crop.tif
├── copernicus.tif (or demCOP30.tif)
├── ground_cover/
│   ├── *.tif (CORINE land cover)
│   └── *.dbf (CORINE lookup table)
└── sentinel_data/
    └── *.tif (RGB imagery)
```

---

## Differences from Original

| Aspect | Original (saocom_v3.ipynb) | Complete (saocom_analysis_complete.ipynb) |
|--------|----------------------------|-------------------------------------------|
| Cells | 64 | 64 (identical) |
| Sections | 27 | 27 (identical) |
| Functionality | All features | All features (100% preserved) |
| Outputs | Had execution outputs | Cleared (ready for fresh run) |
| Size | 51 MB (with outputs) | ~220 KB (cleared) |

**The complete notebook is functionally identical to the original.**

---

## When to Use Each Notebook

**For students:**
- Use `saocom_analysis_complete.ipynb`

**For archival/backup:**
- Keep `saocom_v3.ipynb` (original)

**For reference only:**
- `saocom_analysis_clean.ipynb` (incomplete refactoring attempt)

---

## QA Status

**saocom_analysis_complete.ipynb:**
- Structural tests: PASS
- Critical workflow tests: PASS
- Formatting: PASS
- Completeness: 100%
- Status: READY FOR USE

See `QA_COMPLETE_REPORT.md` for full QA details.

---

## Support Files

The `src/` directory contains modular code (for future use):
- `utils.py` - Raster utilities
- `preprocessing.py` - DEM processing
- `calibration.py` - Height calibration
- `outlier_detection.py` - Outlier filtering
- `statistics_prog.py` - Statistical functions
- `landcover.py` - CORINE classification
- `visualization.py` - Plotting functions

Note: The complete notebook uses inline functions (from original), not the src/ modules.

---

## Next Steps

1. Review `QA_COMPLETE_REPORT.md` for full details
2. Run `saocom_analysis_complete.ipynb` with your data
3. Verify all outputs are generated
4. Share notebook with students

**The notebook is ready for use!**
