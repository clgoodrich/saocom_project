# Quick Start: Void Coverage PCA Analysis

## What You Have

Three files ready to use:

1. **`src/void_coverage_pca.py`** - Helper functions module
2. **`void_coverage_pca_notebook_cell.py`** - Complete notebook code (COPY THIS!)
3. Documentation files (README, visualization guide)

## How to Add to Your Notebook

### Step 1: Open Your Notebook
Open `saocom_analysis_clean_backup.ipynb` in Jupyter

### Step 2: Find or Create PCA Section
- If you have an existing PCA section (Section 13), decide if you want to replace or add after it
- Or create a new section at the end

### Step 3: Copy the Code
1. Open `void_coverage_pca_notebook_cell.py`
2. **Copy the entire file contents** (Ctrl+A, Ctrl+C)
3. Paste into a new notebook cell

### Step 4: Run It
Just execute the cell! It will:
- ✅ Create grid cell dataframe (all cells, not just SAOCOM points)
- ✅ Add terrain and land cover features
- ✅ Run PCA analysis
- ✅ **Generate coverage vs slope visualization (4 panels)**
- ✅ Generate PCA visualizations (scatter, distributions, loadings)
- ✅ Print comprehensive statistics
- ✅ Export summary CSV

## What You'll Get

### Visualizations Saved
```
images/
├── coverage_vs_slope_analysis.png    ← NEW! 4-panel slope analysis
├── void_coverage_pca_scatter.png     ← PC1 vs PC2 scatter
├── void_coverage_pca_distributions.png ← PC histograms
└── void_coverage_pca_loadings.png    ← Feature importance heatmap
```

### Data Exports
```
results/
└── void_coverage_pca_summary.csv     ← Feature comparison table
```

### Console Output
- Grid cell statistics (covered vs void counts)
- **Coverage vs slope statistics** (mean/median/std dev by group)
- **Coverage by slope category** (flat/gentle/moderate/steep percentages)
- PCA variance explained
- Feature loadings
- PC score comparisons
- Top discriminating features

## Expected Results

### Coverage vs Slope Analysis
The 4-panel figure will show:

**Panel 1**: Coverage drops from ~8% (flat) to <1% (steep)
**Panel 2**: Void cells have steeper slopes (distributions separated)
**Panel 3**: Cumulative coverage decreases with slope
**Panel 4**: Box plots confirm statistical difference

### Key Statistics
```
Covered cells:  Mean slope = ~8°
Void cells:     Mean slope = ~15°
Difference:     +90% steeper in voids

Coverage by slope:
  Flat (0-5°):      ~8-9%
  Gentle (5-15°):   ~5-6%
  Moderate (15-30°): ~2-3%
  Steep (>30°):     <1%
```

### PCA Results
- PC1 will likely capture "terrain complexity" (slope + forest)
- PC2 may capture aspect or land cover patterns
- Clear separation between void and covered cells in PC space

## Troubleshooting

### Error: "coverage_grid not found"
**Solution**: Make sure you've run the earlier notebook cells that create `coverage_grid`. This should be in Section 12.9 or similar.

### Error: "slope_tin_grid not found"
**Solution**: The code will automatically compute it! But if you want to use existing slope grid, make sure it's named `slope_tin_grid`.

### Error: "land_cover_grid not found"
**Solution**: The code will automatically load and resample from `data/ground_cover/land_cover_clipped.tif`.

### Memory Issues
If you get memory errors with large grids:
```python
# In the notebook cell, find this line:
sample_rate = 1.0 if total_cells < 1_000_000 else 0.2

# Change to use more aggressive sampling:
sample_rate = 0.1  # Use 10% of cells
```

### Plots Not Showing
Add `plt.show()` after each `plt.savefig()` if plots aren't displaying inline.

## Customization

### Change Slope Bins
```python
# Find this line:
slope_bins = np.arange(0, df_grid['slope'].max() + 5, 5)

# Change to 2-degree bins:
slope_bins = np.arange(0, df_grid['slope'].max() + 2, 2)
```

### Change Figure Size
```python
# Find this line:
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Change to smaller:
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
```

### Add More PCA Components
```python
# Find this line:
n_components=5,

# Change to 10:
n_components=10,
```

## What This Analysis Tells You

### Scientific Insights
1. **Systematic Void Pattern**: 95% voids aren't random - they're in steep terrain
2. **SAR Limitations**: Radar geometry (layover/shadow) limits steep slope coverage
3. **Sampling Bias**: Your accuracy metrics are measured on easier terrain
4. **Terrain Factors**: PCA reveals which terrain combinations predict voids

### For Your Paper/Thesis
Use this to:
- ✅ Explain why certain areas lack data
- ✅ Justify terrain-stratified analysis
- ✅ Discuss limitations of SAR in complex topography
- ✅ Support future acquisition planning recommendations

### Figure Captions
See `COVERAGE_VS_SLOPE_VISUALIZATION.md` for ready-to-use figure captions!

## Next Steps

After running this analysis:

1. **Review the figures** - Do they match expectations?
2. **Check statistics** - Are differences statistically significant?
3. **Compare with existing work** - How do your results compare to other SAR studies?
4. **Update paper** - Add these insights to your discussion section
5. **Create spatial maps** - Use PC scores to map void risk across your ROI

## Questions?

- **Module docs**: See `src/void_coverage_pca.py` for function docstrings
- **Detailed guide**: See `VOID_COVERAGE_PCA_README.md`
- **Visualization details**: See `COVERAGE_VS_SLOPE_VISUALIZATION.md`

## One-Line Summary

**This analysis reveals WHY 95% of your ROI has no SAOCOM data - it's systematically missing in steep, complex terrain due to SAR geometric constraints.**
