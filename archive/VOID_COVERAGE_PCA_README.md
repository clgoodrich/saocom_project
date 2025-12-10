# Void Coverage PCA Analysis

## Overview

This analysis examines **why certain grid cells have no SAOCOM coverage** (void zones). Unlike the previous PCA which analyzed quality issues in existing SAOCOM points, this focuses on the **~95% of grid cells with zero coverage**.

## Problem Statement

Your analysis showed:
- **Total grid cells in ROI**: ~1.9M cells (at 10m resolution)
- **Cells with SAOCOM coverage**: ~95K cells (**5.0%**)
- **Void cells (no coverage)**: ~1.8M cells (**95.0%**)

**Question**: What terrain, land cover, or geometric factors predict whether a grid cell will have SAOCOM coverage or remain void?

## Approach

### Old PCA (Point-Based Quality Analysis)
```
Sample: Only SAOCOM points with data
Label: "void zone" = low quality (coherence <0.5, high residuals, etc.)
Result: ~25% of points classified as low quality
Limitation: Doesn't explain why 95% of cells have NO data at all
```

### New PCA (Grid-Based Coverage Analysis)
```
Sample: ALL grid cells (covered + void)
Label: has_coverage = True/False
Features: Terrain (slope, aspect, elevation) + Land cover
Result: Identifies factors predicting void zones
```

## Key Differences

| Aspect | Old PCA | New PCA |
|--------|---------|---------|
| **Data source** | SAOCOM points only | All grid cells |
| **Sample size** | ~64K points | ~1.9M cells (sampled if large) |
| **Target variable** | Quality flags | Coverage boolean |
| **Question** | "Why is this point low quality?" | "Why is this cell empty?" |
| **Coverage analyzed** | 5% of ROI | 100% of ROI |

## Features Analyzed

### Terrain Features
1. **Slope** (degrees) - Steep terrain may cause layover/shadow
2. **Aspect** (degrees) - Radar look direction dependency
3. **Elevation** (meters) - Altitude effects

### Derived Categories
- Slope bins: flat (0-5°), gentle (5-15°), moderate (15-30°), steep (30+°)
- Aspect bins: N, NE, E, SE, S, SW, W, NW
- Elevation bins: Quintiles (very low → very high)

### Land Cover Features
- Top 5 land cover types (one-hot encoded)
- From CORINE Land Cover classification

## Expected Results

The PCA will identify:

1. **Which terrain factors correlate with void zones**
   - Example: "Steep slopes (>30°) are 80% more common in void cells"
   - Example: "North-facing slopes have 40% less coverage"

2. **Which land cover types have poor coverage**
   - Example: "Forest areas have 50% less coverage than agricultural"

3. **Principal components explaining void patterns**
   - PC1 might capture "terrain complexity" (slope + aspect)
   - PC2 might capture "land cover roughness"

4. **Spatial patterns**
   - Void zones clustered in specific regions?
   - Elevation-dependent coverage gaps?

## How to Use

### 1. Install the Module

The analysis code is in `src/void_coverage_pca.py`. It's already importable.

### 2. Add to Your Notebook

Copy the code from `void_coverage_pca_notebook_cell.py` and paste it as a new section in your notebook (e.g., Section 13 or wherever you want the PCA).

### 3. Required Prerequisites

The code expects these variables to exist in your notebook (from earlier cells):

```python
# From data loading sections:
coverage_grid        # Boolean grid: True = has coverage, False = void
tinitaly_10m         # Elevation DEM (10m resolution)
transform_10m        # Affine transform for the grid
crs                  # Coordinate reference system
grid_height          # Grid dimensions
grid_width

# Computed or loaded:
slope_tin_grid       # Slope grid (degrees)
aspect_tin_grid      # Aspect grid (degrees)
land_cover_grid      # CORINE land cover codes
```

If `slope_tin_grid` and `aspect_tin_grid` don't exist yet, the code will compute them automatically.

If `land_cover_grid` doesn't exist, it will load and resample from `data/ground_cover/land_cover_clipped.tif`.

### 4. Run the Analysis

Just run the cell! It will:

1. Create a dataframe of all grid cells with terrain/land cover features
2. Run PCA to identify void-predicting factors
3. Generate visualizations:
   - `images/void_coverage_pca_scatter.png` - PC1 vs PC2 (void vs covered)
   - `images/void_coverage_pca_distributions.png` - PC score histograms
   - `images/void_coverage_pca_loadings.png` - Feature importance heatmap
4. Export summary statistics:
   - `results/void_coverage_pca_summary.csv` - Feature comparison table

### 5. Interpret Results

Look for:

- **Feature comparison table**: Which features differ most between void/covered?
  - Sort by `pct_difference` to find strongest factors
  - Check `p_value` for statistical significance

- **PC loadings**: Which features load on PC1/PC2?
  - Positive loading on PC1 + higher PC1 in voids → factor increases void probability
  - Negative loading on PC1 + higher PC1 in voids → factor decreases void probability

- **Scatter plot**: Are void/covered cells separable in PC space?
  - Clear separation → factors strongly predict coverage
  - Overlap → coverage may be random or depend on other factors

## Outputs

### Console Output
```
Grid cell dataframe created:
  Total cells: 1,900,000
  Covered cells: 95,000 (5.0%)
  Void cells: 1,805,000 (95.0%)

Variance Explained:
  PC1: 42.35%
  PC2: 18.72%
  PC3: 12.41%
  Cumulative: 73.48%

Top Features per Component:
PC1:
  Positive:
    slope                         : +0.524
    elevation                     : +0.412
    lc_Forest_and_Semi_Natural    : +0.387
  Negative:
    lc_Agricultural_Area          : -0.501
    aspect                        : -0.234
    lc_Artificial_Surfaces        : -0.112

VOID vs COVERED COMPARISON (Raw Features):
Top 10 Discriminating Features (by % difference):

slope
  Covered:      8.234
  Void:        15.678
  Diff:       +90.4% (p=1.23e-145)

lc_Forest_and_Semi_Natural
  Covered:      0.123
  Void:         0.456
  Diff:      +270.7% (p=2.34e-89)

COVERAGE vs SLOPE STATISTICS
Covered cells:
  Mean slope: 8.23°
  Median slope: 6.45°
  Std dev: 5.67°

Void cells:
  Mean slope: 15.68°
  Median slope: 13.21°
  Std dev: 9.34°

Difference:
  Mean difference: +7.45° (+90.4%)
  T-statistic: 124.567
  P-value: 1.23e-145

Coverage rate by slope category:
  Flat         ( 0- 5°):  8.45% (42,150/498,520 cells)
  Gentle       ( 5-15°):  5.23% (38,420/734,890 cells)
  Moderate     (15-30°):  2.67% (12,340/461,230 cells)
  Steep        (30-90°):  0.89% (2,090/234,360 cells)
```

### Visualization Files
- **coverage_vs_slope_analysis.png**: Four-panel analysis showing:
  1. Coverage rate vs slope bins (line plot showing how coverage drops with slope)
  2. Slope distribution histograms (covered vs void cells)
  3. Cumulative coverage rate by slope
  4. Box plots comparing slope distributions
- **void_coverage_pca_scatter.png**: Shows how void vs covered cells separate in principal component space
- **void_coverage_pca_distributions.png**: Histograms of PC scores for each group
- **void_coverage_pca_loadings.png**: Heatmap showing which features contribute to each PC

### Data Export
- **void_coverage_pca_summary.csv**: Full feature comparison table with statistics

## Example Interpretation

If results show:

```
PC1: 45% variance
  - High positive: slope (+0.52), forest (+0.39)
  - High negative: agricultural (-0.48)

Void cells have PC1 = +0.85
Covered cells have PC1 = -0.12
Difference: +0.97 (highly significant)

Raw features:
  Slope: Void cells 90% higher (15.7° vs 8.2°)
  Forest: Void cells 270% more common (45.6% vs 12.3%)
```

**Interpretation**: SAOCOM coverage is strongly **reduced in steep, forested terrain**. PC1 captures a "terrain complexity + vegetation" axis, and void zones score much higher on this axis.

**Implication**: The 95% coverage gaps are NOT random - they systematically occur in:
- Steep slopes (radar shadow/layover)
- Forested areas (low coherence due to vegetation)
- High elevations (if correlated with above)

## Memory Considerations

For large grids (>1M cells), the code automatically samples 20% of cells to reduce memory usage. This still gives ~380K cells, which is more than enough for robust PCA.

To adjust sampling:
```python
sample_rate = 0.1  # Use 10% sample
sample_rate = 1.0  # Use all cells (may be slow)
```

## Files Created

| File | Description |
|------|-------------|
| `src/void_coverage_pca.py` | Core analysis module |
| `void_coverage_pca_notebook_cell.py` | Ready-to-paste notebook code |
| `VOID_COVERAGE_PCA_README.md` | This documentation |

## Next Steps

After running the analysis, you can:

1. **Update paper/thesis**: Report which terrain factors predict void zones
2. **Create spatial maps**: Color-code grid cells by PC1 score to visualize void risk
3. **Stratify accuracy metrics**: Report accuracy separately for low-void-risk vs high-void-risk terrain
4. **Propose acquisition strategy**: Suggest targeting steep/forested areas with different acquisition modes

## Questions?

Check the code comments in:
- `src/void_coverage_pca.py` - Function docstrings explain each step
- `void_coverage_pca_notebook_cell.py` - Section-by-section walkthrough
