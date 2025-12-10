# Control Points Identification Guide

## Overview

This guide explains how to identify and use high-quality control points where SAOCOM, Copernicus, and TINItaly DEMs all agree within a specified tolerance (typically ±2 meters).

## What Are Control Points?

**Control points** are locations where multiple independent elevation measurements agree closely, indicating:
- High measurement confidence
- Stable terrain
- Good geometric conditions
- Minimal systematic errors

These points are valuable for:
- **Calibration validation**: Verify calibration quality
- **Accuracy assessment**: Independent validation dataset
- **Ground truth planning**: Identify locations for field measurements
- **Quality control**: Spatial distribution of reliable measurements

## Module: `src/control_points.py`

### Key Functions

#### `identify_control_points(gdf, tolerance=2.0, ...)`

Identifies points where all three DEMs agree within ±tolerance meters.

**Agreement Criterion:**
```
max(SAOCOM, Copernicus, TINItaly) - min(SAOCOM, Copernicus, TINItaly) ≤ tolerance
```

**Parameters:**
- `gdf`: GeoDataFrame with SAOCOM data and sampled DEM heights
- `tolerance`: Maximum elevation difference (default: 2.0 meters)
- `calibrated`: Use calibrated SAOCOM heights (default: True)

**Returns:**
- GeoDataFrame with control points
- Added columns: `elevation_range`, `mean_elevation`, `std_elevation`

**Example:**
```python
from src.control_points import identify_control_points

# Find control points with ±2m agreement
control_pts = identify_control_points(saocom_gdf, tolerance=2.0)

print(f"Found {len(control_pts)} control points")
print(f"Percentage: {len(control_pts)/len(saocom_gdf)*100:.1f}%")
```

#### `analyze_control_point_distribution(control_pts, all_pts)`

Analyzes spatial and statistical distribution of control points.

**Returns Dictionary:**
- `count`: Number of control points
- `fraction`: Fraction of total points
- `spatial_density`: Points per km²
- `mean_elevation`: Mean elevation
- `elevation_range`: Min-max elevation range
- `mean_agreement`: Mean DEM disagreement
- `by_slope`: Distribution by slope class (if available)

#### `calculate_control_point_bias(control_pts, ...)`

Calculates SAOCOM accuracy metrics at control points.

**Returns:**
- `mean_bias`: Mean residual
- `rmse`: Root mean square error
- `nmad`: Normalized median absolute deviation
- `std_bias`: Standard deviation

#### `recommend_calibration_points(control_pts, n_points=10, method='distributed')`

Selects subset of control points for calibration or validation.

**Methods:**
- `'distributed'`: Spatially distributed using K-means clustering
- `'best'`: Lowest elevation disagreement
- `'random'`: Random sample

#### `export_control_points(control_pts, output_path, format='GeoJSON')`

Exports control points to file.

**Supported Formats:**
- `'GeoJSON'` - GeoJSON format (default)
- `'Shapefile'` - ESRI Shapefile
- `'GeoPackage'` - GeoPackage
- `'CSV'` - CSV with lat/lon columns

## Notebook Workflow

### Location
Pre-written cells: `notebooks/control_points_identification_cells.py`

### Cell Sequence

| Cell | Description | Outputs |
|------|-------------|---------|
| 1 | Import module, set tolerance | Configuration |
| 2 | Identify control points | Control points GeoDataFrame |
| 3 | Analyze distribution | Statistics dictionary |
| 4 | Calculate bias at control points | Accuracy metrics |
| 5 | Load Sentinel-2 imagery | RGB array |
| 6 | Main visualization: Sentinel overlay | `control_points_sentinel_overlay.png` |
| 7 | Multi-panel analysis dashboard | `control_points_analysis_dashboard.png` |
| 8 | Spatial clustering & calibration points | `recommended_calibration_points.png` |
| 9 | Export to files | GeoJSON, CSV, Shapefile |
| 10 | Summary statistics | Printed table |

## Usage Instructions

### Step 1: Set Tolerance

```python
TOLERANCE = 2.0  # meters
USE_CALIBRATED_SAOCOM = True
```

**Tolerance Guidelines:**
- **±1.0m**: Very strict, high-quality only (expect 5-15% of points)
- **±2.0m**: Standard, good quality (expect 15-30% of points)
- **±3.0m**: Moderate, more coverage (expect 30-50% of points)
- **±5.0m**: Loose, maximum coverage (expect 50-70% of points)

### Step 2: Run Identification

```python
control_points = identify_control_points(
    saocom_gdf,
    tolerance=TOLERANCE,
    calibrated=USE_CALIBRATED_SAOCOM
)
```

### Step 3: Analyze Results

```python
stats = analyze_control_point_distribution(control_points, saocom_gdf)
bias_stats = calculate_control_point_bias(control_points)
```

### Step 4: Visualize

Run visualization cells to generate:
- Control points overlaid on Sentinel-2 imagery
- Multi-panel analysis dashboard
- Recommended calibration points map

## Expected Results

### Typical Control Point Percentages

**Flat Terrain (<10° slope):**
- ±2m tolerance: 30-50% of points
- High agreement expected

**Moderate Terrain (10-20° slope):**
- ±2m tolerance: 15-30% of points
- Good agreement in stable areas

**Steep Terrain (>20° slope):**
- ±2m tolerance: 5-15% of points
- Lower agreement due to geometry effects

### Quality Indicators

**Good Control Point Set:**
- ✓ 15-30% of points (±2m tolerance)
- ✓ Spatially distributed across study area
- ✓ Cover range of elevations
- ✓ SAOCOM RMSE < 3m at control points
- ✓ Mean bias < 1m

**Poor Control Point Set:**
- ⚠️ <5% of points
- ⚠️ Clustered in small area
- ⚠️ Limited elevation range
- ⚠️ SAOCOM RMSE > 5m at control points
- ⚠️ Large systematic bias

## Interpretation

### High Control Point Density

**Indicates:**
- Good overall DEM agreement
- Stable terrain conditions
- Successful calibration
- High-quality SAOCOM data

### Low Control Point Density

**Possible Causes:**
- Steep, complex terrain (expected)
- Calibration issues (check bias)
- DEM resolution mismatch
- Systematic errors in one DEM
- Shadow/layover effects (check radar geometry)

**Solutions:**
- Increase tolerance (e.g., ±3m)
- Check calibration quality
- Review outlier filtering
- Analyze spatial patterns

## Visualizations

### Main Map (Sentinel Overlay)

Shows:
- All SAOCOM points (gray background)
- Control points (colored by agreement quality)
- Sentinel-2 RGB imagery
- Statistics text box

**What to Look For:**
- Spatial clustering vs. distribution
- Association with land cover types
- Elevation patterns
- Shadow/illumination effects

### Analysis Dashboard

Six-panel figure showing:
1. **Sentinel overlay**: Spatial distribution
2. **Agreement histogram**: DEM disagreement distribution
3. **SAOCOM residuals**: Accuracy at control points
4. **Slope distribution**: Terrain characteristics
5. **Elevation vs. agreement**: Correlation analysis

### Recommended Calibration Points

Shows:
- Spatially distributed subset (typically 10-20 points)
- Useful for independent validation
- Field measurement prioritization

## Export Formats

### GeoJSON
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "geometry": {"type": "Point", "coordinates": [lon, lat]},
      "properties": {
        "elevation_range": 0.8,
        "mean_elevation": 123.4,
        "HEIGHT_RELATIVE": 123.5,
        ...
      }
    }
  ]
}
```

### CSV
```csv
longitude,latitude,elevation_range,mean_elevation,HEIGHT_RELATIVE,...
11.0234,45.4567,0.8,123.4,123.5,...
```

### Shapefile
Standard ESRI Shapefile with point geometry.

## Advanced Use Cases

### 1. Independent Validation Set

Reserve control points for validation (don't use in calibration):

```python
# Split into calibration and validation
from sklearn.model_selection import train_test_split

cal_pts, val_pts = train_test_split(
    control_points,
    test_size=0.3,
    random_state=42
)
```

### 2. Stratified Sampling

Sample control points by terrain type:

```python
# Get control points from each land cover class
stratified_pts = []
for lc_class in control_points['CLC_L1'].unique():
    class_pts = control_points[control_points['CLC_L1'] == lc_class]
    sample = class_pts.sample(n=min(10, len(class_pts)))
    stratified_pts.append(sample)

stratified_control = pd.concat(stratified_pts)
```

### 3. Temporal Stability Check

If you have multi-temporal SAOCOM data:

```python
# Check if same points are control points across acquisitions
# (Requires multiple acquisition datasets)
control_pts_t1 = identify_control_points(saocom_gdf_date1)
control_pts_t2 = identify_control_points(saocom_gdf_date2)

# Find persistent control points
persistent = gpd.sjoin(control_pts_t1, control_pts_t2, how='inner')
```

### 4. Ground Truth Planning

Prioritize field measurements:

```python
# Get distributed calibration points
field_targets = recommend_calibration_points(
    control_points,
    n_points=20,
    method='distributed'
)

# Export for GPS
export_control_points(
    field_targets,
    'field_measurements/gps_targets.gpx',  # if GPX export added
    format='GeoJSON'
)
```

## Integration with Workflow

### Timing in Analysis

**Insert control point analysis:**

1. ✓ After DEM sampling
2. ✓ After calibration
3. ✓ After outlier filtering
4. **⟹ Before final accuracy assessment** ⟸ Insert here
5. Before radar shadow analysis
6. Before publication plots

### Combined Stratifications

**Control Points + Terrain:**
```python
# Analyze by slope class
for slope_class in ['flat', 'moderate', 'steep']:
    class_control = control_points[control_points['slope_class'] == slope_class]
    stats = calculate_control_point_bias(class_control)
    print(f"{slope_class}: RMSE = {stats['rmse']:.2f}m")
```

**Control Points + Land Cover:**
```python
# Analyze by land cover
for lc in control_points['CLC_L1'].unique():
    lc_control = control_points[control_points['CLC_L1'] == lc]
    stats = calculate_control_point_bias(lc_control)
    print(f"{lc}: n={len(lc_control)}, RMSE = {stats['rmse']:.2f}m")
```

## Troubleshooting

### Issue: No control points found

**Causes:**
- Tolerance too strict
- Calibration not applied
- Missing DEM values
- Severe systematic errors

**Solutions:**
1. Increase tolerance (±3m or ±5m)
2. Verify calibration: `'HEIGHT_CALIBRATED' in saocom_gdf.columns`
3. Check for NaN values: `saocom_gdf[['copernicus_height', 'tinitaly_height']].isna().sum()`
4. Review calibration bias

### Issue: Too many control points (>70%)

**Causes:**
- Tolerance too loose
- Flat terrain (expected)
- Correlated errors between DEMs

**Solutions:**
- Decrease tolerance for stricter quality
- Normal for flat areas
- Check if DEMs are truly independent

### Issue: Control points clustered in one area

**Causes:**
- Spatial variation in DEM quality
- Shadow effects in mountains
- Land cover influence

**Solutions:**
- Analyze spatial patterns (why clustering?)
- Use `recommend_calibration_points()` for distribution
- Check radar geometry in excluded areas

### Issue: High RMSE at control points

**Causes:**
- Calibration failure
- Systematic errors
- DEM resolution mismatch

**Solutions:**
1. Recalibrate SAOCOM
2. Check for systematic bias patterns
3. Verify DEM sampling methods

## References

### Relevant Literature

1. **Höhle & Höhle (2009)**: "Accuracy assessment of digital elevation models by means of robust statistical methods"
   - NMAD and robust statistics at control points

2. **Santillan & Makinano-Santillan (2016)**: "Vertical accuracy assessment of 30-m resolution ALOS, ASTER, and SRTM global DEMs over Northeastern Mindanao"
   - Multi-DEM comparison methodology

3. **Grohmann (2018)**: "Evaluation of TanDEM-X DEMs on selected Brazilian sites"
   - Control point selection strategies

## Best Practices

### Control Point Selection
✓ Use calibrated SAOCOM heights
✓ Apply outlier filtering before identification
✓ Document tolerance threshold used
✓ Report control point percentage
✓ Analyze spatial distribution

### Quality Control
✓ Check for spatial clustering
✓ Verify elevation range coverage
✓ Assess slope distribution
✓ Calculate independent accuracy metrics
✓ Compare with full dataset statistics

### Reporting
✓ State tolerance criterion clearly
✓ Report control point count and percentage
✓ Include spatial distribution map
✓ Present accuracy metrics at control points
✓ Compare with overall accuracy

---

**Version**: 1.0
**Last Updated**: 2025-10-29
**Author**: SAOCOM Project Team
