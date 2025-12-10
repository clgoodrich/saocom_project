# Radar Shadow Analysis for SAOCOM DEM Validation

## Overview

This document describes the radar shadow analysis capabilities added to the SAOCOM DEM validation project. Radar geometry significantly affects the quality of SAR/InSAR-derived elevation measurements, particularly in areas with steep terrain.

## Background

### SAR Geometry Effects

Synthetic Aperture Radar (SAR) is a side-looking sensor that illuminates the terrain at an oblique angle. This geometry creates several effects that can degrade elevation measurements:

1. **Shadow**: Areas where radar cannot reach due to terrain blocking (local incidence angle > 90°)
2. **Layover**: Areas where steep slopes facing the radar cause severe geometric distortion (local incidence angle < ~20°)
3. **Foreshortening**: Compression of terrain features on slopes facing the radar

### Why This Matters

Areas affected by poor radar geometry typically show:
- Higher elevation errors (increased RMSE, NMAD)
- Systematic biases
- Lower measurement density
- Reduced coherence in InSAR processing

Understanding these effects allows you to:
- Stratify accuracy analysis by geometric quality
- Identify areas where DEM is unreliable
- Improve interpretation of validation results
- Mask problematic areas for applications

## Module: `src/radar_geometry.py`

### Key Functions

#### `calculate_local_incidence_angle(slope, aspect, radar_incidence, radar_azimuth)`
Calculates the local incidence angle, which is the angle between the radar look vector and the terrain surface normal.

**Parameters:**
- `slope`: Terrain slope in degrees (0-90°)
- `aspect`: Terrain aspect in degrees (0-360°, 0=North)
- `radar_incidence`: Radar incidence angle from vertical (default: 35° for SAOCOM)
- `radar_azimuth`: Radar look direction (default: 192° for descending orbit)

**Returns:**
- Local incidence angle in degrees
  - < 20°: Potential layover
  - 30-60°: Optimal geometry
  - > 90°: Shadow

**Example:**
```python
from src.radar_geometry import calculate_local_incidence_angle

local_inc = calculate_local_incidence_angle(
    slope=slope_array,
    aspect=aspect_array,
    radar_incidence=35.0,
    radar_azimuth=192.0
)
```

#### `identify_shadow_areas(local_incidence, shadow_threshold=90.0)`
Identifies areas in radar shadow.

**Returns:**
- Boolean mask: True for shadow areas

#### `identify_layover_areas(local_incidence, layover_threshold=20.0)`
Identifies areas with potential layover distortion.

**Returns:**
- Boolean mask: True for layover areas

#### `classify_geometric_quality(local_incidence, slope, ...)`
Classifies terrain into 5 geometric quality categories.

**Returns:**
- Classification array with values:
  - 0: Optimal (flat, well-illuminated)
  - 1: Acceptable (moderate slopes, good geometry)
  - 2: Foreshortening (steep slopes facing radar)
  - 3: Shadow (radar blocked)
  - 4: Layover (severe distortion)

#### `analyze_shadow_statistics(gdf, local_incidence_col, residual_col)`
Computes accuracy metrics stratified by shadow/illumination conditions.

**Returns:**
- Dictionary with statistics for categories: optimal, acceptable, steep, shadow, layover

## SAOCOM Geometry Parameters

### Typical Values

| Parameter | Typical Range | Default Used |
|-----------|---------------|--------------|
| Incidence Angle | 20-50° | 35° |
| Look Azimuth (Descending) | 185-200° | 192° |
| Look Azimuth (Ascending) | 5-20° | 12° |

### Determining Your Acquisition Geometry

1. **Check SAOCOM metadata**: Look for incidence angle in product metadata
2. **Orbit direction**:
   - Descending: Sensor moves south, looks southwest (~192° azimuth)
   - Ascending: Sensor moves north, looks northeast (~12° azimuth)
3. **Adjust parameters** in the notebook cells based on your data

## Notebook Cells

### Location
Pre-written cells are available in:
- `notebooks/radar_shadow_analysis_cells.py`

### Usage

1. **Copy cells** from the Python file into your Jupyter notebook
2. **Adjust parameters** (RADAR_INCIDENCE, RADAR_AZIMUTH) for your acquisition
3. **Run cells sequentially**

### Cell Sequence

| Cell | Description | Outputs |
|------|-------------|---------|
| 1 | Import radar geometry module | - |
| 2 | Set SAOCOM geometry parameters | Configuration display |
| 3 | Load/calculate terrain derivatives | Slope, aspect arrays |
| 4 | Calculate local incidence angle | Local incidence array |
| 5 | Identify shadow/layover areas | Shadow/layover masks |
| 6 | Classify geometric quality | Quality classification |
| 7 | Add geometric data to points | Enhanced GeoDataFrame |
| 8 | Analyze accuracy by geometry | Statistics table |
| 9 | Visualize geometry maps | `radar_geometry_analysis.png` |
| 10 | Visualize residuals by shadow | `shadow_effect_on_accuracy.png` |
| 11 | Create summary table | Formatted statistics |
| 12 | Save geometric rasters | GeoTIFF outputs |

## Expected Outputs

### Visualizations

1. **`images/radar_geometry_analysis.png`**
   - Left: Local incidence angle map with shadow overlay
   - Right: Geometric quality classification

2. **`images/shadow_effect_on_accuracy.png`**
   - 4-panel figure showing:
     - Residuals vs local incidence angle (hexbin)
     - Residual distribution: illuminated vs shadow
     - Box plot of residuals by geometric quality
     - RMSE vs local incidence angle

### Raster Outputs

Saved to `topography_outputs/radar_geometry/`:
- `local_incidence_angle.tif`: Local incidence angle map
- `geometric_quality.tif`: Quality classification (0-4)
- `shadow_mask.tif`: Binary shadow mask

### Statistics

**Shadow Effect Summary Table** includes:
- Point counts per category
- Bias, RMSE, NMAD per category
- Mean local incidence angle per category

## Interpretation Guide

### Expected Results

**Well-illuminated areas (30-60° local incidence):**
- Lower RMSE and NMAD
- Bias close to zero
- Highest point density

**Shadow areas (>90° local incidence):**
- Significantly higher errors or no data
- Sparse or missing measurements
- Potential for outliers

**Layover areas (<20° local incidence):**
- Moderate to high errors
- Possible systematic biases
- Geometric distortion artifacts

### Typical Accuracy Degradation

Based on SAR literature, expect:
- **Optimal geometry**: Baseline accuracy (e.g., RMSE = 2-3m)
- **Foreshortening**: 1.5-2× worse (RMSE = 4-6m)
- **Near shadow**: 2-3× worse (RMSE = 6-9m)
- **Shadow**: No valid data or extreme errors

## Integration with Main Analysis

### Adding to Existing Workflow

Insert shadow analysis after:
1. DEM loading and preprocessing
2. Terrain derivative calculation
3. Outlier filtering

Before:
1. Final accuracy assessment
2. Land cover stratification
3. Publication-ready plots

### Recommended Stratifications

For comprehensive analysis, stratify accuracy by:
1. **Geometric quality** (5 classes)
2. **Land cover type** × **Geometric quality** (combined)
3. **Slope class** × **Shadow/illuminated** (combined)

## References

### Key Papers

1. **Massonnet & Feigl (1998)**: "Radar interferometry and its application to changes in the Earth's surface"
   - Fundamental SAR geometry concepts

2. **Zebker et al. (1994)**: "Topographic mapping from interferometric SAR observations"
   - Shadow and layover effects on InSAR DEMs

3. **Toutin (2002)**: "Impact of terrain slope and aspect on radargrammetric DEM accuracy"
   - Quantification of geometric effects on DEM accuracy

4. **Goyal et al. (2018)**: "Effect of terrain slope on InSAR-based DEM generation"
   - Recent analysis of slope-dependent errors

### SAOCOM Documentation

- SAOCOM User's Manual
- CONAE technical documentation
- Sentinel-1 documentation (similar L-band geometry)

## Troubleshooting

### Issue: All points show as "Optimal" quality

**Cause**: Incorrect radar geometry parameters
**Solution**: Verify incidence angle and azimuth from metadata

### Issue: Unexpected shadow patterns

**Cause**: Wrong orbit direction (ascending vs descending)
**Solution**: Check if azimuth should be ~12° (ascending) or ~192° (descending)

### Issue: No shadow areas detected

**Cause**: Terrain too flat for shadow effects
**Solution**: Normal for gentle terrain; verify with slope statistics

### Issue: Extremely high shadow percentage (>30%)

**Cause**: Very steep terrain or incorrect geometry
**Solution**: Verify parameters and check if mountainous area is expected

## Advanced Topics

### Custom Geometric Quality Thresholds

Modify thresholds based on your requirements:

```python
geometric_quality = classify_geometric_quality(
    local_incidence,
    slope,
    shadow_thresh=85.0,      # More strict shadow threshold
    layover_thresh=25.0,     # More lenient layover threshold
    steep_slope_thresh=25.0  # Different steep slope definition
)
```

### Radar Brightness Correction

The module includes `calculate_radar_brightness()` for:
- Understanding expected backscatter variations
- Potential radiometric normalization
- Coherence prediction

### Multi-Temporal Analysis

For time-series analysis:
- Shadow areas are **constant** (topography doesn't change)
- Use quality mask to filter all acquisitions consistently
- Track coherence vs geometric quality over time

## Contact & Support

For questions about radar shadow analysis:
1. Review this documentation
2. Check `src/radar_geometry.py` function docstrings
3. Consult SAR geometry literature
4. Review SAOCOM acquisition metadata

---

**Version**: 1.0
**Last Updated**: 2025-10-29
**Author**: SAOCOM Project Team
