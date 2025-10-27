# SAOCOM Data Columns Reference

## Input CSV Columns

Your SAOCOM CSV file contains these columns:

```
['ID', 'SVET', 'LVET', 'LAT', 'LAT2', 'LON', 'LON2', 'HEIGHT',
 'HEIGHT WRT DEM', 'SIGMA HEIGHT', 'COHER']
```

## Column Usage in Analysis

### Coordinates (Priority Order)
The notebook uses coordinates in this priority:
1. **LAT2, LON2** (preferred - higher precision)
2. **LAT, LON** (fallback if LAT2/LON2 not available)

**Coordinate System:**
- Input: Geographic coordinates (EPSG:4326 - WGS84 lat/lon)
- Converted to: UTM Zone 32N (EPSG:32632) for analysis

### Height Columns

| Original Column | Renamed To | Purpose |
|----------------|------------|---------|
| `HEIGHT` | `HEIGHT_RELATIVE` | Relative InSAR height (requires calibration) |
| `HEIGHT WRT DEM` | *(not renamed)* | Height with respect to DEM reference |
| `SIGMA HEIGHT` | *(not renamed)* | Height uncertainty/error estimate |

### Quality Metric

| Column | Description | Range | Usage |
|--------|-------------|-------|-------|
| `COHER` | Temporal coherence | 0-1 | Quality indicator (higher = better) |

**Coherence Thresholds:**
- `≥ 0.8`: High quality - used for calibration
- `≥ 0.7`: Good quality - included in analysis
- `< 0.7`: Lower quality - may be filtered out

## Column Transformations in Notebook

### Cell 6: Data Loading
```python
# 1. Load CSV
saocom_df = pd.read_csv(SAOCOM_CSV)

# 2. Use best available coordinates
lat_col = 'LAT2' if 'LAT2' in saocom_df.columns else 'LAT'
lon_col = 'LON2' if 'LON2' in saocom_df.columns else 'LON'

# 3. Create GeoDataFrame (lat/lon)
geometry = [Point(xy) for xy in zip(saocom_df[lon_col], saocom_df[lat_col])]
saocom_gdf = gpd.GeoDataFrame(saocom_df, geometry=geometry, crs='EPSG:4326')

# 4. Convert to UTM
saocom_gdf = saocom_gdf.to_crs('EPSG:32632')

# 5. Rename HEIGHT → HEIGHT_RELATIVE
saocom_gdf['HEIGHT_RELATIVE'] = saocom_gdf['HEIGHT']
```

### Columns Added During Analysis

| Column | Added In | Description |
|--------|----------|-------------|
| `geometry` | Cell 6 | Point geometry (UTM coordinates) |
| `HEIGHT_RELATIVE` | Cell 6 | Copy of HEIGHT column |
| `tinitaly_height` | Cell 17 | Sampled TINItaly DEM height |
| `copernicus_height` | Cell 17 | Sampled Copernicus DEM height |
| `HEIGHT_ABSOLUTE_TIN` | Cell 20 | Calibrated height (TINItaly reference) |
| `HEIGHT_ABSOLUTE_COP` | Cell 22 | Calibrated height (Copernicus reference) |
| `diff_tinitaly` | Cell 20 | Residual: SAOCOM - TINItaly |
| `diff_copernicus` | Cell 22 | Residual: SAOCOM - Copernicus |
| `outlier_score` | Cell 24 | Isolation Forest anomaly score |
| `slope_tin` | Cell 36 | Terrain slope from TINItaly |
| `aspect_tin` | Cell 36 | Terrain aspect from TINItaly |
| `slope_category` | Cell 38 | Slope classification |
| `corine_code` | Cell 40 | CORINE land cover code |
| `land_cover` | Cell 40 | CORINE Level 1 category |

## Expected Data Types

```python
ID                   int64      # Point identifier
SVET                 object     # (check your data)
LVET                 object     # (check your data)
LAT                  float64    # Latitude (degrees)
LAT2                 float64    # Latitude high precision (degrees)
LON                  float64    # Longitude (degrees)
LON2                 float64    # Longitude high precision (degrees)
HEIGHT               float64    # Relative height (meters)
HEIGHT WRT DEM       float64    # Height w.r.t. DEM (meters)
SIGMA HEIGHT         float64    # Height uncertainty (meters)
COHER                float64    # Coherence (0-1)
geometry             geometry   # Point (UTM)
```

## Quick Checks After Loading

```python
# After Cell 6, verify:
print(f"Coordinate columns used: {lat_col}, {lon_col}")
print(f"CRS: {saocom_gdf.crs}")
print(f"HEIGHT_RELATIVE created: {'HEIGHT_RELATIVE' in saocom_gdf.columns}")
print(f"COHER range: [{saocom_gdf['COHER'].min():.2f}, {saocom_gdf['COHER'].max():.2f}]")
print(f"Bounds (UTM): {saocom_gdf.total_bounds}")
```

## Common Issues & Solutions

### Issue: KeyError - Column not found

**Error:**
```
KeyError: 'EASTING'
```

**Solution:**
The notebook now uses LAT2/LON2 → LAT/LON, not EASTING/NORTHING. Make sure you're using the updated Cell 6.

### Issue: All heights are the same after calibration

**Check:**
```python
# Verify HEIGHT_RELATIVE has variation
print(saocom_gdf['HEIGHT_RELATIVE'].describe())
```

If all values are the same, check the original HEIGHT column in your CSV.

### Issue: No high-coherence points for calibration

**Error:**
```
Warning: Only X points with COHER >= 0.8
```

**Solution:**
Lower the coherence threshold:
```python
offset, rmse, n = calibrate_heights(
    saocom_gdf,
    ref_col='tinitaly_height',
    out_col='HEIGHT_ABSOLUTE_TIN',
    coherence_threshold=0.6  # Lowered from 0.8
)
```

## Summary

✅ Use **LAT2/LON2** for coordinates (falls back to LAT/LON)
✅ **HEIGHT** column becomes **HEIGHT_RELATIVE**
✅ **COHER** column used for quality filtering
✅ All coordinates converted to **UTM Zone 32N** for analysis
✅ Calibration creates **HEIGHT_ABSOLUTE_TIN** and **HEIGHT_ABSOLUTE_COP**
