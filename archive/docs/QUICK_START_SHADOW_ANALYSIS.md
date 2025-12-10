# Quick Start: Radar Shadow Analysis

## 🚀 5-Minute Setup

### Step 1: Copy Cells to Notebook

Open `notebooks/radar_shadow_analysis_cells.py` and copy all cells into your Jupyter notebook after the terrain analysis section.

### Step 2: Configure Radar Geometry

```python
# SAOCOM geometry parameters
RADAR_INCIDENCE = 35.0   # degrees (typical: 20-50°)
RADAR_AZIMUTH = 192.0    # degrees (192° = descending, 12° = ascending)
```

### Step 3: Run the Cells

Execute all cells sequentially. They will:
- Calculate local incidence angles
- Identify shadow and layover areas
- Analyze accuracy by geometric quality
- Generate visualizations

### Step 4: Review Outputs

**Images:**
- `images/radar_geometry_analysis.png` - Geometry maps
- `images/shadow_effect_on_accuracy.png` - Accuracy analysis

**Rasters:**
- `topography_outputs/radar_geometry/*.tif` - Geometric quality layers

**Statistics:**
- Printed table showing RMSE/NMAD by illumination category

## 📊 Expected Results

### Typical Findings

```
Geometric Quality Classification:
  Optimal:        45.2% of area  (RMSE: 2.5m)
  Acceptable:     38.7% of area  (RMSE: 3.8m)
  Foreshortening:  8.4% of area  (RMSE: 5.2m)
  Shadow:          5.1% of area  (RMSE: 8.9m)
  Layover:         2.6% of area  (RMSE: 6.7m)
```

### Key Insights

✅ **Shadow areas show degraded accuracy** (2-3× higher RMSE)
✅ **Layover areas have systematic biases**
✅ **Optimal geometry areas meet accuracy specs**
✅ **~40-50% of mountainous terrain in shadow/layover**

## 🔧 Adjusting Parameters

### If your area is flat (< 10° slope)
- Shadow effects minimal
- Focus on land cover stratification instead

### If shadow area > 30%
- Check radar geometry parameters
- Verify orbit direction (ascending/descending)
- Consider if very mountainous terrain

### If no shadow detected
- Terrain likely too gentle
- Verify slope calculation is working
- Check that incidence angle is reasonable (20-50°)

## 📈 Integration with Main Analysis

### Add to Final Report

Include these metrics in your validation report:
1. Geometric quality distribution (% area per class)
2. Accuracy metrics stratified by geometry
3. Visualization showing shadow-affected areas

### Masking Recommendations

For applications requiring high accuracy:
```python
# Filter out problematic geometry
reliable_data = saocom_gdf[saocom_gdf['geometric_quality'] <= 1]
# Keep only Optimal (0) and Acceptable (1) quality
```

## 📚 Full Documentation

See `docs/RADAR_SHADOW_ANALYSIS.md` for:
- Detailed theory
- Function references
- Advanced usage
- Troubleshooting

## ❓ Common Questions

**Q: What incidence angle should I use?**
A: Check SAOCOM metadata. Typical range is 20-50°, use 35° if unknown.

**Q: How do I know orbit direction?**
A: Descending (most common) ≈ 192° azimuth, Ascending ≈ 12° azimuth.

**Q: Shadow areas have no data, is that normal?**
A: Yes! Radar cannot see into shadow. Those areas should be masked.

**Q: Should I remove shadow areas from analysis?**
A: For accuracy statistics, report them separately. For applications, yes, mask them out.

## 🎯 Next Steps

After running shadow analysis:
1. ✅ Compare accuracy: shadow vs illuminated areas
2. ✅ Update filtering: remove/flag shadow areas
3. ✅ Combine with land cover: geometric quality × land cover
4. ✅ Add to publication figures: show shadow overlay on maps
5. ✅ Document in methods: mention geometric quality stratification

---

**Need Help?** Check `docs/RADAR_SHADOW_ANALYSIS.md` or review the function docstrings in `src/radar_geometry.py`.
