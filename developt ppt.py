"""
make_saocom_presentation.py

Generates SAOCOM InSAR DEM Validation Study PPTX using python-pptx.

Place this script in your project root. It will attempt to include images
from the "results/" folder (as produced by your notebook). If images are missing,
the slide text is still created.

Dependencies:
    pip install python-pptx pillow pandas
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Cm
from pathlib import Path
from PIL import Image
import os
import textwrap

# -------------------------
# Configuration / constants
# -------------------------
OUTFILE = "SAOCOM_DEM_Validation_Study.pptx"
RESULTS_DIR = Path("results")
prs = Presentation()
prs.slide_width = Inches(13.33)  # widescreen 16:9 (approx)
prs.slide_height = Inches(7.5)

# Reusable text formatting helper
def add_title_and_subtitle(slide, title, subtitle=None):
    """Adds a title (and optional subtitle) to any slide, even blank ones."""
    title_box = slide.shapes.title
    if title_box is None:
        # Create a new title text box manually at top of slide
        left = Inches(0.5)
        top = Inches(0.3)
        width = prs.slide_width - Inches(1.0)
        height = Inches(1.0)
        title_box = slide.shapes.add_textbox(left, top, width, height)
    tf = title_box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    if subtitle:
        left = Inches(0.5)
        top = Inches(1.2)
        width = prs.slide_width - Inches(1.0)
        height = Inches(1.0)
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf2 = txBox.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(14)
        p2.font.italic = True


def add_bullet_slide(title, bullets, notes=None):
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    add_title_and_subtitle(slide, title)
    body = slide.shapes.placeholders[1].text_frame
    body.clear()
    for b in bullets:
        p = body.add_paragraph()
        p.text = b
        p.level = 0
        p.font.size = Pt(14)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide

def safe_add_image(slide, image_path, left, top, width=None, height=None):
    if image_path and image_path.exists():
        try:
            # python-pptx needs images; optionally resize preserving aspect
            slide.shapes.add_picture(str(image_path), left, top, width=width, height=height)
            return True
        except Exception as e:
            print(f"Warning: could not insert image {image_path}: {e}")
            return False
    return False

def add_two_column_text(slide, left_col_lines, right_col_lines):
    left = Inches(0.5)
    top = Inches(1.3)
    mid = Inches(6.6)
    width = Inches(6.0)
    height = Inches(5.5)
    tb1 = slide.shapes.add_textbox(left, top, width, height).text_frame
    tb1.word_wrap = True
    tb1.clear()
    for line in left_col_lines:
        p = tb1.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
    tb2 = slide.shapes.add_textbox(mid, top, width, height).text_frame
    tb2.word_wrap = True
    tb2.clear()
    for line in right_col_lines:
        p = tb2.add_paragraph()
        p.text = line
        p.font.size = Pt(12)

# -------------------------
# Numerical results (from notebook)
# -------------------------
# These values were extracted from your notebook text and integrated into slides.
TINITALY_OFFSET = 4.308  # meters (median offset as reported)
TINITALY_OFFSET_N = 46920
COPERNICUS_OFFSET = 4.761
COPERNICUS_OFFSET_N = 46939

REF_RMSE = 4.68  # TINITALY vs Copernicus
REF_NMAD = 2.18

SAOCOM_VS_TIN_RMSE = 15.25
SAOCOM_VS_TIN_NMAD = 5.24

SAOCOM_VS_COP_RMSE = 15.40
SAOCOM_VS_COP_NMAD = 4.79

TOTAL_CELLS = 520380
OCCUPIED_CELLS = 67613
VOID_CELLS = 452767
VOID_PERCENT = 87.0

STUDY_AREA_KM2 = 52.04

# -------------------------
# Build slides
# -------------------------

# Slide 1: Title & Study Overview
slide_layout = prs.slide_layouts[0]  # title slide layout
slide = prs.slides.add_slide(slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "SAOCOM InSAR Digital Elevation Model Validation Study"
subtitle.text = "Land cover-stratified accuracy and spatial coverage analysis — Verona region"
slide.notes_slide.notes_text_frame.text = "Study components: point cloud validation, land cover-stratified assessment, void analysis, multi-dataset comparison."

# Slide 2: Datasets & Study Area
bullets = [
    "SAOCOM: L-band InSAR point cloud (CSV with LAT2, LON2, HEIGHT, COHER).",
    "TINITALY: 10 m reference DEM.",
    "Copernicus GLO-30: 30 m reference DEM.",
    "CORINE 2018: Land cover classification (level 3).",
    "Sentinel-2: RGB imagery (bands B04/B03/B02).",
    "Coordinate System: UTM Zone 32N (EPSG:32632).",
    f"Study area: convex hull around SAOCOM points in Verona (~{STUDY_AREA_KM2} km²)."
]
add_bullet_slide("Datasets & Study Area", bullets, notes="All datasets reprojected to EPSG:32632 and resampled to 10 m grid for comparison.")

# Slide 3: Processing Parameters
bullets = [
    "Coherence filtering: γ ≥ 0.3 to remove temporally unstable points.",
    "Target grid resolution: 10 m, NODATA value: -9999.",
    "Resampling: Cubic for continuous DEMs; Nearest neighbor for CORINE (categorical).",
    "k-NN isolated point removal: k=5, distance threshold=100 m.",
    "Valid-data mask: rasterized convex hull from SAOCOM points."
]
add_bullet_slide("Processing Parameters", bullets, notes="Coherence & k-NN parameters chosen to balance point retention and noise suppression; cubic resampling preserves terrain continuity.")

# Slide 4: Data Preprocessing Steps
bullets = [
    "SAOCOM: CSV loaded, columns standardized, numeric casting, invalid points removed (LAT2/LON2 != 0).",
    "Applied coherence filter (γ ≥ 0.3) then reprojected to UTM32N.",
    "Removed spatially-isolated points using k-NN (k=5, 100 m).",
    "Reference DEMs: reprojected to UTM32N and resampled to 10 m using cubic interpolation.",
    "Masked DEMs to SAOCOM convex hull and sampled DEM values at SAOCOM point pixel indices."
]
add_bullet_slide("Data Preprocessing Steps", bullets)

# Slide 5: Height Calibration Methodology
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Height Calibration Methodology", subtitle=None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
t = body.add_paragraph()
t.text = "Calibration approach: constant offset correction using robust median on stable points."
t.level = 0
t.font.size = Pt(14)
for line in [
    "Stable point criteria: Coherence ≥ 0.8, valid reference height, |HEIGHT_RELATIVE| < 1000 m (outlier guard).",
    f"TINITALY offset = median(Ref - SAOCOM_rel) = {TINITALY_OFFSET:.3f} m (N stable points = {TINITALY_OFFSET_N:,}).",
    f"Copernicus offset = {COPERNICUS_OFFSET:.3f} m (N stable points = {COPERNICUS_OFFSET_N:,}).",
    "SAOCOM_Absolute = SAOCOM_Relative + offset.",
    "Median used to avoid influence of long-tail outliers."
]:
    p = body.add_paragraph()
    p.text = line
    p.level = 0
    p.font.size = Pt(12)

# Slide 6: Reference DEM Cross-Comparison
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Reference DEM Cross-Comparison", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "Comparison: TINITALY (10 m) vs Copernicus (30 m resampled to 10 m).",
    f"Valid pixels compared: (masked region) — benchmark agreement.",
    f"Mean diff / Median diff ~ near zero; RMSE = {REF_RMSE:.2f} m; NMAD = {REF_NMAD:.2f} m.",
    "Computed: min/max, std, MAE, correlation and directional breakdown (TINITALY higher / Copernicus higher / roughly equal using NMAD tolerance)."
]:
    p = body.add_paragraph()
    p.text = line
    p.level = 0
    p.font.size = Pt(12)

# Slide 7: Statistical Metrics Definitions
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Statistical Metrics Definitions", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
metrics = [
    "Bias (ME): mean(Dataset1 - Dataset2).",
    "RMSE: sqrt(mean((Δh)^2)) — sensitive to outliers.",
    "MAE: mean(|Δh|).",
    "Std Dev: standard deviation of differences.",
    "NMAD: 1.4826 × median(|Δh - median(Δh)|) — robust spread estimator.",
    "LE68/LE90/LE95: percentiles of |Δh| (68/90/95)."
]
for m in metrics:
    p = body.add_paragraph()
    p.text = m
    p.level = 0
    p.font.size = Pt(12)

# Slide 8: Height Statistics Summary
slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only + blank
add_title_and_subtitle(slide, "Height Statistics Summary", None)
left = Inches(0.4); top = Inches(1.3); width = Inches(12.5); height = Inches(5.8)
tb = slide.shapes.add_textbox(left, top, width, height).text_frame
tb.clear()
summary_text = f"""
Summary statistics (selected):
- SAOCOM (relative) sample stats at points: (N = {len(range(1))})  -- see output CSVs for full table.
- TINITALY @ SAOCOM points: [mean, median, std, min, max].
- Copernicus @ SAOCOM points: [mean, median, std, min, max].
Comparison summary:
- SAOCOM - TINITALY: RMSE = {SAOCOM_VS_TIN_RMSE:.2f} m, NMAD = {SAOCOM_VS_TIN_NMAD:.2f} m.
- SAOCOM - Copernicus: RMSE = {SAOCOM_VS_COP_RMSE:.2f} m, NMAD = {SAOCOM_VS_COP_NMAD:.2f} m.
Note: Full numerical tables were exported to results/ as CSV for reproducibility.
"""
for line in textwrap.dedent(summary_text).strip().split("\n"):
    p = tb.add_paragraph()
    p.text = line.strip()
    p.font.size = Pt(12)

# Slide 9: Visualization - Reference DEM Comparison
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
add_title_and_subtitle(slide, "Visualization - Reference DEM Comparison", None)
# Try to place an 8-panel figure if exists
eight_panel_img = RESULTS_DIR / "reference_dem_8panel.png"
if safe_add_image(slide, eight_panel_img, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    pass
else:
    # fallback text
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.0), Inches(5.5)).text_frame
    tx.text = "8-panel figure (TINITALY map, Copernicus map, differences, histograms) — saved in results/ (reference_dem_8panel.png) if available."
    tx.paragraphs[0].font.size = Pt(12)

# Slide 10: Coverage Analysis
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Coverage Analysis", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    f"Total 10 m cells (study area): {TOTAL_CELLS:,}",
    f"SAOCOM occupied cells: {OCCUPIED_CELLS:,}",
    f"Void cells (no data): {VOID_CELLS:,} — Void percentage: {VOID_PERCENT:.1f}%",
    "Void mask raster saved: results/saocom_void_mask.tif (boolean raster: 0=data, 1=void)."
]:
    p = body.add_paragraph()
    p.text = line
    p.level = 0
    p.font.size = Pt(12)

# Slide 11: CORINE Land Cover Processing
bullets = [
    "DBF lookup used to remap CORINE Value → CODE_18 → LABEL3 (level 3 classes).",
    "Remapped raster resampled to 10 m using nearest neighbor and masked to hull.",
    "Colorblind-friendly palette applied (grouped by 1xx/2xx/3xx/4xx/5xx).",
    "Classes present: see results/corine_classes_present.csv; maps exported as landcover_{code}_{name}.png"
]
add_bullet_slide("CORINE Land Cover Processing", bullets)

# Slide 12: Land Cover Sampling at SAOCOM Points
bullets = [
    "CORINE raster sampled at each SAOCOM point location via row/col of 10 m grid transform.",
    "Points outside grid bounds and NoData were removed.",
    "Resulting analysis table: saocom_lc_analysis DataFrame (exported to results/)."
]
add_bullet_slide("Land Cover Sampling at SAOCOM Points", bullets)

# Slide 13: Height Residuals by Land Cover
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Height Residuals by Land Cover", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "Grouped by CORINE Level 3 classes; minimum 50 points per class enforced.",
    "Metrics: N_points, median_diff_m, mean_diff_m, std_dev_m, NMAD_m.",
    "Residual = Calibrated SAOCOM Height - TINITALY reference.",
    "Results show urban areas with low spread; vineyards and orchards show large spread and high NMAD."
]:
    p = body.add_paragraph()
    p.text = line
    p.level = 0
    p.font.size = Pt(12)

# Slide 14: Void Analysis by Land Cover
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Void Analysis by Land Cover", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "Per-class metrics: total cells, void_cells, area_km2, void_area_km2, pct_LC_is_void, pct_of_total_voids.",
    "Filter: area > 1.0 km² to remove tiny patches.",
    "Top void classes (examples): Vineyards (99.7% void), Non-irrigated arable land (98.6%)."
]:
    p = body.add_paragraph()
    p.text = line
    p.level = 0
    p.font.size = Pt(12)

# Slide 15: Violin Plot Analysis - Overall Comparison
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_title_and_subtitle(slide, "Violin Plot Analysis - Overall Comparison", None)
vp_img = RESULTS_DIR / "violin_overall.png"
if not safe_add_image(slide, vp_img, Inches(0.5), Inches(1.4), height=Inches(5.7)):
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(12.0), Inches(5.7)).text_frame
    tx.text = ("Violin plots (SAOCOM-TIN, SAOCOM-COP) saved in results/violin_overall.png if generated. "
               "Plots include coherence-binned violins and overall distributions.")
    tx.paragraphs[0].font.size = Pt(12)

# Slide 16: Violin Plot Analysis - By Land Cover
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Violin Plot Analysis - By Land Cover", None)
body=slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "Common CORINE codes with N≥50 plotted (outliers trimmed to 0.5th-99.5th percentile).",
    "Synchronized X axis across TIN/COP comparisons with 5% padding for comparability.",
    "Three modes: TINITALY-only, Copernicus-only, side-by-side comparison."
]:
    p=body.add_paragraph(); p.text=line; p.font.size=Pt(12)

# Slide 17: CORINE Land Cover Map
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_and_subtitle(slide, "CORINE Land Cover Map", None)
lc_img = RESULTS_DIR / "corine_map_masked.png"
if not safe_add_image(slide, lc_img, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(5.4)).text_frame.text = (
        "CORINE map (masked) should be in results/corine_map_masked.png. Legend shows only classes present; scale bar lower right."
    )

# Slide 18: Sentinel-2 RGB Processing
add_bullet_slide("Sentinel-2 RGB Processing", [
    "Bands: B04 (Red), B03 (Green), B02 (Blue) reprojected to UTM32N and resampled to 10 m (bilinear).",
    "Stacked, percentile-normalized (2nd-98th) and clipped 0-1 for consistent visualization backgrounds.",
    "Saved normalized RGB as results/sentinel_rgb_norm.png for use in overlays."
])

# Slide 19: Individual Land Cover Overlays
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Individual Land Cover Overlays", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "One map per present land cover class with Sentinel background (70% opacity) and hatching overlays.",
    "Filled polygons with black outlines and hatching; files: landcover_{code}_{safe_name}.png (44 files)."
]:
    p = body.add_paragraph(); p.text=line; p.font.size=Pt(12)

# Slide 20: SAOCOM Comparison - Gridded Analysis
add_bullet_slide("SAOCOM Comparison - Gridded Analysis", [
    "Filter: COHER ≥ 0.3, valid reference heights, elevation range 50–850 m for stable comparisons.",
    "Difference grids created by nearest neighbor interpolation and masked to hull.",
    "Metrics: n_points, mean, median, std, rmse, mae, nmad, min, max, correlation."
])

# Slide 21: SAOCOM Comparison Visualization
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_and_subtitle(slide, "SAOCOM Comparison Visualization", None)
cmp_img = RESULTS_DIR / "saocom_comparison_directional.png"
if not safe_add_image(slide, cmp_img, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(5.2)).text_frame.text = (
        "Directional comparison figure (6-panel) saved as results/saocom_comparison_directional.png."
    )

# Slide 22: Scatter Plot Comparisons
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_and_subtitle(slide, "Scatter Plot Comparisons", None)
scatter_img = RESULTS_DIR / "scatter_comparisons.png"
if not safe_add_image(slide, scatter_img, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(5.2)).text_frame.text = (
        "Scatter plots SAOCOM vs TIN, SAOCOM vs Copernicus and TIN vs Copernicus saved as results/scatter_comparisons.png."
    )

# Slide 23: Void Zones by Land Cover - Map
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_and_subtitle(slide, "Void Zones by Land Cover - Map", None)
void_map = RESULTS_DIR / "voids_by_landcover_map.png"
if not safe_add_image(slide, void_map, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(5.2)).text_frame.text = (
        "Void-by-landcover map saved in results/voids_by_landcover_map.png"
    )

# Slide 24: Void Zones by Land Cover - Charts
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Void Zones by Land Cover - Charts", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "Chart 1: Top landcover classes by % of class void (descending).",
    "Chart 2: Top classes by % of total void area (largest absolute contributors).",
    "Charts saved as results/voids_by_landcover_charts.png."
]:
    p = body.add_paragraph(); p.text=line; p.font.size=Pt(12)

# Slide 25: Land Cover with Voids - Outlined Visualization
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_and_subtitle(slide, "Land Cover with Voids - Outlined Visualization", None)
outline_img = RESULTS_DIR / "landcover_voids_outlined.png"
if not safe_add_image(slide, outline_img, Inches(0.4), Inches(1.4), height=Inches(5.9)):
    slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12.0), Inches(5.2)).text_frame.text = (
        "Detailed landcover-with-voids map: results/landcover_voids_outlined.png"
    )

# Slide 26: Individual Coverage/Void Maps per Class
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Individual Coverage/Void Maps per Class", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
for line in [
    "One per class: coverage (hatching), void (white with dot hatching), saved as landcover_{code}_{name}_coverage_void.png",
    "Statistics shown: total area, coverage area, void area, coverage %, void %."
]:
    p = body.add_paragraph(); p.text=line; p.font.size = Pt(12)

# Slide 27: Coverage Summary Table
slide = prs.slides.add_slide(prs.slide_layouts[5])
add_title_and_subtitle(slide, "Coverage Summary Table", None)
# Insert a short table summary text (full table available as CSV)
left = Inches(0.5); top = Inches(1.5); width = Inches(12.0); height = Inches(5.5)
tb = slide.shapes.add_textbox(left, top, width, height).text_frame
tb.clear()
tb.text = ("Full table of per-class coverage and void % exported to results/coverage_by_landcover.csv.\n"
           "Columns: code, name, coverage_pct, void_pct, total_km2. Sorted by void_pct descending.")
tb.paragraphs[0].font.size = Pt(12)

# Slide 28: Processing Outputs Summary
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Processing Outputs Summary", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
outputs = [
    "Resampled DEMs: results/tinitaly_10m.tif, results/copernicus_10m.tif",
    "Masked DEMs: results/tinitaly_10m_masked.tif, results/copernicus_10m_masked.tif",
    "CORINE remapped: results/corine_remapped.tif, results/corine_10m_masked.tif",
    "Void mask: results/saocom_void_mask.tif",
    "Figures: saocom_comparison_directional.png, voids_by_landcover_map.png, voids_by_landcover_charts.png",
    "Individual maps: landcover_{code}_{name}.png (44 files) and *_coverage_void.png"
]
for o in outputs:
    p = body.add_paragraph(); p.text = o; p.font.size = Pt(12)

# Slide 29: Code Structure Summary
slide = prs.slides.add_slide(prs.slide_layouts[1])
add_title_and_subtitle(slide, "Code Structure Summary", None)
body = slide.shapes.placeholders[1].text_frame
body.clear()
workflow = [
    "1) Data loading & standardization (SAOCOM CSV, TINITALY, Copernicus, CORINE, Sentinel).",
    "2) CRS standardization to EPSG:32632 and resampling to 10 m grid.",
    "3) Coherence filtering, k-NN cleaning, hull mask generation.",
    "4) Height calibration via median offset (stable points, γ ≥ 0.8).",
    "5) Statistical validation: classical & robust metrics (RMSE, NMAD, MAE, LE percentiles).",
    "6) Land cover stratification and void analysis; comprehensive visualizations exported."
]
for w in workflow:
    p = body.add_paragraph(); p.text = w; p.font.size = Pt(12)

# Final slide: Contact / Notes
slide = prs.slides.add_slide(prs.slide_layouts[5])
add_title_and_subtitle(slide, "Notes & How to Re-run", None)
left = Inches(0.5); top = Inches(1.6); width = Inches(12.5); height = Inches(5.2)
tb = slide.shapes.add_textbox(left, top, width, height).text_frame
tb.clear()
instructions = [
    "To reproduce: run the Jupyter notebook provided (saocom_v3_merged) in a Python environment with geopandas, rasterio, scipy, scikit-learn, matplotlib.",
    "This PPTX script reads figures and files from results/. If results/ is not present, slides still contain textual summaries.",
    f"PPTX saved as: {OUTFILE}",
    "Contact: use the notebook's author metadata for provenance and data lineage."
]
for line in instructions:
    p = tb.add_paragraph(); p.text = line; p.font.size = Pt(12)

# -------------------------
# Save presentation
# -------------------------
prs.save(OUTFILE)
print(f"Presentation saved to {OUTFILE}")
