"""
Standalone script to generate Word report from existing notebook results.
Run this after executing saocom_analysis_clean.ipynb to create the Word document.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.export_utils import export_statistics_to_word, create_summary_statistics_dict
from src.statistics_prog import nmad

# Define paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'
IMAGES_DIR = PROJECT_ROOT / 'images'
SAOCOM_CLEANED_SHP = RESULTS_DIR / 'saocom_cleaned.shp'

print("="*70)
print("SAOCOM VALIDATION - WORD REPORT GENERATOR")
print("="*70)
print()

# Check if cleaned data exists
if not SAOCOM_CLEANED_SHP.exists():
    print(f"ERROR: Cleaned SAOCOM shapefile not found at: {SAOCOM_CLEANED_SHP}")
    print("Please run the full notebook first to generate the cleaned dataset.")
    exit(1)

print(f"[1/5] Loading cleaned SAOCOM data...")
saocom_cleaned = gpd.read_file(SAOCOM_CLEANED_SHP)
print(f"      Loaded {len(saocom_cleaned):,} points")

print(f"\n[2/5] Computing residuals and statistics...")
# Compute residuals
residuals_tin = saocom_cleaned['diff_tinitaly'].dropna().values
residuals_cop = saocom_cleaned['diff_cop'].dropna().values

# Compute NMAD
nmad_tin = nmad(residuals_tin)
nmad_cop = nmad(residuals_cop)

print(f"      TINItaly:   n={len(residuals_tin):,}, NMAD={nmad_tin:.2f}m")
print(f"      Copernicus: n={len(residuals_cop):,}, NMAD={nmad_cop:.2f}m")

# Prepare overall statistics
overall_stats = create_summary_statistics_dict(
    residuals_tin, residuals_cop, nmad_tin, nmad_cop
)

print(f"\n[3/5] Computing stratified statistics...")
# Slope statistics
slope_stats = None
if 'slope_category' in saocom_cleaned.columns:
    slope_stats = saocom_cleaned.groupby('slope_category')['diff_tinitaly'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('nmad', lambda x: nmad(x.dropna()))
    ])
    print(f"      Slope categories: {len(slope_stats)}")

# Land cover statistics
lc_stats = None
if 'land_cover' in saocom_cleaned.columns:
    lc_stats = saocom_cleaned.groupby('land_cover')['diff_tinitaly'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('nmad', lambda x: nmad(x.dropna()))
    ])
    print(f"      Land cover types: {len(lc_stats)}")

# Height statistics
height_stats = None
if 'height_category' in saocom_cleaned.columns:
    height_stats = saocom_cleaned.groupby('height_category')['diff_tinitaly'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('nmad', lambda x: nmad(x.dropna()))
    ])
    print(f"      Height categories: {len(height_stats)}")

# Shadow statistics (if available)
shadow_stats = None
if 'shadow_type' in saocom_cleaned.columns or 'geometry_type' in saocom_cleaned.columns:
    category_col = 'shadow_type' if 'shadow_type' in saocom_cleaned.columns else 'geometry_type'
    shadow_groups = saocom_cleaned.groupby(category_col)['diff_tinitaly']
    shadow_stats = {}
    for category, group in shadow_groups:
        vals = group.dropna().values
        if len(vals) > 0:
            shadow_stats[category] = {
                'count': len(vals),
                'bias': float(np.mean(vals)),
                'rmse': float(np.sqrt(np.mean(vals**2))),
                'nmad': float(nmad(vals))
            }
    print(f"      Shadow/geometry categories: {len(shadow_stats)}")

print(f"\n[4/5] Generating Word document...")
word_output_path = RESULTS_DIR / 'saocom_validation_report.docx'

export_statistics_to_word(
    output_path=word_output_path,
    overall_stats=overall_stats,
    slope_stats=slope_stats,
    landcover_stats=lc_stats,
    height_stats=height_stats,
    shadow_stats=shadow_stats,
    images_dir=IMAGES_DIR,
    include_images=True,
    title='SAOCOM DEM Validation Analysis - Comprehensive Report'
)

print(f"\n[5/5] Complete!")
print()
print("="*70)
print("REPORT GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nWord document saved to:")
print(f"  {word_output_path.absolute()}")
print()
print("The report includes:")
print("  - Overall validation statistics (TINItaly & Copernicus)")
print("  - Stratified statistics (slope, land cover, elevation)")
print("  - Shadow/geometry analysis (if available)")
print("  - Key figures and visualizations")
print()
