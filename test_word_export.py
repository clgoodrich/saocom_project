"""Quick test script for Word export functionality."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.export_utils import export_statistics_to_word, create_summary_statistics_dict

# Create sample data
np.random.seed(42)
residuals_tin = np.random.normal(0, 2, 1000)
residuals_cop = np.random.normal(0.5, 3, 1000)
nmad_tin = 1.4826 * np.median(np.abs(residuals_tin - np.median(residuals_tin)))
nmad_cop = 1.4826 * np.median(np.abs(residuals_cop - np.median(residuals_cop)))

# Prepare overall statistics
overall_stats = create_summary_statistics_dict(
    residuals_tin, residuals_cop, nmad_tin, nmad_cop
)

# Create sample slope statistics
slope_stats = pd.DataFrame({
    'count': [500, 300, 150, 50],
    'mean': [0.2, 0.5, 1.2, 2.5],
    'std': [1.5, 2.0, 3.0, 4.5],
    'nmad': [1.2, 1.8, 2.5, 3.8]
}, index=['Flat (0-5°)', 'Gentle (5-15°)', 'Moderate (15-30°)', 'Steep (>30°)'])
slope_stats.index.name = 'slope_category'

# Create sample land cover statistics
lc_stats = pd.DataFrame({
    'count': [400, 350, 200, 50],
    'mean': [0.1, 0.3, 0.8, 1.5],
    'std': [1.2, 1.8, 2.5, 3.2],
    'nmad': [1.0, 1.5, 2.2, 2.8]
}, index=['Forest', 'Grassland', 'Urban', 'Agricultural'])
lc_stats.index.name = 'land_cover'

# Create sample shadow statistics
shadow_stats = {
    'Illuminated': {'count': 800, 'bias': 0.1, 'rmse': 1.5, 'nmad': 1.2},
    'Shadow': {'count': 150, 'bias': 1.5, 'rmse': 3.5, 'nmad': 2.8},
    'Layover': {'count': 50, 'bias': 2.5, 'rmse': 4.5, 'nmad': 3.5}
}

# Test output path
output_path = Path('results/test_export.docx')

print("Testing Word export functionality...")
print("=" * 60)

try:
    export_statistics_to_word(
        output_path=output_path,
        overall_stats=overall_stats,
        slope_stats=slope_stats,
        landcover_stats=lc_stats,
        shadow_stats=shadow_stats,
        images_dir='images',
        include_images=False,  # Skip images for quick test
        title='SAOCOM Validation - Test Export'
    )
    print("\n" + "=" * 60)
    print("SUCCESS! Word export test completed.")
    print("=" * 60)
    print(f"\nTest document created at: {output_path.absolute()}")
    print("\nYou can now run the full notebook to generate the complete report!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
