"""
Fix cell 39 to handle slope variable correctly - MINIMAL CHANGE
"""

import json
from pathlib import Path

notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells")

# Find and fix cell 39 (the local incidence calculation)
# Cell 39 should be the one that says "# Calculate local incidence angles"
target_cell_index = 39

# Check if this is the right cell
cell_39 = nb['cells'][target_cell_index]
source_text = ''.join(cell_39.get('source', ''))

if '# Calculate local incidence angles' in source_text:
    print(f"Found target cell at index {target_cell_index}")

    # Replace with more robust version that checks for slope
    new_source = [
        "# Calculate local incidence angles\n",
        "# Check if slope and aspect exist, if not calculate them\n",
        "if 'slope' not in locals() or 'aspect' not in locals():\n",
        "    print(\"Calculating slope and aspect from TINItaly DEM...\")\n",
        "    from src.preprocessing import calculate_terrain_derivatives\n",
        "    from src.utils import load_dem_array\n",
        "    import rasterio\n",
        "    \n",
        "    tinitaly_path = 'data/tinitaly/tinitaly_crop.tif'\n",
        "    with rasterio.open(tinitaly_path) as src:\n",
        "        dem_array = src.read(1)\n",
        "        dem_transform = src.transform\n",
        "    \n",
        "    slope, aspect = calculate_terrain_derivatives(dem_array, cellsize=10, nodata=-9999)\n",
        "    print(\"✓ Calculated slope and aspect\")\n",
        "\n",
        "local_incidence = calculate_local_incidence_angle(slope, aspect, RADAR_INCIDENCE, RADAR_AZIMUTH)\n",
        "shadow_mask = identify_shadow_areas(local_incidence)\n",
        "geometric_quality = classify_geometric_quality(local_incidence, slope)\n",
        "\n",
        "print(f\"Shadow: {np.sum(shadow_mask)/np.sum(~np.isnan(local_incidence))*100:.1f}% of area\")"
    ]

    # Update the cell
    nb['cells'][target_cell_index]['source'] = new_source

    # Save backup
    import shutil
    backup = notebook_path.with_suffix('.ipynb.backup2')
    shutil.copy2(notebook_path, backup)
    print(f"Backup: {backup}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FIXED: Cell 39 now checks for slope/aspect")
    print(f"If missing, it will calculate them automatically")
    print(f"{'='*60}")
else:
    print(f"ERROR: Cell 39 is not the expected cell")
    print(f"Content: {source_text[:100]}")
