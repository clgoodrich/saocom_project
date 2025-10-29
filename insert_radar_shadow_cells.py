"""
CAREFUL insertion of ONLY radar shadow cells - does NOT modify existing cells
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_cell_count = len(nb['cells'])
print(f"Original notebook: {original_cell_count} cells")

# Find insertion point AFTER slope calculation
insertion_index = None
for i in range(len(nb['cells'])):
    if nb['cells'][i]['cell_type'] == 'code':
        source = ''.join(nb['cells'][i].get('source', ''))
        # Look for slope sampling or after slope calculation
        if 'sample slope and aspect' in source.lower() or 'saocom_cleaned[\'slope\']' in source.lower():
            insertion_index = i + 1
            print(f"Found slope sampling at cell {i}, will insert after it")
            break

if insertion_index is None:
    print("Could not find slope calculation, using default position 38")
    insertion_index = 38

# Radar shadow cells to INSERT
radar_shadow_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## Radar Shadow and Geometry Analysis\n",
            "\n",
            "Analyze radar geometry effects: shadow, layover, foreshortening."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Radar Shadow Analysis\n",
            "from src.radar_geometry import (\n",
            "    calculate_local_incidence_angle,\n",
            "    identify_shadow_areas,\n",
            "    classify_geometric_quality,\n",
            "    analyze_shadow_statistics\n",
            ")\n",
            "\n",
            "RADAR_INCIDENCE = 35.0\n",
            "RADAR_AZIMUTH = 192.0\n",
            "\n",
            "print(f\"\\nRadar Geometry: {RADAR_INCIDENCE}° incidence, {RADAR_AZIMUTH}° azimuth\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate local incidence angles\n",
            "local_incidence = calculate_local_incidence_angle(slope, aspect, RADAR_INCIDENCE, RADAR_AZIMUTH)\n",
            "shadow_mask = identify_shadow_areas(local_incidence)\n",
            "geometric_quality = classify_geometric_quality(local_incidence, slope)\n",
            "\n",
            "print(f\"Shadow: {np.sum(shadow_mask)/np.sum(~np.isnan(local_incidence))*100:.1f}% of area\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sample at SAOCOM points\n",
            "from rasterio.transform import rowcol\n",
            "\n",
            "def sample_at_points(gdf, raster, transform):\n",
            "    values = []\n",
            "    for geom in gdf.geometry:\n",
            "        row, col = rowcol(transform, geom.x, geom.y)\n",
            "        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:\n",
            "            values.append(raster[row, col])\n",
            "        else:\n",
            "            values.append(np.nan)\n",
            "    return np.array(values)\n",
            "\n",
            "saocom_cleaned['local_incidence'] = sample_at_points(saocom_cleaned, local_incidence, dem_transform)\n",
            "saocom_cleaned['is_shadow'] = sample_at_points(saocom_cleaned, shadow_mask.astype(float), dem_transform).astype(bool)\n",
            "saocom_cleaned['geometric_quality'] = sample_at_points(saocom_cleaned, geometric_quality, dem_transform).astype(int)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze accuracy by geometry\n",
            "shadow_stats = analyze_shadow_statistics(saocom_cleaned, 'local_incidence', 'diff_tinitaly')\n",
            "\n",
            "print(\"\\nAccuracy by Geometry:\")\n",
            "for cat, stats in shadow_stats.items():\n",
            "    if stats['count'] > 0:\n",
            "        print(f\"{cat:12s}: n={stats['count']:5d}, RMSE={stats['rmse']:.2f}m, NMAD={stats['nmad']:.2f}m\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "# Local incidence\n",
            "im1 = axes[0].imshow(local_incidence, cmap='RdYlGn_r', vmin=0, vmax=90)\n",
            "axes[0].set_title('Local Incidence Angle')\n",
            "plt.colorbar(im1, ax=axes[0], label='Degrees')\n",
            "\n",
            "# Geometric quality\n",
            "from matplotlib.colors import ListedColormap\n",
            "colors = ['#2ecc71', '#f39c12', '#e74c3c', '#34495e', '#9b59b6']\n",
            "im2 = axes[1].imshow(geometric_quality, cmap=ListedColormap(colors), vmin=0, vmax=4)\n",
            "axes[1].set_title('Geometric Quality')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('images/radar_geometry_analysis.png', dpi=300, bbox_inches='tight')\n",
            "print(\"\\n✓ Saved: images/radar_geometry_analysis.png\")\n",
            "plt.show()"
        ]
    }
]

# ONLY INSERT - do not modify existing cells
new_cells = nb['cells'][:insertion_index] + radar_shadow_cells + nb['cells'][insertion_index:]
nb['cells'] = new_cells

new_cell_count = len(nb['cells'])
print(f"\\nNew notebook: {new_cell_count} cells")
print(f"Added: {new_cell_count - original_cell_count} cells")
print(f"Insertion point: {insertion_index}")

# Backup
import shutil
backup = notebook_path.with_suffix('.ipynb.backup')
shutil.copy2(notebook_path, backup)
print(f"\\nBackup: {backup}")

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\\n{'='*60}")
print(f"SUCCESS - Added {len(radar_shadow_cells)} cells ONLY")
print(f"Original {original_cell_count} cells preserved")
print(f"{'='*60}")
