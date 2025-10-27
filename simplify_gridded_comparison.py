"""
Replace the complex gridded comparison with a simpler working version
"""

import json
from pathlib import Path

def simplify_gridded():
    """Replace with simpler gridded comparison"""

    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find and replace the gridded comparison cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'diff_grid_tin, _ = create_difference_grid' in source:
                print(f"Found gridded comparison in cell {i} - replacing with simpler version")

                # Simpler version that just creates grids from point data
                new_code = """# Create gridded difference maps (simplified)
print("Creating gridded difference maps...")

# Create simple gridded visualization using scatter plot rasterization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# TINItaly grid - create from point residuals
valid_tin_pts = saocom_cleaned[saocom_cleaned['diff_tinitaly'].notna()]
if len(valid_tin_pts) > 0:
    vmin, vmax = np.percentile(valid_tin_pts['diff_tinitaly'], [2, 98])

    # Create gridded view using hexbin
    hb1 = axes[0].hexbin(
        valid_tin_pts.geometry.x,
        valid_tin_pts.geometry.y,
        C=valid_tin_pts['diff_tinitaly'],
        gridsize=100,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        reduce_C_function=np.mean
    )
    plt.colorbar(hb1, ax=axes[0], label='Difference (m)')
    axes[0].set_title('SAOCOM - TINItaly (Gridded)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')
    axes[0].set_aspect('equal')

# Copernicus grid
valid_cop_pts = saocom_cleaned[saocom_cleaned['diff_copernicus'].notna()]
if len(valid_cop_pts) > 0:
    vmin2, vmax2 = np.percentile(valid_cop_pts['diff_copernicus'], [2, 98])

    hb2 = axes[1].hexbin(
        valid_cop_pts.geometry.x,
        valid_cop_pts.geometry.y,
        C=valid_cop_pts['diff_copernicus'],
        gridsize=100,
        cmap='RdBu_r',
        vmin=vmin2,
        vmax=vmax2,
        reduce_C_function=np.mean
    )
    plt.colorbar(hb2, ax=axes[1], label='Difference (m)')
    axes[1].set_title('SAOCOM - Copernicus (Gridded)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Easting (m)')
    axes[1].set_ylabel('Northing (m)')
    axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(IMAGES_DIR / 'gridded_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
"""

                cell['source'] = [new_code]
                print("Replaced with simplified version!")
                break

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Saved fixed notebook")

if __name__ == '__main__':
    simplify_gridded()
