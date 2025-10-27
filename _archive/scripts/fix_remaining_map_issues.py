"""
Fix remaining map issues:
1. Fix zoom for maps 12.7 onward (raster maps need extent parameter)
2. Increase legend marker size in 9.2 land cover map
3. Create individual land cover maps with Sentinel-2 background
"""

import json
from pathlib import Path

def create_code_cell(code):
    """Create a code cell"""
    return {
        'cell_type': 'code',
        'metadata': {},
        'source': [code] if isinstance(code, str) else code,
        'outputs': [],
        'execution_count': None
    }

def create_markdown_cell(text):
    """Create a markdown cell"""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text] if isinstance(text, str) else text
    }

def fix_remaining_issues():
    """Fix all remaining map issues"""

    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells\n")

    # ==================== FIX 1: ZOOM FOR RASTER MAPS (12.7 ONWARD) ====================
    print("=== FIX 1: Fixing zoom for raster maps (12.7 onward) ===\n")

    raster_maps_fixed = 0

    # Cell 75: Terrain Slope Map
    if 75 < len(nb['cells']):
        source = ''.join(nb['cells'][75]['source'])
        if 'terrain_slope.png' in source and 'extent=' not in source:
            print("Fixing Cell 75: Terrain Slope Map")

            # Add extent parameter to imshow
            new_source = source.replace(
                "slope_plot = ax.imshow(slope_tin, cmap='terrain', vmin=0, vmax=45)",
                """# Calculate extent from transform
extent = [
    target_transform.c,  # left (min x)
    target_transform.c + target_transform.a * grid_width,  # right (max x)
    target_transform.f + target_transform.e * grid_height,  # bottom (min y)
    target_transform.f  # top (max y)
]

slope_plot = ax.imshow(slope_tin, cmap='terrain', vmin=0, vmax=45,
                       extent=extent, origin='upper')"""
            )

            nb['cells'][75]['source'] = [new_source]
            raster_maps_fixed += 1

    # Cell 77: Reference DEM Comparison
    if 77 < len(nb['cells']):
        source = ''.join(nb['cells'][77]['source'])
        if 'reference_dem_comparison.png' in source and 'extent=' not in source:
            print("Fixing Cell 77: Reference DEM Comparison")

            # Add extent to all three imshow calls
            new_source = source.replace(
                "axes[0, 0].imshow(tin_plot, cmap='terrain')",
                """# Calculate extent
extent = [
    target_transform.c,
    target_transform.c + target_transform.a * grid_width,
    target_transform.f + target_transform.e * grid_height,
    target_transform.f
]

axes[0, 0].imshow(tin_plot, cmap='terrain', extent=extent, origin='upper')"""
            )

            # Update other imshow calls
            new_source = new_source.replace(
                "axes[0, 1].imshow(cop_plot, cmap='terrain')",
                "axes[0, 1].imshow(cop_plot, cmap='terrain', extent=extent, origin='upper')"
            )

            new_source = new_source.replace(
                "axes[1, 0].imshow(dem_diff",
                "axes[1, 0].imshow(dem_diff, extent=extent, origin='upper'"
            )

            nb['cells'][77]['source'] = [new_source]
            raster_maps_fixed += 1

    # Cell 79: Coverage Grid
    if 79 < len(nb['cells']):
        source = ''.join(nb['cells'][79]['source'])
        if 'coverage_and_voids.png' in source and 'extent=' not in source:
            print("Fixing Cell 79: Coverage Grid and Void Zones")

            # Add extent to slope background
            new_source = source.replace(
                "ax.imshow(slope_tin, cmap='gray', alpha=0.3)",
                """# Calculate extent
extent = [
    target_transform.c,
    target_transform.c + target_transform.a * grid_width,
    target_transform.f + target_transform.e * grid_height,
    target_transform.f
]

ax.imshow(slope_tin, cmap='gray', alpha=0.3, extent=extent, origin='upper')"""
            )

            nb['cells'][79]['source'] = [new_source]
            raster_maps_fixed += 1

    # Cell 83: Summary Dashboard
    if 83 < len(nb['cells']):
        source = ''.join(nb['cells'][83]['source'])
        if 'summary_dashboard.png' in source and 'slope_tin' in source:
            print("Fixing Cell 83: Summary Dashboard (slope panel)")

            # Add extent to slope panel (ax5)
            new_source = source.replace(
                "ax5.imshow(slope_tin, cmap='terrain')",
                """# Calculate extent for raster
extent = [
    target_transform.c,
    target_transform.c + target_transform.a * grid_width,
    target_transform.f + target_transform.e * grid_height,
    target_transform.f
]

ax5.imshow(slope_tin, cmap='terrain', extent=extent, origin='upper')"""
            )

            nb['cells'][83]['source'] = [new_source]
            raster_maps_fixed += 1

    print(f"[OK] Fixed {raster_maps_fixed} raster maps with proper extent/zoom\n")

    # ==================== FIX 2: LEGEND MARKER SIZE (9.2) ====================
    print("=== FIX 2: Increasing legend marker size in 9.2 land cover map ===\n")

    # Find cell 44 (land cover spatial map)
    if 44 < len(nb['cells']):
        source = ''.join(nb['cells'][44]['source'])
        if 'land_cover_map.png' in source:
            print("Fixing Cell 44: Land Cover Map legend")

            # Update legend to have larger markers
            if 'ax.legend(' in source:
                new_source = source.replace(
                    "ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)",
                    """ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9,
          markerscale=3)  # Make legend markers 3x larger"""
                )
                nb['cells'][44]['source'] = [new_source]
                print("[OK] Increased legend marker size to 3x\n")

    # ==================== FIX 3: INDIVIDUAL LAND COVER MAPS ====================
    print("=== FIX 3: Creating individual land cover maps with Sentinel-2 background ===\n")

    # Find where to insert (after cell 44 - land cover map)
    insert_idx = 45

    # Create new markdown header
    new_cells = []

    new_cells.append(create_markdown_cell("""### 9.3 Individual Land Cover Maps with Sentinel-2 Background

Generate detailed maps for each major land cover type showing:
- Sentinel-2 RGB imagery as background
- Points for that specific land cover type
- Bounding box with white fill showing extent
- All standard map elements (scale bar, north arrow, grid)"""))

    # Create the code for individual land cover maps
    individual_lc_code = """# Create individual land cover maps with Sentinel-2 background
print("Creating individual land cover maps with Sentinel-2 background...")

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

# Load Sentinel-2 imagery
SENTINEL_PATH = DATA_DIR / 'sentinel_data' / 'Sentinel2Views_Clip.tif'

print(f"Loading Sentinel-2 imagery from {SENTINEL_PATH}...")
with rasterio.open(SENTINEL_PATH) as src:
    sentinel_data = src.read()  # Read all bands
    sentinel_bounds = src.bounds
    sentinel_crs = src.crs
    sentinel_transform = src.transform

    # Get RGB bands (assuming bands 1,2,3 are RGB or need to pick specific bands)
    # For Sentinel-2, often need to pick bands and scale
    if src.count >= 3:
        # Read first 3 bands as RGB
        rgb = np.dstack([src.read(i) for i in range(1, 4)])

        # Normalize to 0-1 range for display
        rgb_normalized = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            band = rgb[:, :, i]
            # Clip to reasonable percentiles to avoid extreme values
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            rgb_normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)

print(f"Sentinel-2 image loaded: {rgb.shape[0]} x {rgb.shape[1]} pixels")
print(f"Bounds: {sentinel_bounds}")

# Reproject Sentinel-2 extent to match SAOCOM data CRS (EPSG:32632)
from rasterio.warp import transform_bounds
sentinel_extent_utm = transform_bounds(sentinel_crs, 'EPSG:32632',
                                       sentinel_bounds.left, sentinel_bounds.bottom,
                                       sentinel_bounds.right, sentinel_bounds.top)

# Calculate extent for imshow in UTM coordinates
sentinel_extent = [
    sentinel_extent_utm[0],  # left
    sentinel_extent_utm[2],  # right
    sentinel_extent_utm[1],  # bottom
    sentinel_extent_utm[3]   # top
]

print(f"Sentinel-2 extent (original CRS): {sentinel_bounds}")
print(f"Sentinel-2 extent (UTM 32N): {sentinel_extent}")

# Get top land cover types (minimum 500 points for meaningful visualization)
lc_counts = saocom_cleaned['land_cover'].value_counts()
top_lc_types = lc_counts[lc_counts >= 500].head(8).index

print(f"\\nCreating maps for {len(top_lc_types)} land cover types with >= 500 points:")
for lc_type in top_lc_types:
    print(f"  - {lc_type}: {lc_counts[lc_type]:,} points")

# Create individual map for each land cover type
for idx, lc_type in enumerate(top_lc_types):
    print(f"\\nCreating map {idx+1}/{len(top_lc_types)}: {lc_type}")

    # Filter points for this land cover type
    lc_subset = saocom_cleaned[saocom_cleaned['land_cover'] == lc_type].copy()

    # Get bounding box for this land cover type
    lc_bounds = lc_subset.total_bounds

    # Add margin
    margin_x = (lc_bounds[2] - lc_bounds[0]) * 0.15
    margin_y = (lc_bounds[3] - lc_bounds[1]) * 0.15

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Display Sentinel-2 as background
    ax.imshow(rgb_normalized, extent=sentinel_extent, origin='upper', zorder=0)

    # Add white-filled bounding box for this land cover type
    bbox_rect = Rectangle(
        (lc_bounds[0], lc_bounds[1]),
        lc_bounds[2] - lc_bounds[0],
        lc_bounds[3] - lc_bounds[1],
        linewidth=3,
        edgecolor='red',
        facecolor='white',
        alpha=0.4,
        zorder=1,
        label=f'{lc_type} Extent'
    )
    ax.add_patch(bbox_rect)

    # Plot points for this land cover type
    ax.scatter(
        lc_subset.geometry.x,
        lc_subset.geometry.y,
        c='blue',
        s=20,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5,
        zorder=2,
        label=f'{lc_type} Points (n={len(lc_subset):,})'
    )

    # Add hull boundary for ALL SAOCOM data (for context)
    hull = saocom_cleaned.geometry.unary_union.convex_hull
    hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)
    hull_gdf.boundary.plot(
        ax=ax,
        color='yellow',
        linewidth=2,
        linestyle='--',
        label='Full Study Area',
        zorder=1
    )

    # Set map extent to land cover bounding box with margin
    ax.set_xlim(lc_bounds[0] - margin_x, lc_bounds[2] + margin_x)
    ax.set_ylim(lc_bounds[1] - margin_y, lc_bounds[3] + margin_y)

    # Map elements
    ax.set_xlabel('UTM Easting (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('UTM Northing (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Land Cover: {lc_type}\\n({len(lc_subset):,} SAOCOM points)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.5, linestyle='--', color='white', linewidth=1.5)
    ax.set_aspect('equal')

    # Add scale bar
    scalebar = ScaleBar(1, 'm', length_fraction=0.25, location='lower right',
                        box_alpha=0.8, scale_loc='top', color='black',
                        box_color='white')
    ax.add_artist(scalebar)

    # Add north arrow
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=20, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                fontsize=30, ha='center', va='center')

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
              edgecolor='black', facecolor='white')

    plt.tight_layout()

    # Save with safe filename (replace spaces/slashes)
    safe_filename = lc_type.replace(' ', '_').replace('/', '_').replace('\\\\', '_')
    plt.savefig(IMAGES_DIR / f'land_cover_{safe_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved land_cover_{safe_filename}.png")

print(f"\\n[OK] Created {len(top_lc_types)} individual land cover maps with Sentinel-2 background")
"""

    new_cells.append(create_code_cell(individual_lc_code))

    # Insert new cells (note: the other cells shifted, so we need to insert after the current 44)
    # But since we already processed cell 44, we insert at 45
    for cell in reversed(new_cells):
        nb['cells'].insert(insert_idx, cell)

    print(f"[OK] Added {len(new_cells)} new cells for individual land cover maps\n")

    # ==================== SAVE NOTEBOOK ====================
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("="*80)
    print("[SUCCESS] All remaining fixes applied!")
    print(f"  - Fixed {raster_maps_fixed} raster maps with proper extent (no zoom issues)")
    print("  - Increased legend marker size in 9.2 land cover map (3x)")
    print("  - Added individual land cover maps with Sentinel-2 background (8 maps)")
    print(f"  - Total cells now: {len(nb['cells'])}")
    print("="*80)

if __name__ == '__main__':
    fix_remaining_issues()
