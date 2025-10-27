"""
Build the FINAL notebook: saocom_analysis_final.ipynb

This will have ALL functionality from the original but be:
- Well organized with clear sections
- Educational with explanatory markdown
- Easy to follow
- Properly commented
- Student-friendly
"""

import json
from pathlib import Path

def create_cell(cell_type, source, metadata=None):
    """Create a notebook cell"""
    cell = {
        'cell_type': cell_type,
        'metadata': metadata or {},
        'source': source if isinstance(source, list) else [source]
    }
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell

def create_notebook():
    """Create the final notebook"""

    cells = []

    # Title
    cells.append(create_cell('markdown', [
        '# SAOCOM InSAR Height Validation Analysis\n',
        '\n',
        '**Project:** Validation of SAOCOM satellite InSAR-derived heights against reference DEMs\n',
        '\n',
        '**Reference DEMs:**\n',
        '- TINItaly (10m resolution) - High accuracy reference\n',
        '- Copernicus DEM (30m resolution, resampled to 10m)\n',
        '\n',
        '**Analysis includes:**\n',
        '- Height calibration and residual analysis\n',
        '- Outlier detection using machine learning\n',
        '- Land cover classification (CORINE)\n',
        '- Terrain analysis (slope, aspect)\n',
        '- Void zone analysis\n',
        '- Comprehensive statistical validation\n',
        '- Publication-quality visualizations\n',
        '\n',
        '---\n'
    ]))

    # Section 1: Setup
    cells.append(create_cell('markdown', [
        '## 1. Setup and Configuration\n',
        '\n',
        'Import required libraries and configure paths and parameters.\n'
    ]))

    cells.append(create_cell('code', [
        '# Core libraries\n',
        'import sys\n',
        'import os\n',
        'import warnings\n',
        'warnings.filterwarnings(\'ignore\')\n',
        '\n',
        '# Data handling\n',
        'import numpy as np\n',
        'import pandas as pd\n',
        'from pathlib import Path\n',
        '\n',
        '# Geospatial\n',
        'import geopandas as gpd\n',
        'import rasterio\n',
        'from rasterio.warp import reproject, Resampling, calculate_default_transform\n',
        'from rasterio.transform import from_bounds, rowcol\n',
        'from rasterio.mask import mask\n',
        'from rasterio import features\n',
        'from rasterio.features import shapes\n',
        'from shapely.geometry import Point, box, shape\n',
        '\n',
        '# Visualization\n',
        'import matplotlib.pyplot as plt\n',
        'import matplotlib.patches as mpatches\n',
        'from matplotlib.colors import ListedColormap, BoundaryNorm\n',
        'from matplotlib_scalebar.scalebar import ScaleBar\n',
        'import seaborn as sns\n',
        '\n',
        '# Other\n',
        'from dbfread import DBF\n',
        '\n',
        'print("All libraries imported successfully")\n'
    ]))

    cells.append(create_cell('code', [
        '# Project paths\n',
        'DATA_DIR = Path("data")\n',
        'RESULTS_DIR = Path("results")\n',
        'IMAGES_DIR = Path("images")\n',
        '\n',
        '# Create output directories\n',
        'RESULTS_DIR.mkdir(exist_ok=True)\n',
        'IMAGES_DIR.mkdir(exist_ok=True)\n',
        '\n',
        '# Analysis parameters\n',
        'COHERENCE_THRESHOLD = 0.3  # Minimum coherence for data inclusion\n',
        'CALIBRATION_COHERENCE = 0.8  # High coherence for calibration\n',
        'NODATA = -9999  # NoData value for rasters\n',
        'GRID_SIZE = 10  # Target grid resolution (meters)\n',
        'TARGET_CRS = \'EPSG:32632\'  # UTM Zone 32N\n',
        '\n',
        'print(f"Data directory: {DATA_DIR.absolute()}")\n',
        'print(f"Results will be saved to: {RESULTS_DIR.absolute()}")\n',
        'print(f"Figures will be saved to: {IMAGES_DIR.absolute()}")\n'
    ]))

    # CORINE definitions (from original)
    cells.append(create_cell('code', [
        '# CORINE Land Cover class definitions (Level 3)\n',
        'CORINE_CLASSES = {\n',
        '    111: \'Continuous urban fabric\', 112: \'Discontinuous urban fabric\',\n',
        '    121: \'Industrial or commercial units\', 122: \'Road and rail networks and associated land\',\n',
        '    123: \'Port areas\', 124: \'Airports\', 131: \'Mineral extraction sites\',\n',
        '    132: \'Dump sites\', 133: \'Construction sites\', 141: \'Green urban areas\',\n',
        '    142: \'Sport and leisure facilities\', 211: \'Non-irrigated arable land\',\n',
        '    212: \'Permanently irrigated land\', 213: \'Rice fields\', 221: \'Vineyards\',\n',
        '    222: \'Fruit trees and berry plantations\', 223: \'Olive groves\',\n',
        '    231: \'Pastures\', 241: \'Annual crops associated with permanent crops\',\n',
        '    242: \'Complex cultivation patterns\', 243: \'Agriculture/natural vegetation mix\',\n',
        '    244: \'Agro-forestry areas\', 311: \'Broad-leaved forest\',\n',
        '    312: \'Coniferous forest\', 313: \'Mixed forest\', 321: \'Natural grasslands\',\n',
        '    322: \'Moors and heathland\', 323: \'Sclerophyllous vegetation\',\n',
        '    324: \'Transitional woodland-shrub\', 331: \'Beaches, dunes, sands\',\n',
        '    332: \'Bare rocks\', 333: \'Sparsely vegetated areas\', 334: \'Burnt areas\',\n',
        '    335: \'Glaciers and perpetual snow\', 411: \'Inland marshes\',\n',
        '    412: \'Peat bogs\', 421: \'Salt marshes\', 422: \'Salines\',\n',
        '    423: \'Intertidal flats\', 511: \'Water courses\', 512: \'Water bodies\',\n',
        '    521: \'Coastal lagoons\', 522: \'Estuaries\', 523: \'Sea and ocean\'\n',
        '}\n',
        '\n',
        'print(f"CORINE classes defined: {len(CORINE_CLASSES)} classes")\n'
    ]))

    # Data file discovery
    cells.append(create_cell('code', [
        '# Discover data files\n',
        'file_discovery = {\n',
        '    \'saocom\': ("saocom_csv", "*.csv"),\n',
        '    \'tinitaly\': ("tinitaly", "*.tif"),\n',
        '    \'copernicus\': ("", "copernicus*.tif"),\n',
        '    \'corine\': ("ground_cover", "*.tif"),\n',
        '    \'sentinel\': ("sentinel_data", "*.tif")\n',
        '}\n',
        '\n',
        'files_found = {}\n',
        'for key, (subdir, pattern) in file_discovery.items():\n',
        '    search_dir = DATA_DIR / subdir if subdir else DATA_DIR\n',
        '    file_list = list(search_dir.glob(pattern))\n',
        '    if file_list:\n',
        '        files_found[key] = file_list[0]\n',
        '        print(f"Found {key:12s}: {file_list[0].name}")\n',
        '    else:\n',
        '        print(f"WARNING: {key:12s} not found in {search_dir}")\n',
        '\n',
        '# Store paths as variables\n',
        'saocom_path = files_found.get(\'saocom\')\n',
        'tinitaly_path = files_found.get(\'tinitaly\')\n',
        'copernicus_path = files_found.get(\'copernicus\')\n',
        'corine_path = files_found.get(\'corine\')\n',
        'sentinel_path = files_found.get(\'sentinel\')\n',
        '\n',
        '# Find CORINE DBF lookup table\n',
        'if corine_path:\n',
        '    corine_dbf_candidates = list((DATA_DIR / "ground_cover").glob(f"{corine_path.stem}.vat.dbf"))\n',
        '    corine_dbf_path = corine_dbf_candidates[0] if corine_dbf_candidates else None\n',
        '    if corine_dbf_path:\n',
        '        print(f"Found CORINE DBF: {corine_dbf_path.name}")\n'
    ]))

    # Section 2: Load SAOCOM Data
    cells.append(create_cell('markdown', [
        '## 2. Load SAOCOM InSAR Data\n',
        '\n',
        'Load SAOCOM point measurements from CSV and convert to GeoDataFrame.\n',
        '\n',
        '**Data columns:**\n',
        '- LAT2, LON2: High-precision geographic coordinates\n',
        '- HEIGHT: Relative InSAR height (requires calibration)\n',
        '- COHER: Temporal coherence (quality metric, 0-1)\n',
        '- SIGMA HEIGHT: Height uncertainty estimate\n'
    ]))

    cells.append(create_cell('code', [
        '# Load SAOCOM CSV\n',
        'saocom_df = pd.read_csv(saocom_path)\n',
        'print(f"Loaded {len(saocom_df):,} SAOCOM points")\n',
        'print(f"Columns: {list(saocom_df.columns)}")\n',
        '\n',
        '# Use highest precision coordinates available\n',
        'lat_col = \'LAT2\' if \'LAT2\' in saocom_df.columns else \'LAT\'\n',
        'lon_col = \'LON2\' if \'LON2\' in saocom_df.columns else \'LON\'\n',
        'print(f"Using coordinates: {lat_col}, {lon_col}")\n',
        '\n',
        '# Create GeoDataFrame\n',
        'geometry = [Point(xy) for xy in zip(saocom_df[lon_col], saocom_df[lat_col])]\n',
        'saocom_gdf = gpd.GeoDataFrame(saocom_df, geometry=geometry, crs=\'EPSG:4326\')\n',
        '\n',
        '# Transform to UTM\n',
        'saocom_gdf = saocom_gdf.to_crs(TARGET_CRS)\n',
        'print(f"Transformed to {TARGET_CRS}")\n',
        '\n',
        '# Create HEIGHT_RELATIVE column\n',
        'saocom_gdf[\'HEIGHT_RELATIVE\'] = saocom_gdf[\'HEIGHT\']\n',
        '\n',
        '# Display summary\n',
        'print(f"\\nData summary:")\n',
        'print(f"  Bounds (UTM): {saocom_gdf.total_bounds}")\n',
        'print(f"  Coherence range: [{saocom_gdf[\'COHER\'].min():.2f}, {saocom_gdf[\'COHER\'].max():.2f}]")\n',
        'print(f"  Height range: [{saocom_gdf[\'HEIGHT\'].min():.2f}, {saocom_gdf[\'HEIGHT\'].max():.2f}] m")\n'
    ]))

    # Continue building...
    # This is getting long, let me create the full notebook programmatically

    # Save notebook
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    output_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_final.ipynb')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Created notebook with {len(cells)} cells (so far)")
    print(f"Saved to: {output_path}")
    print()
    print("NOTE: This is just the beginning - continuing to build full notebook...")

    return output_path

if __name__ == '__main__':
    create_notebook()
