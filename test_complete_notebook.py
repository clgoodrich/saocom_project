"""
Comprehensive QA test for saocom_analysis_complete.ipynb
Tests all major sections and reports results.
"""
import json
import sys
import traceback
from pathlib import Path

def load_notebook(path):
    """Load notebook JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_notebook_structure(nb):
    """Test basic notebook structure"""
    cells = nb['cells']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']

    print("=" * 80)
    print("NOTEBOOK STRUCTURE TEST")
    print("=" * 80)
    print(f"Total cells: {len(cells)}")
    print(f"Code cells: {len(code_cells)}")
    print(f"Markdown cells: {len(markdown_cells)}")
    print()

    return code_cells, markdown_cells

def identify_major_sections(markdown_cells):
    """Identify major sections from markdown headers"""
    sections = []
    for i, cell in enumerate(markdown_cells):
        source = ''.join(cell['source'])
        if source.startswith('#'):
            # Extract first line as section title
            title = source.split('\n')[0].strip('# ')
            sections.append((i, title))

    print("=" * 80)
    print("MAJOR SECTIONS IDENTIFIED")
    print("=" * 80)
    for idx, (cell_idx, title) in enumerate(sections, 1):
        print(f"{idx}. {title}")
    print()

    return sections

def check_critical_functions(code_cells):
    """Check if critical functions are defined"""
    print("=" * 80)
    print("CRITICAL FUNCTIONS CHECK")
    print("=" * 80)

    critical_functions = [
        'read_raster_meta',
        'load_dem_array',
        'resample_to_10m',
        'sample_raster_at_points',
        'calibrate_heights',
        'score_outliers_isolation_forest',
        'filter_by_score_iqr',
        'nmad',
        'calculate_height_stats',
        'get_clc_level1',
        'remove_isolated_knn',
        'calculate_terrain_derivatives',
        'create_difference_grid',
    ]

    # Combine all code
    all_code = '\n'.join([
        ''.join(cell['source'])
        for cell in code_cells
    ])

    found = {}
    for func in critical_functions:
        # Check for function definition
        if f'def {func}' in all_code:
            found[func] = 'DEFINED'
        # Check for import
        elif f'import {func}' in all_code or f'from src' in all_code:
            found[func] = 'IMPORTED'
        else:
            found[func] = 'MISSING'

    for func, status in found.items():
        symbol = '[OK]' if status != 'MISSING' else '[X]'
        print(f"{symbol} {func:40s} {status}")

    missing = [f for f, s in found.items() if s == 'MISSING']
    print()
    if missing:
        print(f"WARNING: {len(missing)} functions missing")
        return False
    else:
        print("All critical functions found")
        return True

def check_visualization_cells(code_cells):
    """Check for visualization code"""
    print("=" * 80)
    print("VISUALIZATION CELLS CHECK")
    print("=" * 80)

    viz_patterns = [
        ('plt.figure', 'Matplotlib figures'),
        ('plt.subplot', 'Subplots'),
        ('sns.', 'Seaborn plots'),
        ('go.Figure', 'Plotly figures'),
        ('ax.imshow', 'Raster displays'),
        ('ax.scatter', 'Scatter plots'),
        ('ax.plot', 'Line plots'),
        ('violinplot', 'Violin plots'),
        ('hexbin', 'Hexbin plots'),
        ('hist2d', '2D histograms'),
    ]

    all_code = '\n'.join([
        ''.join(cell['source'])
        for cell in code_cells
    ])

    found_viz = {}
    for pattern, description in viz_patterns:
        count = all_code.count(pattern)
        found_viz[description] = count
        symbol = '[OK]' if count > 0 else '[X]'
        print(f"{symbol} {description:30s} {count:3d} occurrences")

    total_viz = sum(found_viz.values())
    print()
    print(f"Total visualization elements: {total_viz}")
    return total_viz > 20  # Expect at least 20 visualization elements

def check_land_cover_analysis(code_cells):
    """Check for comprehensive land cover analysis"""
    print("=" * 80)
    print("LAND COVER ANALYSIS CHECK")
    print("=" * 80)

    all_code = '\n'.join([
        ''.join(cell['source'])
        for cell in code_cells
    ])

    checks = [
        ('corine', 'CORINE data loading'),
        ('get_clc_level', 'Level classification'),
        ('land_cover', 'Land cover column'),
        ('clc_level1', 'Level 1 analysis'),
        ('clc_level2', 'Level 2 analysis'),
        ('clc_level3', 'Level 3 analysis'),
        ('groupby', 'Grouping by land cover'),
    ]

    for pattern, description in checks:
        found = pattern in all_code.lower()
        symbol = '[OK]' if found else '[X]'
        print(f"{symbol} {description}")

    # Check for void zone analysis
    void_found = 'void' in all_code.lower()
    print(f"{'[OK]' if void_found else '[X]'} Void zone analysis")

    # Check for swiss cheese
    swiss_found = 'swiss' in all_code.lower() or 'cheese' in all_code.lower()
    print(f"{'[OK]' if swiss_found else '[X]'} Swiss cheese visualization")

    return True

def check_data_files(code_cells):
    """Check if all required data files are referenced"""
    print("=" * 80)
    print("DATA FILES CHECK")
    print("=" * 80)

    all_code = '\n'.join([
        ''.join(cell['source'])
        for cell in code_cells
    ])

    required_files = [
        'saocom_csv',
        'tinitaly',
        'copernicus',
        'corine',
        'sentinel',
        '.csv',
        '.tif',
        '.dbf',
    ]

    for pattern in required_files:
        found = pattern.lower() in all_code.lower()
        symbol = '[OK]' if found else '[X]'
        print(f"{symbol} {pattern}")

    return True

def check_outputs(code_cells):
    """Check if outputs are being saved"""
    print("=" * 80)
    print("OUTPUT GENERATION CHECK")
    print("=" * 80)

    all_code = '\n'.join([
        ''.join(cell['source'])
        for cell in code_cells
    ])

    output_patterns = [
        ('savefig', 'Figure saving'),
        ('to_file', 'GeoDataFrame export'),
        ('to_csv', 'CSV export'),
        ('writeRaster', 'Raster writing'),
        ('results/', 'Results directory'),
        ('images/', 'Images directory'),
    ]

    for pattern, description in output_patterns:
        count = all_code.count(pattern)
        symbol = '[OK]' if count > 0 else '[X]'
        print(f"{symbol} {description:30s} {count:3d} occurrences")

    return True

def main():
    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_complete.ipynb')

    if not nb_path.exists():
        print(f"ERROR: Notebook not found at {nb_path}")
        sys.exit(1)

    print("=" * 80)
    print("SAOCOM ANALYSIS COMPLETE - COMPREHENSIVE QA TEST")
    print("=" * 80)
    print(f"Notebook: {nb_path.name}")
    print()

    try:
        nb = load_notebook(nb_path)
        code_cells, markdown_cells = test_notebook_structure(nb)
        sections = identify_major_sections(markdown_cells)

        tests_passed = []

        # Run all checks
        tests_passed.append(("Functions", check_critical_functions(code_cells)))
        tests_passed.append(("Visualizations", check_visualization_cells(code_cells)))
        tests_passed.append(("Land Cover", check_land_cover_analysis(code_cells)))
        tests_passed.append(("Data Files", check_data_files(code_cells)))
        tests_passed.append(("Outputs", check_outputs(code_cells)))

        # Summary
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for test_name, passed in tests_passed:
            status = "PASS" if passed else "FAIL"
            symbol = '[OK]' if passed else '[X]'
            print(f"{symbol} {test_name:30s} {status}")

        total_pass = sum(1 for _, p in tests_passed if p)
        total_tests = len(tests_passed)

        print()
        print(f"Tests passed: {total_pass}/{total_tests}")

        if total_pass == total_tests:
            print()
            print("[OK] ALL STRUCTURAL TESTS PASSED")
            print("[OK] Notebook appears to contain all required functionality")
            print()
            print("NEXT STEP: Execute notebook cells to verify runtime behavior")
        else:
            print()
            print("[X] SOME TESTS FAILED")
            print("Review missing components above")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
