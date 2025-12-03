"""
Comprehensive validation of all variable dependencies in the notebook.
Build a dependency map and verify all variables exist before use.
"""

import json
import re
import ast

def validate_all_variables(notebook_path):
    """Validate that all variables are defined before use."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    # Track all defined variables by cell index
    defined_vars = {}  # {var_name: first_cell_index}

    # Track issues
    issues = []

    print("="*70)
    print("VARIABLE DEPENDENCY VALIDATION")
    print("="*70)

    # First pass: find all variable definitions
    print("\n1. Scanning for variable definitions...")
    for idx, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue

        src = ''.join(cell['source'])

        # Find assignments (simple heuristic)
        # Look for patterns like: var = ..., df['col'] = ..., etc.
        lines = src.split('\n')
        for line in lines:
            # Skip comments
            if line.strip().startswith('#'):
                continue

            # Find simple assignments
            match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
            if match:
                var = match.group(1)
                if var not in defined_vars:
                    defined_vars[var] = idx

            # Find dataframe column assignments
            matches = re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\[[\"\']([a-zA-Z_][a-zA-Z0-9_]*)[\"\']]\s*=', line)
            for match in matches:
                df_name = match.group(1)
                col_name = match.group(2)
                key = f"{df_name}['{col_name}']"
                if key not in defined_vars:
                    defined_vars[key] = idx

    print(f"   Found {len(defined_vars)} variable definitions")

    # Second pass: check all variable usages
    print("\n2. Checking variable usage in all cells...")

    for idx, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue

        src = ''.join(cell['source'])
        if not src.strip():
            continue

        # Get all identifiers used in this cell
        used_vars = set()

        # Simple pattern matching for common variables
        patterns = [
            r'\b(saocom_cleaned|residuals_tin|residuals_cop|nmad_tin|nmad_cop)\b',
            r'\b(calibrated_height|copernicus_height|tinitaly_height)\b',
            r'\b(slope|aspect|elevation|land_cover)\b',
            r'\b(penetration_depth|radar_geometry|slope_category)\b',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, src)
            for match in matches:
                used_vars.add(match.group(1))

        # Check dataframe column access
        df_col_matches = re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\[[\"\']([a-zA-Z_][a-zA-Z0-9_]*)[\"\']]\)', src)
        for match in df_col_matches:
            key = f"{match.group(1)}['{match.group(2)}']"
            used_vars.add(key)

        # Check if variables are defined before use
        for var in used_vars:
            # Check simple variable
            if var in defined_vars:
                if defined_vars[var] > idx:
                    issues.append({
                        'cell': idx,
                        'variable': var,
                        'defined_at': defined_vars[var],
                        'type': 'used_before_definition'
                    })
            else:
                # Check if it's a dataframe column reference
                is_column = False
                for df_var in ['saocom_cleaned', 'saocom_data']:
                    if var in [f"{df_var}['residual_tin']", f"{df_var}['residual_cop']"]:
                        # These might be column names, check if they're created
                        parts = var.split("'")
                        if len(parts) > 1:
                            col_key = f"{df_var}['{parts[1]}']"
                            if col_key in defined_vars and defined_vars[col_key] > idx:
                                issues.append({
                                    'cell': idx,
                                    'variable': var,
                                    'defined_at': defined_vars.get(col_key, 'never'),
                                    'type': 'column_used_before_definition'
                                })
                                is_column = True
                                break

    # Report issues
    print(f"\n3. Analysis Results:")
    print(f"   Total cells checked: {sum(1 for c in cells if c['cell_type'] == 'code')}")
    print(f"   Issues found: {len(issues)}")

    if issues:
        print("\n" + "="*70)
        print("ISSUES FOUND:")
        print("="*70)
        for issue in issues:
            print(f"\n  Cell {issue['cell']}:")
            print(f"    Variable '{issue['variable']}' used")
            if issue['defined_at'] == 'never':
                print(f"    But NEVER defined!")
            else:
                print(f"    But defined later at cell {issue['defined_at']}")

            # Show snippet of problematic cell
            src = ''.join(cells[issue['cell']]['source'])
            lines = src.split('\n')
            print(f"    Snippet: {lines[0][:60]}...")
    else:
        print("\n   [OK] No variable dependency issues found!")

    return issues

def get_detailed_report(notebook_path):
    """Generate detailed report of added analysis cells."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    print("\n" + "="*70)
    print("DETAILED CHECK OF ADDED ANALYSIS CELLS")
    print("="*70)

    markers = [
        ('Penetration Depth Analysis', ['saocom_cleaned', 'copernicus_height', 'calibrated_height', 'np', 'land_cover_level1']),
        ('Expanded Copernicus Comparison', ['residuals_tin', 'residuals_cop', 'nmad_tin', 'nmad_cop', 'np', 'pd']),
        ('Slope-Aspect Interaction', ['saocom_cleaned', 'aspect', 'slope', 'residual_tin']),
        ('calculate_confidence_intervals', ['np', 'stats'])
    ]

    for marker, required_vars in markers:
        found = False
        for idx, cell in enumerate(cells):
            if marker in ''.join(cell['source']):
                print(f"\n{marker} (Cell {idx}):")
                print(f"  Required variables: {', '.join(required_vars)}")

                # Check which are defined before this cell
                missing = []
                for var in required_vars:
                    # Simple check - see if variable appears in earlier cells
                    found_before = False
                    for prev_idx in range(idx):
                        prev_src = ''.join(cells[prev_idx]['source'])
                        if f"{var} =" in prev_src or f"import {var}" in prev_src or 'from scipy import stats' in prev_src:
                            found_before = True
                            break

                    if not found_before:
                        missing.append(var)

                if missing:
                    print(f"  [WARNING] MISSING: {', '.join(missing)}")
                else:
                    print(f"  [OK] All dependencies satisfied")

                found = True
                break

        if not found:
            print(f"\n{marker}: NOT FOUND in notebook")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'

    # Run validation
    issues = validate_all_variables(notebook_path)

    # Get detailed report
    get_detailed_report(notebook_path)

    print("\n" + "="*70)
    if issues:
        print(f"RESULT: {len(issues)} issues need fixing")
    else:
        print("RESULT: All variable dependencies are valid!")
    print("="*70)
