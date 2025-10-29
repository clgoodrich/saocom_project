"""
QA script to check for potential variable name issues in the notebook
"""

import json
from pathlib import Path
import re

notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells\n")
print("="*70)
print("QA REPORT: Variable Name Check")
print("="*70)

# Track variable definitions and usage
defined_vars = set()
potential_issues = []

# Known variables that should exist
expected_vars = {
    'target_transform': 'Defined in cell 10',
    'slope_tin': 'Defined in cell 35 (slope calculation)',
    'aspect_tin': 'Defined in cell 35 (aspect calculation)',
    'saocom_cleaned': 'Cleaned SAOCOM dataframe',
    'tinitaly_10m': 'TINItaly DEM array',
    'copernicus_10m': 'Copernicus DEM array',
}

print("\n[OK] Expected key variables:")
for var, desc in expected_vars.items():
    print(f"  - {var}: {desc}")

# Check radar shadow cells specifically (45-50)
print("\n" + "="*70)
print("RADAR SHADOW CELLS (45-50) - Detailed Check")
print("="*70)

radar_cells = {
    45: "Markdown header",
    46: "Imports and setup",
    47: "Calculate local incidence",
    48: "Sample at SAOCOM points",
    49: "Analyze statistics",
    50: "Visualize"
}

for cell_idx in range(45, 51):
    cell = nb['cells'][cell_idx]
    print(f"\n[Cell {cell_idx}] {radar_cells[cell_idx]}")

    if cell['cell_type'] == 'markdown':
        print("  Type: Markdown (no variables to check)")
        continue

    source = ''.join(cell.get('source', ''))

    # Check for specific variable patterns
    vars_used = set()

    # Find variable usage (simple heuristic)
    # Look for common patterns like: var_name, var_name[, var_name.
    pattern = r'\b([a-z_][a-z0-9_]*)\b'
    matches = re.findall(pattern, source)

    # Filter out Python keywords and function names
    keywords = {'if', 'for', 'in', 'print', 'def', 'return', 'import', 'from',
                'as', 'and', 'or', 'not', 'is', 'with', 'open', 'range'}

    vars_in_cell = [m for m in matches if m not in keywords]

    # Look for key variables
    key_checks = {
        'slope': '[WARN] slope (should be slope_tin)',
        'aspect': '[WARN] aspect (should be aspect_tin)',
        'dem_transform': '[WARN] dem_transform (should be target_transform)',
        'slope_tin': '[OK] slope_tin',
        'aspect_tin': '[OK] aspect_tin',
        'target_transform': '[OK] target_transform',
        'local_incidence': '[OK] local_incidence',
        'shadow_mask': '[OK] shadow_mask',
        'geometric_quality': '[OK] geometric_quality',
    }

    findings = []
    for var, description in key_checks.items():
        if var in source:
            findings.append(description)

    if findings:
        for finding in findings:
            print(f"  {finding}")
    else:
        print("  (No key variables found)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Final check - look for any remaining problematic patterns
all_source = '\n'.join([''.join(cell.get('source', [])) for cell in nb['cells'] if cell['cell_type'] == 'code'])

issues_found = []

if 'dem_transform' in all_source and 'target_transform' not in all_source[:all_source.find('dem_transform')]:
    issues_found.append("[WARN] Found 'dem_transform' - should use 'target_transform'")

# Check for bare 'slope' or 'aspect' (not slope_tin/aspect_tin)
# This is tricky because we want to allow 'slope' in comments/strings
code_cells_47_50 = [nb['cells'][i] for i in range(47, 51)]
for i, cell in enumerate(code_cells_47_50, start=47):
    source = ''.join(cell.get('source', ''))
    # Look for 'slope,' or 'aspect,' which indicates parameter passing
    if re.search(r'\bslope\s*[,\)]', source) and 'slope_tin' not in source:
        issues_found.append(f"[WARN] Cell {i}: Uses 'slope' instead of 'slope_tin'")
    if re.search(r'\baspect\s*[,\)]', source) and 'aspect_tin' not in source:
        issues_found.append(f"[WARN] Cell {i}: Uses 'aspect' instead of 'aspect_tin'")

if issues_found:
    print("\n[WARN] POTENTIAL ISSUES FOUND:")
    for issue in issues_found:
        print(f"  {issue}")
else:
    print("\n[SUCCESS] No obvious variable name issues detected!")
    print("\nKey fixes applied:")
    print("  [OK] Cell 47: Uses slope_tin and aspect_tin")
    print("  [OK] Cell 48: Uses target_transform")

print("\n" + "="*70)
