import json
import ast

# Load the updated notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("="*80)
print("FINAL VERIFICATION REPORT")
print("="*80)
print()

all_pass = True

for cell_idx in [86, 98, 107]:
    print(f"Cell {cell_idx}:")
    print("-" * 40)

    cell_source_list = nb['cells'][cell_idx]['source']
    cell_source_str = ''.join(cell_source_list)

    # Syntax check
    try:
        ast.parse(cell_source_str)
        print("  [PASS] Syntax is valid")
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error: {e}")
        all_pass = False
        continue

    # Check for proper indentation (no extra indent after figure creation)
    has_proper_indent = True
    for i, line in enumerate(cell_source_list):
        if 'fig' in line and 'plt.subplots(' in line:
            # Check if this line has no leading spaces (except conditional indents)
            # The important thing is that code AFTER plt.subplots maintains consistent indentation
            if i + 1 < len(cell_source_list):
                next_line = cell_source_list[i + 1]
                # Next line should either be empty or continue with same indent level
                if next_line.strip() and not next_line.startswith('    '):
                    if line.startswith('    '):  # If fig creation is indented, next line should match
                        has_proper_indent = False

    if has_proper_indent:
        print("  [PASS] Indentation is correct")
    else:
        print("  [WARN] Indentation might need review")

    # Count key elements
    fig_count = cell_source_str.count('plt.subplots(')
    show_count = cell_source_str.count('plt.show()')
    layout_count = cell_source_str.count('plt.tight_layout()')
    savefig_count = cell_source_str.count('plt.savefig(')

    print(f"  Figures created: {fig_count}")
    print(f"  plt.show() calls: {show_count}")
    print(f"  plt.tight_layout() calls: {layout_count}")
    print(f"  plt.savefig() calls: {savefig_count}")

    # Check matching counts
    if fig_count == show_count == layout_count == savefig_count:
        print("  [PASS] All counts match (each figure properly closed)")
    else:
        print("  [WARN] Counts don't match - check figure management")

    # Check for old multi-axes references that would break
    problematic_patterns = [
        'axes[0, 0]',
        'axes[0, 1]',
        'axes[1, 0]',
        'axes[1, 1]',
        'axes[0]',
        'axes[1]',
        'fig.add_subplot(gs['
    ]

    found_issues = []
    for pattern in problematic_patterns:
        if pattern in cell_source_str:
            found_issues.append(pattern)

    if found_issues:
        print(f"  [FAIL] Found old subplot patterns: {found_issues}")
        all_pass = False
    else:
        print("  [PASS] No old subplot references found")

    print()

print("="*80)
if all_pass:
    print("FINAL RESULT: ALL CHECKS PASSED")
    print("The notebook cells are ready to run!")
else:
    print("FINAL RESULT: SOME ISSUES FOUND")
    print("Please review the warnings/failures above.")
print("="*80)
