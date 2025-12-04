import json
import ast
import sys

# Set UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load the updated notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("Verifying syntax of updated cells...\n")

for cell_idx in [86, 98, 107]:
    print(f"{'='*80}")
    print(f"CELL {cell_idx} - Syntax Check")
    print(f"{'='*80}")

    cell_source = ''.join(nb['cells'][cell_idx]['source'])

    try:
        ast.parse(cell_source)
        print(f"[OK] Cell {cell_idx}: SYNTAX OK")
        print(f"  Lines of code: {len(nb['cells'][cell_idx]['source'])}")

        # Count figures created
        fig_count = cell_source.count('plt.subplots(')
        print(f"  Figures created: {fig_count}")

        # Count plt.show() calls
        show_count = cell_source.count('plt.show()')
        print(f"  plt.show() calls: {show_count}")

        # Count plt.tight_layout() calls
        layout_count = cell_source.count('plt.tight_layout()')
        print(f"  plt.tight_layout() calls: {layout_count}")

    except SyntaxError as e:
        print(f"[ERROR] Cell {cell_idx}: SYNTAX ERROR")
        print(f"  Error: {e}")
        print(f"  Line {e.lineno}: {e.text}")

    print()

print("="*80)
print("Verification complete!")
