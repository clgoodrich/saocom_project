"""
Fix the gridded comparison cell in saocom_analysis_clean.ipynb
The create_difference_grid function returns a tuple (grid, points) but code treats it as single value
"""

import json
from pathlib import Path

def fix_gridded_comparison():
    """Fix the gridded comparison code"""

    # Load notebook
    nb_path = Path('C:/Users/colto/Documents/GitHub/saocom_project/saocom_analysis_clean.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells")

    # Find and fix cell 57 (gridded comparison code)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'diff_grid_tin = create_difference_grid' in source and 'diff_grid_cop = create_difference_grid' in source:
                print(f"Found gridded comparison in cell {i}")

                # Fix the source code to unpack the tuple
                new_source = source.replace(
                    'diff_grid_tin = create_difference_grid(',
                    'diff_grid_tin, _ = create_difference_grid('
                ).replace(
                    'diff_grid_cop = create_difference_grid(',
                    'diff_grid_cop, _ = create_difference_grid('
                )

                # Update the cell
                cell['source'] = new_source.split('\n') if '\n' in new_source else [new_source]
                print("Fixed!")
                break

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Saved fixed notebook to {nb_path.name}")

if __name__ == '__main__':
    fix_gridded_comparison()
