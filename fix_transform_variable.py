"""
Fix cell 48: Change dem_transform to target_transform
"""

import json
from pathlib import Path
import shutil

notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells")

# Fix cell 48
cell_48 = nb['cells'][48]
source_text = ''.join(cell_48.get('source', ''))

if 'dem_transform' in source_text:
    print(f"Found dem_transform in cell 48")

    # Replace dem_transform with target_transform
    new_source = source_text.replace('dem_transform', 'target_transform')

    # Convert back to list format with newlines preserved
    nb['cells'][48]['source'] = new_source.split('\n')
    nb['cells'][48]['source'] = [line + '\n' if i < len(nb['cells'][48]['source'])-1 else line
                                   for i, line in enumerate(nb['cells'][48]['source'])]

    # Backup
    backup = notebook_path.with_suffix('.ipynb.backup4')
    shutil.copy2(notebook_path, backup)
    print(f"Backup: {backup}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FIXED: Cell 48 now uses target_transform")
    print(f"{'='*60}")
else:
    print(f"ERROR: dem_transform not found in cell 48")
