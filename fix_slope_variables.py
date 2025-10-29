"""
Minimal fix: Change slope/aspect to slope_tin/aspect_tin in cell 47
"""

import json
from pathlib import Path
import shutil

notebook_path = Path('saocom_analysis_clean.ipynb')
print(f"Loading: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells")

# Fix cell 47
cell_47 = nb['cells'][47]
source_text = ''.join(cell_47.get('source', ''))

if 'calculate_local_incidence_angle' in source_text and 'slope,' in source_text:
    print(f"Found target cell at index 47")

    # Simple replacement: slope -> slope_tin, aspect -> aspect_tin
    new_source = source_text.replace(
        "    slope,\n    aspect,",
        "    slope_tin,\n    aspect_tin,"
    ).replace(
        "geometric_quality = classify_geometric_quality(local_incidence, slope)",
        "geometric_quality = classify_geometric_quality(local_incidence, slope_tin)"
    )

    # Convert back to list format
    nb['cells'][47]['source'] = new_source.split('\n')
    # Re-add newlines except for last line
    nb['cells'][47]['source'] = [line + '\n' if i < len(nb['cells'][47]['source'])-1 else line
                                   for i, line in enumerate(nb['cells'][47]['source'])]

    # Backup
    backup = notebook_path.with_suffix('.ipynb.backup3')
    shutil.copy2(notebook_path, backup)
    print(f"Backup: {backup}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"FIXED: Cell 47 now uses slope_tin and aspect_tin")
    print(f"{'='*60}")
else:
    print(f"ERROR: Cell 47 is not the expected cell")
    print(f"Content: {source_text[:100]}")
