import json

# Load the updated notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Write output to file
with open(r'C:\users\colto\documents\github\saocom_project\fixed_cell_86_preview.txt', 'w', encoding='utf-8') as out:
    out.write("="*80 + "\n")
    out.write("CELL 86 - FIXED VERSION (First 40 lines)\n")
    out.write("="*80 + "\n\n")

    cell_source = ''.join(nb['cells'][86]['source'])
    lines = cell_source.split('\n')

    for i, line in enumerate(lines[:40], 1):
        out.write(f"{i:3d}: {line}\n")

    out.write(f"\n... ({len(lines)} total lines)\n")

print("Preview written to fixed_cell_86_preview.txt")
