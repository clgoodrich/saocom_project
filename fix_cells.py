import json
import sys

# Load notebooks
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_backup.ipynb', encoding='utf-8') as f:
    backup_nb = json.load(f)

with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    current_nb = json.load(f)

# Write to file with UTF-8
with open(r'C:\users\colto\documents\github\saocom_project\cells_content.txt', 'w', encoding='utf-8') as out:
    for cell_idx in [86, 98, 107]:
        out.write(f"\n{'='*80}\n")
        out.write(f"CELL {cell_idx} FROM BACKUP\n")
        out.write(f"{'='*80}\n")
        cell_source = ''.join(backup_nb['cells'][cell_idx]['source'])
        out.write(cell_source)
        out.write(f"\n{'='*80}\n\n")

print("Cell contents written to cells_content.txt")
