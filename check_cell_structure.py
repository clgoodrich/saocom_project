import json

# Load the updated notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Check cell 86 structure
print("Cell 86 source structure:")
print(f"Type: {type(nb['cells'][86]['source'])}")
print(f"Length: {len(nb['cells'][86]['source'])}")
print(f"\nFirst 10 elements:")
for i, line in enumerate(nb['cells'][86]['source'][:10], 1):
    print(f"{i}: {repr(line)}")
