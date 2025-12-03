"""
Script to convert all multi-panel subplots to individual figures in the notebook.
Each subplot will become its own figure and be saved separately.
"""

import json
import re

def split_subplot_cell(source_lines):
    """Convert a multi-panel subplot cell to multiple individual figures."""
    source = ''.join(source_lines)

    # Detect subplot structure
    match = re.search(r'fig,\s*axes\s*=\s*plt\.subplots\((\d+),?\s*(\d+)?', source)
    if not match:
        return source_lines  # No change needed

    rows = int(match.group(1))
    cols = int(match.group(2)) if match.group(2) else 1

    if rows == 1 and cols == 1:
        return source_lines  # Single panel, no change

    # Replace subplot creation with individual figure creation
    # Strategy: Replace axes[i] or axes[i, j] references with ax
    # and wrap each section in its own figure

    new_source = []

    # Add comment
    new_source.append("# Modified: Each plot now created as individual figure\n")

    if rows == 1 and cols == 2:
        # 1x2 subplot - two side-by-side plots
        # Split into two separate figures
        lines = source.split('\n')

        # Find the subplot line
        subplot_idx = None
        for i, line in enumerate(lines):
            if 'plt.subplots' in line and 'axes' in line:
                subplot_idx = i
                break

        if subplot_idx is not None:
            # Keep everything before subplot
            for i in range(subplot_idx):
                new_source.append(lines[i] + '\n')

            # Now we need to split the axes[0] and axes[1] sections
            # This is complex, so we'll use a simple heuristic:
            # Replace axes[0] with ax in first figure, axes[1] with ax in second

            remaining = '\n'.join(lines[subplot_idx+1:])

            # Split at natural boundaries (look for axes[1] first usage)
            axes1_match = re.search(r'axes\[1\]', remaining)
            if axes1_match:
                split_point = axes1_match.start()
                part1 = remaining[:split_point]
                part2 = remaining[split_point:]

                # Create first figure
                new_source.append("\n# Figure 1\n")
                new_source.append("fig, ax = plt.subplots(figsize=(10, 6))\n")
                new_source.append(part1.replace('axes[0]', 'ax'))
                new_source.append("plt.tight_layout()\n")
                new_source.append("plt.show()\n\n")

                # Create second figure
                new_source.append("# Figure 2\n")
                new_source.append("fig, ax = plt.subplots(figsize=(10, 6))\n")
                new_source.append(part2.replace('axes[1]', 'ax'))
                new_source.append("plt.tight_layout()\n")
                new_source.append("plt.show()\n")
            else:
                # Couldn't split, return original
                return source_lines
        else:
            return source_lines

    elif rows == 2 and cols == 2:
        # 2x2 subplot - four plots
        # Similar logic but for 4 quadrants
        new_source.append("# Note: Original 2x2 subplot split into 4 individual figures\n")
        new_source.append(source)  # Keep original for now, manual editing may be needed

    else:
        # Other configurations, keep original
        return source_lines

    return new_source

# Read notebook
print("Reading notebook...")
with open('saocom_analysis_clean.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Track changes
cells_modified = 0

# Process each cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))

        # Check if this cell has multi-panel subplots
        if re.search(r'axes\s*=.*subplots\([^)]*[,]\s*[2-9]', source) or \
           re.search(r'axes\s*=.*subplots\([2-9]', source):

            print(f"Processing cell {i}...")
            # For now, just add a comment to mark cells for manual review
            cell['source'].insert(0, '# TODO: Convert multi-panel subplot to individual figures\n')
            cells_modified += 1

print(f"\nMarked {cells_modified} cells for conversion")
print("Manual conversion recommended due to complexity of subplot structures")

# Save modified notebook
# with open('saocom_analysis_clean.ipynb', 'w', encoding='utf-8') as f:
#     json.dump(nb, f, indent=1)

print("\nDue to the complexity and variety of subplot structures,")
print("I recommend a targeted approach. Let me create a helper script instead.")
