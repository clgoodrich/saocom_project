"""
Final cleanup to convert remaining subplot patterns to individual figures.
"""

import json
import re

def convert_remaining_subplots(notebook_path):
    """Convert any remaining subplot patterns (2x1, etc.) to individual figures."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Check for remaining "fig, axes = plt.subplots" patterns
            if 'fig, axes = plt.subplots' in source and 'axes[' in source:
                print(f"\nFound remaining subplot at cell {idx}")

                # Detect dimensions
                match = re.search(r'fig, axes = plt.subplots\((\d+),\s*(\d+)', source)
                if match:
                    rows = int(match.group(1))
                    cols = int(match.group(2))
                    print(f"  Dimensions: {rows}x{cols}")

                    if rows == 2 and cols == 1:
                        # Convert 2x1 subplot
                        new_source = convert_2x1_subplot(source)
                        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
                        modified_count += 1
                        print(f"  [OK] Converted 2x1 subplot")
                    elif rows == 1 and cols == 2:
                        # Convert 1x2 subplot (should have been caught, but just in case)
                        new_source = convert_1x2_subplot(source)
                        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
                        modified_count += 1
                        print(f"  [OK] Converted 1x2 subplot")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified_count} cells")
    print(f"[DONE] Saved to: {notebook_path}")

def convert_2x1_subplot(source):
    """Convert 2x1 subplot to two individual figures."""
    lines = source.split('\n')
    new_lines = []
    first_fig_created = False
    second_fig_created = False

    for line in lines:
        # Skip the original subplot creation
        if 'fig, axes = plt.subplots(2, 1' in line:
            continue

        # Replace axes[0] with individual figure
        if 'axes[0]' in line:
            if not first_fig_created:
                new_lines.append('# First figure')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(16, 6))')
                first_fig_created = True
            new_line = line.replace('axes[0]', 'ax1')
            new_lines.append(new_line)

        # Replace axes[1] with individual figure
        elif 'axes[1]' in line:
            if not second_fig_created:
                # Close first figure
                if first_fig_created:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Second figure')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(16, 6))')
                second_fig_created = True
            new_line = line.replace('axes[1]', 'ax2')
            new_lines.append(new_line)

        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def convert_1x2_subplot(source):
    """Convert 1x2 subplot to two individual figures."""
    lines = source.split('\n')
    new_lines = []
    first_fig_created = False
    second_fig_created = False

    for line in lines:
        # Skip the original subplot creation
        if 'fig, axes = plt.subplots(1, 2' in line:
            continue

        # Replace axes[0] with individual figure
        if 'axes[0]' in line:
            if not first_fig_created:
                new_lines.append('# First figure')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(8, 6))')
                first_fig_created = True
            new_line = line.replace('axes[0]', 'ax1')
            new_lines.append(new_line)

        # Replace axes[1] with individual figure
        elif 'axes[1]' in line:
            if not second_fig_created:
                if first_fig_created:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Second figure')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(8, 6))')
                second_fig_created = True
            new_line = line.replace('axes[1]', 'ax2')
            new_lines.append(new_line)

        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    convert_remaining_subplots(notebook_path)
