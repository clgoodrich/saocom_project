"""
Rebuild the notebook properly from the original, applying conversions correctly.
This preserves proper line formatting while converting subplots.
"""

import json
import re

def rebuild_notebook(original_path, output_path):
    """Rebuild notebook with proper formatting."""

    with open(original_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = 0

    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue

        # Get source as list of lines (preserving structure)
        if isinstance(cell['source'], str):
            lines = cell['source'].split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

        source_lines = cell['source'][:]

        # Check for subplot patterns
        has_subplots = any('fig, axes = plt.subplots' in line for line in source_lines)

        if has_subplots:
            # Determine subplot type
            subplot_line = next((line for line in source_lines if 'fig, axes = plt.subplots' in line), None)

            if subplot_line:
                # Extract dimensions
                match = re.search(r'plt\.subplots\s*\(\s*(\d+)\s*,\s*(\d+)', subplot_line)
                if match:
                    rows, cols = int(match.group(1)), int(match.group(2))
                    print(f"Cell {idx}: Converting {rows}x{cols} subplot")

                    if rows == 1 and cols == 2:
                        new_lines = convert_1x2_properly(source_lines)
                        cell['source'] = new_lines
                        modified += 1
                    elif rows == 2 and cols == 1:
                        new_lines = convert_2x1_properly(source_lines)
                        cell['source'] = new_lines
                        modified += 1
                    elif rows == 2 and cols == 2:
                        new_lines = convert_2x2_properly(source_lines)
                        cell['source'] = new_lines
                        modified += 1

        # Check for gridspec
        has_gridspec = any('add_gridspec' in line for line in source_lines)
        if has_gridspec:
            print(f"Cell {idx}: Converting gridspec")
            new_lines = convert_gridspec_properly(source_lines)
            cell['source'] = new_lines
            modified += 1

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified} cells")
    print(f"[DONE] Saved to: {output_path}")

def convert_1x2_properly(source_lines):
    """Convert 1x2 subplot while preserving line structure."""
    new_lines = []
    in_first = False
    in_second = False

    for line in source_lines:
        # Skip subplot creation
        if 'fig, axes = plt.subplots(1, 2' in line:
            continue

        # Handle axes[0]
        if 'axes[0]' in line:
            if not in_first:
                new_lines.append('# First figure\n')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(8, 6))\n')
                in_first = True
            new_lines.append(line.replace('axes[0]', 'ax1'))

        # Handle axes[1]
        elif 'axes[1]' in line:
            if not in_second:
                if in_first:
                    new_lines.append('plt.tight_layout()\n')
                    new_lines.append('plt.show()\n')
                    new_lines.append('\n')
                new_lines.append('# Second figure\n')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(8, 6))\n')
                in_second = True
            new_lines.append(line.replace('axes[1]', 'ax2'))

        # Keep other lines
        elif 'plt.tight_layout()' in line or 'plt.show()' in line:
            if in_second:  # Only add for the last figure
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Ensure last figure has show
    if in_second and not any('plt.show()' in l for l in new_lines[-3:]):
        # Remove trailing newlines
        while new_lines and new_lines[-1].strip() == '':
            new_lines.pop()
        new_lines.append('plt.tight_layout()\n')
        new_lines.append('plt.show()\n')

    return new_lines

def convert_2x1_properly(source_lines):
    """Convert 2x1 subplot while preserving line structure."""
    new_lines = []
    in_first = False
    in_second = False

    for line in source_lines:
        if 'fig, axes = plt.subplots(2, 1' in line:
            continue

        if 'axes[0]' in line:
            if not in_first:
                new_lines.append('# First figure\n')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(16, 6))\n')
                in_first = True
            new_lines.append(line.replace('axes[0]', 'ax1'))

        elif 'axes[1]' in line:
            if not in_second:
                if in_first:
                    new_lines.append('plt.tight_layout()\n')
                    new_lines.append('plt.show()\n')
                    new_lines.append('\n')
                new_lines.append('# Second figure\n')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(16, 6))\n')
                in_second = True
            new_lines.append(line.replace('axes[1]', 'ax2'))

        elif 'plt.tight_layout()' in line or 'plt.show()' in line:
            if in_second:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if in_second and not any('plt.show()' in l for l in new_lines[-3:]):
        while new_lines and new_lines[-1].strip() == '':
            new_lines.pop()
        new_lines.append('plt.tight_layout()\n')
        new_lines.append('plt.show()\n')

    return new_lines

def convert_2x2_properly(source_lines):
    """Convert 2x2 subplot."""
    new_lines = []
    current_subplot = None

    for line in source_lines:
        if 'fig, axes = plt.subplots(2, 2' in line:
            continue

        if 'axes[0, 0]' in line or 'axes[0][0]' in line:
            if current_subplot and current_subplot != '00':
                new_lines.append('plt.tight_layout()\n')
                new_lines.append('plt.show()\n')
                new_lines.append('\n')
            if current_subplot != '00':
                new_lines.append('# Figure 1\n')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(8, 6))\n')
                current_subplot = '00'
            new_lines.append(line.replace('axes[0, 0]', 'ax1').replace('axes[0][0]', 'ax1'))

        elif 'axes[0, 1]' in line or 'axes[0][1]' in line:
            if current_subplot and current_subplot != '01':
                new_lines.append('plt.tight_layout()\n')
                new_lines.append('plt.show()\n')
                new_lines.append('\n')
            if current_subplot != '01':
                new_lines.append('# Figure 2\n')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(8, 6))\n')
                current_subplot = '01'
            new_lines.append(line.replace('axes[0, 1]', 'ax2').replace('axes[0][1]', 'ax2'))

        elif 'axes[1, 0]' in line or 'axes[1][0]' in line:
            if current_subplot and current_subplot != '10':
                new_lines.append('plt.tight_layout()\n')
                new_lines.append('plt.show()\n')
                new_lines.append('\n')
            if current_subplot != '10':
                new_lines.append('# Figure 3\n')
                new_lines.append('fig3, ax3 = plt.subplots(figsize=(8, 6))\n')
                current_subplot = '10'
            new_lines.append(line.replace('axes[1, 0]', 'ax3').replace('axes[1][0]', 'ax3'))

        elif 'axes[1, 1]' in line or 'axes[1][1]' in line:
            if current_subplot and current_subplot != '11':
                new_lines.append('plt.tight_layout()\n')
                new_lines.append('plt.show()\n')
                new_lines.append('\n')
            if current_subplot != '11':
                new_lines.append('# Figure 4\n')
                new_lines.append('fig4, ax4 = plt.subplots(figsize=(8, 6))\n')
                current_subplot = '11'
            new_lines.append(line.replace('axes[1, 1]', 'ax4').replace('axes[1][1]', 'ax4'))

        elif 'plt.tight_layout()' in line or 'plt.show()' in line:
            if current_subplot:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return new_lines

def convert_gridspec_properly(source_lines):
    """Convert gridspec subplot."""
    new_lines = []
    current_ax = None

    for line in source_lines:
        # Skip gridspec and figure creation
        if 'add_gridspec' in line or ('fig = plt.figure' in line and 'figsize' in line):
            continue

        # Detect subplot creation
        match = re.search(r'(ax\d+)\s*=\s*fig\.add_subplot', line)
        if match:
            ax_name = match.group(1)
            if current_ax:
                new_lines.append('plt.tight_layout()\n')
                new_lines.append('plt.show()\n')
                new_lines.append('\n')
            new_lines.append(f'# Figure for {ax_name}\n')
            new_lines.append(f'fig, {ax_name} = plt.subplots(figsize=(10, 8))\n')
            current_ax = ax_name
            continue

        if 'plt.tight_layout()' in line or 'plt.show()' in line:
            if current_ax:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return new_lines

if __name__ == '__main__':
    original = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb'
    output = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    rebuild_notebook(original, output)
