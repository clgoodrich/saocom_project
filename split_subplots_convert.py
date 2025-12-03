"""
Convert all subplot figures in a Jupyter notebook to standalone individual figures.
"""

import json
import re
from pathlib import Path

def convert_subplots_to_individual(notebook_path):
    """Convert all subplot patterns to individual figures."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            original_source = source

            # Pattern 1: Convert "fig, axes = plt.subplots(1, 2, ...)" to individual figures
            # Match the subplot creation and subsequent axes usage
            if 'fig, axes = plt.subplots' in source and ', 2' in source:
                # Check if it's a 1x2 or 2x1 subplot
                match = re.search(r'fig, axes = plt.subplots\(([\d\s,]+).*?\)', source)
                if match:
                    dims = match.group(1).strip()

                    if '1, 2' in dims or '1,2' in dims:  # 1 row, 2 columns
                        # Split into two separate figures
                        new_source = convert_1x2_subplot(source)
                        cell['source'] = new_source.split('\n')
                        if new_source != original_source:
                            modified_count += 1
                            print(f"[OK] Converted 1x2 subplot")

                    elif '2, 2' in dims or '2,2' in dims:  # 2 rows, 2 columns
                        new_source = convert_2x2_subplot(source)
                        cell['source'] = new_source.split('\n')
                        if new_source != original_source:
                            modified_count += 1
                            print(f"[OK] Converted 2x2 subplot")

            # Pattern 2: Convert GridSpec patterns
            elif 'GridSpec' in source and 'fig.add_subplot' in source:
                new_source = convert_gridspec_subplot(source)
                cell['source'] = new_source.split('\n')
                if new_source != original_source:
                    modified_count += 1
                    print(f"[OK] Converted GridSpec subplot")

    # Save modified notebook
    output_path = notebook_path.replace('.ipynb', '_solo_figs.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified_count} cells")
    print(f"[DONE] Saved to: {output_path}")
    return output_path

def convert_1x2_subplot(source):
    """Convert 1x2 subplot to two individual figures."""
    lines = source.split('\n')
    new_lines = []
    in_first_plot = False
    in_second_plot = False
    first_plot_lines = []
    second_plot_lines = []

    for line in lines:
        # Skip the original subplot creation
        if 'fig, axes = plt.subplots(1, 2' in line:
            continue

        # Replace axes[0] with individual figure
        if 'axes[0]' in line:
            if not in_first_plot:
                new_lines.append('# First figure')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(8, 6))')
                in_first_plot = True
            new_line = line.replace('axes[0]', 'ax1')
            new_lines.append(new_line)

        # Replace axes[1] with individual figure
        elif 'axes[1]' in line:
            if not in_second_plot:
                # Close first figure before starting second
                if in_first_plot:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Second figure')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(8, 6))')
                in_second_plot = True
            new_line = line.replace('axes[1]', 'ax2')
            new_lines.append(new_line)

        # Handle tight_layout and show
        elif 'plt.tight_layout()' in line or 'plt.show()' in line:
            if in_second_plot:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def convert_2x2_subplot(source):
    """Convert 2x2 subplot to four individual figures."""
    lines = source.split('\n')
    new_lines = []
    current_subplot = None

    for line in lines:
        # Skip the original subplot creation
        if 'fig, axes = plt.subplots(2, 2' in line:
            continue

        # Replace axes[0, 0] with individual figure
        if 'axes[0, 0]' in line or 'axes[0][0]' in line:
            if current_subplot != '00':
                if current_subplot is not None:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Figure 1')
                new_lines.append('fig1, ax1 = plt.subplots(figsize=(8, 6))')
                current_subplot = '00'
            new_line = line.replace('axes[0, 0]', 'ax1').replace('axes[0][0]', 'ax1')
            new_lines.append(new_line)

        # Replace axes[0, 1]
        elif 'axes[0, 1]' in line or 'axes[0][1]' in line:
            if current_subplot != '01':
                if current_subplot is not None:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Figure 2')
                new_lines.append('fig2, ax2 = plt.subplots(figsize=(8, 6))')
                current_subplot = '01'
            new_line = line.replace('axes[0, 1]', 'ax2').replace('axes[0][1]', 'ax2')
            new_lines.append(new_line)

        # Replace axes[1, 0]
        elif 'axes[1, 0]' in line or 'axes[1][0]' in line:
            if current_subplot != '10':
                if current_subplot is not None:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Figure 3')
                new_lines.append('fig3, ax3 = plt.subplots(figsize=(8, 6))')
                current_subplot = '10'
            new_line = line.replace('axes[1, 0]', 'ax3').replace('axes[1][0]', 'ax3')
            new_lines.append(new_line)

        # Replace axes[1, 1]
        elif 'axes[1, 1]' in line or 'axes[1][1]' in line:
            if current_subplot != '11':
                if current_subplot is not None:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append('# Figure 4')
                new_lines.append('fig4, ax4 = plt.subplots(figsize=(8, 6))')
                current_subplot = '11'
            new_line = line.replace('axes[1, 1]', 'ax4').replace('axes[1][1]', 'ax4')
            new_lines.append(new_line)

        # Handle tight_layout and show
        elif 'plt.tight_layout()' in line or 'plt.show()' in line:
            if current_subplot is not None:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def convert_gridspec_subplot(source):
    """Convert GridSpec subplot to individual figures."""
    lines = source.split('\n')
    new_lines = []
    current_ax = None
    ax_mapping = {}
    ax_counter = 1

    for line in lines:
        # Skip GridSpec creation and figure creation
        if 'GridSpec' in line and '=' in line:
            continue
        if 'fig = plt.figure' in line:
            continue

        # Detect new subplot creation
        match = re.search(r'(ax\d+)\s*=\s*fig\.add_subplot', line)
        if match:
            ax_name = match.group(1)
            if ax_name not in ax_mapping:
                if current_ax is not None:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')
                    new_lines.append('')
                new_lines.append(f'# Figure {ax_counter}')
                new_lines.append(f'fig{ax_counter}, {ax_name} = plt.subplots(figsize=(10, 8))')
                ax_mapping[ax_name] = ax_counter
                current_ax = ax_name
                ax_counter += 1
            continue

        # Handle tight_layout and show
        if 'plt.tight_layout()' in line or 'plt.show()' in line:
            if current_ax is not None:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb'
    convert_subplots_to_individual(notebook_path)
