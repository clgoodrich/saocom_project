"""
Convert GridSpec multi-panel figures to individual standalone figures.
"""

import json
import re

def convert_gridspec_figures(notebook_path):
    """Convert GridSpec patterns to individual figures."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Check if this cell contains GridSpec
            if 'GridSpec' in source and 'fig.add_subplot' in source:
                print("Found GridSpec cell, converting...")

                lines = source.split('\n')
                new_lines = []
                subplot_counter = 0
                in_subplot = False
                current_ax_name = None

                for i, line in enumerate(lines):
                    # Skip GridSpec and figure creation
                    if 'GridSpec(' in line or 'plt.figure(' in line or 'fig = plt.figure' in line:
                        print(f"  Skipping line: {line[:60]}...")
                        continue

                    # Detect new subplot: ax1 = fig.add_subplot(...)
                    match = re.search(r'(ax\d+)\s*=\s*fig\.add_subplot', line)
                    if match:
                        ax_name = match.group(1)
                        subplot_counter += 1

                        # Close previous subplot if exists
                        if in_subplot:
                            new_lines.append('plt.tight_layout()')
                            new_lines.append('plt.show()')
                            new_lines.append('')

                        # Start new subplot
                        print(f"  Creating figure {subplot_counter} for {ax_name}")
                        new_lines.append(f'# Figure {subplot_counter}')
                        new_lines.append(f'fig{subplot_counter}, {ax_name} = plt.subplots(figsize=(10, 8))')
                        in_subplot = True
                        current_ax_name = ax_name
                        continue

                    # Keep all other lines
                    new_lines.append(line)

                # Add final show
                if in_subplot:
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')

                # Update the cell
                new_source = '\n'.join(new_lines)
                cell['source'] = new_source.split('\n')
                modified_count += 1
                print(f"  [OK] Converted GridSpec with {subplot_counter} subplots\n")

    # Save modified notebook
    output_path = notebook_path.replace('.ipynb', '_gridspec_fixed.ipynb')
    if 'solo_figs' in notebook_path:
        # Already processed, just update the same file
        output_path = notebook_path

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified_count} cells with GridSpec")
    print(f"[DONE] Saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    convert_gridspec_figures(notebook_path)
