"""
Convert fig.add_gridspec multi-panel figures to individual standalone figures.
"""

import json
import re

def convert_gridspec_figures(notebook_path):
    """Convert fig.add_gridspec patterns to individual figures."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Check if this cell contains add_gridspec and add_subplot
            if ('add_gridspec' in source or 'GridSpec' in source) and 'fig.add_subplot' in source:
                print(f"\nFound gridspec cell at index {idx}, converting...")

                lines = source.split('\n')
                new_lines = []
                subplot_counter = 0
                in_subplot = False
                current_ax_name = None

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Skip figure creation and gridspec
                    if ('fig = plt.figure' in line or
                        'gs = fig.add_gridspec' in line or
                        'GridSpec(' in line):
                        print(f"  Skipping: {line[:70]}...")
                        continue

                    # Detect new subplot: ax1 = fig.add_subplot(gs[...])
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

                    # Keep all other non-empty lines
                    new_lines.append(line)

                # Add final show
                if in_subplot and new_lines:
                    # Remove any existing plt.tight_layout() or plt.show() at the end
                    while new_lines and (new_lines[-1].strip() == '' or
                                        'plt.tight_layout()' in new_lines[-1] or
                                        'plt.show()' in new_lines[-1]):
                        new_lines.pop()
                    new_lines.append('plt.tight_layout()')
                    new_lines.append('plt.show()')

                # Update the cell
                new_source = '\n'.join(new_lines)
                cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
                modified_count += 1
                print(f"  [OK] Converted gridspec with {subplot_counter} subplots")

    # Save modified notebook
    output_path = notebook_path

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified_count} cells with gridspec")
    print(f"[DONE] Saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    convert_gridspec_figures(notebook_path)
