"""
Add plt.tight_layout() and plt.show() immediately after each figure is completed.
"""

import json
import re

def add_show_after_figures(notebook_path):
    """Add plt.show() after each figure in the notebook."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified_count = 0

    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Skip if no figure creation
            if 'plt.subplots' not in source and 'plt.figure' not in source:
                continue

            # Skip if already has plt.show() at the end
            if source.rstrip().endswith('plt.show()'):
                continue

            lines = source.split('\n')
            new_lines = []

            # Track when we've created a figure
            fig_created = False
            last_plot_line = -1

            for i, line in enumerate(lines):
                new_lines.append(line)

                # Detect figure creation
                if ('plt.subplots' in line or 'plt.figure' in line) and '=' in line:
                    fig_created = True
                    last_plot_line = i

                # Track plotting commands
                elif fig_created and any(cmd in line for cmd in [
                    '.plot(', '.scatter(', '.bar(', '.barh(', '.hist(',
                    '.imshow(', '.contour(', '.pcolor(', '.set_', '.text(',
                    '.axhline(', '.axvline(', '.legend(', '.colorbar(',
                    '.set_xlim', '.set_ylim', '.set_title', '.set_xlabel', '.set_ylabel'
                ]):
                    last_plot_line = i

            # If we created a figure and don't already have plt.show() at the end
            if fig_created and last_plot_line >= 0:
                # Remove trailing empty lines
                while new_lines and new_lines[-1].strip() == '':
                    new_lines.pop()

                # Check if last line is already plt.show() or plt.tight_layout()
                has_show = any('plt.show()' in l for l in new_lines[-3:])
                has_tight = any('plt.tight_layout()' in l for l in new_lines[-3:])

                if not has_tight:
                    new_lines.append('plt.tight_layout()')
                if not has_show:
                    new_lines.append('plt.show()')

                # Update cell
                new_source = '\n'.join(new_lines)
                if new_source != source.rstrip():
                    cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
                    modified_count += 1
                    print(f"[OK] Added plt.show() to cell {idx}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Modified {modified_count} cells")
    print(f"[DONE] Saved to: {notebook_path}")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    add_show_after_figures(notebook_path)
