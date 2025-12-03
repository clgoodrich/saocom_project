"""
Fix incorrectly formatted cells where source is a string instead of list of strings.
"""

import json

def fix_cell_formatting(notebook_path):
    """Fix cells with incorrect source formatting."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    fixed_count = 0
    cells = notebook['cells']

    for idx, cell in enumerate(cells):
        # Check if source is a string instead of list
        if isinstance(cell['source'], str):
            # Convert to list of lines
            lines = cell['source'].split('\n')
            # Add newline to all but last line
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            fixed_count += 1
            print(f"[OK] Fixed cell {idx} - converted string to list")

        # Also check for improper list formatting (missing newlines)
        elif isinstance(cell['source'], list) and len(cell['source']) > 1:
            needs_fix = False
            for i, line in enumerate(cell['source'][:-1]):  # All but last line
                if not line.endswith('\n'):
                    needs_fix = True
                    break

            if needs_fix:
                # Reconstruct properly
                full_source = ''.join(cell['source'])
                lines = full_source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
                fixed_count += 1
                print(f"[OK] Fixed cell {idx} - added missing newlines")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Fixed {fixed_count} cells")
    print(f"[DONE] Saved to: {notebook_path}")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    fix_cell_formatting(notebook_path)
