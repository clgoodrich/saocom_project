"""
Fix all cells where figure creation was incorrectly inserted inside function calls.
"""

import json
import re

def fix_all_function_calls(notebook_path):
    """Fix all broken function calls in notebook."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_count = 0

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        src = ''.join(cell['source'])

        # Look for pattern: function_name(\nfig, ax = plt.subplots
        # This indicates figure creation was inserted inside a function call
        pattern = r'(\w+)\s*\(\s*\n\s*#\s*\w+\s+figure\s*\n\s*fig\d*,\s*ax\d*\s*=\s*plt\.subplots'

        if re.search(pattern, src):
            print(f"\nCell {idx}: Found broken function call pattern")

            # Extract function calls and fix them
            lines = src.split('\n')
            new_lines = []
            in_broken_call = False
            func_name = None
            fig_num = 1

            i = 0
            while i < len(lines):
                line = lines[i]

                # Check if this starts a function call
                match = re.match(r'^(plot_\w+)\s*\(', line)
                if match:
                    func_name = match.group(1)
                    # Check if next lines have fig creation
                    if i+1 < len(lines) and 'fig' in lines[i+1] and 'plt.subplots' in lines[i+1]:
                        in_broken_call = True
                        # Insert fig creation BEFORE function call
                        new_lines.append(f'# Figure {fig_num}')
                        new_lines.append(f'fig{fig_num}, ax{fig_num} = plt.subplots(figsize=(8, 6))')
                        # Now start function call properly
                        new_lines.append(f'{func_name}(')
                        new_lines.append(f'    ax{fig_num},')
                        # Skip the embedded fig creation and comment
                        i += 1
                        while i < len(lines) and ('fig' in lines[i] or '# First' in lines[i] or '# Second' in lines[i]):
                            i += 1
                        i -= 1  # Back up one
                        fig_num += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

                i += 1

            if in_broken_call:
                cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
                fixed_count += 1
                print(f"  [OK] Fixed cell {idx}")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Fixed {fixed_count} cells")
    return fixed_count

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb'
    fix_all_function_calls(notebook_path)
