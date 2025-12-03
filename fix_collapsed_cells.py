"""
Fix cells where all content is collapsed into a single line without newlines.
"""

import json
import re

def fix_collapsed_cells(notebook_path):
    """Fix cells where newlines were lost."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    fixed_count = 0
    cells = notebook['cells']

    for idx, cell in enumerate(cells):
        if isinstance(cell['source'], list):
            # Join to see full content
            full_source = ''.join(cell['source'])

            # Check if it's a collapsed cell (single line but should be multi-line)
            # Indicators: multiple statements without newlines
            if len(cell['source']) == 1 and len(full_source) > 100:
                # Try to detect if this needs fixing
                # Look for patterns like: statement}statement or statement'statement
                needs_fixing = False

                # Code patterns that indicate missing newlines
                code_patterns = [
                    r'\}[a-zA-Z_#]',  # } followed by letter or #
                    r'\)[a-zA-Z_#]',  # ) followed by letter or #
                    r'\'[a-zA-Z_#]',  # ' followed by letter or #
                    r'\"[a-zA-Z_#]',  # " followed by letter or #
                ]

                for pattern in code_patterns:
                    if re.search(pattern, full_source):
                        needs_fixing = True
                        break

                if needs_fixing:
                    print(f"[FIXING] Cell {idx} - collapsed content detected")

                    # Try to reconstruct by adding newlines at logical points
                    fixed_source = full_source

                    # Add newlines after closing braces/parens when followed by identifier
                    fixed_source = re.sub(r'\}([a-zA-Z_#])', r'}\n\1', fixed_source)
                    fixed_source = re.sub(r'(\w)\n# ', r'\1\n\n# ', fixed_source)  # Double space before comments

                    # Add newlines after statements ending with quotes followed by identifier
                    fixed_source = re.sub(r'(\')([a-zA-Z_#])', r'\1\n\2', fixed_source)
                    fixed_source = re.sub(r'(\")([a-zA-Z_#])', r'\1\n\2', fixed_source)

                    # Add newlines after closing parens in typical patterns
                    fixed_source = re.sub(r'\)([a-zA-Z_][a-zA-Z_0-9]*\s*=)', r')\n\1', fixed_source)
                    fixed_source = re.sub(r'\)(plt\.)', r')\n\1', fixed_source)
                    fixed_source = re.sub(r'\)(fig)', r')\n\1', fixed_source)
                    fixed_source = re.sub(r'\)(if\s)', r')\n\1', fixed_source)

                    # Check if we actually improved it
                    if '\n' in fixed_source and fixed_source != full_source:
                        lines = fixed_source.split('\n')
                        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
                        fixed_count += 1
                        print(f"  [OK] Split into {len(lines)} lines")
                    else:
                        print(f"  [SKIP] Could not auto-fix - manual intervention needed")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Fixed {fixed_count} collapsed cells")
    print(f"[DONE] Saved to: {notebook_path}")

    # Report any remaining issues
    remaining_issues = []
    for idx, cell in enumerate(cells):
        if isinstance(cell['source'], list) and len(cell['source']) == 1:
            content = cell['source'][0]
            if len(content) > 200 and content.count('\n') == 0:
                remaining_issues.append(idx)

    if remaining_issues:
        print(f"\n[WARNING] {len(remaining_issues)} cells may still need manual fixing:")
        print(f"  Cell indices: {remaining_issues[:10]}")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    fix_collapsed_cells(notebook_path)
