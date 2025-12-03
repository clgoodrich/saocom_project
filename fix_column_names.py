"""
Fix column names in added analysis cells to match actual dataframe columns.
"""

import json

def fix_column_names(notebook_path):
    """Fix incorrect column references in analysis cells."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    fixed = 0

    for idx, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue

        src = ''.join(cell['source'])

        # Fix penetration depth cell
        if 'Penetration Depth Analysis' in src:
            print(f"Fixing Penetration Depth cell {idx}")

            # Replace incorrect column name
            new_src = src.replace('calibrated_height', 'HEIGHT_ABSOLUTE_TIN')
            new_src = new_src.replace('"HEIGHT_ABSOLUTE_TIN"', '"HEIGHT_ABSOLUTE_COP"')  # Use Copernicus for penetration
            new_src = new_src.replace('["HEIGHT_ABSOLUTE_TIN"]', '["HEIGHT_ABSOLUTE_COP"]')

            # Actually, penetration should be Copernicus - SAOCOM, so:
            # Use Copernicus height directly and HEIGHT_ABSOLUTE_COP
            new_src = src.replace(
                'penetration_depth = saocom_cleaned["copernicus_height"] - saocom_cleaned["calibrated_height"]',
                'penetration_depth = saocom_cleaned["copernicus_height"] - saocom_cleaned["HEIGHT_ABSOLUTE_COP"]'
            )

            lines = new_src.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            fixed += 1

        # Fix slope-aspect cell
        if 'Slope-Aspect Interaction' in src and 'radar_geometry' in src:
            print(f"Fixing Slope-Aspect cell {idx}")

            # Fix residual_tin reference - should be diff_tinitaly
            new_src = src.replace('residual_tin', 'diff_tinitaly')
            new_src = new_src.replace('"residual_tin"', '"diff_tinitaly"')
            new_src = new_src.replace("'residual_tin'", "'diff_tinitaly'")

            lines = new_src.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            fixed += 1

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Fixed {fixed} cells with incorrect column names")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    fix_column_names(notebook_path)
