"""
Fix the order of analysis cells to ensure variables exist before they're used.
"""

import json

def fix_cell_order(notebook_path):
    """Reorder cells so analyses come after variable creation."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    # Find key locations
    residuals_created_idx = None
    calibrated_height_idx = None
    penetration_idx = None
    copernicus_comparison_idx = None
    ci_function_idx = None
    slope_aspect_idx = None

    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])

            if 'residuals_tin = ' in src and 'dropna()' in src:
                residuals_created_idx = idx
                print(f"Found residuals_tin creation at cell {idx}")

            if 'Penetration Depth Analysis' in src:
                penetration_idx = idx
                print(f"Found Penetration Depth Analysis at cell {idx}")

            if 'Expanded Copernicus Comparison' in src:
                copernicus_comparison_idx = idx
                print(f"Found Copernicus Comparison at cell {idx}")

            if 'Confidence Interval Calculation' in src:
                ci_function_idx = idx
                print(f"Found CI functions at cell {idx}")

            if 'Slope-Aspect Interaction' in src and 'radar_geometry' in src:
                slope_aspect_idx = idx
                print(f"Found Slope-Aspect Interaction at cell {idx}")

    # Remove cells that need to be moved
    cells_to_move = []

    # Remove in reverse order to preserve indices
    if copernicus_comparison_idx and copernicus_comparison_idx < residuals_created_idx:
        print(f"\nMoving Copernicus Comparison from {copernicus_comparison_idx} to after {residuals_created_idx}")
        cells_to_move.append(('copernicus', cells.pop(copernicus_comparison_idx)))
        # Adjust indices
        if penetration_idx and penetration_idx > copernicus_comparison_idx:
            penetration_idx -= 1
        if residuals_created_idx > copernicus_comparison_idx:
            residuals_created_idx -= 1
        if slope_aspect_idx and slope_aspect_idx > copernicus_comparison_idx:
            slope_aspect_idx -= 1

    if penetration_idx and penetration_idx <= residuals_created_idx:
        print(f"Moving Penetration Depth from {penetration_idx} to after {residuals_created_idx}")
        cells_to_move.append(('penetration', cells.pop(penetration_idx)))
        # Adjust indices
        if residuals_created_idx > penetration_idx:
            residuals_created_idx -= 1
        if slope_aspect_idx and slope_aspect_idx > penetration_idx:
            slope_aspect_idx -= 1

    # Insert cells in the right place (after residuals are created)
    # Find the new index (after residuals and a few following cells)
    insert_idx = residuals_created_idx + 3  # Give some space

    for name, cell in reversed(cells_to_move):
        print(f"Inserting {name} at position {insert_idx}")
        cells.insert(insert_idx, cell)

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"\n[DONE] Fixed cell order")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    fix_cell_order(notebook_path)
