"""
Add remaining enhancements:
1. RMSE to land cover tables
2. Slope-aspect interaction (create new section)
"""

import json

def add_remaining_enhancements(notebook_path):
    """Add remaining analytical enhancements."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    # 1. Add RMSE to land cover tables
    add_rmse_to_tables(cells)

    # 2. Add slope-aspect interaction analysis
    add_slope_aspect_section(cells)

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("[DONE] Remaining enhancements added")

def add_rmse_to_tables(cells):
    """Add RMSE column to land cover statistics tables."""

    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Find land cover statistics with groupby and agg
            if 'land_cover' in source and 'groupby' in source and 'agg' in source:
                if 'rmse' not in source.lower():
                    lines = source.split('\n')
                    new_lines = []

                    for i, line in enumerate(lines):
                        new_lines.append(line)

                        # After nmad line, add rmse
                        if "'nmad'" in line.lower() and 'lambda' in line:
                            # Get the same indentation
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(' ' * indent + "'rmse': lambda x: np.sqrt(np.mean(x**2)),")
                            print(f"[OK] Added RMSE to cell {idx}")

                    # Update cell
                    if len(new_lines) > len(lines):
                        cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])

def add_slope_aspect_section(cells):
    """Add new section for slope-aspect interaction analysis."""

    # Find a good insertion point (after land cover analysis)
    insert_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'land_cover' in source and 'level1' in source:
                insert_idx = idx + 2  # Insert after land cover section
                break

    if insert_idx is None:
        print("[SKIP] Could not find insertion point for slope-aspect")
        return

    # Create markdown header
    markdown_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## Terrain Analysis: Slope-Aspect Interaction\n',
            '\n',
            'Analysis of how terrain slope and aspect (relative to radar look direction) affect SAOCOM height accuracy.\n'
        ]
    }

    # Create analysis cell
    analysis_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Slope-Aspect Interaction Analysis\n',
            '# Analyze residuals based on terrain geometry relative to radar\n',
            'print("\\n" + "="*60)\n',
            'print("TERRAIN GEOMETRY ANALYSIS")\n',
            'print("="*60)\n',
            '\n',
            '# Check if slope and aspect columns exist\n',
            'if "slope" not in saocom_cleaned.columns:\n',
            '    print("\\n[INFO] Slope data not available. Computing from DEM...")\n',
            '    # Note: This requires topographic derivatives to be computed\n',
            '    print("[SKIP] Slope-aspect analysis requires topographic processing.")\n',
            'else:\n',
            '    # SAOCOM ascending pass looks from approximately 98 degrees (East)\n',
            '    # Radar-facing slopes: aspects roughly 45-135 degrees (NE to SE)\n',
            '    # Radar-away slopes: aspects roughly 225-315 degrees (SW to NW)\n',
            '    \n',
            '    def classify_radar_geometry(aspect):\n',
            '        """Classify aspect relative to radar look direction."""\n',
            '        if pd.isna(aspect):\n',
            '            return "Unknown"\n',
            '        elif 45 <= aspect <= 135:\n',
            '            return "Radar-facing"\n',
            '        elif 225 <= aspect <= 315:\n',
            '            return "Radar-away"\n',
            '        else:\n',
            '            return "Oblique"\n',
            '    \n',
            '    # Classify slope steepness\n',
            '    def classify_slope(slope):\n',
            '        """Classify slope into categories."""\n',
            '        if pd.isna(slope):\n',
            '            return "Unknown"\n',
            '        elif slope < 5:\n',
            '            return "Flat (0-5°)"\n',
            '        elif slope < 15:\n',
            '            return "Gentle (5-15°)"\n',
            '        elif slope < 30:\n',
            '            return "Moderate (15-30°)"\n',
            '        else:\n',
            '            return "Steep (>30°)"\n',
            '    \n',
            '    if "aspect" in saocom_cleaned.columns:\n',
            '        saocom_cleaned["radar_geometry"] = saocom_cleaned["aspect"].apply(classify_radar_geometry)\n',
            '    else:\n',
            '        print("[INFO] Aspect data not available")\n',
            '        saocom_cleaned["radar_geometry"] = "Unknown"\n',
            '    \n',
            '    saocom_cleaned["slope_category"] = saocom_cleaned["slope"].apply(classify_slope)\n',
            '    \n',
            '    # Statistics by radar geometry\n',
            '    print("\\n1. Residual Statistics by Radar Geometry:")\n',
            '    print("-" * 60)\n',
            '    radar_geom_stats = saocom_cleaned.groupby("radar_geometry")["residual_tin"].agg([\n',
            '        ("count", "count"),\n',
            '        ("bias", "mean"),\n',
            '        ("std", "std"),\n',
            '        ("nmad", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),\n',
            '        ("rmse", lambda x: np.sqrt(np.mean(x**2)))\n',
            '    ]).round(3)\n',
            '    print(radar_geom_stats)\n',
            '    \n',
            '    # Cross-tabulation: slope category × radar geometry\n',
            '    print("\\n2. Residuals by Slope Category and Radar Geometry:")\n',
            '    print("-" * 60)\n',
            '    slope_radar_stats = saocom_cleaned.groupby(["slope_category", "radar_geometry"])["residual_tin"].agg([\n',
            '        ("n", "count"),\n',
            '        ("bias", "mean"),\n',
            '        ("nmad", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),\n',
            '        ("rmse", lambda x: np.sqrt(np.mean(x**2)))\n',
            '    ]).round(3)\n',
            '    print(slope_radar_stats)\n',
            '    \n',
            '    # Visualization 1: Boxplot by radar geometry\n',
            '    fig, ax = plt.subplots(figsize=(10, 6))\n',
            '    saocom_cleaned.boxplot(column="residual_tin", by="radar_geometry", ax=ax)\n',
            '    ax.set_title("Residuals by Radar Geometry", fontweight="bold", fontsize=14)\n',
            '    ax.set_xlabel("Radar Geometry", fontweight="bold")\n',
            '    ax.set_ylabel("Residual (m)", fontweight="bold")\n',
            '    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Zero error")\n',
            '    ax.grid(alpha=0.3)\n',
            '    plt.suptitle("")  # Remove default title\n',
            '    plt.tight_layout()\n',
            '    plt.show()\n',
            '    \n',
            '    # Visualization 2: Heatmap of NMAD by slope and radar geometry\n',
            '    fig, ax = plt.subplots(figsize=(10, 6))\n',
            '    \n',
            '    # Pivot table for heatmap\n',
            '    heatmap_data = saocom_cleaned.groupby(["slope_category", "radar_geometry"])["residual_tin"].apply(\n',
            '        lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))\n',
            '    ).unstack(fill_value=np.nan)\n',
            '    \n',
            '    import matplotlib.pyplot as plt\n',
            '    im = ax.imshow(heatmap_data.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=5)\n',
            '    \n',
            '    # Set ticks and labels\n',
            '    ax.set_xticks(range(len(heatmap_data.columns)))\n',
            '    ax.set_yticks(range(len(heatmap_data.index)))\n',
            '    ax.set_xticklabels(heatmap_data.columns)\n',
            '    ax.set_yticklabels(heatmap_data.index)\n',
            '    \n',
            '    # Add values as text\n',
            '    for i in range(len(heatmap_data.index)):\n',
            '        for j in range(len(heatmap_data.columns)):\n',
            '            val = heatmap_data.values[i, j]\n',
            '            if not np.isnan(val):\n',
            '                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontweight="bold")\n',
            '    \n',
            '    ax.set_title("NMAD (m) by Slope Category and Radar Geometry", fontweight="bold", fontsize=14)\n',
            '    ax.set_xlabel("Radar Geometry", fontweight="bold")\n',
            '    ax.set_ylabel("Slope Category", fontweight="bold")\n',
            '    \n',
            '    # Colorbar\n',
            '    cbar = plt.colorbar(im, ax=ax)\n',
            '    cbar.set_label("NMAD (m)", fontweight="bold")\n',
            '    \n',
            '    plt.tight_layout()\n',
            '    plt.show()\n',
            '    \n',
            '    print("\\n[COMPLETE] Slope-aspect interaction analysis finished.")\n'
        ]
    }

    # Insert both cells
    cells.insert(insert_idx, markdown_cell)
    cells.insert(insert_idx + 1, analysis_cell)
    print(f"[OK] Added slope-aspect interaction section at cells {insert_idx}-{insert_idx+1}")

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    add_remaining_enhancements(notebook_path)
