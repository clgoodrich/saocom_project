"""
Enhance the SAOCOM analysis notebook with additional statistical analyses:
1. Add RMSE to land cover tables
2. Compute penetration depth (SAOCOM - Copernicus)
3. Add slope-aspect interaction analysis
4. Expand Copernicus comparison
5. Add confidence intervals to statistics
"""

import json
import re

def add_enhancements(notebook_path):
    """Add analytical enhancements to the notebook."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Find the cell after land cover analysis to add new analyses
    cells = notebook['cells']

    # 1. Find and enhance land cover statistics table (add RMSE)
    enhance_land_cover_table(cells)

    # 2. Add penetration depth analysis
    add_penetration_depth_analysis(cells)

    # 3. Add slope-aspect interaction analysis
    add_slope_aspect_interaction(cells)

    # 4. Expand Copernicus comparison
    enhance_copernicus_comparison(cells)

    # 5. Add confidence intervals function
    add_confidence_intervals(cells)

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("[DONE] Enhancements added to notebook")

def enhance_land_cover_table(cells):
    """Add RMSE column to land cover statistics tables."""

    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Find land cover statistics calculations
            if 'land_cover_level' in source and 'groupby' in source and 'agg' in source:
                # Check if RMSE is already there
                if 'rmse' not in source.lower():
                    lines = source.split('\n')
                    new_lines = []

                    for line in lines:
                        new_lines.append(line)

                        # After bias and NMAD, add RMSE
                        if "'nmad':" in line or "'NMAD':" in line:
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(' ' * indent + "'rmse': lambda x: np.sqrt(np.mean(x**2)),")

                    # Update cell
                    cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
                    print(f"[OK] Added RMSE to land cover table at cell {idx}")

def add_penetration_depth_analysis(cells):
    """Add analysis of penetration depth (SAOCOM - Copernicus difference)."""

    # Find where to insert (after Copernicus residual computation)
    insert_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'residuals_cop' in source and '=' in source:
                insert_idx = idx + 1
                break

    if insert_idx is None:
        print("[SKIP] Could not find Copernicus residual calculation")
        return

    # Create new cell for penetration depth analysis
    penetration_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Penetration Depth Analysis\n',
            '# Compute the difference between SAOCOM and Copernicus as a proxy for penetration\n',
            'print("\\nPenetration Depth Analysis (SAOCOM - Copernicus)")\n',
            'print("="*60)\n',
            '\n',
            '# Compute penetration depth (positive = SAOCOM below Copernicus)\n',
            'penetration_depth = saocom_cleaned["copernicus_height"] - saocom_cleaned["calibrated_height"]\n',
            'saocom_cleaned["penetration_depth"] = penetration_depth\n',
            '\n',
            '# Overall statistics\n',
            'print(f"Mean penetration: {np.mean(penetration_depth):.3f} m")\n',
            'print(f"Median penetration: {np.median(penetration_depth):.3f} m")\n',
            'print(f"Std Dev: {np.std(penetration_depth):.3f} m")\n',
            'print(f"NMAD: {1.4826 * np.median(np.abs(penetration_depth - np.median(penetration_depth))):.3f} m")\n',
            '\n',
            '# Penetration by land cover\n',
            'print("\\nPenetration depth by land cover (Level 1):")\n',
            'penetration_by_lc = saocom_cleaned.groupby("land_cover_level1")["penetration_depth"].agg([\n',
            '    ("count", "count"),\n',
            '    ("mean", "mean"),\n',
            '    ("median", "median"),\n',
            '    ("std", "std")\n',
            ']).round(3)\n',
            'print(penetration_by_lc)\n',
            '\n',
            '# Visualization\n',
            'fig, ax = plt.subplots(figsize=(12, 6))\n',
            'saocom_cleaned.boxplot(column="penetration_depth", by="land_cover_level1", ax=ax)\n',
            'ax.set_title("Penetration Depth by Land Cover Type", fontweight="bold")\n',
            'ax.set_xlabel("Land Cover")\n',
            'ax.set_ylabel("Penetration Depth (m)")\n',
            'ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)\n',
            'plt.suptitle("")  # Remove default title\n',
            'plt.xticks(rotation=45, ha="right")\n',
            'plt.tight_layout()\n',
            'plt.show()\n'
        ]
    }

    cells.insert(insert_idx, penetration_cell)
    print(f"[OK] Added penetration depth analysis at cell {insert_idx}")

def add_slope_aspect_interaction(cells):
    """Add slope-aspect interaction analysis (radar-facing vs away)."""

    # Find where to insert (after slope/aspect analysis)
    insert_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'aspect' in source and 'slope' in source and 'groupby' in source:
                insert_idx = idx + 1
                break

    if insert_idx is None:
        print("[SKIP] Could not find slope/aspect analysis")
        return

    # Create new cell for slope-aspect interaction
    interaction_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Slope-Aspect Interaction Analysis\n',
            '# Analyze residuals based on whether slope faces toward/away from radar\n',
            'print("\\nSlope-Aspect Interaction Analysis")\n',
            'print("="*60)\n',
            '\n',
            '# SAOCOM looks from ~98 degrees (East)\n',
            '# Radar-facing: aspects roughly 45-135 degrees (NE to SE)\n',
            '# Radar-away: aspects roughly 225-315 degrees (SW to NW)\n',
            '\n',
            'def classify_radar_geometry(aspect):\n',
            '    """Classify aspect relative to radar look direction (98 deg)."""\n',
            '    if 45 <= aspect <= 135:\n',
            '        return "Radar-facing"\n',
            '    elif 225 <= aspect <= 315:\n',
            '        return "Radar-away"\n',
            '    else:\n',
            '        return "Oblique"\n',
            '\n',
            'saocom_cleaned["radar_geometry"] = saocom_cleaned["aspect"].apply(classify_radar_geometry)\n',
            '\n',
            '# Statistics by radar geometry\n',
            'radar_geom_stats = saocom_cleaned.groupby("radar_geometry")["residual_tin"].agg([\n',
            '    ("count", "count"),\n',
            '    ("bias", "mean"),\n',
            '    ("std", "std"),\n',
            '    ("nmad", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),\n',
            '    ("rmse", lambda x: np.sqrt(np.mean(x**2)))\n',
            ']).round(3)\n',
            '\n',
            'print("\\nResidual statistics by radar geometry:")\n',
            'print(radar_geom_stats)\n',
            '\n',
            '# Stratify by slope class and radar geometry\n',
            'print("\\nResiduals by slope class and radar geometry:")\n',
            'slope_radar_stats = saocom_cleaned.groupby(["slope_class", "radar_geometry"])["residual_tin"].agg([\n',
            '    ("count", "count"),\n',
            '    ("bias", "mean"),\n',
            '    ("nmad", lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))\n',
            ']).round(3)\n',
            'print(slope_radar_stats)\n',
            '\n',
            '# Visualization\n',
            'fig, ax = plt.subplots(figsize=(10, 6))\n',
            'saocom_cleaned.boxplot(column="residual_tin", by="radar_geometry", ax=ax)\n',
            'ax.set_title("Residuals by Radar Geometry", fontweight="bold")\n',
            'ax.set_xlabel("Radar Geometry")\n',
            'ax.set_ylabel("Residual (m)")\n',
            'ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)\n',
            'plt.suptitle("")\n',
            'plt.tight_layout()\n',
            'plt.show()\n'
        ]
    }

    cells.insert(insert_idx, interaction_cell)
    print(f"[OK] Added slope-aspect interaction analysis at cell {insert_idx}")

def enhance_copernicus_comparison(cells):
    """Add expanded Copernicus comparison section."""

    # Find where Copernicus is mentioned
    insert_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'copernicus' in source.lower() and 'residual' in source.lower():
                insert_idx = idx + 1
                break

    if insert_idx is None:
        print("[SKIP] Could not find Copernicus comparison section")
        return

    copernicus_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Expanded Copernicus vs TINItaly Comparison\n',
            'print("\\nExpanded Copernicus Comparison")\n',
            'print("="*60)\n',
            '\n',
            '# Compare residuals: SAOCOM-TINItaly vs SAOCOM-Copernicus\n',
            'print("\\nResidual comparison:")\n',
            'comparison_stats = pd.DataFrame({\n',
            '    "Reference": ["TINItaly (10m)", "Copernicus (30m)"],\n',
            '    "Bias": [np.mean(residuals_tin), np.mean(residuals_cop)],\n',
            '    "NMAD": [nmad_tin, nmad_cop],\n',
            '    "RMSE": [np.sqrt(np.mean(residuals_tin**2)), np.sqrt(np.mean(residuals_cop**2))],\n',
            '    "Std Dev": [np.std(residuals_tin), np.std(residuals_cop)]\n',
            '}).round(3)\n',
            'print(comparison_stats)\n',
            '\n',
            '# Correlation between the two residual sets\n',
            'correlation = np.corrcoef(residuals_tin, residuals_cop)[0, 1]\n',
            'print(f"\\nCorrelation between residuals: {correlation:.4f}")\n',
            '\n',
            '# Difference between residuals (systematic difference between DEMs)\n',
            'residual_difference = residuals_tin - residuals_cop\n',
            'print(f"\\nSystematic difference (TINItaly - Copernicus):")\n',
            'print(f"  Mean: {np.mean(residual_difference):.3f} m")\n',
            'print(f"  Std: {np.std(residual_difference):.3f} m")\n',
            '\n',
            '# Scatter plot: TINItaly vs Copernicus residuals\n',
            'fig, ax = plt.subplots(figsize=(10, 8))\n',
            'ax.scatter(residuals_tin, residuals_cop, alpha=0.1, s=1)\n',
            'ax.plot([-50, 50], [-50, 50], "r--", label="1:1 line")\n',
            'ax.set_xlabel("SAOCOM - TINItaly Residual (m)", fontweight="bold")\n',
            'ax.set_ylabel("SAOCOM - Copernicus Residual (m)", fontweight="bold")\n',
            'ax.set_title(f"Residual Comparison (r={correlation:.3f})", fontweight="bold")\n',
            'ax.grid(alpha=0.3)\n',
            'ax.legend()\n',
            'ax.set_aspect("equal")\n',
            'plt.tight_layout()\n',
            'plt.show()\n',
            '\n',
            '# Bland-Altman plot for DEM comparison\n',
            'fig, ax = plt.subplots(figsize=(12, 6))\n',
            'mean_residuals = (residuals_tin + residuals_cop) / 2\n',
            'diff_residuals = residuals_tin - residuals_cop\n',
            'ax.scatter(mean_residuals, diff_residuals, alpha=0.1, s=1)\n',
            'ax.axhline(y=np.mean(diff_residuals), color="red", linestyle="-", label=f"Mean: {np.mean(diff_residuals):.2f}m")\n',
            'ax.axhline(y=np.mean(diff_residuals) + 1.96*np.std(diff_residuals), color="red", linestyle="--", label=f"+1.96 SD: {np.mean(diff_residuals) + 1.96*np.std(diff_residuals):.2f}m")\n',
            'ax.axhline(y=np.mean(diff_residuals) - 1.96*np.std(diff_residuals), color="red", linestyle="--", label=f"-1.96 SD: {np.mean(diff_residuals) - 1.96*np.std(diff_residuals):.2f}m")\n',
            'ax.set_xlabel("Mean of Residuals (m)", fontweight="bold")\n',
            'ax.set_ylabel("Difference (TINItaly - Copernicus) (m)", fontweight="bold")\n',
            'ax.set_title("Bland-Altman: DEM Comparison", fontweight="bold")\n',
            'ax.legend()\n',
            'ax.grid(alpha=0.3)\n',
            'plt.tight_layout()\n',
            'plt.show()\n'
        ]
    }

    cells.insert(insert_idx, copernicus_cell)
    print(f"[OK] Added expanded Copernicus comparison at cell {insert_idx}")

def add_confidence_intervals(cells):
    """Add confidence interval calculations to statistics."""

    # Add CI helper function near the beginning
    insert_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import' in source and 'numpy' in source:
                insert_idx = idx + 1
                break

    if insert_idx is None:
        print("[SKIP] Could not find imports section")
        return

    ci_function_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Confidence Interval Calculation Functions\n',
            'from scipy import stats\n',
            '\n',
            'def calculate_confidence_intervals(data, confidence=0.95):\n',
            '    """Calculate confidence intervals for key statistics.\n',
            '    \n',
            '    Args:\n',
            '        data: array of residuals\n',
            '        confidence: confidence level (default 0.95)\n',
            '    \n',
            '    Returns:\n',
            '        dict with statistics and their CIs\n',
            '    """\n',
            '    n = len(data)\n',
            '    mean = np.mean(data)\n',
            '    std = np.std(data, ddof=1)\n',
            '    se_mean = std / np.sqrt(n)\n',
            '    \n',
            '    # t-distribution for mean CI\n',
            '    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)\n',
            '    mean_ci = (mean - t_crit * se_mean, mean + t_crit * se_mean)\n',
            '    \n',
            '    # Bootstrap for NMAD CI\n',
            '    nmad = 1.4826 * np.median(np.abs(data - np.median(data)))\n',
            '    nmad_bootstrap = []\n',
            '    for _ in range(1000):\n',
            '        sample = np.random.choice(data, size=n, replace=True)\n',
            '        nmad_bootstrap.append(1.4826 * np.median(np.abs(sample - np.median(sample))))\n',
            '    nmad_ci = (np.percentile(nmad_bootstrap, (1-confidence)/2*100),\n',
            '               np.percentile(nmad_bootstrap, (1+confidence)/2*100))\n',
            '    \n',
            '    # RMSE CI (using chi-square for variance)\n',
            '    rmse = np.sqrt(np.mean(data**2))\n',
            '    chi2_lower = stats.chi2.ppf((1-confidence)/2, n)\n',
            '    chi2_upper = stats.chi2.ppf((1+confidence)/2, n)\n',
            '    rmse_ci = (rmse * np.sqrt(n/chi2_upper), rmse * np.sqrt(n/chi2_lower))\n',
            '    \n',
            '    return {\n',
            '        "mean": mean,\n',
            '        "mean_ci": mean_ci,\n',
            '        "nmad": nmad,\n',
            '        "nmad_ci": nmad_ci,\n',
            '        "rmse": rmse,\n',
            '        "rmse_ci": rmse_ci,\n',
            '        "std": std\n',
            '    }\n',
            '\n',
            'def format_with_ci(value, ci, decimals=3):\n',
            '    """Format a value with its confidence interval."""\n',
            '    return f"{value:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"\n',
            '\n',
            'print("Confidence interval functions loaded.")\n'
        ]
    }

    cells.insert(insert_idx, ci_function_cell)
    print(f"[OK] Added confidence interval functions at cell {insert_idx}")

    # Now add CI calculations to main statistics
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Find overall statistics calculation
            if 'print("Overall Statistics")' in source or 'Overall validation statistics' in source:
                lines = source.split('\n')
                new_lines = []

                for line in lines:
                    new_lines.append(line)

                    # After printing NMAD, add CI calculations
                    if 'nmad_tin' in line and 'print' in line and 'CI' not in source:
                        new_lines.append('')
                        new_lines.append('# Add confidence intervals')
                        new_lines.append('ci_stats = calculate_confidence_intervals(residuals_tin)')
                        new_lines.append('print(f"\\nWith 95% Confidence Intervals:")')
                        new_lines.append('print(f"Bias: {format_with_ci(ci_stats[\\"mean\\"], ci_stats[\\"mean_ci\\"])}")')
                        new_lines.append('print(f"NMAD: {format_with_ci(ci_stats[\\"nmad\\"], ci_stats[\\"nmad_ci\\"])}")')
                        new_lines.append('print(f"RMSE: {format_with_ci(ci_stats[\\"rmse\\"], ci_stats[\\"rmse_ci\\"])}")')

                        cell['source'] = [line + '\n' for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines else [])
                        print(f"[OK] Added CIs to overall statistics at cell {idx}")
                        break

if __name__ == '__main__':
    notebook_path = r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean_solo_figs.ipynb'
    add_enhancements(notebook_path)
