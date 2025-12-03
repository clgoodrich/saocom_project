"""
Generate a horizontal, compact flowchart for SAOCOM DEM Validation workflow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Define colors
COLORS = {
    'start_end': '#2ECC71',
    'input_output': '#3498DB',
    'process': '#E8F4F8',
    'decision': '#F39C12',
    'analysis': '#9B59B6',
    'export': '#27AE60',
    'text': '#2C3E50'
}

def create_box(ax, x, y, width, height, text, box_type='process', fontsize=7):
    """Create a flowchart box with appropriate shape and color."""

    if box_type == 'start_end':
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            facecolor=COLORS['start_end'],
            edgecolor=COLORS['text'], linewidth=2
        )
        fontweight = 'bold'
        fontsize = fontsize + 1

    elif box_type == 'input_output':
        points = np.array([
            [x - width/2 + 0.08, y - height/2],
            [x + width/2 + 0.08, y - height/2],
            [x + width/2 - 0.08, y + height/2],
            [x - width/2 - 0.08, y + height/2]
        ])
        box = patches.Polygon(
            points, closed=True,
            facecolor=COLORS['input_output'],
            edgecolor=COLORS['text'], linewidth=1.5
        )
        fontweight = 'normal'

    elif box_type == 'decision':
        points = np.array([
            [x, y - height/2],
            [x + width/2, y],
            [x, y + height/2],
            [x - width/2, y]
        ])
        box = patches.Polygon(
            points, closed=True,
            facecolor=COLORS['decision'],
            edgecolor=COLORS['text'], linewidth=1.5
        )
        fontweight = 'bold'

    elif box_type == 'analysis':
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=COLORS['analysis'],
            edgecolor=COLORS['text'], linewidth=1.5, alpha=0.7
        )
        fontweight = 'normal'

    elif box_type == 'export':
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=COLORS['export'],
            edgecolor=COLORS['text'], linewidth=1.5, alpha=0.8
        )
        fontweight = 'bold'

    else:  # process
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=COLORS['process'],
            edgecolor=COLORS['text'], linewidth=1.5
        )
        fontweight = 'normal'

    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=COLORS['text'], wrap=True)

    return box

def draw_arrow(ax, x1, y1, x2, y2, style='solid'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=15,
        linewidth=1.5, color=COLORS['text'],
        linestyle=style
    )
    ax.add_patch(arrow)

def create_flowchart():
    """Create the horizontal SAOCOM validation flowchart."""

    fig, ax = plt.subplots(figsize=(28, 12))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(14, 11.5, 'SAOCOM DEM Validation Workflow',
            fontsize=16, fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(14, 11, 'InSAR Height Validation Against Reference DEMs',
            fontsize=10, ha='center', style='italic', color=COLORS['text'])

    # Box dimensions
    bw, bh = 1.4, 0.6  # Standard box width and height
    gap = 1.6  # Horizontal gap between boxes

    # Row positions
    row1 = 9.5
    row2 = 7.5
    row3 = 5.5
    row4 = 3.5
    row5 = 1.5

    x = 0.8

    # START
    create_box(ax, x, row1, 1.2, 0.7, 'START', 'start_end', 7)
    x_prev = x
    x += gap

    # Load SAOCOM
    create_box(ax, x, row1, bw, bh, 'Load SAOCOM\nCSV Data', 'input_output', 6)
    draw_arrow(ax, x_prev + 0.6, row1, x - bw/2, row1)
    x_prev = x
    x += gap

    # Remove isolated points
    create_box(ax, x, row1, bw, bh, 'Remove\nIsolated Points', 'process', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1)
    x_prev = x
    x += gap

    # Load DEMs (parallel)
    create_box(ax, x, row1+0.8, bw, bh, 'Load TINItaly\n(10m)', 'input_output', 6)
    create_box(ax, x, row1-0.8, bw, bh, 'Load Copernicus\n(30m)', 'input_output', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1+0.8)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1-0.8)
    x_prev_dem = x
    x += gap

    # Resample DEMs (parallel)
    create_box(ax, x, row1+0.8, bw, bh, 'Resample\nto 10m', 'process', 6)
    create_box(ax, x, row1-0.8, bw, bh, 'Upsample\n30m→10m', 'process', 6)
    draw_arrow(ax, x_prev_dem + bw/2, row1+0.8, x - bw/2, row1+0.8)
    draw_arrow(ax, x_prev_dem + bw/2, row1-0.8, x - bw/2, row1-0.8)
    x_prev = x
    x += gap

    # Sample at points
    create_box(ax, x, row1, bw, bh, 'Sample DEMs\nat Points', 'process', 6)
    draw_arrow(ax, x_prev + bw/2, row1+0.8, x - bw/2, row1)
    draw_arrow(ax, x_prev + bw/2, row1-0.8, x - bw/2, row1)
    x_prev = x
    x += gap

    # Control points (decision)
    create_box(ax, x, row1, 1.2, 0.7, 'Select\nControl Points\n(coh≥0.8)', 'decision', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - 0.6, row1)
    x_prev = x
    x += gap

    # Calibration (parallel)
    create_box(ax, x, row1+0.8, bw, bh, 'Calibrate to\nTINItaly', 'process', 6)
    create_box(ax, x, row1-0.8, bw, bh, 'Calibrate to\nCopernicus', 'process', 6)
    draw_arrow(ax, x_prev + 0.6, row1, x - bw/2, row1+0.8)
    draw_arrow(ax, x_prev + 0.6, row1, x - bw/2, row1-0.8)
    x_prev = x
    x += gap

    # Compute residuals
    create_box(ax, x, row1, bw, bh, 'Compute\nResiduals', 'process', 6)
    draw_arrow(ax, x_prev + bw/2, row1+0.8, x - bw/2, row1)
    draw_arrow(ax, x_prev + bw/2, row1-0.8, x - bw/2, row1)
    x_prev = x
    x += gap

    # Outlier detection
    create_box(ax, x, row1, bw, bh, 'Outlier\nDetection', 'process', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1)
    x_prev = x
    x += gap

    # Clean dataset
    create_box(ax, x, row1, bw, bh, 'Clean\nDataset', 'process', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1)
    x_prev = x
    x += gap

    # Core metrics
    create_box(ax, x, row1, bw, bh, 'Calculate\nBias/RMSE/NMAD', 'analysis', 6)
    draw_arrow(ax, x_prev + bw/2, row1, x - bw/2, row1)

    # Branch down to terrain analysis
    draw_arrow(ax, x, row1 - bh/2, x, row2 + bh/2)
    x_terrain = x

    # Second row - Terrain & LC analysis
    x2 = 0.8
    create_box(ax, x2, row2, bw, bh, 'Compute\nSlope/Aspect', 'process', 6)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Accuracy by\nSlope', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Radar\nShadow', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Load CORINE\nLand Cover', 'input_output', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Accuracy by\nLand Cover', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Accuracy by\nElevation', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Accuracy by\nCoherence', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)
    x2 += gap

    create_box(ax, x2, row2, bw, bh, 'Coverage &\nVoid Analysis', 'analysis', 6)
    draw_arrow(ax, x2 - gap + bw/2, row2, x2 - bw/2, row2)

    # Connect first row to second row
    draw_arrow(ax, x_terrain, row1 - bh/2, 0.8, row2)

    # Third row - Visualization
    draw_arrow(ax, x2, row2 - bh/2, x2, row3 + bh/2)

    x3 = 0.8
    create_box(ax, x3, row3, bw, bh, 'Scatter &\nBland-Altman', 'process', 6)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'Residual\nMaps', 'process', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'Hexbin\nDensity', 'process', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'Gridded\nComparison', 'process', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'Violin &\nDistribution', 'process', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'PCA Void\nFactors', 'analysis', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)
    x3 += gap

    create_box(ax, x3, row3, bw, bh, 'Summary\nDashboard', 'process', 6)
    draw_arrow(ax, x3 - gap + bw/2, row3, x3 - bw/2, row3)

    # Connect second row to third row
    draw_arrow(ax, x2, row2 - bh/2, 0.8, row3)

    # Fourth row - Exports
    draw_arrow(ax, x3, row3 - bh/2, x3, row4 + bh/2)

    x4 = 4
    create_box(ax, x4, row4, bw, bh, 'Export\nShapefile', 'export', 6)
    x4 += gap + 0.5

    create_box(ax, x4, row4, bw, bh, 'Export\nCSV Stats', 'export', 6)
    draw_arrow(ax, x4 - gap - 0.5 + bw/2, row4, x4 - bw/2, row4)
    x4 += gap + 0.5

    create_box(ax, x4, row4, bw, bh, 'Export\nWord Report', 'export', 6)
    draw_arrow(ax, x4 - gap - 0.5 + bw/2, row4, x4 - bw/2, row4)
    x4 += gap + 0.5

    # Connect third row to exports
    draw_arrow(ax, x3, row3 - bh/2, x4 - gap - 0.5 - bw/2, row4)

    # END
    create_box(ax, x4, row4, 1.2, 0.7, 'END', 'start_end', 7)
    draw_arrow(ax, x4 - gap - 0.5 + bw/2, row4, x4 - 0.6, row4)

    # Legend at bottom
    legend_y = 0.5
    legend_x = 2
    ax.text(legend_x - 0.8, legend_y, 'Legend:', fontsize=8, fontweight='bold', color=COLORS['text'])

    create_box(ax, legend_x, legend_y, 1, 0.3, 'Start/End', 'start_end', 6)
    legend_x += 1.4
    create_box(ax, legend_x, legend_y, 1, 0.3, 'Input/Output', 'input_output', 6)
    legend_x += 1.4
    create_box(ax, legend_x, legend_y, 1, 0.3, 'Process', 'process', 6)
    legend_x += 1.4
    create_box(ax, legend_x, legend_y, 1, 0.3, 'Analysis', 'analysis', 6)
    legend_x += 1.4
    create_box(ax, legend_x, legend_y, 1, 0.3, 'Export', 'export', 6)

    plt.tight_layout()
    return fig

# Generate and save
print("Generating horizontal SAOCOM workflow flowchart...")
fig = create_flowchart()
output_path = 'docs/saocom_workflow_flowchart.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Flowchart saved to: {output_path}")

pdf_path = 'docs/saocom_workflow_flowchart.pdf'
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"[OK] PDF version saved to: {pdf_path}")

plt.close()
print("Done!")
