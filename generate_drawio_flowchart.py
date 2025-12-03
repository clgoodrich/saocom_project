"""
Generate SAOCOM workflow flowchart in draw.io format.
Creates an editable .drawio file that can be opened in draw.io/diagrams.net.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_drawio_flowchart():
    """Create flowchart in draw.io XML format."""

    # Create root mxfile element
    mxfile = ET.Element('mxfile', {
        'host': 'app.diagrams.net',
        'modified': '2024-12-03T00:00:00.000Z',
        'agent': 'Python Script',
        'version': '22.1.0',
        'type': 'device'
    })

    # Create diagram
    diagram = ET.SubElement(mxfile, 'diagram', {
        'id': 'saocom_workflow',
        'name': 'SAOCOM Validation Workflow'
    })

    # Create mxGraphModel
    model = ET.SubElement(diagram, 'mxGraphModel', {
        'dx': '1434',
        'dy': '843',
        'grid': '1',
        'gridSize': '10',
        'guides': '1',
        'tooltips': '1',
        'connect': '1',
        'arrows': '1',
        'fold': '1',
        'page': '1',
        'pageScale': '1',
        'pageWidth': '3300',
        'pageHeight': '1400',
        'math': '0',
        'shadow': '0'
    })

    # Create root
    root = ET.SubElement(model, 'root')

    # Add default parent cells
    ET.SubElement(root, 'mxCell', {'id': '0'})
    ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})

    # Cell ID counter
    cell_id = 2

    # Style definitions
    styles = {
        'start_end': 'rounded=1;whiteSpace=wrap;html=1;fillColor=#2ECC71;strokeColor=#2C3E50;strokeWidth=2;fontSize=11;fontStyle=1;',
        'input_output': 'shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fillColor=#3498DB;strokeColor=#2C3E50;strokeWidth=1.5;fontSize=10;',
        'process': 'rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F4F8;strokeColor=#2C3E50;strokeWidth=1.5;fontSize=10;',
        'decision': 'rhombus;whiteSpace=wrap;html=1;fillColor=#F39C12;strokeColor=#2C3E50;strokeWidth=1.5;fontSize=10;fontStyle=1;',
        'analysis': 'rounded=1;whiteSpace=wrap;html=1;fillColor=#9B59B6;strokeColor=#2C3E50;strokeWidth=1.5;fontSize=10;opacity=70;',
        'export': 'rounded=1;whiteSpace=wrap;html=1;fillColor=#27AE60;strokeColor=#2C3E50;strokeWidth=1.5;fontSize=10;fontStyle=1;opacity=80;',
        'arrow': 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#2C3E50;endArrow=classic;'
    }

    # Box dimensions
    box_w, box_h = 120, 60
    gap_x = 140

    # Starting positions
    start_x = 50
    row1_y = 100
    row2_y = 250
    row3_y = 400
    row4_y = 550

    # Row 1: Main processing pipeline
    x = start_x

    boxes = []

    # START
    boxes.append({
        'id': cell_id, 'text': 'START', 'x': x, 'y': row1_y, 'w': 100, 'h': 60, 'style': 'start_end'
    })
    cell_id += 1
    x += gap_x

    # Load SAOCOM
    boxes.append({
        'id': cell_id, 'text': 'Load SAOCOM<br/>CSV Data', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'input_output'
    })
    cell_id += 1
    x += gap_x

    # Remove isolated
    boxes.append({
        'id': cell_id, 'text': 'Remove<br/>Isolated Points', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x += gap_x

    # Load TINItaly (above)
    boxes.append({
        'id': cell_id, 'text': 'Load TINItaly<br/>(10m)', 'x': x, 'y': row1_y - 80, 'w': box_w, 'h': box_h, 'style': 'input_output'
    })
    tin_load_id = cell_id
    cell_id += 1

    # Load Copernicus (below)
    boxes.append({
        'id': cell_id, 'text': 'Load Copernicus<br/>(30m)', 'x': x, 'y': row1_y + 80, 'w': box_w, 'h': box_h, 'style': 'input_output'
    })
    cop_load_id = cell_id
    cell_id += 1
    x += gap_x

    # Resample TINItaly
    boxes.append({
        'id': cell_id, 'text': 'Resample<br/>to 10m', 'x': x, 'y': row1_y - 80, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    tin_resample_id = cell_id
    cell_id += 1

    # Upsample Copernicus
    boxes.append({
        'id': cell_id, 'text': 'Upsample<br/>30m → 10m', 'x': x, 'y': row1_y + 80, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cop_resample_id = cell_id
    cell_id += 1
    x += gap_x

    # Sample at points
    boxes.append({
        'id': cell_id, 'text': 'Sample DEMs<br/>at Points', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    sample_id = cell_id
    cell_id += 1
    x += gap_x

    # Control points (decision)
    boxes.append({
        'id': cell_id, 'text': 'Select<br/>Control Points<br/>(coh ≥ 0.8)', 'x': x, 'y': row1_y, 'w': 110, 'h': 80, 'style': 'decision'
    })
    decision_id = cell_id
    cell_id += 1
    x += gap_x

    # Calibrate TINItaly
    boxes.append({
        'id': cell_id, 'text': 'Calibrate to<br/>TINItaly', 'x': x, 'y': row1_y - 80, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    calib_tin_id = cell_id
    cell_id += 1

    # Calibrate Copernicus
    boxes.append({
        'id': cell_id, 'text': 'Calibrate to<br/>Copernicus', 'x': x, 'y': row1_y + 80, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    calib_cop_id = cell_id
    cell_id += 1
    x += gap_x

    # Compute residuals
    boxes.append({
        'id': cell_id, 'text': 'Compute<br/>Residuals', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    residuals_id = cell_id
    cell_id += 1
    x += gap_x

    # Outlier detection
    boxes.append({
        'id': cell_id, 'text': 'Outlier<br/>Detection', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x += gap_x

    # Clean dataset
    boxes.append({
        'id': cell_id, 'text': 'Clean<br/>Dataset', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x += gap_x

    # Core metrics
    boxes.append({
        'id': cell_id, 'text': 'Calculate<br/>Bias/RMSE/NMAD', 'x': x, 'y': row1_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    metrics_id = cell_id
    cell_id += 1

    # Row 2: Stratified analysis
    x2 = start_x

    boxes.append({
        'id': cell_id, 'text': 'Compute<br/>Slope/Aspect', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Accuracy by<br/>Slope', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Radar<br/>Shadow', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Load CORINE<br/>Land Cover', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'input_output'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Accuracy by<br/>Land Cover', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Accuracy by<br/>Elevation', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Accuracy by<br/>Coherence', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x2 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Coverage &<br/>Void Analysis', 'x': x2, 'y': row2_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    row2_end_id = cell_id
    cell_id += 1

    # Row 3: Visualization
    x3 = start_x

    boxes.append({
        'id': cell_id, 'text': 'Scatter &<br/>Bland-Altman', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Residual<br/>Maps', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Hexbin<br/>Density', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Gridded<br/>Comparison', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Violin &<br/>Distribution', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'PCA Void<br/>Factors', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'analysis'
    })
    cell_id += 1
    x3 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Summary<br/>Dashboard', 'x': x3, 'y': row3_y, 'w': box_w, 'h': box_h, 'style': 'process'
    })
    row3_end_id = cell_id
    cell_id += 1

    # Row 4: Exports
    x4 = start_x + gap_x * 3

    boxes.append({
        'id': cell_id, 'text': 'Export<br/>Shapefile', 'x': x4, 'y': row4_y, 'w': box_w, 'h': box_h, 'style': 'export'
    })
    cell_id += 1
    x4 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Export<br/>CSV Stats', 'x': x4, 'y': row4_y, 'w': box_w, 'h': box_h, 'style': 'export'
    })
    cell_id += 1
    x4 += gap_x

    boxes.append({
        'id': cell_id, 'text': 'Export<br/>Word Report', 'x': x4, 'y': row4_y, 'w': box_w, 'h': box_h, 'style': 'export'
    })
    cell_id += 1
    x4 += gap_x

    # END
    boxes.append({
        'id': cell_id, 'text': 'END', 'x': x4, 'y': row4_y, 'w': 100, 'h': 60, 'style': 'start_end'
    })
    end_id = cell_id
    cell_id += 1

    # Add all boxes to XML
    for box in boxes:
        cell = ET.SubElement(root, 'mxCell', {
            'id': str(box['id']),
            'value': box['text'],
            'style': styles[box['style']],
            'vertex': '1',
            'parent': '1'
        })
        ET.SubElement(cell, 'mxGeometry', {
            'x': str(box['x']),
            'y': str(box['y']),
            'width': str(box['w']),
            'height': str(box['h']),
            'as': 'geometry'
        })

    # Add connections (arrows) between sequential boxes
    for i in range(len(boxes) - 1):
        # Only connect boxes in the same row (simple linear connections)
        if abs(boxes[i]['y'] - boxes[i+1]['y']) < 70 and abs(boxes[i]['x'] - boxes[i+1]['x']) < 200:
            edge = ET.SubElement(root, 'mxCell', {
                'id': str(cell_id),
                'style': styles['arrow'],
                'edge': '1',
                'parent': '1',
                'source': str(boxes[i]['id']),
                'target': str(boxes[i+1]['id'])
            })
            ET.SubElement(edge, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
            cell_id += 1

    return mxfile

# Generate and save
print("Generating draw.io flowchart...")
mxfile = create_drawio_flowchart()

# Convert to pretty XML string
xml_str = ET.tostring(mxfile, encoding='unicode')
dom = minidom.parseString(xml_str)
pretty_xml = dom.toprettyxml(indent='  ')

# Remove extra blank lines
pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])

# Save to file
output_path = 'docs/saocom_workflow_flowchart.drawio'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(pretty_xml)

print(f"[OK] Draw.io flowchart saved to: {output_path}")
print()
print("To open in draw.io:")
print("  1. Go to https://app.diagrams.net/")
print("  2. Click 'Open Existing Diagram'")
print(f"  3. Select the file: {output_path}")
print("  4. Edit, rearrange, and customize as needed!")
print()
print("All shapes are editable and you can:")
print("  - Move boxes around")
print("  - Change colors and styles")
print("  - Add/remove connections")
print("  - Adjust text and formatting")
