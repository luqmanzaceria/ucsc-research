import xml.etree.ElementTree as ET
import csv
import re

def parse_svg(svg_file):
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Namespace map to handle SVG namespace
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    lines = []

    # Find all path elements
    paths = root.findall('.//svg:path', namespaces)
    
    for path in paths:
        # Extract the 'd' attribute which contains the path data
        path_data = path.get("d")

        # Regular expression to match coordinates
        coord_pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
        
        # Split the path data into commands and coordinates
        segments = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|' + coord_pattern, path_data)

        head = []
        coordinates = []
        for segment in segments:
            if segment in 'MmLlHhVvCcSsQqTtAaZz':
                if coordinates:
                    # Process the collected coordinates
                    paired_coordinates = [[coordinates[i], -float(coordinates[i + 1])] for i in range(0, len(coordinates), 2)]
                    if paired_coordinates:
                        if not head:
                            head.extend(paired_coordinates[0])
                        else:
                            lines.extend([paired_coordinates[i] + paired_coordinates[i + 1] for i in range(len(paired_coordinates) - 1)])
                    coordinates = []
            else:
                # Append numeric value
                coordinates.append(float(segment))

    return lines

def save_to_csv(lines, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for line in lines:
            csv_writer.writerow(line)

if __name__ == "__main__":
    # Define your SVG file path
    svg_file = "new_reassembled_E2_3.svg"
    
    # Define your CSV file name
    csv_file = "new_reassembled_E2_3.csv"

    # Parse SVG and save to CSV
    lines = parse_svg(svg_file)
    save_to_csv(lines, csv_file)
