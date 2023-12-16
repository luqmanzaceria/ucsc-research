import xml.etree.ElementTree as ET
import csv

def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    lines = []

    # Find all path elements
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    # print(paths)
    
    for path in paths:
        # Extract the 'd' attribute which contains the path data
        path_data = path.get("d")

        # Split the path data into commands and coordinates
        commands = path_data.split()
        
        # Extract coordinates from commands
        coordinates = []
        head = []
        for command in commands:
            if command[0] == "M":
                command = command[1:]
                coords = list(map(float, command.split(',')))
                coords[1] = -coords[1]
                head = []
                head.extend(coords)
            else:
                if command[0] == "C":
                    command = command[1:]
                    coordinates = []
                coords = list(map(float, command.split(',')))
                coords[1] = - coords[1]
                coordinates.extend(coords)

        # Pair coordinates to form lines
        paired_coordinates = [[coordinates[i], coordinates[i + 1]] for i in range(0, len(coordinates), 2)]
        # Add lines to the list
        head.extend(paired_coordinates[0])
        lines.append(head)
        lines.extend([paired_coordinates[i] + paired_coordinates[i + 1] for i in range(len(paired_coordinates) - 1)])
    return lines

def save_to_csv(lines, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for line in lines:
            csv_writer.writerow(line)

if __name__ == "__main__":
    svg_file = "reassembled_E2_3.svg"
    csv_file = "reassembled_E2_3.csv"

    lines = parse_svg(svg_file)
    save_to_csv(lines, csv_file)