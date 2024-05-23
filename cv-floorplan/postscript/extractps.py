import re
import matplotlib.pyplot as plt
import numpy as np

# https://www.freeconvert.com/pdf-to-ps/download

def parse_ps_file(ps_content):
    # Define the regex pattern to capture 'moveto', 'lineto', and 'stroke'
    command_pattern = re.compile(r'(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)\s+(m|l|S)')

    paths = []  # List to hold all paths
    current_path = []  # Temporary storage for the current path points

    for match in command_pattern.finditer(ps_content):
        x, y, cmd = float(match.group(1)), float(match.group(2)), match.group(3)
        
        if cmd == 'm':  # moveto command
            if current_path:  # if there's an ongoing path, store it before starting a new one
                paths.append(current_path)
                current_path = []
            current_path.append((x, y))  # Start a new path
        elif cmd == 'l':  # lineto command
            current_path.append((x, y))  # Add point to current path
        elif cmd == 'S':  # stroke command
            if current_path:
                paths.append(current_path)  # Store the finished path
                current_path = []  # Reset the path for the next series of commands

    if current_path:  # If there's any remaining path after the last stroke
        paths.append(current_path)

    return paths

def distance_point_to_line(point, line):
    """Calculate the distance from a point to a line given in rho, theta format."""
    rho, theta = line
    x, y = point
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho)

def evaluate_hough_lines(extracted_paths, hough_lines, threshold=5):
    """Evaluate how well the Hough lines fit the extracted lines with an accuracy percentage."""
    close_points = 0
    total_points = 0
    for path in extracted_paths:
        for point in path:
            distances = [distance_point_to_line(point, line) for line in hough_lines]
            min_distance = min(distances)
            if min_distance <= threshold:
                close_points += 1
            total_points += 1
    print(str(close_points) + "/" + str(total_points))
    accuracy_percentage = (close_points / total_points) * 100 if total_points > 0 else 0
    return accuracy_percentage

# Load Hough lines from a file
def load_hough_lines(filename):
    hough_lines = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            rho, theta = float(parts[0]), float(parts[1])
            hough_lines.append((rho, theta))
    return hough_lines

# Reading the content from a PS file (ensure the file path is correct)
with open('E2_2_another.ps', 'r') as file:
    content = file.read()

# Parse the PS file content
extracted_lines = parse_ps_file(content)

# Output the extracted lines
if extracted_lines:
    for line in extracted_lines:
        print(line)
else:
    print("No lines extracted.")

# Read Hough lines from file
hough_lines = load_hough_lines('hough_lines.txt')

# Calculate the minimum and maximum coordinates of the extracted points
min_x = min(point[0] for path in extracted_lines for point in path)
max_x = max(point[0] for path in extracted_lines for point in path)
min_y = min(point[1] for path in extracted_lines for point in path)
max_y = max(point[1] for path in extracted_lines for point in path)

# Plot the extracted points and Hough lines
plt.figure(figsize=(10, 10))
i = 0
for path in extracted_lines:
    print("path ", i)
    i += 1
    x_coords = [point[0] for point in path]
    y_coords = [max_y - point[1] for point in path]  # Invert the y-coordinates
    plt.scatter(x_coords, y_coords, s=10, color='blue')

for rho, theta in hough_lines:
    print("hough line")
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(max_y - (y0 + 1000 * (a)))  # Invert the y-coordinate
    x2 = int(x0 - 1000 * (-b))
    y2 = int(max_y - (y0 - 1000 * (a)))  # Invert the y-coordinate
    plt.plot((x1, x2), (y1, y2), 'r-')

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Extracted Points and Hough Lines')
plt.tight_layout()
plt.show()

# Compute accuracy
accuracy = evaluate_hough_lines(extracted_lines, hough_lines)
print(f"Accuracy of Hough lines fitting: {accuracy:.2f}%")