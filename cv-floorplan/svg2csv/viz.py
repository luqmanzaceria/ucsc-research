import csv
import numpy as np
from sympy import Plane
import matplotlib.pyplot as plt

# Returns a 3D array [[[x1, y1], [x2, y2]]]
def load_map_csv(input_path):
    map_data = []
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Check if the row has the expected 4 columns
            if len(row) == 4:
                try:
                    p1 = np.array([float(row[0]), float(row[1])])
                    p2 = np.array([float(row[2]), float(row[3])])
                    map_data.append(np.array([p1, p2]))
                    print(f"Got the row: {row}")
                except ValueError:
                    # Handle the exception if conversion to float fails
                    print(f"Skipping row due to conversion error: {row}")
            # else:
            #     print(f"Skipping incomplete row: {row}")
    
    return np.array(map_data)

def draw_segments(planes):
    if isinstance(planes[0], Plane):
        planes = [plane.to_2D() for plane in planes]
    
    # Create the plot
    for seg in planes:
        plt.plot(*zip(*seg), color='black')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    floorplan = load_map_csv("./new_reassembled_E2_3.csv")
    draw_segments(floorplan)