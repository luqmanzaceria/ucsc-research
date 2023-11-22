import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from itertools import product
import process_tiles

def tile(filename, dir_in, dir_out, d):
    img_path = os.path.join(dir_in, filename)
    img = Image.open(img_path)
    w, h = img.size
    print(f"Original image dimensions (Width x Height): {w} x {h}")
    tiled_images_info = []

    grid = product(range(0, h, d), range(0, w, d))
    for i, j in grid:
        box = (j, i, min(j+d, w), min(i+d, h))
        tile_img = img.crop(box)
        tile_name = f'{os.path.splitext(filename)[0]}_tile_{i}_{j}.png'
        tile_path = os.path.join(dir_out, tile_name)
        tile_img.save(tile_path)

        # Check dimensions of the tile
        tile_width, tile_height = tile_img.size
        print(f"Tile {tile_name} dimensions (Width x Height): {tile_width} x {tile_height}")

        tiled_images_info.append((i, j, tile_path))

    return tiled_images_info, w, h

def resize_tile(tile_img, new_width, new_height):
    # Resize the tile image to the new dimensions
    return cv2.resize(tile_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def join_images(dir_out, original_width, original_height, tile_size):
    # Create a new blank image with the original dimensions
    full_image = np.zeros((original_height, original_width), dtype=np.uint8)

    # Initialize a variable to store the prefix
    prefix = None

    # Compile a regular expression pattern to detect the file pattern
    file_pattern = re.compile(r"(.*?)_tile_(\d+)_(\d+).png")

    # Scan directory for files and determine the prefix
    for filename in os.listdir(dir_out):
        match = file_pattern.match(filename)
        if match:
            prefix, y, x = match.groups()
            break  # We just need to detect the prefix once

    if prefix is None:
        raise ValueError("No matching files found for the given pattern.")

    # Update the tile pattern to include the detected prefix
    tile_pattern = re.compile(rf"{re.escape(prefix)}_tile_(\d+)_(\d+).png")

    # Dictionary to hold tile images
    tiles = {}

    # Scan directory for matching files using the new pattern
    for filename in os.listdir(dir_out):
        match = tile_pattern.match(filename)
        if match:
            y, x = map(int, match.groups())
            tile_path = os.path.join(dir_out, filename)
            tile_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            tiles[(y, x)] = tile_img

    # Iterate over the expected tile positions
    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):
            # Calculate new tile size considering the edge cases
            new_tile_width = min(tile_size, original_width - x)
            new_tile_height = min(tile_size, original_height - y)

            # Check if the tile image exists in the dictionary
            if (y, x) in tiles:
                # Resize the tile to the new size
                resized_tile_img = resize_tile(tiles[(y, x)], new_tile_width, new_tile_height)
                # Place the resized tile into the full image
                full_image[y:y + new_tile_height, x:x + new_tile_width] = resized_tile_img

    return full_image


def main():
    parser = argparse.ArgumentParser(description='Process floor plans and generate output images.')
    parser.add_argument('input_folder', help='Path to the folder containing floor plan images.')
    parser.add_argument('output_folder', help='Path to the folder where output images will be saved.')
    args = parser.parse_args()

    tile_size = 256

    intermediate_folder = os.path.join(args.input_folder, "intermediate_images")
    os.makedirs(intermediate_folder, exist_ok=True)

    binary_folder = os.path.join(args.output_folder, "binary_images")
    os.makedirs(binary_folder, exist_ok=True)

    # Create the reassembled folder
    reassembled_folder = os.path.join(args.output_folder, "reassembled")
    os.makedirs(reassembled_folder, exist_ok=True)

    for floor_plan_name in os.listdir(args.input_folder):
        input_path = os.path.join(args.input_folder, floor_plan_name)

        if os.path.isfile(input_path):
            tiled_images_info, original_width, original_height = tile(floor_plan_name, args.input_folder, intermediate_folder, tile_size)

            for i, j, tile_path in tiled_images_info:
                intermediate_path = os.path.join(intermediate_folder, os.path.basename(tile_path))
                binary_path = os.path.join(binary_folder, "binary_" + os.path.basename(tile_path))
                output_image_path = os.path.join(args.output_folder, "processed_" + os.path.basename(tile_path))

                process_tiles.process_imgNEWALG(tile_path, output_image_path, mode='RGB')

                # Check dimensions after processing
                processed_img = Image.open(output_image_path)
                processed_width, processed_height = processed_img.size
                print(f"Processed image {os.path.basename(output_image_path)} dimensions (Width x Height): {processed_width} x {processed_height}")

                binary_image_needed = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
                morphology_img = cv2.morphologyEx(binary_image_needed, cv2.MORPH_OPEN, kernel, iterations=1)

                final_result = cv2.adaptiveThreshold(morphology_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
                cv2.imwrite(binary_path, final_result)

                # Check dimensions after final processing
                final_img = cv2.imread(binary_path)
                final_height, final_width = final_img.shape[:2]
                print(f"Final binary image {os.path.basename(binary_path)} dimensions (Width x Height): {final_width} x {final_height}")

            # Reassemble the final binary images into a single image
            full_binary_image = join_images(binary_folder, original_width, original_height, tile_size)
                
            # Save the reassembled image
            reassembled_path = os.path.join(args.output_folder, "reassembled", f"reassembled_{floor_plan_name}")
            cv2.imwrite(reassembled_path, full_binary_image)
            print(f"Reassembled image saved to: {reassembled_path}")



if __name__ == '__main__':
    main()