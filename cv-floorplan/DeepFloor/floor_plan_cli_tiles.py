import argparse
import os
from processors.fp_model_initializer import init_model
# from processors.fp_compartment_counter import get_floor_plan_compartments
from processors.fp_model_prediction_post_processing import process_img
from processors.fp_in_mem_floor_plans import InMemFloorPlans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from itertools import product


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
    # Create a new blank image with double the original dimensions
    full_image = np.zeros((original_height * 2, original_width * 2), dtype=np.uint8)

    # Iterate over the tiles based on their expected positions
    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):
            # Calculate new tile size considering the edge cases
            new_tile_width = min(tile_size, original_width - x) * 2
            new_tile_height = min(tile_size, original_height - y) * 2

            # Build the filename based on expected naming convention
            tile_name = f"binary_e2_1x1_1_tile_{y}_{x}.png"
            tile_path = os.path.join(dir_out, tile_name)

            # Check if the tile image exists
            if os.path.exists(tile_path):
                # Read the tile image
                tile_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)

                # Resize the tile to double its size, or to the remaining space if it's on the edge
                resized_tile_img = resize_tile(tile_img, new_tile_width, new_tile_height)
                # Place the resized tile into the full image
                full_image[y * 2:y * 2 + new_tile_height, x * 2:x * 2 + new_tile_width] = resized_tile_img
            else:
                print(f"Warning: Tile image not found at {tile_path}")
                # Optionally fill the missing tile space with zeros (black) or any other placeholder

    return full_image


def main():
    parser = argparse.ArgumentParser(description='Process floor plans and generate output images.')
    parser.add_argument('input_folder', help='Path to the folder containing floor plan images.')
    parser.add_argument('output_folder', help='Path to the folder where output images will be saved.')
    args = parser.parse_args()

    path_weights = "/Users/luq/dev/projects/ucsc-research/cv-floorplan/DeepFloor/deepfloor/pretrained/G"
    model = init_model(path_weights=path_weights)

    tile_size = 256

    intermediate_folder = os.path.join(args.input_folder, "intermediate_images")
    os.makedirs(intermediate_folder, exist_ok=True)

    binary_folder = os.path.join(args.output_folder, "binary_images")
    os.makedirs(binary_folder, exist_ok=True)

    # Create the reassembled folder
    reassembled_folder = os.path.join(args.output_folder, "reassembled")
    os.makedirs(reassembled_folder, exist_ok=True)

    floor_plan_in_mem_db = InMemFloorPlans()
    floor_plan_in_mem_db.overwrite_floor_plans(os.listdir(args.input_folder))

    for floor_plan_name in floor_plan_in_mem_db.get_floor_plans():
        input_path = os.path.join(args.input_folder, floor_plan_name)

        if os.path.isfile(input_path):
            tiled_images_info, original_width, original_height = tile(floor_plan_name, args.input_folder, intermediate_folder, tile_size)

            for i, j, tile_path in tiled_images_info:
                intermediate_path = os.path.join(intermediate_folder, os.path.basename(tile_path))
                binary_path = os.path.join(binary_folder, "binary_" + os.path.basename(tile_path))
                output_image_path = os.path.join(args.output_folder, "processed_" + os.path.basename(tile_path))

                processed_image = process_img(tile_path, intermediate_path, model)

                binary_array = np.where(processed_image == 10, 1, 0)
                plt.imsave(output_image_path, binary_array, cmap='gray')

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

                # Reassembly of tiles into the full image can be coded here if needed

            # Reassemble the final binary images into a single image
            full_binary_image = join_images(binary_folder, original_width, original_height, tile_size)
                
            # Save the reassembled image
            reassembled_path = os.path.join(args.output_folder, "reassembled", f"reassembled_{floor_plan_name}")
            cv2.imwrite(reassembled_path, full_binary_image)
            print(f"Reassembled image saved to: {reassembled_path}")



if __name__ == '__main__':
    main()
