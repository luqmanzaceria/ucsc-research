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
    return cv2.resize(tile_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def join_images(dir_out, original_width, original_height, tile_size, image_prefix):
    full_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    tile_pattern = re.compile(rf"processed_{re.escape(image_prefix)}_tile_(\d+)_(\d+).png")
    tiles = {}

    for filename in os.listdir(dir_out):
        match = tile_pattern.match(filename)
        if match:
            y, x = map(int, match.groups())
            tile_path = os.path.join(dir_out, filename)
            tile_img = cv2.imread(tile_path)
            tiles[(y, x)] = tile_img

    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):
            if (y, x) in tiles:
                resized_tile_img = resize_tile(tiles[(y, x)], min(tile_size, original_width - x), min(tile_size, original_height - y))
                full_image[y:y + min(tile_size, original_height - y), x:x + min(tile_size, original_width - x)] = resized_tile_img

    return full_image

def main():
    parser = argparse.ArgumentParser(description='Process floor plans and generate output images.')
    parser.add_argument('input_folder', help='Path to the folder containing floor plan images.')
    parser.add_argument('output_folder', help='Path to the folder where output images will be saved.')
    args = parser.parse_args()

    tile_size = 256

    intermediate_folder = os.path.join(args.input_folder, "intermediate_images")
    os.makedirs(intermediate_folder, exist_ok=True)

    processed_folder = os.path.join(args.output_folder, "processed_images")
    os.makedirs(processed_folder, exist_ok=True)

    reassembled_folder = os.path.join(args.output_folder, "reassembled")
    os.makedirs(reassembled_folder, exist_ok=True)

    for floor_plan_name in os.listdir(args.input_folder):
        input_path = os.path.join(args.input_folder, floor_plan_name)

        if os.path.isfile(input_path):
            tiled_images_info, original_width, original_height = tile(floor_plan_name, args.input_folder, intermediate_folder, tile_size)

            for i, j, tile_path in tiled_images_info:
                output_image_path = os.path.join(processed_folder, "processed_" + os.path.basename(tile_path))
                process_tiles.process_img(tile_path, output_image_path, mode='RGB')
                processed_img = Image.open(output_image_path)
                processed_width, processed_height = processed_img.size
                print(f"Processed image {os.path.basename(output_image_path)} dimensions (Width x Height): {processed_width} x {processed_height}")

            image_prefix = os.path.splitext(floor_plan_name)[0]
            full_image = join_images(processed_folder, original_width, original_height, tile_size, image_prefix)
                
            reassembled_path = os.path.join(reassembled_folder, f"reassembled_{floor_plan_name}")
            cv2.imwrite(reassembled_path, full_image)
            print(f"Reassembled image saved to: {reassembled_path}")

if __name__ == '__main__':
    main()
