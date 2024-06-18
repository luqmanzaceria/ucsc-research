# import matplotlib.pyplot as plt
import keras_ocr
import cv2 as cv
import math
import numpy as np
import os
import sys
import argparse
import re
from PIL import Image
from itertools import product
import process_tiles
import shutil
import random
import json

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

pipeline = keras_ocr.pipeline.Pipeline()

def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path)
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv.inpaint(img, mask, 7, cv.INPAINT_NS)
                 
    return img

# def tile(filename, dir_in, dir_out, d):
#     img_path = os.path.join(dir_in, filename)
#     img = Image.open(img_path)
#     w, h = img.size
#     print(f"Original image dimensions (Width x Height): {w} x {h}")
#     tiled_images_info = []

#     grid = product(range(0, h, d), range(0, w, d))
#     for i, j in grid:
#         box = (j, i, min(j+d, w), min(i+d, h))
#         tile_img = img.crop(box)
#         tile_name = f'{os.path.splitext(filename)[0]}_tile_{i}_{j}.png'
#         tile_path = os.path.join(dir_out, tile_name)
#         tile_img.save(tile_path)

#         # Check dimensions of the tile
#         tile_width, tile_height = tile_img.size
#         print(f"Tile {tile_name} dimensions (Width x Height): {tile_width} x {tile_height}")

#         tiled_images_info.append((i, j, tile_path))

#     return tiled_images_info, w, h

def tile(filename, dir_in, dir_out, d):
    img_path = os.path.join(dir_in, filename)
    img = Image.open(img_path)
    w, h = img.size
    # print(f"Original image dimensions (Width x Height): {w} x {h}")
    tiled_images_info = []

    grid = product(range(0, h, d), range(0, w, d))
    for i, j in grid:
        box = (j, i, min(j+d, w), min(i+d, h))
        tile_img = img.crop(box)
        tile_name = f'{os.path.splitext(os.path.basename(filename))[0]}_tile_{i}_{j}.png'
        tile_path = os.path.join(dir_out, tile_name)
        tile_img.save(tile_path)

        # Check dimensions of the tile
        tile_width, tile_height = tile_img.size
        # print(f"Tile {tile_name} dimensions (Width x Height): {tile_width} x {tile_height}")

        tiled_images_info.append((i, j, tile_path))

    return tiled_images_info, w, h

def resize_tile(tile_img, new_width, new_height):
    # Resize the tile image to the new dimensions
    return cv.resize(tile_img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

def join_images(dir_out, original_width, original_height, tile_size, image_prefix):
    # Create a new blank image with the original dimensions
    full_image = np.zeros((original_height, original_width), dtype=np.uint8)

    # Adjust the regular expression to match the file naming convention
    # This pattern assumes the format 'binary_[image_prefix]_tile_[y]_[x].png'
    tile_pattern = re.compile(rf"binary_{re.escape(image_prefix)}_tile_(\d+)_(\d+).png")

    # Dictionary to hold tile images
    tiles = {}

    # print("=========================")
    # print(f"image_prefix: {image_prefix}")
    # print("=========================")

    # Scan directory for matching files using the updated pattern
    for filename in os.listdir(dir_out):
        match = tile_pattern.match(filename)
        if match:
            y, x = map(int, match.groups())
            tile_path = os.path.join(dir_out, filename)
            tile_img = cv.imread(tile_path, cv.IMREAD_GRAYSCALE)
            tiles[(y, x)] = tile_img

    # Iterate over the expected tile positions
    for y in range(0, original_height, tile_size):
        for x in range(0, original_width, tile_size):
            # Check if the tile image exists in the dictionary
            if (y, x) in tiles:
                # Resize the tile to the new size
                resized_tile_img = resize_tile(tiles[(y, x)], min(tile_size, original_width - x), min(tile_size, original_height - y))
                # Place the resized tile into the full image
                full_image[y:y + min(tile_size, original_height - y), x:x + min(tile_size, original_width - x)] = resized_tile_img

    return full_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    # Get the directory path of the script file
    script_dir = (os.path.dirname(__file__))
    # OCR TEXT REMOVAL

    img_path = sys.argv[1]
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_notext = inpaint_text(img_path, pipeline)

        # Uncomment to show the image with matplotlib
        # plt.imshow(cv.cvtColor(processed_img, cv.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # Generate new file path with "_notext"
        base_path, extension = os.path.splitext(img_path)
        notext_img_path = os.path.join(script_dir, f"{base_path}_notext{extension}")

        # Save the processed image under a new file name
        cv.imwrite(notext_img_path, img_notext)
        # print(f"Processed image saved at: {notext_img_path}")
    else:
        print("The file specified does not exist or is not an image file.")



    # DEEPFLOORPLAN SIMPLIFICATION

    # Dynamically adjust tile size based on the image dimensions
    img_notext = Image.open(notext_img_path)
    w, h = img_notext.size
    tile_size = min(w, h) // 2  # Define the tile size as one-third the minimum dimension

    # Create the intermediate folders relative to the script directory
    intermediate_folder = os.path.join(script_dir, "intermediate_images")
    os.makedirs(intermediate_folder, exist_ok=True)

    binary_folder = os.path.join(script_dir, "binary_images")
    os.makedirs(binary_folder, exist_ok=True)

    processed_folder = os.path.join(script_dir, "processed")
    os.makedirs(processed_folder, exist_ok=True)

    # print(processed_folder)

    # Create the reassembled folder
    # reassembled_folder = os.path.join("reassembled")
    # os.makedirs(reassembled_folder, exist_ok=True)

    # Specify the path to the pretrained model relative to the script directory
    pretrained_model_path = os.path.join(script_dir, "pretrained/")

    tiled_images_info, original_width, original_height = tile(notext_img_path, ".", intermediate_folder, tile_size)
    for i, j, tile_path in tiled_images_info:
        intermediate_path = os.path.join(intermediate_folder, os.path.basename(tile_path))
        binary_path = os.path.join(binary_folder, "binary_" + os.path.basename(tile_path))
        output_image_path = os.path.join(processed_folder, "processed_" + os.path.basename(tile_path))
        # print(output_image_path)
        process_tiles.process_img(tile_path, output_image_path, mode='RGB', pretrained_model_path=pretrained_model_path)
        # Check dimensions after processing
        processed_img = Image.open(output_image_path)
        processed_width, processed_height = processed_img.size
        # print(f"Processed image {os.path.basename(output_image_path)} dimensions (Width x Height): {processed_width} x {processed_height}")
        binary_image_needed = cv.imread(output_image_path, cv.IMREAD_GRAYSCALE)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
        morphology_img = cv.morphologyEx(binary_image_needed, cv.MORPH_OPEN, kernel, iterations=1)
        final_result = cv.adaptiveThreshold(morphology_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
        cv.imwrite(binary_path, final_result)
        # Check dimensions after final processing
        final_img = cv.imread(binary_path)
        final_height, final_width = final_img.shape[:2]
        # print(f"Final binary image {os.path.basename(binary_path)} dimensions (Width x Height): {final_width} x {final_height}")

    # Extract the prefix from the floor plan name
    image_prefix = os.path.splitext(os.path.basename(notext_img_path))[0]
    # Reassemble the final binary images into a single image
    full_binary_image = join_images(binary_folder, original_width, original_height, tile_size, image_prefix)

        
    # Save the reassembled image in the script directory
    reassembled_path = os.path.join(script_dir, f"reassembled_{image_prefix}.png")
    cv.imwrite(reassembled_path, full_binary_image)
    # print(f"Reassembled image saved to: {reassembled_path}")

    # Remove folders and files
    shutil.rmtree(intermediate_folder)
    shutil.rmtree(binary_folder)
    shutil.rmtree(processed_folder)
    os.remove(notext_img_path)
    # print("Done deleting intermediate files!")

    
    # HOUGH TRANSFORM

    filename = reassembled_path
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name] \n')
        exit()

    # Resize the image to make it square
    max_dim = max(src.shape[:2])
    src_square = cv.resize(src, (max_dim, max_dim), interpolation=cv.INTER_LINEAR)

    # Apply morphological opening
    kernel = np.ones((5,5),np.uint8)
    src_square = cv.morphologyEx(src_square, cv.MORPH_OPEN, kernel, iterations=3)

    # Display the result
    # plt.figure(figsize=(10, 10))
    # plt.imshow(src_square)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # Edge detection
    dst = cv.Canny(src_square, 50, 200, None, 3)

    # Convert to BGR image for display
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # Detect lines in the entire image
    lines = cv.HoughLines(dst, 0.1, np.pi / 180, max_dim//10, None, 0, 0)
    all_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            max_len = int(np.sqrt(2) * max_dim)
            pt1 = (int(x0 + max_len*(-b)), int(y0 + max_len*(a)))
            pt2 = (int(x0 - max_len*(-b)), int(y0 - max_len*(a)))
            all_lines.append((pt1, pt2, theta))

    # Non-maximum suppression
    strong_lines = np.zeros([len(all_lines),1,2])
    n2 = 0
    for n1 in range(0, len(all_lines)):
        pt1, pt2, theta = all_lines[n1]
        rho = abs((pt2[1] - pt1[1]) * pt1[0] - (pt2[0] - pt1[0]) * pt1[1]) / np.sqrt((pt2[1] - pt1[1])**2 + (pt2[0] - pt1[0])**2)
        if n1 == 0:
            strong_lines[n2] = [[rho, theta]]
            n2 = n2 + 1
        else:
            if rho < 0:
                rho *= -1
                theta -= np.pi
            closeness_rho = np.isclose(rho, strong_lines[0:n2,0,0], atol=10)  # Decrease atol for rho
            closeness_theta = np.isclose(theta, strong_lines[0:n2,0,1], atol=np.pi/45)  # Decrease atol for theta
            closeness = np.all([closeness_rho, closeness_theta], axis=0)
            if not any(closeness):
                strong_lines[n2] = [[rho, theta]]
                n2 = n2 + 1

    # cdst_copy = cdst.copy()

    # # Draw lines on the image
    # for line in strong_lines[:n2]:
    #     rho, theta = line[0]
    #     a = math.cos(theta)
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     max_len = int(np.sqrt(2) * max_dim)
    #     pt1 = (int(x0 + max_len*(-b)), int(y0 + max_len*(a)))
    #     pt2 = (int(x0 - max_len*(-b)), int(y0 - max_len*(a)))
    #     angle_deg = theta * 180 / np.pi
    #     tolerance = 0.1  # Adjust tolerance as needed
    #     if (abs(angle_deg) <= tolerance) or (abs(abs(angle_deg) - 90) <= tolerance) or (abs(abs(angle_deg) - 180) <= tolerance) or (abs(abs(angle_deg) - 270) <= tolerance):
    #         # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #         color  = (0, 0, 255)
    #         cv.line(cdst, pt1, pt2, color, 7, cv.LINE_AA)

    # Display the result
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cdst)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # # Rescale the image to original size
    # cdst_original = cv.resize(cdst, (src.shape[1], src.shape[0]), interpolation=cv.INTER_LINEAR)

    # # Create a copy of the original-sized image for drawing lines
    # cdst_original_copy = cdst_original.copy()

    # # Draw lines on the original-sized image
    # for line in strong_lines[:n2]:
    #     rho, theta = line[0]
    #     a = math.cos(theta)
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     max_len = int(np.sqrt(2) * max(src.shape[:2]))
    #     pt1 = (int(x0 + max_len*(-b)), int(y0 + max_len*(a)))
    #     pt2 = (int(x0 - max_len*(-b)), int(y0 - max_len*(a)))
    #     angle_deg = theta * 180 / np.pi
    #     tolerance = 0.1  # Adjust tolerance as needed
    #     if (abs(angle_deg) <= tolerance) or (abs(abs(angle_deg) - 90) <= tolerance) or (abs(abs(angle_deg) - 180) <= tolerance) or (abs(abs(angle_deg) - 270) <= tolerance):
    #         color = (0, 0, 255)
    #         cv.line(cdst_original_copy, pt1, pt2, color, 7, cv.LINE_AA)

    # Display the result
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cdst_original_copy)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    os.remove(reassembled_path)

    result = {
    'detectedLines': strong_lines[:n2].tolist(),
    'reassembledPath': reassembled_path
    }

    # Redirecting debug info to stderr
    print("Debug information", file=sys.stderr)

    # Ensuring only JSON is printed to stdout
    print(json.dumps(result))

    return (json.dumps(result))

    

if __name__ == '__main__':
    main()