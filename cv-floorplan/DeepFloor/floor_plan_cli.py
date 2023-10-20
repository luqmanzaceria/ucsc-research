# process_floor_plans.py
import argparse
import os
from processors.fp_model_initializer import init_model
from processors.fp_in_mem_floor_plans import InMemFloorPlans
# from processors.fp_compartment_counter import get_floor_plan_compartments
from processors.fp_model_prediction_post_processing import process_img
import matplotlib.pyplot as plt

import numpy as np

import cv2

def main():

    parser = argparse.ArgumentParser(
        description='Process floor plans and generate output images.')
    parser.add_argument(
        'input_folder', help='Path to the folder containing floor plan images.')
    parser.add_argument(
        'output_folder', help='Path to the folder where output images will be saved.')
    args = parser.parse_args()

    # Initialize the model
    path_weights = "/Users/user/Documents/SIP_Tests/deep-floor-plan-recognition/deepfloor/pretrained/G"
    model = init_model(path_weights=path_weights)

    intermediate_folder = os.path.join(args.input_folder,"intermediate_images")

    binary_folder = os.path.join(args.output_folder,"binary_images")



    # Load floor plans from the input folder
    floor_plan_in_mem_db = InMemFloorPlans()
    
    floor_plan_in_mem_db.overwrite_floor_plans(os.listdir(args.input_folder))

    # Process each floor plan
    for floor_plan_name in floor_plan_in_mem_db.get_floor_plans():

        input_path = os.path.join(args.input_folder, floor_plan_name)

        if os.path.isfile(input_path):

            intermediate_path = os.path.join(intermediate_folder,floor_plan_name)

            binary_path = os.path.join(binary_folder,"binary_" + floor_plan_name)

            output_image_path = os.path.join(
                args.output_folder, "processed_" + floor_plan_name)

            print("************************\n" + input_path)
            #output_path = os.path.join(args.output_folder, floor_plan_name)

            # FIX ISSUE OF USING SAME INPUT IMAGE
            # Use cv2 to show this processed image 
            # Try to change rgb to bgr 
            # Print out the unique values in the image (numpy array)
            # filter out irrelevant like if 1 = window etc. to get only walls in binary image
        
            processed_image = process_img(input_path,intermediate_path, model)

            condition = processed_image == 10

            binary_array = np.where(condition,1,0)

            plt.imsave(os.path.join(output_image_path),binary_array)

            binary_image_needed = cv2.imread(
                os.path.join(output_image_path))
            
            bgr_class_image = cv2.cvtColor(binary_image_needed, cv2.COLOR_RGB2BGR)
            gray_class_image = cv2.cvtColor(bgr_class_image, cv2.COLOR_BGR2GRAY)
            
            #1
            # denoised_image = cv2.medianBlur(gray_class_image, 5)
            #2 - 2nd best
            # smoothed_image = cv2.GaussianBlur(gray_class_image, (5, 5), 0)
            #3
            # kernel = np.ones((5, 5), np.uint8)
            # eroded_image = cv2.erode(gray_class_image, kernel, iterations=1)
            # dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
            
            #4 - best so far
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            morphology_img = cv2.morphologyEx(
                gray_class_image, cv2.MORPH_OPEN, kernel, iterations=1)

            #5
            # blur = cv2.GaussianBlur(gray_class_image, (0, 0), sigmaX=33, sigmaY=33)
            # divide = cv2.divide(gray_class_image, blur, scale=255)
            # thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            final_result = cv2.adaptiveThreshold(
                morphology_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
            
            cv2.imwrite(binary_path,final_result)


            # # Get compartment counts
            # # compartments = get_floor_plan_compartments(processed_image)


            


if __name__ == '__main__':
    main()
