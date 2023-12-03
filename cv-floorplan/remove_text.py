import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
import os

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
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return img

def main():
    input_folder = 'input'  # Specify your input folder path

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            img_path = os.path.join(input_folder, filename)
            processed_img = inpaint_text(img_path, pipeline)

            # plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()

            # Overwrite the original file with the processed image
            cv2.imwrite(img_path, processed_img)
            print(f"Processed image overwritten at: {img_path}")

if __name__ == '__main__':
    main()
