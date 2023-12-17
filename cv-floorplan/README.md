# Computer Vision Wall Extraction From Floorplans

## Original Image (Example):
![Original Unprocessed Floorplan Image](https://github.com/luqmanzaceria/ucsc-research/blob/main/cv-floorplan/E2_3.png)

## DeepFloor
Extracts structured features from floorplan images by processing with an image recognition model. Had issues with blotches in the output image. Derived from this [repository](https://github.com/whchien/deep-floor-plan-recognition).
![DeepFloor Output Image](https://github.com/luqmanzaceria/ucsc-research/blob/main/cv-floorplan/DeepFloor_E2_3.png)

## original_dfp
The above DeepFloor's image recognition model originates from the code in this [repository](https://github.com/zlzeng/DeepFloorplan). Utilizing this code was a successful approach because the output images lacked blotches.
![Original_dfp Output Image](https://github.com/luqmanzaceria/ucsc-research/blob/main/cv-floorplan/originaldfp_E2_3.png)

## remove_text.py
A script that uses Keras OCR and Computer Vision to remove text from floorplans for initial processing.

## RBG Research
A low-level image processing approach to wall extraction
![RBG Research Output Image](https://github.com/luqmanzaceria/ucsc-research/blob/main/cv-floorplan/RBG_E2_3.png)

## SVG to CSV
A script to convert SVG to CSV of coordinates and another script to visualize this CSV of coordinates with Matplotlib.
