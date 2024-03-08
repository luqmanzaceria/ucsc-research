import cv2
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# Load the image in, convert to black and white
image = cv2.imread("reassembled_E2_2.png",0)
_,image = cv2.threshold(image,250,255,cv2.THRESH_BINARY_INV)
plt.subplot(2,2,1)
plt.imshow(image,cmap="gray")
plt.title("Back and White Source Image")


image_to_compute = image # only do this to make experimenting with commenting out parts later

# Get the derivatives
# For some reason taking the x and y at the same time did not work right
# This may not be necassary in your case, if there is a lot of noise in the image
#   it may help
# Comment out the sobel operations and the morphology and see what happens
# Sobel of image
img_sobel_x = cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0,ksize=3)
img_sobel_y = cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1,ksize=3)
img_sobel = np.abs(img_sobel_x) + np.abs(img_sobel_y)
_, image_to_compute = cv2.threshold(img_sobel, .001,1, cv2.THRESH_BINARY)
plt.subplot(2,2,2)
plt.imshow(image_to_compute,cmap="gray")
plt.title("Thresholded Sobel of Input")
# Morphology on image
# again, this part may not be needed, but will help with noisy images
kernel = np.ones((9,9),np.uint8)
closing = cv2.morphologyEx(image_to_compute, cv2.MORPH_CLOSE, kernel)
image_to_compute = (closing*255).astype(np.uint8)
plt.subplot(2,2,3)
plt.imshow(image_to_compute,cmap="gray")
plt.title("Morphlogy on Image")

# Get the lines as 1 pixel wide using skeletonization
# Making all lines skinny makes the compute go faster
def skeletonize(img):
    '''
    https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
    Get all the lines as 1 pixel wide
    '''
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skel = np.zeros(img.shape, np.uint8)
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, opening)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel

skel = skeletonize(image_to_compute)
plt.subplot(2,2,4)
plt.imshow(skel,cmap="gray")
plt.title("Skeletonized Image")


plt.show()


# Now get the indicies where the pixels have a line in them
y,x = np.where(skel>0)
locs = np.array((x,y),dtype=np.float32).T
print("Have {} pixels to find lines through".format(locs.shape))


# Use a RANSAC algorithm to compute the lines
# This will randomly sample points, compute the line through them
#    then find how many pixels are within the max_devaition parameter
# The more points within the max_deviation parameter the better the line fits
def RANSAC_LINEST(points,max_deviation=2.,n_points=3):
    '''
    Approximate a line from the data
    Returns the (slope,y_intercept) of the line that has the most points within max_deviation of the line
    '''

    mask_probabilities = ~points[:,0].mask/points[:,0].count()
    best_coeffs = [0,0]
    best_count = 0
    best_idxs = []
    for it in range(100):
        # choose random points, and compute the line of best fit for them
        idxs = np.random.choice(points.shape[0],size=n_points,p=mask_probabilities,replace=False)
        pts = points[idxs,:]
        poly = np.poly1d(np.polyfit(pts[:,0],pts[:,1],deg=1))
        
        # compute the resulting points and find matches
        computed_ys = poly(points[:,0])
        deltas = np.abs(points[:,1] - computed_ys)
        good_idxs = np.where(deltas < max_deviation)[0]
        if len(good_idxs) > best_count:
            best_count = len(good_idxs)
            best_coeffs = poly.coefficients
            best_idxs = good_idxs
    return best_coeffs,best_idxs

# Now go through and find lines of best fit using RANSAC
# After each line has been found, mask off the points that 
#    were used to create it
line_coeffs = []
completed_points_mask = np.zeros_like(locs,dtype=np.uint8)
locs_masked = ma.masked_array(locs,mask=completed_points_mask)

num_lines_to_find = 5
n_cols = 2
n_rows = 3

for line_idx in range(num_lines_to_find):
    coeffs,idxs = RANSAC_LINEST(locs_masked,max_deviation=5,n_points=2)
    line_coeffs.append(coeffs)
    
    completed_points_mask[idxs,:] = 1
    locs_masked.mask = completed_points_mask

    x_lim = (locs[idxs,0].min(),locs[idxs,0].max())
    xs = np.arange(x_lim[0],x_lim[1],1)
    ys = xs*coeffs[0]+coeffs[1]
    plt.subplot(n_rows,n_cols,line_idx+1)
    plt.title("Estimate for line {}".format(line_idx))
    plt.imshow(image)
    plt.plot(xs,ys)
plt.show()