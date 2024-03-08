import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from itertools import combinations

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None  # No intersection

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


img_bgr = cv2.imread('originaldfp_E2_3.png',1)
img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

_,img = cv2.threshold(img,10,1,cv2.THRESH_BINARY_INV)

skeleton = skeletonize(img,method='lee')
skeleton_bgr = cv2.cvtColor(skeleton.astype(np.uint8),cv2.COLOR_GRAY2BGR)


# fig,ax = plt.subplots(1)
# ax.imshow(skeleton, cmap = 'gray')


#get tips
img_conv = cv2.filter2D(skeleton.astype(np.uint8),-1,np.ones((3,3))) #
img_conv = img_conv*skeleton
img_tips = img_conv == 2
tips = np.array(np.nonzero(img_tips)).T



# fig,ax = plt.subplots(1)
# ax.imshow(skeleton, cmap = 'gray')
# ax.scatter(tips[0,:],tips[1,:],s=4,color='r')


lines = cv2.HoughLines(skeleton.astype(np.uint8), 1, np.pi / 180, 150, None, 0, 0)

lines_points = []

# # Draw the lines
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(skeleton_bgr, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    lines_points.append([pt1,pt2])


line_combinations = combinations(lines_points, 2)


fig,ax = plt.subplots(1)
ax.imshow(img, cmap = 'gray')
ax.scatter(tips[0,:],tips[1,:],s=4,color='r')

for line_combination in list(line_combinations):
    line1 = line_combination[0]
    line2 = line_combination[1]

    intersection = line_intersection(line1, line2)

    if intersection is not None and intersection[0] < img.shape[1] and intersection[1] < img.shape[0]:
        ax.scatter(intersection[0], intersection[1], s=4, color='b')


# fig,ax = plt.subplots(1)
# ax.imshow(skeleton_bgr,'gray')

plt.show()