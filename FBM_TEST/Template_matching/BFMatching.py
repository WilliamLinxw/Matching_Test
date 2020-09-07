import numpy as np 
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('redbing_zip.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('pieces_zip.jpeg', cv2.IMREAD_GRAYSCALE)

# create SIFT and detect/compute
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
matches = bf.match(des1, des2)
# matches = bf.knnMatch(des1, des2, k = 2)

matches = sorted(matches, key = lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:25], img2, flags = 2)

plt.imshow(img3), plt.show()