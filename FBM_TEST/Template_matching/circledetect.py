import cv2
import numpy as np
import math

pieces = cv2.imread('pieces_2_zip.jpeg')
gray_img = cv2.cvtColor(pieces, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

w, h, c = cimg.shape
maxR = math.ceil(min(w, h) / 16)
minR = math.floor(max(w, h) / 30)
minD = math.floor(minR)

# method cv2.HOUGH_GRADIENT 也就是霍夫圆检测，梯度法
# dp 计数器的分辨率图像像素分辨率与参数空间分辨率的比值（官方文档上写的是图像分辨率与累加器分辨率的比值，它把参数空间认为是一个累加器，毕竟里面存储的都是经过的像素点的数量），dp=1，则参数空间与图像像素空间（分辨率）一样大，dp=2，参数空间的分辨率只有像素空间的一半大
# minDist 圆心之间最小距离，如果距离太小，会产生很多相交的圆，如果距离太大，则会漏掉正确的圆
# param1 canny检测的双阈值中的高阈值，低阈值是它的一半
# param2 最小投票数（基于圆心的投票数）
# minRadius 需要检测院的最小半径
# maxRadius 需要检测院的最大半径
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minD, param1=130, param2=21, minRadius=minR, maxRadius=maxR)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(pieces, (i[0], i[1]), i[2], (0,255,0), 2)
    cv2.circle(pieces, (i[0], i[1]), 2, (0,0,255), 3)

cv2.imshow('HoughCircles', pieces)
cv2.waitKey()
cv2.destroyAllWindows()