import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('redbing_zip.png', 0)
img2 = cv2.imread('pieces_zip.jpeg', 0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# surf = cv2.xfeatures2d.SURF_create(500)
# kp1, des1 = surf.detectAndCompute(img1, None)
# kp2, des2 = surf.detectAndCompute(img2, None)

# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_KETREE = 0
index_params = dict(algorithm = FLANN_INDEX_KETREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k = 2)

# store all the good matches as per Lowe's ratio test
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],[w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print('Not enough matches are found - %d / %d' %(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = matchesMask, flags = 2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray')
plt.show()

# #基于FLANN的匹配器(FLANN based Matcher)定位图片
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
 
# MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
# template = cv2.imread('template_adjusted.jpg',0) # queryImage
# target = cv2.imread('target.jpg',0) # trainImage
# # Initiate SIFT detector创建sift检测器
# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(template,None)
# kp2, des2 = sift.detectAndCompute(target,None)
# #创建设置FLANN匹配
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# #舍弃大于0.7的匹配
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
# if len(good)>MIN_MATCH_COUNT:
#     # 获取关键点的坐标
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     #计算变换矩阵和MASK
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = template.shape
#     # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     target = cv2.polylines(target,[np.int32(dst)],True, 255, 3, cv2.LINE_AA)
# else:
#     print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#     matchesMask = None
# draw_params = dict(matchColor=(0,255,0), 
#                    singlePointColor=None,
#                    matchesMask=matchesMask, 
#                    flags=2)
# result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
# plt.imshow(result, 'gray')
# plt.show()