import cv2
from matplotlib import pyplot as plt

template = cv2.imread('template_adjusted.jpg', 0)
target = cv2.imread('multi_target.jpg', 0)
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(template, None)#计算template中的特征点和描述符
kp2,des2 = orb.detectAndCompute(target, None) #计算target中的
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #建立匹配关系
mathces = bf.match(des1, des2) #匹配描述符
mathces = sorted(mathces, key = lambda x:x.distance) #据距离来排序
result = cv2.drawMatches(template, kp1, target, kp2, mathces[:40], None, flags=2) #画出匹配关系
plt.imshow(result), plt.show() #matplotlib描绘出来