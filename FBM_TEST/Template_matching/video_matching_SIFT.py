import cv2
import numpy as np
import time

# read the template chessboard
image = cv2.imread('Template_target/chessboard_template.jpg', 0)

# read the video
def readVideo(videopath, image):
    cap = cv2.VideoCapture('Template_target/chessboard_video.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        # print('frame:',frame.shape)
        if ret == True:
            start = time.time()
            found = matchSift(image, frame)
            end = time.time()
            print(end - start)
            width, height = found.shape[:2]
            size = (int(height / 2), int(width / 2))
            found = cv2.resize(found, size, interpolation = cv2.INTER_AREA)
            cv2.imshow('found', found)
            print('------------')
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()

def matchSift(findimg, img):
    gray1 = cv2.cvtColor(findimg, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN matcher parameters
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(des1, des2, k = 2)

    # prepare an empty mask to draw good matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    # David G. Lowe's ratio test, populate the mask
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    
    drawParams = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), matchesMask = matchesMask, flags = 0)

    resultImage = cv2.drawMatchesKnn(findimg, kp1, img, kp2, matches, None, **drawParams)

    return resultImage

# def matchSift(findimg, img):
#     """转换成灰度图片"""
#     gray1 = cv2.cvtColor(findimg, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     """创建SIFT对象"""
#     sift = cv2.xfeatures2d.SIFT_create()
#     """创建FLAN匹配器"""
#     matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
#     """检测关键点并计算键值描述符"""
#     kpts1, descs1 = sift.detectAndCompute(gray1, None)
#     kpts2, descs2 = sift.detectAndCompute(gray2, None)
#     """KnnMatt获得Top2"""
#     matches = matcher.knnMatch(descs1, descs2, 2)
#     """根据他们的距离排序"""
#     matches = sorted(matches, key=lambda x: x[0].distance)
#     """比率测试，以获得良好的匹配"""
#     good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
#     canvas = img.copy()
#     """发现单应矩阵"""
#     """当有足够的健壮匹配点对（至少个MIN_MATCH_COUNT）时"""
#     if len(good) >= MIN_MATCH_COUNT:
#         """从匹配中提取出对应点对"""
#         """小对象的查询索引，场景的训练索引"""
#         src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#         """利用匹配点找到CV2.RANSAC中的单应矩阵"""
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         """计算图1的畸变，也就是在图2中的对应的位置"""
#         h, w = findimg.shape[:2]
#         pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, M)
#         """绘制边框"""
#         cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
#     else:
#         print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#         return img
#     return canvas

if __name__ == "__main__":
    videopath = 'Template_target/chessboard_video.mp4'
    imagepath = 'Template_target/chessboard_template.jpg'
    # MIN_MATCH_COUNT = 10
    image = cv2.imread(imagepath)
    readVideo(videopath, image)