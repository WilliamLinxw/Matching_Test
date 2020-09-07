import time
import cv2

time1 = time.time()

# image = cv2.imread('Template_target/pieces.jpeg')
image = cv2.imread('pieces_2.jpeg')

res = cv2.resize(image, (756, 1008), interpolation=cv2.INTER_AREA)

# cv2.imwrite('Template_target/pieces_zip.jpeg', res)
cv2.imwrite('pieces_2_zip.jpeg', res)

time2 = time.time()
print('Time consumed: ', time2 - time1)