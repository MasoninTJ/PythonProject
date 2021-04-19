import cv2 as cv
import numpy as np

# 图像加法
x = np.uint8([250])
y = np.uint8([10])

# OpenCV加法和Numpy加法之间有区别
print(cv.add(x, y))
print(x + y)

"""
推荐使用cv.add()
"""

# 图像融合
# y = ax + (1-a)x

img1 = cv.imread('image/pic_0005.jpg')
img1 = cv.resize(img1, (500, 300))

img2 = cv.imread('image/pic_0007.jpg')
img2 = cv.resize(img2, (500, 300))

dst = cv.addWeighted(img1, 0.3, img2, 0.7, 0)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 按位运算

"""
cv.bitwise_not
cv.bitwise_and
cv.bitwise_or
cv.bitwise_xor
"""
# 可以用来提取图像
