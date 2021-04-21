import cv2 as cv
import numpy as np

"""
平移旋转缩放，透视，仿射
"""
img = cv.imread('image/pic_0009.jpg')
cv.imshow('image', img)
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('img1', res)
# 或者
height, width = img.shape[:2]
res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
cv.imshow('img2', res)

src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rows, cols = src.shape
M = np.float32([[1, 0, 100], [0, 1, 50]])  # (1,0)方向平移100，(0,1)方向平移50
dst = cv.warpAffine(src, M, (cols, rows))
cv.imshow('dst', dst)

M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
dst = cv.warpAffine(img, M, (cols, rows))
cv.imshow('dst2', dst)

cv.waitKey(0)
cv.destroyAllWindows()
