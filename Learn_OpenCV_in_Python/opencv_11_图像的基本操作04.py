import cv2 as cv

"""
拆分和合并图像通道
"""

img = cv.imread('image/pic_0008.jpg')

b, g, r = cv.split(img)
img = cv.merge((b, g, r))

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

"""
cv.split()是一项耗时的操作（就时间而言）。因此，仅在必要时才这样做。否则请进行Numpy索引。
"""
