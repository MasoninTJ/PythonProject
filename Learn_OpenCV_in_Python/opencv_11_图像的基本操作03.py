import cv2 as cv

"""
图片的ROI，类似于numpy的切片处理
"""
img = cv.imread('image/pic_0005.jpg')

roi = img[50:150, 150:250]
print(roi.shape)
cv.imshow('roi', roi)
cv.waitKey(0)
cv.destroyAllWindows()
