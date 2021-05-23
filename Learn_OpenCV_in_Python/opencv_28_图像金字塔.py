import cv2
import numpy as np

img = cv2.imread('image/pic_0012.jpg', cv2.IMREAD_GRAYSCALE)

lower_reso = cv2.pyrDown(img)
height_reso = cv2.pyrUp(img)

cv2.imshow('img', img)
cv2.imshow('lower_reso', lower_reso)
cv2.imshow('height_reso', height_reso)
cv2.waitKey(0)
cv2.destroyAllWindows()