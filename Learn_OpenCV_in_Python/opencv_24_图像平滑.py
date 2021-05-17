import cv2
import numpy as np

img = cv2.imread('image/pic_0012.jpg', cv2.IMREAD_GRAYSCALE)

# 这里所用的内核为5*5的平均
kernel = np.ones((5, 5), np.float32) / 25
dst_1 = cv2.filter2D(img, -1, kernel)  # 均值滤波，只是使用2D卷积的函数来进行操作
dst_2 = cv2.blur(img, (5, 5))  # 均值滤波
dst_3 = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
dst_4 = cv2.medianBlur(img, 5)  # 中值滤波
dst_5 = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波

# 该函数用于数组的拼接，0表示纵向，1表示横向
# hstack和vstack
result = np.concatenate((img, dst_1, dst_2, dst_3, dst_4, dst_5), axis=1)
cv2.imshow('图像平滑对比', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
