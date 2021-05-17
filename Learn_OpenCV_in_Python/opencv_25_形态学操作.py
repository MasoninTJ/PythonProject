import cv2
import numpy as np

img = cv2.imread('image/pic_0015.jpg', cv2.IMREAD_GRAYSCALE)
complement = np.ones(img.shape[::-1], np.uint8) * 255  # tips,这里不指定uint8会显示异常
white_column_split = np.ones((img.shape[0], 1), np.uint8) * 255  # 1列的白色分隔符

kernel = np.ones((5, 5), np.uint8)

# 侵蚀和扩张
erosion = cv2.erode(img, kernel, iterations=1)  # 侵蚀
dilation = cv2.dilate(img, kernel, iterations=1)  # 扩张
result_1 = np.concatenate((img, white_column_split, erosion, white_column_split, dilation), axis=1)

white_row_split = np.ones((1, result_1.shape[1]), np.uint8) * 255  # 1行的白色分隔符

# 开运算 = 先侵蚀后扩张
# 闭运算 = 先扩张再侵蚀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
result_2 = np.concatenate((img, white_column_split, opening, white_column_split, closing), axis=1)

# 形态学梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 顶帽 = 输入图像和图像开运算之差
# 黑帽 = 输入图像和图像闭运算之差
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

result_3 = np.concatenate((gradient, white_column_split, tophat, white_column_split, blackhat), axis=1)

result = np.concatenate((result_1, white_row_split, result_2, white_row_split, result_3), axis=0)

cv2.imshow('形态学操作', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
