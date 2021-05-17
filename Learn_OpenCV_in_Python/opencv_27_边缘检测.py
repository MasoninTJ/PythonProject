import cv2
import numpy as np

img = cv2.imread('image/pic_0015.jpg', cv2.IMREAD_GRAYSCALE)
complement = np.ones(img.shape[::-1], np.uint8) * 255  # tips,这里不指定uint8会显示异常
white_column_split = np.ones((img.shape[0], 1), np.uint8) * 255  # 1列的白色分隔符

edges = cv2.Canny(img, 100, 200)
result = np.concatenate((img, white_column_split, edges), axis=1)
cv2.imshow('gradient', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
