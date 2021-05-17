import cv2
import numpy as np

img = cv2.imread('image/pic_0015.jpg', cv2.IMREAD_GRAYSCALE)
complement = np.ones(img.shape[::-1], np.uint8) * 255  # tips,这里不指定uint8会显示异常
white_column_split = np.ones((img.shape[0], 1), np.uint8) * 255  # 1列的白色分隔符

laplacian = cv2.Laplacian(img, cv2.CV_64F)
print(laplacian.dtype)
laplacian = cv2.convertScaleAbs(laplacian) # 将图像转换到np.uint8
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobely = cv2.convertScaleAbs(sobely)

result = np.concatenate((img, white_column_split, laplacian, white_column_split, sobelx, white_column_split, sobely), axis=1)
cv2.imshow('gradient', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
