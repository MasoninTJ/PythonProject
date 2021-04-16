import cv2

# 读灰度图
image = cv2.imread(r'image/pic_0001.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', image)
cv2.waitKey(0)

# 读彩色图
image2 = cv2.imread('image/pic_0002.jpg', cv2.IMREAD_COLOR)
cv2.imshow('image', image2)

cv2.waitKey(0)
cv2.destroyAllWindows()


