import cv2

img_rgb = cv2.imread('image/pic_0012.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
cv2.drawContours(img_rgb, contours, -1, (0,255,0), 1)
cv2.imshow('img', img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
