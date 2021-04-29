import cv2 as cv
import numpy as np

img_rgb = cv.imread('image/mario.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('image/mario_coin.png', 0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.98
loc = np.where(res >= threshold)
print(len(list(zip(*loc[::-1]))))
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
cv.imshow('res.png', img_rgb)

cv.waitKey(0)
cv.destroyAllWindows()