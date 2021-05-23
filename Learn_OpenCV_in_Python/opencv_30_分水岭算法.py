import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def imshow(name, img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.title(name, fontsize='xx-large')
    plt.imshow(temp)


img_rgb = cv2.imread('count_1.bmp', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # 右边有点小瑕疵，去掉
ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
erode = cv2.erode(img_bin, kernel, iterations=4)

dist_img = cv2.distanceTransform(erode,  cv2.DIST_L2, 5)
dist_img = np.uint8(dist_img)

ret, sure_fg = cv2.threshold(dist_img, 0.7* dist_img.max(), 255, cv2.THRESH_BINARY)
sure_bg = cv2.dilate(erode, kernel, iterations=4)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers1 = cv2.connectedComponents(sure_fg)
markers = markers1 + 1
markers[unknown == 255] = 0

markers3 = cv2.watershed(img_rgb, markers)

img_rgb[markers3 == -1] = [0, 255, 255]
img_rgb[dist_img > 60] = [0,255,0]
cv2.imwrite('result.png', img_rgb)
plt.imshow(dist_img, cmap='gray')
plt.show()


