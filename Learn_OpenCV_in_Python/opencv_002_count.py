import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((7, 7), np.uint8)

img = cv2.imread('image/pic_0003.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow('原始图片', img)
h, w = img.shape

ret, img_bin = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
img_bin = cv2.resize(img_bin, (w // 4, h // 4))
cv2.imshow('二值化图片', img_bin)

erosion = cv2.erode(img_bin, kernel, iterations=1)  #
cv2.imshow('erosion', erosion)

dist_img = cv2.distanceTransform(erosion, cv2.DIST_L1, cv2.DIST_MASK_3)  # 距离变换
cv2.imshow('距离变换', dist_img)
dist_output = cv2.normalize(dist_img, 0, 1.0, cv2.NORM_MINMAX)  # 归一化
cv2.imshow('dist_output', dist_output * 80)

ret1, th2 = cv2.threshold(dist_output * 80, 0.3, 255, cv2.THRESH_BINARY)
cv2.imshow('th2', th2)

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
opening = np.array(opening, np.uint8)
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓提取
count = 0
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    circle_img = cv2.circle(opening, center, radius, (255, 255, 255), 1)
    area = cv2.contourArea(cnt)
    area_circle = 3.14 * radius * radius
    # print(area/area_circle)
    if area / area_circle <= 0.5:
        # img = cv2.drawContours(img, cnt, -1, (0,0,255), 5)#差（红色）
        img = cv2.putText(img, 'bad', center, font, 0.5, (0, 0, 255))
    elif area / area_circle >= 0.6:
        # img = cv2.drawContours(img, cnt, -1, (0,255,0), 5)#优（绿色）
        img = cv2.putText(img, 'good', center, font, 0.5, (0, 0, 255))
    else:
        # img = cv2.drawContours(img, cnt, -1, (255,0,0), 5)#良（蓝色）
        img = cv2.putText(img, 'normal', center, font, 0.5, (0, 0, 255))
    count += 1
img = cv2.putText(img, ('sum=' + str(count)), (50, 50), font, 1, (255, 0, 0))
cv2.imshow('circle_img', img)
print('玉米粒共有：', count)

cv2.waitKey(0)
cv2.destroyAllWindows()
