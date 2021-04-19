import cv2
import numpy as np

# 绘制线
img = np.zeros((512, 512, 3), np.uint8)
red = (0, 0, 255)  # BGR
green = (0, 255, 0)
blue = (255, 0, 0)
cv2.line(img, (0, 0), (511, 511), red, 5)
cv2.rectangle(img, (50, 50), (451, 451), green, 10)
cv2.circle(img, (255, 255), 50, blue, -1)

"""
cv2.ellipse()   绘制椭圆
cv2.polylines() 绘制多边形
"""
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (75, 255), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('image', img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
