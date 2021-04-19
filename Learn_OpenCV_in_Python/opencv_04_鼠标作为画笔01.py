import cv2 as cv
import numpy as np

"""
cv2.setMouseCallback()
"""


def list_of_event():
    """
    列出所有可用的活动
    @return:
    """
    events = [i for i in dir(cv) if 'EVENT' in i]
    for i in events:
        print(i)


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:  # 鼠标左键双击
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


# 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while True:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
