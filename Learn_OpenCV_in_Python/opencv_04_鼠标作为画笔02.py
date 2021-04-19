import numpy as np
import cv2 as cv

drawing = False  # 如果按下鼠标，则为真
ix, iy = -1, -1


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)


# 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while True:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
