import cv2 as cv
import numpy as np

"""
颜色空间转换
比较常用的是BGR-GRAY-HSV
HSV的色相范围为[0,179]，饱和度范围为[0,255]，值范围为[0,255]
"""


def list_of_color():
    """
    列出所有可用的空间
    @return:
    """
    flag = [i for i in dir(cv) if i.startswith('COLOR_')]
    for i in flag:
        print(i)


def video_hsv():
    cap = cv.VideoCapture(0)
    while True:
        # 读取帧
        _, frame = cap.read()
        # 转换颜色空间 BGR 到 HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 定义HSV中蓝色的范围
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        # 设置HSV的阈值使得只取蓝色
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        # 将掩膜和图像逐像素相加
        res = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    # 如何找到所需要的HSV值
    green = np.uint8([[[0, 255, 0]]])
    hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
    print(hsv_green)
