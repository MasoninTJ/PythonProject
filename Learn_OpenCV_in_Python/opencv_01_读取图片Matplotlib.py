import cv2
from matplotlib import pyplot as plt

"""
使用Matplotlib库来显示图片，亲测这种方式有点卡卡的
"""

img = cv2.imread('image/pic_0003.bmp', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # 隐藏刻度值
plt.show()
