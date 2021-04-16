import cv2
from matplotlib import pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])  # numpy的ravel函数功能是将多维数组降为一维数组
    plt.show()


img = cv2.imread('image/pic_0003.bmp', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
plot_demo(img)

"""
OpenCV加载的彩色图片是BGR模式的。但是Matplotlib显示的是RGB模式的。
所以彩色图片如果是通过OpenCV读入的在Matplotlib里会显示不正确。
"""
