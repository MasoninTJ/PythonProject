import cv2
from matplotlib import pyplot as plt


def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])  # numpy的ravel函数功能是将多维数组降为一维数组
    plt.show()


img = cv2.imread('image/pic_0003.bmp', 0)
# hist = cv2.calcHist(img, [0], None, 256, [0, 256])

ret, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
plot_demo(img)
