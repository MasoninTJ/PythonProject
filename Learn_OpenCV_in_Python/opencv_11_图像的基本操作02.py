import cv2 as cv

img = cv.imread('image/pic_0005.jpg')
# 如果图像是灰度的，则返回的元组仅包含行数和列数，因此这是检查加载的图像是灰度还是彩色的好方法。
print(f'图像的尺寸：{img.shape}')
print(f'图像总像素：{img.size}')
# img.dtype在调试时非常重要，因为OpenCV-Python代码中的大量错误是由无效的数据类型引起的。
print(f'图像数据类型：{img.dtype}')
