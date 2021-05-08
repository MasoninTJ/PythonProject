import cv2

"""
cv2.imread()函数用于读取图片
- cv2.IMREAD_COLOR 加载彩色图像。任何图像的透明度都会被忽视。它是默认标志。BGR
- cv2.IMREAD_GRAYSCALE 以灰度模式加载图像
- cv2.IMREAD_UNCHANGED  加载图像，包括alpha通道

cv2.imshow()
cv2.imwrite()
"""

img = cv2.imread(r'C:\Users\ZerosZhang\Documents\PycharmCode\Learn_OpenCV_in_Python\image\pic_0010.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)  # 图像尺寸
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
elif key == ord('s'):
    cv2.imwrite('img_test_save.png', img)
    cv2.destroyAllWindows()
