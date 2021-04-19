import cv2 as cv
from matplotlib import pyplot as plt

"""
图像设置边框，填充

cv2.copyMakerBorder()

边框类型：
cv.BORDER_CONSTANT - 添加恒定的彩色边框。value -边框的颜色
cv.BORDER_REFLECT - 边框将是边框元素的镜像，如下所示： fedcba | abcdefgh | hgfedcba
cv.BORDER_REFLECT_101或cv.BORDER_DEFAULT与上述相同，但略有变化，例如： gfedcb | abcdefgh | gfedcba
cv.BORDER_REPLICATE最后一个元素被复制，像这样： aaaaaa | abcdefgh | hhhhhhh
cv.BORDER_WRAP难以解释，它看起来像这样： cdefgh | abcdefgh | abcdefg


"""

BLUE = [255, 0, 0]
img1 = cv.imread('image/pic_0009.jpg')
plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')

constant = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=BLUE)
plt.subplot(232), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

reflect = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REFLECT)
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')

reflect101 = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REFLECT_101)
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')

replicate = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_REPLICATE)
plt.subplot(235), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')

wrap = cv.copyMakeBorder(img1, 50, 50, 50, 50, cv.BORDER_WRAP)
plt.subplot(236), plt.imshow(wrap, 'gray'), plt.title('WRAP')

plt.show()
