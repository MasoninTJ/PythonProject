import cv2

#定义图片显示函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)#等待时间
    cv2.destroyAllWindows()

img=cv2.imread('31.jpg')
#cv_show('img',img)

img1=img[0:400, 0:200]

cv_show('img1',img1)