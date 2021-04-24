import cv2


def read_image(image_path):
    """
    读取图片并窗口显示
    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(path, image):
    """
    保存代码
    :param path:
    :param image:
    :return:
    """
    cv2.imwrite(path, image)
