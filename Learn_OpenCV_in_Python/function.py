import cv2


def read_image(image_path):
    img = cv2.imread(image_path)
    cv2.imshow('image', image_path)


def save_image(path, image):
    """
    保存代码
    :param path:
    :param image:
    :return:
    """
    cv2.imwrite(path, image)
