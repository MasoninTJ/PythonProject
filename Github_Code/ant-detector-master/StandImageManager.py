import os
import time
import cv2


class StandImageManager(object):
    SAVE_PATH = "data/settings/"
    standerImages = []

    # 初始化
    def __init__(self):
        super(object, self).__init__()
        for file_name in os.listdir(self.SAVE_PATH):
            if file_name.endswith('.jpg'):
                image = cv2.imread(self.SAVE_PATH+file_name)
                self.standerImages.append([file_name, image])

    # 增加标准图片
    def add(self, image):
        # 存储标准图片,以时间戳为文件名，精确到0.1秒
        file_name = str(int(round(time.time() * 10))) + ".jpg"
        cv2.imwrite(self.SAVE_PATH + file_name, image)
        self.standerImages.append([file_name, image])
        return file_name

    def remove(self, file_name):
        for standerImage in self.standerImages:
            if standerImage[0] == file_name:
                self.standerImages.remove(standerImage)
                break
        os.remove(self.SAVE_PATH + file_name)
