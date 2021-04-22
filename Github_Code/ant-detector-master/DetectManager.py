import numpy
import cv2
import time
from datetime import datetime


class DetectManager(object):
    LOG_PATH = "data/log/single/"
    standerImages = None
    settings = None
    standers = None  # 每个标准图对应每个设置的遮罩，遮罩后的图像
    detected = None
    detectResult = None
    averageResult = None
    threshold = None
    final_result = None
    log_file = None
    # 初始化

    def __init__(self):
        super(object, self).__init__()
        self.settingsCount = 0
        self.standerImageCount = 0
        self.minMasks = None
        self.log_file = open(self.LOG_PATH + datetime.now().date().strftime('%Y%m%d') +
                             '.log', 'a', encoding='utf-8')

    # 检测前的初始化
    def init(self, stander_images, settings):
        try:
            self.standerImages = stander_images
            self.settings = settings
            self.settingsCount = len(self.settings)
            self.standerImageCount = len(self.standerImages)
            # 存放每个设置的最小遮罩，按最小外接正矩形剪裁的遮罩
            self.minMasks = [None] * self.settingsCount
            # 每个设置每个标准图的预处理数据，二维数组，形状为[settingsCount,standerImageCount]
            self.standers = [
                [None] * self.standerImageCount for i in range(self.settingsCount)]
            for i in range(self.settingsCount):
                setting = settings[i]
                detect_type = setting[2]
                box = setting[6]
                min_mask = self._cut_by_box(setting[5], box)
                self.minMasks[i] = min_mask
                for j in range(self.standerImageCount):
                    self.standers[i][j] = self._image_init(
                        detect_type, stander_images[j][1], box, min_mask)
        except BaseException as error:
            print(error)

    # 图像预处理,根据不同的检测类型，进行不同的预处理
    def _image_init(self, detect_type, image, box, min_mask):
        if detect_type == 0:  # 模板检测
            # 剪裁，遮罩
            return self._cut_by_box(image, box) & min_mask
        elif detect_type == 1:  # 灰度特征检测
            # 剪裁，遮罩，转灰度
            return cv2.cvtColor(self._cut_by_box(image, box) & min_mask, cv2.COLOR_BGR2GRAY)
        elif detect_type == 2:  # 颜色特征检测
            # 剪裁，遮罩，转HSV，提取色相
            hsv_image = cv2.cvtColor(self._cut_by_box(
                image, box) & min_mask, cv2.COLOR_BGR2HSV)
            return hsv_image[:, :, 0]  # 提取色相
        elif detect_type == 3:  # 灰度分布检测
            # 剪裁，转灰度，直方图，归一化
            gray_image = cv2.cvtColor(
                self._cut_by_box(image, box), cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_image], [0], min_mask[:, :, 0], [
                                128], [0, 256])  # 形状128X1
            return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)  # 归一化
        elif detect_type == 4:  # 颜色分布检测
            # 剪裁，转HSV，色相H和饱和度S直方图，归一化
            hsv_image = cv2.cvtColor(
                self._cut_by_box(image, box), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_image], [0], min_mask[:, :, 0], [
                                128], [0, 256])  # 形状128X128
            # 归一化 形状128X128
            return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)

    # 检测并输出结果
    def detect(self, image):
        detect_results = numpy.zeros(
            (self.settingsCount, self.standerImageCount))
        average_results = numpy.zeros(self.settingsCount)
        final_results = numpy.zeros(self.settingsCount)
        for i in range(self.settingsCount):
            setting = self.settings[i]
            detect_type = setting[2]
            box = setting[6]
            target = self._image_init(
                detect_type, image, box, self.minMasks[i])
            for j in range(self.standerImageCount):
                detect_results[i][j] = self._detect(detect_type, target, self.standers[i][j],
                                                    setting[3])
            # 平均结果
            average_result = numpy.mean(detect_results[i])
            average_results[i] = average_result
            # 最终结果，未超过阈值检测通过结果为1，否则为0
            if average_result > setting[4]:
                final_results[i] = 1
            # 打印检测结果
            self.detected = "检测:" + setting[1]
            self.detectResult = '检测结果:{}'.format(detect_results[i])
            self.averageResult = "平均值:" + str(average_result)
            self.threshold = "阈值:" + str(setting[4])
            self.final_result = "结果:" + str(final_results[i])
            self._output(self.detected)
            self._output(self.detectResult)
            self._output(self.averageResult +
                         self.threshold + self.final_result)
        return [final_results, average_results, detect_results]

    # 执行单项检测
    def _detect(self, detect_type, target, stander, argument):
        # detect_type:0 模板检测，1特征检测，2颜色检测
        if detect_type == 0:  # 模板检测
            res = cv2.matchTemplate(target, stander, cv2.TM_CCOEFF_NORMED)
            return res[0, 0]
        elif detect_type == 1 or detect_type == 2:  # 灰度特征检测 与 颜色特征检测
            return self._features_detect(target, stander, argument)
        elif detect_type == 3 or detect_type == 4:  # 灰度分布检测 与 颜色分布检测
            return self._histograms_detect(target, stander, argument)

    # 分布检测
    def _histograms_detect(self, target, stander, argument):
        return cv2.compareHist(target, stander, 0)  # 比较直方图

    # 特征检测
    def _features_detect(self, target, stander, argument):
        try:
            # 使用SIFT检测角点
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(stander, None)
            kp2, des2 = sift.detectAndCompute(target, None)
            # 设置FLANN匹配器参数，algorithm设置可参考https://docs.opencv.org/3.1.0/dc/d8c/namespacecvflann.html
            indexParams = dict(algorithm=0, trees=3)
            searchParams = dict(checks=50)
            # 定义FLANN匹配器
            matcher = cv2.FlannBasedMatcher(indexParams, searchParams)
            # 使用 KNN 算法实现匹配
            matches = matcher.knnMatch(des1, des2, k=2)
            # 计算正确匹配数量
            good_match_count = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < argument * n.distance:  # argument 官方建议取值0.7
                    good_match_count += 1
            return good_match_count / len(kp1)  # 好匹配占比
        except BaseException as error:
            print(error)
        return 0

    # 输出并写日志
    def _output(self, string):
        print(string)
        self.log_file.write(string + '\n')
        self.log_file.flush()

    # 图像剪裁
    def _cut_by_box(self, image, box):
        return image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
