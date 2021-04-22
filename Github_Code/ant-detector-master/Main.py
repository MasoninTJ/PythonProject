import sys
import cv2
import numpy as np
import time
import socket
import threading
import inspect
import ctypes
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMessageBox
from DetectManager import DetectManager
from SettingManager import SettingManager
from view.MainWindow import MainWindow
from CameraManager import CameraManager
from StandImageManager import StandImageManager


class Main(object):
    CORRECT_PATH = "data/test/correct/"
    ERROR_PATH = "data/test/error/"
    type = 0
    cameraTypeId = 0
    # 设置计数器的值为 10
    correct_list_result = []
    correct_avg_result = []
    error_list_result = []
    error_avg_result = []
    max_result = ''
    min_result = ''
    avg_result = []
    LOG_PATH = "data/log/batch/"

    # 初始化
    def __init__(self):
        self.mainWindow = MainWindow()
        # 相机相关初始化
        self.cameraManager = CameraManager()
        self.cameraManager.imageCapturedSignal.connect(self._image_captured)
        self.currentImage = None
        self.mainWindow.openCameraSignal.connect(self._open_camera)
        self.mainWindow.closeCameraSignal.connect(self._close_camera)
        self.mainWindow.systemSettingDialog.cameraTypeSignal.connect(
            self.get_cameraTypeId)
        # 标准图片相关初始化
        self.standerImageManager = StandImageManager()
        self._load_stander_image()
        self.mainWindow.addStanderImageButton.clicked.connect(
            self._add_stander_image)
        self.mainWindow.standerImageListWidget.itemDoubleClicked.connect(
            self._stander_image_selected)
        self.mainWindow.removeStanderImageButton.clicked.connect(
            self._remove_stander_image)
        self.mainWindow.closeButton.clicked.connect(self._close_stander_image)
        # 检测设置相关初始化
        self.settingManager = SettingManager()
        self._load_setting()
        self.mainWindow.settingDialog.addSettingSignal.connect(
            self._add_setting)
        self.mainWindow.settingDialog.modifySettingSignal.connect(
            self._modify_setting)
        self.mainWindow.settingDialog.removeSettingSignal.connect(
            self._remove_setting)
        self.mainWindow.closeButton.clicked.connect(self._close_stander_image)
        # 检测相关初始化
        self.detectManager = DetectManager()
        self.mainWindow.manualDetectButton.clicked.connect(self._manual_detect)
        self.mainWindow.startAutoSingnal.connect(self._start_auto)
        self.mainWindow.closeAutoSingnal.connect(self._close_auto)

        # 测试图相关初始化
        self._load_test_image()
        self.mainWindow.addCorrectImageButton.clicked.connect(
            lambda: self._add_test_image(0))
        self.mainWindow.addErrorImageButton.clicked.connect(
            lambda: self._add_test_image(1))
        self.mainWindow.correctImageListWidget.itemDoubleClicked.connect(
            self._test_correct_image_selected)
        self.mainWindow.errorImageListWidget.itemDoubleClicked.connect(
            self._test_error_image_selected)
        self.mainWindow.correctImageListWidget.itemDoubleClicked.connect(
            self._test_image_detect)
        self.mainWindow.errorImageListWidget.itemDoubleClicked.connect(
            self._test_image_detect)
        self.mainWindow.removeTestImageButton.clicked.connect(
            self._remove_test_image)
        self.mainWindow.cleanCorrectImageListButton.clicked.connect(
            lambda: self._clean_image_list(0))
        self.mainWindow.cleanErrorImageListButton.clicked.connect(
            lambda: self._clean_image_list(1))
        self.mainWindow.importCorrectImageButton.clicked.connect(
            lambda: self._import_image(0))
        self.mainWindow.importErrorImageButton.clicked.connect(
            lambda: self._import_image(1))
        self.mainWindow.batchDetectButton.clicked.connect(
            self._batch_image_detect)
        self.log_file = open(self.LOG_PATH + datetime.now().date().strftime('%Y%m%d') +
                             '.log', 'a', encoding='utf-8')
        self.udp_socket = None
        self.address = None
        self.server_th = None
        self.msg = None
        self.mainWindow.systemSettingDialog.settings = self.settingManager.settings

    def show(self):
        # 窗体运行
        self.mainWindow.show()

    # 输出错误信息
    def show_error(self, msg):
        self.mainWindow.show_error(msg)

    # 输出警告信息
    def show_warning(self, msg):
        self.mainWindow.show_warning(msg)

    # 输出信息
    def show_message(self, msg):
        self.mainWindow.show_message(msg)

    # 获取相机类型参数
    def get_cameraTypeId(self, cameraTypeId):
        self.cameraTypeId = cameraTypeId

    def _start_auto(self):
        self.mainWindow.set_auto_status(True)
        self._auto_detect()

    def _close_auto(self):
        self.mainWindow.set_auto_status(False)
        if self.udp_socket:
            self._stop_thread(self.server_th)
            self.udp_socket.close()

    # 启动相机
    def _open_camera(self):
        if self.cameraManager.open(self.cameraTypeId):
            self.mainWindow.set_camera_status(True)
            self.cameraManager.start_capture()
            self.show_message("相机启动")
        else:
            self.show_error("相机启动失败")

    # 关闭相机
    def _close_camera(self):
        self.cameraManager.close()
        self.mainWindow.set_camera_status(False)
        self.show_message("相机关闭")

    # 相机捕获到画面
    def _image_captured(self, image):
        if image is None:
            self._close_camera()
            self.show_error("获取画面失败")
        else:
            self.currentImage = image
            self.mainWindow.show_image(image)

    # 加载已存在的标准图片
    def _load_stander_image(self):
        for standerImage in self.standerImageManager.standerImages:
            self.mainWindow.add_stander_image_item(
                standerImage[0], standerImage[1])

    # 添加标准图片
    def _add_stander_image(self):
        if self.currentImage is not None:
            file_name = self.standerImageManager.add(self.currentImage)
            self.mainWindow.add_stander_image_item(
                file_name, self.currentImage)

    # 标准图片被选中的事件处理
    def _stander_image_selected(self, selected_item):
        if self.cameraManager.is_opened():
            self.cameraManager.stop_capture()
        self.mainWindow.show_image(selected_item.image)

    # 关闭标准图片
    def _close_stander_image(self):
        if self.cameraManager.is_opened():
            self.cameraManager.start_capture()

    # 删除标准图片
    def _remove_stander_image(self):
        if not self.mainWindow.show_confirm("确定要删除此标准图片吗？"):
            return
        self._close_stander_image()
        self.standerImageManager.remove(
            self.mainWindow.selectedStanderImageItem.text())
        self.mainWindow.remove_stander_image_item()

    def _add_setting(self, setting):
        self.settingManager.add(setting)
        self.mainWindow.add_setting_item(setting)

    def _modify_setting(self, setting):
        self.settingManager.modify(setting)
        self.mainWindow.modify_setting_item(setting)

    def _remove_setting(self, setting):
        self.settingManager.remove(setting)
        self.mainWindow.remove_setting_item()

    # 加载设置
    def _load_setting(self):
        for setting in self.settingManager.settings:
            self.mainWindow.add_setting_item(setting)

    def _manual_detect(self):
        self.detectManager.init(
            self.standerImageManager.standerImages, self.settingManager.settings)
        self._detect_result(self.currentImage)

    def _auto_detect(self):
        detect_single = self.mainWindow.systemSettingDialog.detect_signal

        # 信号值等于 1 为实时检测
        # 信号值等于 0 为外部信号检测
        if detect_single == 1:
            self.detectManager.init(
                self.standerImageManager.standerImages, self.mainWindow.systemSettingDialog.pre_setting)
            self._internal_detect()
        elif detect_single == 0:
            self.detectManager.init(
                self.standerImageManager.standerImages, self.settingManager.settings)
            self._external_detect()
        else:
            self.show_error("请先设置自动检测方式在进行操作")

    def _internal_detect(self):
        self.show_message("正在进行预检测\t")
        result = self.detectManager.detect(self.currentImage)
        if 1.0 in result[0]:
            self.show_message("预检测通过，开始正式检测\t")
            self._manual_detect()
        else:
            self.show_message("预检测不通过，无法开始正式检测\t")

        if self.mainWindow.isAutoStarted == True:
            self._internal_detect()
        else:
            return

    def _external_detect(self):
        IP = self.mainWindow.systemSettingDialog.IP
        port = self.mainWindow.systemSettingDialog.port
        self.show_message("正在接受信号")
        if self.mainWindow._isCameraOpened == True:
            self._udp_server_start(IP, port)

    def _udp_server_start(self, IP, port):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = (str(IP), int(port))
        self.udp_socket.bind(address)
        self.server_th = threading.Thread(target=self._udp_server_concurrency)
        self.server_th.start()

    def _udp_server_concurrency(self):
        while True:
            recv_msg = self.udp_socket.recvfrom(1024)
            self.msg = recv_msg[0]
            if b'1' in recv_msg[0]:
                self._detect_result(self.currentImage)
            else:
                self.show_message("未检测到信号")

    def _async_raise(self, tid, exctype):
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            tid, ctypes.py_object(exctype))

    def _stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    def _test_image_detect(self, selected_item):
        # 双击后进行检测
        # 当选中时从文件夹中取出图片数据
        if self.type == 1:
            folder_path = self.ERROR_PATH
        else:
            folder_path = self.CORRECT_PATH

        image = self._img_read(folder_path + selected_item.text())
        self.detectManager.init(
            self.standerImageManager.standerImages, self.settingManager.settings)
        self._detect_result(image)

    def _detect_result(self, image):
        setting = self.settingManager.settings
        detect_result = self.detectManager.detect(image)
        for i in range(len(setting)):
            msg = "检测:" + str(setting[i][1]) + "\n"
            msg += "结果:" + str(detect_result[0][i]) + "\n"
            msg += "平均值:" + str(detect_result[1][i]) + "\n"
            msg += "检测结果:" + str(detect_result[2][i]) + "\n"
            self.show_message(msg)

    def _load_test_image(self):
        # 加载已存在的正确测试图片
        for testCorrectImage in self.mainWindow.testCorrectImages:
            self.mainWindow.add_test_item(testCorrectImage, 0)
        # 加载已存在的错误测试图片
        for testErrorImage in self.mainWindow.testErrorImages:
            self.mainWindow.add_test_item(testErrorImage, 1)

    def _add_test_image(self, type):
        if self.currentImage is not None:
            self.mainWindow.add_test_image(self.currentImage, type)

    def _test_correct_image_selected(self, selected_item):
        self._test_image_selected(selected_item, 0)

    def _test_error_image_selected(self, selected_item):
        self._test_image_selected(selected_item, 1)

    def _test_image_selected(self, item, type):
        if type == 0:
            # 告知检测模块检测的是哪一列表的图片
            # 以及得到对应的路径
            # 0: 正确列表
            # 1: 错误列表
            current_path = self.CORRECT_PATH
            self.type = 0
        else:
            current_path = self.ERROR_PATH
            self.type = 1

        image = self._img_read(current_path + item.text())
        if self.cameraManager.is_opened():
            self.cameraManager.stop_capture()
        self.mainWindow.show_image(image)

    def _batch_image_detect(self):
        if self.mainWindow.testCorrectImages and self.mainWindow.testErrorImages:
            self.detectManager.init(
                self.standerImageManager.standerImages, self.settingManager.settings)
            self.correct_list_result.clear()
            self.correct_avg_result.clear()
            self.error_list_result.clear()
            self.error_avg_result.clear()

            for testImage in self.mainWindow.testCorrectImages:
                target_file = self.CORRECT_PATH + testImage
                self._image_list_detect(target_file, 0)

            for testImage in self.mainWindow.testErrorImages:
                target_file = self.ERROR_PATH + testImage
                self._image_list_detect(target_file, 1)

            localtime = time.strftime("%H:%M:%S", time.localtime())
            self.log_file.write("本地时间: " + localtime + '\n')
            self.log_file.write("正确列表检查结果" + '\n')
            self.show_message("本地时间: " + localtime)
            self._batch_detect_result(0)
            self.log_file.write("错误列表检查结果" + '\n')
            self._batch_detect_result(1)
            self.show_warning("检测完毕，请在控制界面查看结果。")
        else:
            self.show_warning("列表为空，无法进行批量检查。")

    def _image_list_detect(self, target, type):
        if type == 0:
            list_results = self.correct_list_result
            avg_results = self.correct_avg_result
        else:
            list_results = self.error_list_result
            avg_results = self.error_avg_result

        image = self._img_read(target)
        detect_result = self.detectManager.detect(image)
        list_results.append(detect_result[0])
        avg_results.append(detect_result[1])

    def _batch_detect_result(self, type):
        list_settings = self.settingManager.settings
        if type == 0:
            result_title = "正确列表批量检测结果"
            list_results = self.correct_list_result
            list_avg = self.correct_avg_result
        else:
            result_title = "错误列表批量检测结果"
            list_results = self.error_list_result
            list_avg = self.error_avg_result
        self.show_message(result_title)
        for i in range(len(list_settings)):
            # 每次遍历前需要清空数组
            self.avg_result = []
            right_count = 0
            wrong_count = 0
            msg = "检测:" + str(list_settings[i][1]) + "\n"
            for j in range(len(list_results)):
                # 查找第 j 个数组中第 i 个设置的结果
                if list_results[j][i] == 1.0:
                    right_count += 1
                else:
                    wrong_count += 1
            msg += "正确个数:" + str(right_count) + "\n"
            msg += "错误个数:" + str(wrong_count) + "\n"
            for k in range(len(list_avg)):
                self.avg_result.append(list_avg[k][i])
            self.max_result = np.around(np.max(self.avg_result), decimals=3)
            self.min_result = np.around(np.min(self.avg_result), decimals=3)
            self.avg_result = np.around(np.mean(self.avg_result), decimals=3)
            msg += "平均值:" + str(self.avg_result) + "\n"
            msg += "最大值:" + str(self.max_result) + "\n"
            msg += "最小值" + str(self.min_result) + "\n"
            self.show_message(msg)
            self.log_file.write(msg)
            self.log_file.flush()
        # 释放掉内存中的临时数组
        list_results.clear()
        list_avg.clear()

    def _close_test_image(self):
        # 关闭测试图片
        if self.cameraManager.is_opened():
            self.cameraManager.start_capture()

    def _remove_test_image(self):
        # 删除测试图片
        if not self.mainWindow.show_confirm("确定要删除此标准图片吗？"):
            return
        self._close_test_image()
        self.mainWindow.remove_test_image(
            self.mainWindow.selectedTestImageItem.text())

    def _clean_image_list(self, list_type):
        self.mainWindow.clean_list(list_type)

    def _import_image(self, list_type):
        self.mainWindow.import_test_image(list_type)

    @staticmethod
    def _img_read(image_path):
        # 封装图片读取操作
        # 主要针对中文命名的图片进行转码使得能够正常读取
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        return image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
