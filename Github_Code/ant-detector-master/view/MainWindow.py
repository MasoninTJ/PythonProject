import cv2
import numpy
import os
import time
import shutil
from threading import Thread, Semaphore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QListWidgetItem, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets
from ui.MainWindow import Ui_MainWindow
from view.MaskGraphicItem import MaskGraphicItem
from view.SettingDialog import SettingDialog
from view.SystemSettingDialog import SystemSettingDialog


class MainWindow(QMainWindow, Ui_MainWindow):
    openCameraSignal = pyqtSignal()  # 打开相机信号，利用信号机制实现解耦
    closeCameraSignal = pyqtSignal()  # 关闭相机信号
    startAutoSingnal = pyqtSignal()
    closeAutoSingnal = pyqtSignal()
    
    testCorrectImages = []
    testErrorImages = []
    CORRECT_PATH = "data/test/correct/"
    ERROR_PATH = "data/test/error/"
    cwd = None
    currentItemType = None
    folder_path = None
    test_images = None
    image_list = None
    # 设置计数器的值为 10
    sem = Semaphore(10)

    # 初始化
    def __init__(self, parent=None):
        self.cwd = os.getcwd()
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # 系统设置初始化
        self.systemSettingDialog = SystemSettingDialog(self)
        self.systemSettingsButton.clicked.connect(
            self._processing_systemSettings)
        # 相机相关初始化
        self._isCameraOpened = False
        self.cameraButton.clicked.connect(self._on_camera_action)
        # 图像显示区相关初始化
        self._scene = QGraphicsScene()  # 创建场景,用于显示图片
        self.graphicsView.setScene(self._scene)  # 将场景添加至视图
        self.currentImage = None
        self.zoom_scale = 1  # 图像的缩放比例
        self.imageGraphicItem = QGraphicsPixmapItem()  # 用于在scene中显示Image
        self.imageGraphicItem.setZValue(0)  # 设置显示在0层
        self._scene.addItem(self.imageGraphicItem)
        # 关闭按钮位置设置
        self.closeButton.move(self.graphicsView.width() / 2, 10)
        self.closeButton.setVisible(False)
        # 标准图片相关初始化
        self.standerImageListWidget.itemDoubleClicked.connect(
            self._stander_image_selected)
        self.closeButton.clicked.connect(self._close_stander_image)
        self.selectedStanderImageItem = None
        # 检测设置相关初始化
        self.settingDialog = SettingDialog(self)
        self.addSettingButton.clicked.connect(self._add_setting)
        self.maskGraphicItem = MaskGraphicItem()  # 用于在scene中显示mask
        self._scene.addItem(self.maskGraphicItem)
        self.settingDialog.resetMaskButton.clicked.connect(self._reset_mask)
        self.settingListWidget.itemDoubleClicked.connect(
            self._setting_selected)
        self.selectedSettingItem = None
        self.isAutoStarted = False
        # 测试图片相关初始化
        self.correctImageListWidget.itemDoubleClicked.connect(self._correct_image_select)
        self.errorImageListWidget.itemDoubleClicked.connect(self._error_image_select)
        self.closeButton.clicked.connect(self._close_test_image)
        self.selectedTestImageItem = None
        self._init_test_list()
        # 自动测试
        self.autoDetectButton.clicked.connect(self._on_auto_action)

    # 打开系统设置
    def _processing_systemSettings(self):
        self.systemSettingDialog.show_add()
        self.systemSettingDialog.reload_setting()

    # 相机控制事件
    def _on_camera_action(self):
        if self._isCameraOpened:
            self.closeCameraSignal.emit()  # 通知外部，处理关闭相机的事务
        else:
            self.openCameraSignal.emit()  # 通知外部，处理启动相机的事务
            self.closeButton.setVisible(False)  # 当相机开启时，"关闭图片"按钮不可见

    # 设置相机相关widget的状态
    def set_camera_status(self, is_opened):
        self._isCameraOpened = is_opened
        if self._isCameraOpened:
            self.cameraButton.setText("关闭相机")
        else:
            self.cameraButton.setText("启动相机")
        self.autoDetectButton.setEnabled(self._isCameraOpened)
        self.manualDetectButton.setEnabled(self._isCameraOpened)
        self.addStanderImageButton.setEnabled(self._isCameraOpened)
        self.addCorrectImageButton.setEnabled(self._isCameraOpened)
        self.addErrorImageButton.setEnabled(self._isCameraOpened)

    # 自动检测控制事件
    def _on_auto_action(self):
        if self.isAutoStarted:
            self.closeAutoSingnal.emit()
        else:
            self.startAutoSingnal.emit()
            
    # 设置自动检测相关 widget 的状态
    def set_auto_status(self, is_started):
        self.isAutoStarted = is_started
        if self.isAutoStarted:
            self.autoDetectButton.setText("关闭检测")
        else:
            self.autoDetectButton.setText("自动检测")

    # 显示消息
    def show_message(self, msg):
        QApplication.processEvents()
        self.textBrowser.append(msg)
        self.textBrowser.moveCursor(
            self.textBrowser.textCursor().End)  # 光标移到到最后

    # 显示错误
    def show_error(self, msg):
        QtWidgets.QMessageBox.information(
            self, "错误", msg, QtWidgets.QMessageBox.Ok)

    # 显示警告
    def show_warning(self, msg):
        QtWidgets.QMessageBox.information(
            self, "注意", msg, QtWidgets.QMessageBox.Ok)

    # 显示确认信息
    def show_confirm(self, msg):
        reply = QtWidgets.QMessageBox.information(
            self, "确认", msg, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            return True
        else:
            return False

    # 显示图像
    def show_image(self, image):
        self.imageGraphicItem.setVisible(True)
        # 关闭键按钮在视图界面居中,注：46是button长度的一半
        self.closeButton.move((self.graphicsView.width() / 2) - 46, 10)
        self.currentImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换图像通道
        image_width = image.shape[1]  # 获取图像大小
        image_height = image.shape[0]
        frame = QImage(image, image_width, image_height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.imageGraphicItem.setPixmap(pix)
        # 根据graphicsView的大小，调整图片的比例
        scale_x = self.graphicsView.width() / image_width
        scale_y = self.graphicsView.height() / image_height
        if scale_x < scale_y:
            self.zoom_scale = scale_x
        else:
            self.zoom_scale = scale_y
        self.imageGraphicItem.setScale(self.zoom_scale)

    # 添加标准图到列表中
    def add_stander_image_item(self, file_name, image):
        item = QListWidgetItem()
        item.setText(file_name)
        item.image = image
        self.standerImageListWidget.addItem(item)

    # 标准图片被双击选中的事件处理
    def _stander_image_selected(self, selected_item):
        self.closeButton.setVisible(True)
        self.selectedStanderImageItem = selected_item
        self.addStanderImageButton.setEnabled(False)
        self.removeStanderImageButton.setEnabled(True)
        self.closeButton.setEnabled(True)
        self.autoDetectButton.setEnabled(False)
        self.manualDetectButton.setEnabled(False)

    # 关闭标准图片，并完成相应的UI界面变化
    def _close_stander_image(self):
        self.selectedStanderImageItem = None
        if self._isCameraOpened:
            self.addStanderImageButton.setEnabled(True)
            self.autoDetectButton.setEnabled(True)
            self.manualDetectButton.setEnabled(True)
        else:
            self.imageGraphicItem.setVisible(False)
        self.removeStanderImageButton.setEnabled(False)
        self.closeButton.setEnabled(False)
        self.closeButton.setVisible(False)

    # 删除标准图片列表中的项目
    def remove_stander_image_item(self):
        row = self.standerImageListWidget.row(self.selectedStanderImageItem)
        self.standerImageListWidget.takeItem(row)
        self._close_stander_image()

    # 增加检测设置的按钮按下的事件处理
    def _add_setting(self):
        if self.currentImage is None:
            self.show_warning("请先开启相机或打开标准图片后，再添加检测设置")
            return
        # 新建mask
        new_mask = numpy.zeros(self.currentImage.shape,
                               self.currentImage.dtype)
        self.show_mask(new_mask)
        self.settingDialog.show_add()
        self.tabWidget.setEnabled(False)

    # 显示mask
    def show_mask(self, mask):
        self.maskGraphicItem.display(mask, self.zoom_scale)

    # 关闭设置对话框后，应调用此方法，更新界面UI状态
    def setting_dialog_closed(self):
        self.maskGraphicItem.setVisible(False)  # 关闭遮罩
        self.tabWidget.setEnabled(True)

    # 重置遮罩
    def _reset_mask(self):
        new_mask = numpy.zeros(self.currentImage.shape,
                               self.currentImage.dtype)
        self.show_mask(new_mask)

    # 添加检测设置到列表中
    def add_setting_item(self, setting):
        item = QListWidgetItem()
        item.setText(setting[1])
        item.setting = setting
        self.settingListWidget.addItem(item)

    # 修改检测设置列表中的项
    def modify_setting_item(self, setting):
        self.selectedSettingItem.setText(setting[1])
        self.selectedSettingItem.setting = setting

    # 删除检测设置列表中的项
    def remove_setting_item(self):
        row = self.settingListWidget.row(self.selectedSettingItem)
        self.settingListWidget.takeItem(row)

    # 检测设置被双击选中的事件处理
    def _setting_selected(self, selected_item):
        if self.currentImage is None:
            self.show_warning("请先开启相机或打开标准图片后，再编辑检测设置")
            return
        self.selectedSettingItem = selected_item
        self.show_mask(selected_item.setting[5])
        self.settingDialog.show_edit(selected_item.setting)
        self.tabWidget.setEnabled(False)

    # 遍历测试图名称
    def _init_test_list(self):
        for file_name in os.listdir(self.CORRECT_PATH):
            if file_name.endswith('.jpg'):
                self.testCorrectImages.append(file_name)
        for file_name in os.listdir(self.ERROR_PATH):
            if file_name.endswith('.jpg'):
                self.testErrorImages.append(file_name)

    def _correct_image_select(self, selected_item):
        self._test_image_selected(selected_item, 0)

    def _error_image_select(self, selected_item):
        self._test_image_selected(selected_item, 1)

    def _test_image_selected(self, selected_item, type):
        # 测试图片被双击选中的事件处理
        self._decide_type(type)
        self.selectedTestImageItem = selected_item
        self.closeButton.setVisible(True)
        self.addCorrectImageButton.setEnabled(False)
        self.addErrorImageButton.setEnabled(False)
        self.removeTestImageButton.setEnabled(True)
        self.closeButton.setEnabled(True)
        self.autoDetectButton.setEnabled(False)
        self.manualDetectButton.setEnabled(False)

    def add_test_item(self, file_name, type):
        # 不直接添加图片数据到内存中
        self._decide_type(type)
        item = QListWidgetItem()
        item.setText(file_name)
        self.image_list.addItem(item)

    def add_test_image(self, image, type):
        # 增加测试图片
        # 存储测试图片,以时间戳为文件名，精确到0.1秒
        self._decide_type(type)
        file_name = str(int(round(time.time() * 10))) + ".jpg"
        cv2.imwrite(self.folder_path + file_name, image)
        self.test_images.append(file_name)
        self.add_test_item(file_name, type)

    def remove_test_image(self, file_name):
        # 删除测试图片列表中的图片
        self._decide_type(self.currentItemType)
        row = self.image_list.row(self.selectedTestImageItem)
        self.image_list.takeItem(row)
        self._close_test_image()
        if file_name in self.test_images:
            self.test_images.remove(file_name)
            os.remove(self.folder_path + file_name)
            self.show_error("删除成功！")
        else:
            self.show_error("删除失败！")

    def clean_list(self, type):
        # 强制清空再新建
        self._decide_type(type)
        self.image_list.clear()
        self.test_images.clear()
        self.removeTestImageButton.setEnabled(False)
        shutil.rmtree(self.folder_path)
        os.mkdir(self.folder_path)
        QApplication.processEvents()

    def import_test_image(self, type):
        # 导入测试图到文件夹
        self._decide_type(type)
        files, file_type = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                                  "测试图选择",
                                                                  self.cwd,  # 起始路径
                                                                  "Image Files (*.jpg)")
        for file in files:
            # 获取一个 semaphore
            self.sem.acquire()
            import_thread = Thread(
                target=self._copy_image, args=(file, self.folder_path))
            import_thread.start()

        QApplication.processEvents()

    def _close_test_image(self):
        # 关闭测试图片，并完成相应的UI界面变化
        self.selectedTestImageItem = None
        if self._isCameraOpened:
            self.addCorrectImageButton.setEnabled(True)
            self.addErrorImageButton.setEnabled(True)
            self.autoDetectButton.setEnabled(True)
            self.manualDetectButton.setEnabled(True)
        else:
            self.imageGraphicItem.setVisible(False)
        self.closeButton.setVisible(False)
        self.removeTestImageButton.setEnabled(False)
        self.selectedStanderImageItem = None

    def _decide_type(self, type):
        # 私有方法
        # 通过 type 的值来判断属于正确还是错误测试图
        # 1 为错误，其他值为正确
        if type == 1:
            self.currentItemType = 1
            self.folder_path = self.ERROR_PATH
            self.test_images = self.testErrorImages
            self.image_list = self.errorImageListWidget
        else:
            self.currentItemType = 0
            self.folder_path = self.CORRECT_PATH
            self.test_images = self.testCorrectImages
            self.image_list = self.correctImageListWidget

    def _copy_image(self, source_file, target_folder):
        shutil.copy(source_file, target_folder)
        file_name = os.path.basename(source_file)
        if file_name not in self.test_images:
            self.test_images.append(file_name)
            item = QListWidgetItem()
            item.setText(file_name)
            self.image_list.addItem(item)
        # 释放 semaphore
        self.sem.release()
