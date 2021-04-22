# 导入程序运行必须模块
import os
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
# 导入designer工具生成的login模块
from mainWindows import Ui_MainWindow
from createModelEvent import MyCreateForm

import cv2 as cv
import numpy as np


class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.create_form = MyCreateForm()   # 创建ROI的子窗口

        self.cwd = os.getcwd()  # 获取当前程序文件位置

        self.button_load_image.clicked.connect(self.load_image)
        self.button_previous_test_image.clicked.connect(self.previous_test_image)
        self.button_next_test_image.clicked.connect(self.next_test_image)
        self.button_create_model.clicked.connect(self.create_model)
        self.button_test_run.clicked.connect(self.test_run)
        self.button_save.clicked.connect(self.save_task)
        self.button_init.clicked.connect(self.init)
        self.button_run.clicked.connect(self.run)
        self.button_end.clicked.connect(self.end)

        self.list_image_test = []

    def load_image(self):
        """
        编辑任务中加载图像按钮，清空测试图像列表，加载新图像，可以加载一个图像，也可以加载多个图像
        @return:
        """
        temp_image_list, filetype = QFileDialog.getOpenFileNames(self, "多文件选择", self.cwd, "bmp (*.bmp)")
        if len(temp_image_list):
            self.list_image_test.clear()
            self.list_image_test = temp_image_list
            self.label_edit_image.setPixmap(QPixmap(self.list_image_test[0]))
            self.label_test_image_num.setText(f'{1}/{len(self.list_image_test)}')

            self.button_previous_test_image.setEnabled(False)
            if len(temp_image_list) == 1:
                self.button_next_test_image.setEnabled(False)
            else:
                self.button_next_test_image.setEnabled(True)
        else:
            pass  # 取消选择

    def previous_test_image(self):
        """
        编辑任务中，上一张测试图片，如果到头了，则按钮显示为灰色
        @return:
        """
        previous_test_image_index = int(self.label_test_image_num.text().split('/')[0]) - 2  # 获取上一张测试图像的下标
        self.label_edit_image.setPixmap(QPixmap(self.list_image_test[previous_test_image_index]))
        self.label_test_image_num.setText(f'{previous_test_image_index + 1}/{len(self.list_image_test)}')

        self.button_next_test_image.setEnabled(True)
        if previous_test_image_index == 0:  # 到第一张图时失去使能
            self.button_previous_test_image.setEnabled(False)

    def next_test_image(self):
        """
        编辑任务中，下一张测试图片，如果到头了，则按钮显示为灰色
        @return:
        """
        next_test_image_index = int(self.label_test_image_num.text().split('/')[0])  # 获取下一张测试图像的下标
        self.label_edit_image.setPixmap(QPixmap(self.list_image_test[next_test_image_index]))
        self.label_test_image_num.setText(f'{next_test_image_index + 1}/{len(self.list_image_test)}')

        self.button_previous_test_image.setEnabled(True)
        if next_test_image_index + 1 == len(self.list_image_test):  # 到最后一张图时失去使能
            self.button_next_test_image.setEnabled(False)

    def create_model(self):
        """
        编辑任务中创建模型，从当前图片上选取ROI可以创建一个模型，也可以创建多个模型
        @return:
        """

        current_test_image_index = int(self.label_test_image_num.text().split('/')[0]) - 1  # 当前测试图片的下标
        current_test_image_np = np.fromfile(self.list_image_test[current_test_image_index], dtype=np.uint8)
        current_test_image = cv.imdecode(current_test_image_np, 0)

        qt_image = QImage(current_test_image.data, current_test_image.shape[1], current_test_image.shape[0], QImage.Format_Grayscale8)
        # 将子窗口显示在屏幕上，并显示图片
        self.create_form.label.setPixmap(QPixmap.fromImage(qt_image))
        self.create_form.scaled_image()
        self.create_form.show()

    def test_run(self):
        """
        以本地图片运行计数
        @return:
        """
        pass

    def save_task(self):
        """
        保存任务，将当前模型保存于任务名的文件夹下
        @return:
        """
        pass

    def init(self):
        """
        初始化硬件
        - 相机
        - 震动盘
        - 传送带
        @return:
        """
        pass

    def run(self):
        """
        持续运行任务，直到3次计数为0结束
        @return:
        """
        pass

    def end(self):
        """
        手动结束任务
        @return:
        """
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle('Fusion')  # 以苹果的风格进行显示

    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
