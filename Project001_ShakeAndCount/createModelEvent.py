import sys

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication

from createModel import Ui_set_roi_model


class MyCreateForm(QWidget, Ui_set_roi_model):
    def __init__(self, parent=None):
        super(MyCreateForm, self).__init__(parent)
        self.setupUi(self)

    def clear_roi(self):
        """
        清除ROI，图片恢复原大小
        @return:
        """
        pass

    def save_roi(self):
        """
        返回ROI的图片
        @return:
        """
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle('Fusion')  # 以苹果的风格进行显示

    # 初始化
    myWin = MyCreateForm()
    # 将窗口控件显示在屏幕上
    myWin.show()

    current_test_image_np = np.fromfile(r'C:\Users\ZerosZhang\Desktop\01_count_077_001.bmp', dtype=np.uint8)
    current_test_image = cv2.imdecode(current_test_image_np, 0)
    image_size_x, image_size_y = current_test_image.shape
    label_size_x, label_size_y = myWin.label.size().width(), myWin.label.size().height()
    scaled = min([label_size_x / image_size_x, label_size_y / image_size_y])
    print(scaled)
    result_image = cv2.resize(current_test_image, (0, 0), fx=scaled, fy=scaled)
    cv2.imshow('scaled image', result_image)
    print(result_image.shape)
    cv2.waitKey(0)
    qt_image = QImage(result_image.data, result_image.shape[1], result_image.shape[0], QImage.Format_Grayscale8)
    myWin.label.setPixmap(QPixmap.fromImage(qt_image))

    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
