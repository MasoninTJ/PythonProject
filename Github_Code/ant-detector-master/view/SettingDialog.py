from PyQt5.QtWidgets import QDialog
import time
import cv2
from PyQt5 import QtGui

from ui.SettingDialog import Ui_SettingDialog
from PyQt5.QtCore import pyqtSignal


class SettingDialog(QDialog, Ui_SettingDialog):
    addSettingSignal = pyqtSignal(object)
    modifySettingSignal = pyqtSignal(object)
    removeSettingSignal = pyqtSignal(object)

    # 初始化
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)
        self.setModal(False)
        self.saveButton.clicked.connect(self._save)
        self._status = 0  # 1添加模式，2编辑模式
        self._currentSetting = None
        self.removeButton.clicked.connect(self._remove)

    # 显示添加界面
    def show_add(self):
        self._status = 1
        self.setWindowTitle("添加")
        self.removeButton.setEnabled(False)
        self.show()

    # 显示编辑界面
    def show_edit(self, setting):
        self._status = 2
        self._currentSetting = setting
        self.setWindowTitle("编辑")
        self.removeButton.setEnabled(True)
        self.titleEdit.setText(self._currentSetting[1])
        self.typeComboBox.setCurrentIndex(self._currentSetting[2])
        self.argumentEdit.setText(str(self._currentSetting[3]))
        self.thresholdEdit.setText(str(self._currentSetting[4]))
        self.show()

    # 保存按钮按下的处理
    def _save(self):
        if self._status == 1:  # 添加
            setting_data = self._get_setting_data()
            if setting_data is None:
                return
            self.addSettingSignal.emit(setting_data)
        else:  # 编辑
            setting_data = self._get_setting_data(self._currentSetting)
            if setting_data is None:
                return
            self.modifySettingSignal.emit(setting_data)
        self.close()

    # 获取并验证数据
    def _get_setting_data(self, setting_data=None):
        if setting_data is None:
            setting_data = [None] * 7  # 定义长度为7的数组
            # setting_data[0] ID
            setting_data[0] = str(int(round(time.time())))
        # setting_data[1] 名称
        if len(self.titleEdit.text()) == 0:
            self.parent().show_error("请输入名称")
            return None
        else:
            setting_data[1] = self.titleEdit.text()
        # setting_data[2] 类型 int
        setting_data[2] = self.typeComboBox.currentIndex()
        # setting_data[3] 参数 float
        try:
            setting_data[3] = float(self.argumentEdit.text())
        except ValueError:
            self.parent().show_error("输入的参数必须是有效的浮点数")
            return None
        # setting_data[4] 阈值 float
        try:
            setting_data[4] = float(self.thresholdEdit.text())
        except ValueError:
            self.parent().show_error("输入的阈值必须是有效的浮点数")
            return None
        # setting_data[5] 检测区域 numpy图像
        setting_data[5] = self.parent().maskGraphicItem.mask
        # setting_data[6] 检测区域的最小外接正矩形 [x,y,w,h]
        setting_data[6] = cv2.boundingRect(setting_data[5][:, :, 0])
        return setting_data

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.parent().setting_dialog_closed()

    def _remove(self):
        if self.parent().show_confirm("确定要删除此检测设置吗？"):
            self.removeSettingSignal.emit(self._currentSetting)
            self.close()


