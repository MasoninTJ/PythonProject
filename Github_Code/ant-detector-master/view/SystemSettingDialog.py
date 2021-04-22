from PyQt5.QtWidgets import QDialog
from PyQt5 import QtGui, QtWidgets

from ui.SystemSettingDialog import Ui_SystemSettingDialog
from PyQt5.QtCore import pyqtSignal


class SystemSettingDialog(QDialog, Ui_SystemSettingDialog):
    cameraTypeSignal = pyqtSignal(object)  # 相机类型信号
    detect_signal = None  # 自动检测方式信号
    settings = []
    IP = None
    port = None
    pre_setting = []

    # 初始化
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)
        self.computer_Camera.setChecked(True)  # 默认初始相机都是“本机相机”
        self.saveButton.clicked.connect(self._save)
        self.label_7.setVisible(False)
        self.trigger_settings.setVisible(False)
        self.label_8.setVisible(False)
        self.label_9.setVisible(False)
        self.external_IP.setVisible(False)
        self.external_port.setVisible(False)
 
        # 对 RadioButton 进行分组
        self.cameraGroup = QtWidgets.QButtonGroup(self)
        self.detectGroup = QtWidgets.QButtonGroup(self)
        self.cameraGroup.addButton(self.computer_Camera)
        self.cameraGroup.addButton(self.external_Camera)
        self.detectGroup.addButton(self.external_signal)
        self.detectGroup.addButton(self.internal_signal)

        self.internal_signal.clicked.connect(lambda: self._select_setting(1))
        self.external_signal.clicked.connect(lambda: self._select_setting(0))

    # 显示添加界面
    def show_add(self):
        self.setWindowTitle("系统设置")
        self.show()

    # 获取系统设置的数据
    # def _get_system_setting_data(self):

    # 保存按钮按下的处理
    def _save(self):
        # 选择本机摄像
        if self.computer_Camera.isChecked():
            cameraTypeId = 0
            self.cameraTypeSignal.emit(cameraTypeId)
        # 选择外设摄像
        if self.external_Camera.isChecked():
            cameraTypeId = 1
            self.cameraTypeSignal.emit(cameraTypeId)
        # 选择外设检测
        if self.external_signal.isChecked() == True:
            self.IP = self.external_IP.text()
            self.port = self.external_port.text()

        self.close()
        self.pre_setting.insert(0, self.trigger_settings.itemData(self.trigger_settings.currentIndex()))

    def _select_setting(self, type):
        if type == 1:
            self.detect_signal = 1
            self.label_7.setVisible(True)
            self.trigger_settings.setVisible(True)
            self.label_8.setVisible(False)
            self.label_9.setVisible(False)
            self.external_IP.setVisible(False)
            self.external_port.setVisible(False)
        else:
            self.detect_signal = 0
            self.label_7.setVisible(False)
            self.trigger_settings.setVisible(False)
            self.label_8.setVisible(True)
            self.label_9.setVisible(True)
            self.external_IP.setVisible(True)
            self.external_port.setVisible(True)
        
    def reload_setting(self):
        self.trigger_settings.clear()
        for setting in self.settings:
            self.trigger_settings.addItem(setting[1], setting)
