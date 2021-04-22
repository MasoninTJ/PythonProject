import cv2
from PyQt5.QtCore import pyqtSignal, QObject, QTimer


class CameraManager(QObject):
    imageCapturedSignal = pyqtSignal(object)  # 画面捕获信号
    _is_capturing = False

    # 初始化
    def __init__(self, parent=None):
        super(QObject, self).__init__(parent)
        self._camera = cv2.VideoCapture()
        self._timer = QTimer()  # 初始化定时器
        self._timer.timeout.connect(self._on_timer)

    def open(self, cameraTypeId):
        self._camera.open(cameraTypeId)  # 初始化摄像头
        return self._camera.isOpened()

    def is_opened(self):
        return self._camera.isOpened()

    def close(self):
        if self._is_capturing:
            self.stop_capture()
        self._camera.release()
        cv2.destroyAllWindows()

    # 开始捕获画面
    def start_capture(self):
        self._timer.start(30)
        self._is_capturing = True

    def stop_capture(self):
        self._timer.stop()
        self._is_capturing = False

    def is_capturing(self):
        return self._is_capturing

    def _on_timer(self):
        flag, image = self._camera.read()
        if flag:
            self.imageCapturedSignal.emit(image)
        else:
            self.imageCapturedSignal.emit(None)

