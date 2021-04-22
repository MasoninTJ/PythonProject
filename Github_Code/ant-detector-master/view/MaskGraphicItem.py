import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem


class MaskGraphicItem(QGraphicsPixmapItem):

    # 初始化
    def __init__(self, parent=None):
        super(QGraphicsPixmapItem, self).__init__(parent)
        self.mask = None
        self.zoom_scale = 1
        self.setVisible(False)
        self.setZValue(1)
        self.setOpacity(0.5)

    def display(self, mask, zoom_scale):
        self.mask = mask
        self.zoom_scale = zoom_scale
        self._update_mask()
        self.setScale(self.zoom_scale)
        self.setVisible(True)

    def _update_mask(self):
        image = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)  # 转换图像通道
        image_width = image.shape[1]
        image_height = image.shape[0]
        frame = QImage(image, image_width, image_height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.setPixmap(pix)

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self._draw_circle(event.pos())

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self._draw_circle(event.pos())

    def _draw_circle(self, center):
        cv2.circle(self.mask, (int(center.x()), int(center.y())), 10, (255, 255, 255), -1)
        self._update_mask()
