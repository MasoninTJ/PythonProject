import cv2
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsScene, QFileDialog, QGraphicsRectItem, QGraphicsItem

"""
重构PyQt5中QGraphicsView控件，用于显示opencv中的图片，支持放大缩小，平移，右键保存
"""


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent=parent)
        self._scene = QGraphicsScene(self)
        self._zoom = 0  # 图片缩放量
        self._empty = True  # 是否包含图片
        self._image = QGraphicsPixmapItem()  # 设置图片
        self._roi = QGraphicsRectItem()  # 设置ROI
        self._roi.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)    # 设置ROI可以选择并拖动

        self._roi.setRect(100, 30, 100, 30)

        self._scene.addItem(self._image)
        self._scene.addItem(self._roi)

        self.setScene(self._scene)
        self.setAlignment(Qt.AlignCenter)  # 居中显示
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖动
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(640, 480)

        self.origin = None
        self.changeRubberBand = True

    def contextMenuEvent(self, event):  # 右键菜单事件
        if not self.has_photo():
            return
        # 将右键菜单的功能暂时注释掉，后续有需要再说，优先实现右键添加ROI功能
        # menu = QMenu()  # 右键菜单栏
        # save_action = QAction('另存为', self)  # 新建一个保存事件
        # save_action.triggered.connect(self.save_current)  # 绑定事件
        # menu.addAction(save_action)  # 将保存事件加入右键彩蛋
        # menu.exec(QCursor.pos())  # 在鼠标的位置进行显示

    def save_current(self):
        """
        保存当前图像
        @return:
        """
        file_name = QFileDialog.getSaveFileName(self, '另存为', './', 'Image files(*.jpg *.gif *.png)')[0]
        print(file_name)
        if file_name:
            self._image.pixmap().save(file_name)

    def get_image(self):
        """
        当前QPixmap转化为QImage
        @return:
        """
        if self.has_photo():
            return self._image.pixmap().toImage()

    def has_photo(self):
        """
        判断是否包含图像
        @return:
        """
        return not self._empty

    def change_image(self, img):
        """
        修改图片
        @param img:
        @return:
        """
        self.update_image(img)
        self.fitInView()

    def update_image(self, img):
        """
        更新图像
        @param img:
        @return:
        """
        self._empty = False
        self._image.setPixmap(img_to_pixmap(img))

    def fitInView(self, rect=QRectF(), scale=True):
        """
        将适配图像到窗口
        @param rect:
        @param scale:
        @return:
        """
        rect = QRectF(self._image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                view_rect = self.viewport().rect()
                scene_rect = self.transform().mapRect(rect)
                factor = min(view_rect.width() / scene_rect.width(), view_rect.height() / scene_rect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def wheelEvent(self, event):
        """
        滚轮
        @param event:
        @return:
        """
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0


def img_to_pixmap(img_cv):
    """
    将cv图片转化为QPixmap
    @param img_cv: opencv 中的图像
    @return:
    """
    if len(img_cv.shape) == 3:  # bgr图像
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = img_cv.shape
        image = QImage(img_cv, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image)
    elif len(img_cv.shape) == 2:  # 灰度图
        h, w = img_cv.shape
        image = QImage(img_cv, w, h, QImage.Format_Grayscale8)
        return QPixmap.fromImage(image)
    else:
        pass


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    graphics = GraphicsView()
    graphics.show()
    sys.exit(app.exec_())
