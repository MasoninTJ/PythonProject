from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem


class GraphicsRectItem(QGraphicsRectItem):
    """
    自定义矩形图元
    """
    def __init__(self):
        super(GraphicsRectItem, self).__init__()
        self.setPen(QPen(Qt.green))
        self.setRect(QRectF(0, 0, 100, 100))
        self.setPos(0, 0)

        self.left_button_down = False
        self.start_pos = QPoint()

    def wheelEvent(self, event):
        """
        滚轮缩放，每次缩放5个像素
        @param event:
        @return:
        """
        self.setCursor(Qt.CrossCursor)
        if event.delta() > 0:
            temp = self.rect()
            temp.adjust(-5, -5, 5, 5)
            self.setRect(temp)
        elif event.delta() < 0:
            if self.rect().height() > 20 and self.rect().width() > 20:
                temp = self.rect()
                temp.adjust(5, 5, -5, -5)
                self.setRect(temp)

    def mousePressEvent(self, event) -> None:
        """
        坐标左键按下，记录起始位置
        @param event:
        @return:
        """
        self.left_button_down = True
        self.start_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        """
        鼠标移动，计算与起始位置的相对偏移，这里使用了简化运算
        @param event:
        @return:
        """
        if self.left_button_down:
            self.setPos(event.scenePos() - self.start_pos)

    def mouseReleaseEvent(self, event) -> None:
        """
        鼠标左键抬起
        @param event:
        @return:
        """
        self.left_button_down = False


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    view = QGraphicsView()
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 500, 500)

    # 实例化对象
    rect1 = GraphicsRectItem()
    scene.addItem(rect1)

    view.setScene(scene)
    view.show()

    sys.exit(app.exec_())
