# -*- coding: utf-8 -*-
import math
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2



#定义图片显示函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)#等待时间
    cv2.destroyAllWindows()

#qt图片转换为opencv    (qimg = self.label.pixmap().toImage())
def qtpixmap_to_cvimg(qimg):
    ptr = qimg.constBits()
    ptr.setsize(qimg.byteCount())
    mat = np.array(ptr).reshape(qimg.height(), qimg.width())
    return mat

# # opencv图片转换为qt
def cvimg_to_qtimg(cvimg):
    height, width, channel = cvimg.shape
    cvimg1 = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cvimg2 = QImage(cvimg1.data, width, height, width * channel, QImage.Format_RGB888)
    return cvimg2

def refreshShow(cvimg):
    # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
    height, width, channel = cvimg.shape
    bytesPerLine = 3 * width
    cvimg2 = QImage(cvimg.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()
    return cvimg2

#继承QMainWindow基类
class MenuDemo(QMainWindow):
    #初始化MenuDemo子类
    def __init__(self, parent=None):
        super(MenuDemo, self).__init__(parent)
        self.setWindowTitle("图像处理")

        self.resize(800, 800)
        self.center()


        #添加文件对象
        bar = self.menuBar()
        file = bar.addMenu("文件")
        #在文件对象下
        load = QAction("载入", self)
        file.addAction(load)
        #file.triggered[QAction].connect(self.processtrigger)  #设置退出菜单属性
        load.triggered.connect(self.getfile)  # 设置文件载入菜单属性
        #先建立保存操作对象，设置快捷关联，在添加到文件菜单下
        save = QAction("保存", self)
        save.setShortcut("Ctrl+s")
        file.addAction(save)
        #先建立对象在添加到菜单
        quit = QAction("退出", self)
        file.addAction(quit)
        quit.triggered.connect(self.fun_Exit)  #设置退出菜单属性
        #编辑菜单
        edit = bar.addMenu("编辑")
        self.befthre = QAction("预处理阈值设置", self)
        edit.addAction(self.befthre)
        self.judthre=QAction("判断阈值设置", self)
        edit.addAction(self.judthre)

        #工具栏
        tb = self.addToolBar("File")
        new = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/new.png"), "新建", self)
        tb.addAction(new)
        open = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/open.png"), "打开", self)
        tb.addAction(open)
        open.triggered.connect(self.getfile)
        save = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/save.png"), "保存", self)
        tb.addAction(save)
        open.triggered.connect(self.getfile)


        #self.dockwin = QDockWidget("文件", self)

        #self.btnm = QPushButton('btn1')


        # self.dockwin.setWidget(self.btnm)
        #
        # self.dockwin.setFloating(False)
        # self.setCentralWidget(QTextEdit())
        #
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dockwin)





        self.label = QLabel(self)
        self.label.setText("3通道图像")
        self.label.setFixedSize(500, 450)
        self.label.move(0, 60)
        self.label.setStyleSheet("QLabel{background:yellow;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )


        self.btn1 = QPushButton(self)
        self.btn1.setText("打开图片")
        self.btn1.move(510, 100)
        self.btn1.clicked.connect(self.getfile)

        self.btn7 = QPushButton(self)
        self.btn7.setText("计算区域设置")
        self.btn7.move(510, 150)
        #self.btn7.clicked.connect(self.resultjud)

        self.btn8 = QPushButton(self)
        self.btn8.setText("载入计算区域")
        self.btn8.move(510, 200)
        self.btn8.clicked.connect(self.Loadarea)


        self.btn2 = QPushButton(self)
        self.btn2.setText("高斯滤波")
        self.btn2.move(510, 250)
        self.btn2.clicked.connect(self.gaussianBlur)

        self.btn3 = QPushButton(self)
        self.btn3.setText("灰度处理")
        self.btn3.move(510, 300)
        self.btn3.clicked.connect(self.imggray)

        self.btn4 = QPushButton(self)
        self.btn4.setText("轮廓提取")
        self.btn4.move(510, 350)
        self.btn4.clicked.connect(self.contoursfun)

        self.btn5 = QPushButton(self)
        self.btn5.setText("图像计算")
        self.btn5.move(510, 400)
        self.btn5.clicked.connect(self.caldata)

        self.btn6 = QPushButton(self)
        self.btn6.setText("结果判断")
        self.btn6.move(510, 450)
        self.btn6.clicked.connect(self.resultjud)



        self.text_browser = QTextBrowser(self)  # 实例化一个QTextBrowser对象
        self.text_browser.setText('\n'+'初始显示')   # 设置编辑框初始化时显示的文本
        self.text_browser.setFixedSize(500, 200)
        self.text_browser.move(0, 550)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))



        #退出菜单响应方法
    def fun_Exit(self):
        response_quit=QApplication.instance()
        response_quit.quit()

    # 文件载入菜单响应方法
    def getfile(self):
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open Image', './__data', '*.png *.jpg *.bmp')
        if fileName == '':
            return
        # 采用opencv函数读取数据
        self.img = cv2.imread(fileName,-1)

        if self.img.size == 1:
            return
        self.refreshShow3()



    #更新图片函数三通道
    def refreshShow3(self):
        height, width, channel = self.img.shape
        bytesPerLine = channel * width
        self.qImg = QImage(self.img.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()
        # 将Qimage显示出来
        self.label.setPixmap(QPixmap.fromImage(self.qImg).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    # 更新图片函数双通道
    def refreshShow2(self):
        rows, columns = self.img.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(self.img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label.setPixmap(QPixmap.fromImage(QImg).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def Loadarea(self):
        print('topY=',topY,'\ntopX',topX,'\nimgheight=',imgheight,'\nimgwidth=',imgwidth)


    #高斯滤波
    def gaussianBlur(self):

        if self.img.size == 1:
            return
        # 对图像做高斯滤波
        self.img= cv2.GaussianBlur(self.img, (3, 3), 1)
        #self.img = cv2.blur(self.img, (5, 5))
        self.refreshShow3()

    #灰度处理
    def imggray(self):
        #self.img = cv2.blur(self.img1, (5, 5))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.refreshShow2()

    #轮廓提取
    def contoursfun(self):
        global res1, contours
        # 边缘检测
        cannypic = cv2.Canny(self.img, 50, 150)
        # 轮廓提取和绘制
        contours, hierachy = cv2.findContours(cannypic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 传入绘制图像，轮廓轮廓索引，颜色模式，线条厚度 # 注意需要copy,要不原图会变。。。
        draw_img = self.img.copy()
        self.img = cv2.drawContours(draw_img, contours, -1, (255, 255, 255), 1)
        self.refreshShow2()

    # 计算缺陷面积，周长，与最小灰度值
    def caldata (self):
        global dpi
        dpi = 96  # 图片的dpi，查看图片属性可得
        global ares_list, length_list, depth_list
        ares_list = []
        length_list = []
        depth_list = []
        count = 0  # 轮廓个数
        margin = 0  # 裁剪边距
        ave = 255
        all_ares = 0  # 总面积
        all_length = 0  # 总周长
        h = self.img.shape[0]  # 图像的高度
        w = self.img.shape[1]  # 图像的宽度
        img_white = np.zeros([h, w, 1], np.uint8)  # 三维图像
        img_white[:, :, 0] = np.ones([h, w]) * 255  # 空白图像
        # 画轮廓
        for i, contour in enumerate(contours):
            cv2.drawContours(self.img, contours, i, (0, 255, 0), 1)  # 在原图中画轮廓
            cv2.drawContours(img_white, contours, i, (0, 0, 0), -1)  # 在空白图中画轮廓
            gray = cv2.bitwise_or(self.img, img_white)
        # 正式计算
        for i, contour in enumerate(contours):

            # 周长面积
            ares1 = cv2.contourArea(contour)  # 计算包围形状的面积
            ares = ares1 / (dpi ** 2) * 25.4 * 25.4
            ares_list.insert(i, ares)
            length1 = cv2.arcLength(contour, True)  # 计算包围形状的周长
            length = length1 / dpi * 25.4
            length_list.insert(i, length)
            print('第' + str(i) + '个轮廓面积=' + str(ares))
            print('第' + str(i) + '个轮廓周长=' + str(length))
            all_ares = all_ares + ares  # 总面积
            all_length = all_length + length  # 总周长

            # 灰度值
            rect = cv2.minAreaRect(contour)  # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            box = np.int0(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
            h, w = self.img.shape[:2]  # 原图像的高和宽
            rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1  # 最小外接矩形的宽和高
            if rect_w <= rect_h:
                x, y = box[1][0], box[1][1]  # 旋转中心
                M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
                rotated_image = cv2.warpAffine(gray, M2, (w * 2, h * 2))
                rotated_canvas = rotated_image[y - margin:y + rect_h + margin + 1, x - margin:x + rect_w + margin + 1]
                pixel_data = np.array(rotated_canvas)
            else:
                x, y = box[2][0], box[2][1]  # 旋转中心
                M2 = cv2.getRotationMatrix2D((x, y), rect[2] + 90, 1)
                rotated_image = cv2.warpAffine(gray, M2, (w * 2, h * 2))
                rotated_canvas = rotated_image[y - margin:y + rect_w + margin + 1, x - margin:x + rect_h + margin + 1]
            # cv2.imwrite("D:/python-data/{}.jpg".format(count), rotated_canvas)
            # cv_show("rotated_canvas", rotated_canvas)
            # hist = cv2.calcHist([rotated_canvas], [0], None, [256], [0, 256])
            # 提取像素值,进而得到平均灰度值
            aspect_ratio = max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])
            if aspect_ratio < 100:
                pixel_data = np.array(rotated_canvas)
                n = np.array(rotated_canvas, dtype=object).shape
                # print("Contour defect #{}".format(count))
                # print("pixel_data=", pixel_data)
                # 创建access_pixels函数
                height = rotated_canvas.shape[0]
                width = rotated_canvas.shape[1]
                # channels =rotated_canvas.shape[2]
                # print("weight:%s,height:%s" % (width, height))
                number = 0
                for row in range(height):  # 遍历像素点
                    for col in range(width):
                        # for channel in range(channels):
                        pv = rotated_canvas[row, col]
                        # if pv <200&pv==200:
                        #     break
                        if pv >= 144:
                            pv = 0
                            number = number + 1
                            rotated_canvas[row, col] = pv
                            new = np.array(rotated_canvas)
                            new_data = np.sum(new)
                            aveb = new_data / (new.shape[0] * new.shape[1] - number)
                            ave = min(ave, aveb)
                            # 深度和灰度关系公式
                            depth = 10000 * math.exp(-0.025 * ave)
                depth_list.insert(i, depth)
                print('第' + str(i) + '个轮廓深度=' + str(depth))
                text='\n'
                text =text+'第' + str(i) + '个轮廓面积=' + str(ares)+'\n'+'第' + str(i) + '个轮廓周长=' + str(length) + "\n"+'第' + str(i) + '个轮廓深度=' + str(depth) + "\n"
                # print(text)
                self.text_browser.insertPlainText(text)
                # text.insert(END, '第' + str(i) + '个轮廓面积=' + str(ares) + "\n")
                # text.insert(END, '第' + str(i) + '个轮廓周长=' + str(length) + "\n")
                # text.insert(END, '第' + str(i) + '个轮廓深度=' + str(depth) + "\n")


    # 缺陷度判断
    def resultjud(self):
        # 判断阈值设置
        # 阈值1：面积判断  thr_val1(面积最小值)
        # 阈值2：周长判断  thr_val2（周长最小值）
        # 阈值3：深度判断  thr_val3（深度最小值）
        # 阈值4：总体判断  thr_all
        #print(ares_list, "\n", length_list, "\n", depth_list)  # 三个数据列表
        # 判断思路：
        # 1.先对每一个单一缺陷的单一形状特征值进行判断，超过阈值的直接判断整个图像为缺陷，
        # 2.如果所有缺陷的三个形状特征值都在阈值之下，则对每个缺陷的三个形状特征值进行加权处理得到总特征值W_form
        # 3.对每个缺陷的总特征值进行判断，超过阈值的直接判断为图像缺陷，否则为图像通过
        # 判断加权公式
        # 加权系数

        if 'A' not in dir():
            return

        w1 = 0.35
        w2 = 0.2
        w3 = 0.45
        #   f= 0.35 * x1 + 0.2 * x2 + 0.45 * x3
        # 总的特征值
        thr_all1 = [w1 * ares_list[i] + w2 * length_list[i] + w3 * depth_list[i] for i in range(0, len(ares_list))]

        if (max(ares_list) > A):
            a = "图像面积缺陷"
        elif (max(length_list) > L):
            a = "图像周长缺陷(面积通过)"
        elif (max(depth_list) > D):
            a = "图像深度缺陷(面积，周长通过)"
        elif (max(thr_all1) > ALL):
            a = "图像整体缺陷(面积，周长，深度通过)"
        else:
            a = "图像通过"
        result="判断结果：" + str(a)
        self.text_browser.insertPlainText(result)



    # 关闭窗口的时候,触发了QCloseEvent，需要重写closeEvent()事件处理程序，这样就可以弹出是否退出的确认窗口
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "退出程序","确定退出图像处理？",QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
        # 判断按钮的选择
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class Computdo(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('计算区域设置')
        self.setGeometry(550, 50, 800, 1000)

        # 全局部件（注意参数 self），用于"承载"全局布局
        wwg = QWidget(self)

        # 全局布局（注意参数 wwg）
        wl = QHBoxLayout(wwg)

        vlayout = QVBoxLayout()
        glayout = QGridLayout()



        # 局部布局添加部件

        self.labtest = QLabel(self)
        self.labtest.setText("图像区域")
        self.labtest.setFixedSize(500, 400)
        self.labtest.setStyleSheet("QLabel{background:yellow;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        self.datashow = QLabel(self)
        self.datashow.setText("图像数据:")
        self.datashow.setFixedSize(500,18)
        self.datashow.setStyleSheet("QLabel{background:yellow;}")


        self.labtest1 = QLabel(self)
        self.labtest1.setText("计算区域")
        self.labtest1.setFixedSize(500, 400)
        self.labtest1.setStyleSheet("QLabel{background:yellow;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )


        self.btntest1= QPushButton("输入图像")
        self.btntest1.clicked.connect(self.inputimg)

        self.btntest2= QPushButton("载入计算区域")
        self.btntest2.clicked.connect(self.imgchange)

        vlayout.addWidget(self.btntest1)
        vlayout.addWidget(self.labtest)
        vlayout.addWidget(self.datashow)
        vlayout.addWidget(self.btntest2)
        vlayout.addWidget(self.labtest1)



        self.labcoo1 = QLabel("图像顶点X坐标")
        self.sp1 = QSpinBox()
        self.sp1.setRange(0, 1000)
        self.sp1.setValue(0)
        self.sp1.valueChanged.connect(self.valuechange1)

        self.labcoo2 = QLabel("图像顶点Y坐标")
        self.sp2 = QSpinBox()
        self.sp2.setRange(0, 1000)
        self.sp2.setValue(0)
        self.sp2.valueChanged.connect(self.valuechange2)

        self.labcoo3 = QLabel("选择图像宽度")
        self.sp3 = QSpinBox()
        self.sp3.setRange(0, 1000)
        self.sp3.setValue(100)
        self.sp3.valueChanged.connect(self.valuechange3)

        self.labcoo4 = QLabel("选择图像高度")
        self.sp4 = QSpinBox()
        self.sp4.setRange(0, 1000)
        self.sp4.setValue(100)
        self.sp4.valueChanged.connect(self.valuechange4)

        glayout.addWidget(self.labcoo1, 1, 1)
        glayout.addWidget(self.sp1, 1, 2)
        glayout.addWidget(self.labcoo2, 2, 1)
        glayout.addWidget(self.sp2, 2, 2)
        glayout.addWidget(self.labcoo3, 3, 1)
        glayout.addWidget(self.sp3, 3, 2)
        glayout.addWidget(self.labcoo4, 4, 1)
        glayout.addWidget(self.sp4,4, 2)

        self.labshow1=QLabel("图像顶点X坐标=")
        self.labshow1.setFixedSize(120,15)
        self.labshow2=QLabel("图像顶点Y坐标=")
        self.labshow2.setFixedSize(120,15)
        self.labshow3=QLabel("选择图像宽度=")
        self.labshow3.setFixedSize(120,15)
        self.labshow4=QLabel("选择图像高度=")
        self.labshow4.setFixedSize(120,15)


        glayout.addWidget(self.labshow1, 5, 1,1,2)
        glayout.addWidget(self.labshow2, 6, 1,1,2)
        glayout.addWidget(self.labshow3, 7, 1,1,2)
        glayout.addWidget(self.labshow4,8, 1,1,2)

        glayout.setSpacing(10)  # 设置间距

        # 这里向局部布局内添加部件,将他加到全局布局
        wl.addLayout(vlayout)
        wl.addLayout(glayout)

    def valuechange1(self):
        global topX
        topX = self.sp1.value()
        self.labshow1.setText("图像顶点X坐标=" + str(self.sp1.value()))

    def valuechange2(self):
        global topY
        topY = self.sp2.value()
        self.labshow2.setText("图像顶点Y坐标=" + str(self.sp2.value()))

    def valuechange3(self):
        global imgwidth
        imgwidth=self.sp3.value()
        self.labshow3.setText("选择图像宽度=" + str(self.sp3.value()))

    def valuechange4(self):
        global imgheight
        imgheight=self.sp4.value()
        self.labshow4.setText("选择图像高度=" + str(self.sp4.value()))

    def inputimg(self):
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open Image', './__data', '*.png *.jpg *.bmp')
        if fileName == '':
            return
        # 采用opencv函数读取数据
        self.sorimg = cv2.imread(fileName,-1)
        height, width, channel = self.sorimg.shape

        self.datashow.setText("图像数据:" + '高度='+str(height)+'宽度='+str(width))


        if self.sorimg.size == 1:
            return
        self.refreshShow3()

    def imgchange(self):

        print(topX, topY, imgwidth, imgheight)
        #self.sorimgnew=self.sorimg.copy()
        self.sorimg3 = cv2.rectangle(self.sorimg, (topY, topX), (topY+imgheight, topX+imgwidth), (255, 255, 255), 1)
        #cv_show('img',self.sorimg3)
        self.refreshShow3()


        self.sorimg2 = self.sorimg[topY:topY+imgheight,topX:topX+imgwidth]
        #cv_show('img',self.sorimg2)

        self.sorimg2 = cv2.cvtColor(self.sorimg2, cv2.COLOR_BGR2RGB)
        self.QtImg = QImage(self.sorimg2.data,
                                  self.sorimg2.shape[1],
                                  self.sorimg2.shape[0],
                                  self.sorimg2.shape[1] * 3,
                                  QImage.Format_RGB888)
        self.labtest1.setPixmap(QPixmap.fromImage(self.QtImg))


    #更新图片函数三通道
    def refreshShow3(self):
        height, width, channel = self.sorimg.shape
        bytesPerLine = channel * width
        self.qImg = QImage(self.sorimg.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()
        # 将Qimage显示出来
        self.labtest.setPixmap(QPixmap.fromImage(self.qImg).scaled(self.labtest.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))





class NewWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('判断阈值设置')
        self.resize(400, 100)

        #分页表格布局
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        self.addTab(self.tab3, "Tab 3")
        self.addTab(self.tab4, "Tab 4")
        self.tab1UI()
        self.tab2UI()
        self.tab3UI()
        self.tab4UI()

    def valuechange1(self):
        global A
        print('面积阈值=%s' % self.sl1.value())
        self.l1.setText(str(self.sl1.value()))
        A=self.sl1.value()

    def valuechange2(self):
        global L
        print('周长阈值=%s' % self.sl2.value())
        self.l2.setText(str(self.sl2.value()))
        L = self.sl2.value()

    def valuechange3(self):
        global D
        print('深度阈值=%s' % self.sl4.value())
        self.l3.setText(str(self.sl3.value()))
        D = self.sl3.value()

    def valuechange4(self):
        global ALL
        print('总阈值=%s' % self.sl4.value())
        self.l4.setText(str(self.sl4.value()))
        ALL = self.sl4.value()

    def tab1UI(self):
        layout = QFormLayout()
        self.sl1 = QSlider(Qt.Horizontal)
        self.sl1.setMinimum(0)
        self.sl1.setMaximum(1000)
        self.sl1.setSingleStep(1)
        self.sl1.setValue(500)
        self.sl1.setTickPosition(QSlider.TicksBelow)
        self.sl1.setTickInterval(100)
        self.sl1.sliderReleased.connect(self.valuechange1)
        self.l1 = QLabel()

        layout.addRow("面积阈值", self.sl1)
        layout.addRow("阈值结果", self.l1)
        self.setTabText(0, "面积阈值")
        self.tab1.setLayout(layout)

    def tab2UI(self):
        layout = QFormLayout()
        self.sl2 = QSlider(Qt.Horizontal)
        self.sl2.setMinimum(0)
        self.sl2.setMaximum(1000)
        self.sl2.setSingleStep(1)
        self.sl2.setValue(500)
        self.sl2.setTickPosition(QSlider.TicksBelow)
        self.sl2.setTickInterval(100)
        self.sl2.sliderReleased.connect(self.valuechange2)
        self.l2 = QLabel()

        layout.addRow("周长阈值", self.sl2)
        layout.addRow("阈值结果", self.l2)

        self.setTabText(1, "周长阈值")
        self.tab2.setLayout(layout)

    def tab3UI(self):
        layout = QFormLayout()
        self.sl3 = QSlider(Qt.Horizontal)
        self.sl3.setMinimum(0)
        self.sl3.setMaximum(1000)
        self.sl3.setSingleStep(1)
        self.sl3.setValue(500)
        self.sl3.setTickPosition(QSlider.TicksBelow)
        self.sl3.setTickInterval(100)
        self.sl3.sliderReleased.connect(self.valuechange3)
        self.l3 = QLabel()

        layout.addRow("深度阈值", self.sl3)
        layout.addRow("阈值结果", self.l3)

        self.setTabText(2, "深度阈值")
        self.tab3.setLayout(layout)

    def tab4UI(self):
        layout = QFormLayout()
        self.sl4 = QSlider(Qt.Horizontal)
        self.sl4.setMinimum(0)
        self.sl4.setMaximum(1000)
        self.sl4.setSingleStep(1)
        self.sl4.setValue(500)
        self.sl4.setTickPosition(QSlider.TicksBelow)
        self.sl4.setTickInterval(100)
        self.sl4.sliderReleased.connect(self.valuechange4)
        self.l4 = QLabel()

        layout.addRow("总阈值", self.sl4)
        layout.addRow("阈值结果", self.l4)

        self.setTabText(3, "总阈值")
        self.tab4.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("D:/python_data/QtUI_data/Chapter04/images/cartoon3.ico"))  # 窗口图标设置
    demo = MenuDemo()
    newWin = NewWindow()
    compu=Computdo()
    demo.show()
    demo.judthre.triggered.connect(newWin.show)
    demo.btn7.clicked.connect(compu.show)
    sys.exit(app.exec_())