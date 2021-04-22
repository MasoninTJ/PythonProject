# -*- coding: utf-8 -*-
import math
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2
from PyQt5.uic.properties import QtCore

#图像标签类
class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    sendmsg2 = pyqtSignal(int, int,int,int)
    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
    # 绘制事件
    def paintEvent(self, event):
        super(MyLabel,self).paintEvent(event)
        painter = QPainter(self)
        painter.begin(self)
        self.rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(self.rect)
        #x,y坐标，矩形宽度，矩形高度
        #print(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        x=self.x0
        y=self.y0
        w=abs(self.x1 - self.x0)
        h=abs(self.y1 - self.y0)
        # 自定义单击信号
        #self.sendmsg2.emit(self.x0, self.y0,abs(self.x1 - self.x0),abs(self.y1 - self.y0))
        self.sendmsg2.emit(x, y, w, h)
        painter.end()


#定义图片显示函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)#等待时间
    cv2.destroyAllWindows()

# 继承QMainWindow基类
class MenuDemo(QMainWindow):
    # 初始化MenuDemo子类
    def __init__(self, parent=None):
        super(MenuDemo, self).__init__(parent)
        self.setWindowTitle("图像处理")
        # 宽×高
        self.resize(1400, 900)
        # 最小窗口尺寸
        self.setMinimumSize(1400,900)
        #self.setMaximumSize(2000, 1000)  # 最大窗口尺寸
        self.center()

        ################ 菜单栏#################
        bar = self.menuBar()
        file = bar.addMenu("文件")
        # 在文件对象下
        load = QAction("载入", self)
        file.addAction(load)
        load.triggered.connect(self.getfile)  # 设置文件载入菜单属性
        # 先建立保存操作对象，设置快捷关联，在添加到文件菜单下
        save = QAction("保存", self)
        save.setShortcut("Ctrl+s")
        file.addAction(save)
        # 先建立对象在添加到菜单
        quit = QAction("退出", self)
        file.addAction(quit)
        quit.triggered.connect(self.fun_Exit)  # 设置退出菜单属性
        # 编辑菜单
        edit = bar.addMenu("编辑")
        self.befthre = QAction("预处理阈值设置", self)
        edit.addAction(self.befthre)
        self.judthre = QAction("判断阈值设置", self)
        edit.addAction(self.judthre)

        ############# 工具栏################
        tb = self.addToolBar("File")
        new = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/new.png"), "新建", self)
        tb.addAction(new)
        open = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/open.png"), "打开", self)
        tb.addAction(open)
        open.triggered.connect(self.getfile)
        save = QAction(QIcon("D:/python_data/QtUI_data/Chapter04/images/save.png"), "保存", self)
        tb.addAction(save)

        ############# 状态栏  ###############
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        #self.statusBar.showMessage(" 菜单选项被点击了", 5000)

        ############主窗口控件####################
        #全局布局
        alllayout=QHBoxLayout()

        #局部布局
        # 右垂直布局
        vlayout1=QVBoxLayout()

        v1_hlayout1=QHBoxLayout() ##右上水平布局
        v1_h1_hlayout1=QVBoxLayout()###右上右垂直布局
        v1_h1_hlayout2 = QVBoxLayout()###右上左垂直布局



        # 左垂直布局
        vlayout2 = QVBoxLayout()

        #图像显示区
        #图像标签
        self.lab1 = MyLabel(self)  # 重定义的label
        self.lab1.setText('图像显示区')
        self.lab1.setAlignment(Qt.AlignLeft)
        self.lab1.setAlignment(Qt.AlignTop)


        self.lab1.setFixedSize(800, 600)
        self.lab1.setStyleSheet("QLabel{background:yellow;}")
        #self.lab1.setScaledContents(True)  # 让图片自适应label大小

        #测试按钮
        self.btn1 = QPushButton("图像自适应标签大小")
        self.btn1.clicked.connect(self.lab1reshow)
        self.btn2 = QPushButton("图像清除并还原初始设置")
        self.btn2.clicked.connect(self.clearimg)


        self.lab2=QLabel('流程编辑区')
        self.lab2.setFixedSize(300, 20)
        self.lab2.setStyleSheet("QLabel{background:yellow;}")
        self.btn5 = QPushButton("流程清空")
        self.btn5.clicked.connect(self.listwidget_clear)
        self.btn6 = QPushButton("统一运行")
        self.btn6.clicked.connect(self.btn6_func)
        self.lab_set=QLabel('功能参数设置区')
        self.lab_set.setFixedSize(300, 20)
        self.lab_set.setStyleSheet("QLabel{background:yellow;}")


        ##########功能设置界面########################

        ##################第0个功能设置界面框架#####################
        self.frame0 = QFrame()
        self.frame0.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame0.setMinimumHeight(285)
        #框架内布局
        self.frame0Layout = QVBoxLayout(self.frame0)

        #控件设置
        self.frame0_lab=QLabel('读入图像区域设置')
        self.frame0_lab.setStyleSheet("QLabel{background:yellow;}")
        self.frame0_lab.setAlignment(Qt.AlignCenter)
        self.frame0_lab.setFixedSize(290, 20)
        # 嵌套网格布局
        self.gridlayout_frame0 = QGridLayout()

        #表单布局内控件
        # x坐标
        self.xslider = QSlider(Qt.Horizontal)
        self.xslider.setRange(0,800)
        self.xslider.setTickPosition(QSlider.TicksBelow)
        self.xslider.setTickInterval(50)
        self.xslider.valueChanged.connect(self.Pagerange)
        # y坐标
        self.yslider = QSlider(Qt.Horizontal)
        self.yslider.setRange(0, 800)
        self.yslider.setTickPosition(QSlider.TicksBelow)
        self.yslider.setTickInterval(50)
        self.yslider.valueChanged.connect(self.Pagerange)
        # 区域宽度
        self.wslider = QSlider(Qt.Horizontal)
        self.wslider.setRange(0, 800)
        self.wslider.setTickPosition(QSlider.TicksBelow)
        self.wslider.setTickInterval(50)
        self.wslider.valueChanged.connect(self.Pagerange)
        # 区域高度
        self.hslider = QSlider(Qt.Horizontal)
        self.hslider.setRange(0, 800)
        self.hslider.setTickPosition(QSlider.TicksBelow)
        self.hslider.setTickInterval(50)
        self.hslider.valueChanged.connect(self.Pagerange)

        self.gridlab1 = QLabel('0')
        self.gridlab1.setStyleSheet("QLabel{background:yellow;}")
        self.gridlab1.setMinimumWidth(20)
        self.gridlab2 = QLabel('0')
        self.gridlab2.setStyleSheet("QLabel{background:yellow;}")
        self.gridlab2.setMinimumWidth(20)
        self.gridlab3 = QLabel('0')
        self.gridlab3.setStyleSheet("QLabel{background:yellow;}")
        self.gridlab3.setMinimumWidth(20)
        self.gridlab4 = QLabel('0')
        self.gridlab4.setStyleSheet("QLabel{background:yellow;}")
        self.gridlab4.setMinimumWidth(20)

        #确认lab1图像截图
        self.okbtn = QPushButton('确认区域')
        self.okbtn.clicked.connect(self.okbtn_fun)
        self.reshowbtn = QPushButton('重置区域')
        self.reshowbtn.clicked.connect(self.reshowbtn_fun)

        #添加控件到网格布局
        self.gridlayout_frame0.addWidget(QLabel('x坐标值：'),0,0)
        self.gridlayout_frame0.addWidget(QLabel('y坐标值：'), 1,0)
        self.gridlayout_frame0.addWidget(QLabel('区域高度：'), 2, 0)
        self.gridlayout_frame0.addWidget(QLabel('区域宽度：'), 3, 0)
        self.gridlayout_frame0.addWidget(self.xslider, 0, 1)
        self.gridlayout_frame0.addWidget(self.yslider, 1, 1)
        self.gridlayout_frame0.addWidget(self.wslider, 2, 1)
        self.gridlayout_frame0.addWidget(self.hslider, 3, 1)
        self.gridlayout_frame0.addWidget(self.gridlab1, 0, 2)
        self.gridlayout_frame0.addWidget(self.gridlab2, 1, 2)
        self.gridlayout_frame0.addWidget(self.gridlab3, 2, 2)
        self.gridlayout_frame0.addWidget(self.gridlab4, 3, 2)
        self.gridlayout_frame0.addWidget(self.okbtn, 5, 0)
        self.gridlayout_frame0.addWidget(self.reshowbtn, 5, 1,1,2)
        self.gridlayout_frame0.setHorizontalSpacing(0)
        self.gridlayout_frame0.setVerticalSpacing(20)

        # 控件添加到布局
        self.frame0Layout.addWidget(self.frame0_lab)
        self.frame0Layout.addLayout(self.gridlayout_frame0)
        self.frame0Layout.addStretch(0)


        # # 参数数值设置滑动条QSlider
        # self.gsSlider=QSlider(Qt.Horizontal)
        # #最小值
        # self.gsSlider.setMinimum(0)
        # #最大值
        # self.gsSlider.setMaximum(255)
        # #设定当前值
        # self.gsSlider.setValue(131)
        # #添加刻度
        # self.gsSlider.setTickPosition(QSlider.TicksBelow)
        # #设置信号连接
        # self.gsSlider.valueChanged.connect(self.Threshold)
        #控件添加到布局
        # self.frame0Layout.addWidget(self.gsSlider)

        ##################第1个功能设置界面框架(滤波处理)################
        self.frame1 = QFrame()
        self.frame1.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame1.setMinimumHeight(290)
        # 框架内布局
        self.frame1Layout = QVBoxLayout(self.frame1)
        self.lineEdit_password1 = QLineEdit()
        self.lineEdit_password1.setPlaceholderText("参数设置二")
        self.pushButton_enter1 = QPushButton()
        self.pushButton_enter1.setText("确认")
        #添加控件到布局
        self.frame1Layout.addWidget(self.lineEdit_password1)
        self.frame1Layout.addWidget(self.pushButton_enter1)
        self.frame1.setVisible(False)

        ##################第2个功能设置界面框架(滤波处理)################
        self.frame2 = QFrame()
        self.frame2.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame2.setMinimumHeight(290)
        # 框架内布局
        self.frame2Layout = QVBoxLayout(self.frame2)
        self.lineEdit_password1 = QLineEdit()
        self.lineEdit_password1.setPlaceholderText("参数设置二")
        self.pushButton_enter1 = QPushButton()
        self.pushButton_enter1.setText("确认")
        #添加控件到布局
        self.frame2Layout.addWidget(self.lineEdit_password1)
        self.frame2Layout.addWidget(self.pushButton_enter1)
        self.frame2.setVisible(False)

        ##################第3个功能设置界面框架()
        self.frame3 = QFrame()
        self.frame3.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame3.setMinimumHeight(290)
        # 框架内布局
        self.frame3Layout = QVBoxLayout(self.frame3)
        self.pus3 = QPushButton()
        self.pus3.setText("确认")
        # 添加控件到布局
        self.frame3Layout.addWidget(self.pus3)
        self.frame3.setVisible(False)

        ##################第4个功能设置界面框架()
        self.frame4 = QFrame()
        self.frame4.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame4.setMinimumHeight(290)
        # 框架内布局
        self.frame4Layout = QVBoxLayout(self.frame4)
        self.pus4 = QPushButton()
        self.pus4.setText("确认")
        # 添加控件到布局
        self.frame4Layout.addWidget(self.pus4)
        self.frame4.setVisible(False)

        ##################第5个功能设置界面框架()
        self.frame5 = QFrame()
        self.frame5.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame5.setMinimumHeight(290)
        # 框架内布局
        self.frame5Layout = QVBoxLayout(self.frame5)
        self.pus5 = QPushButton()
        self.pus5.setText("确认")
        # 添加控件到布局
        self.frame5Layout.addWidget(self.pus5)
        self.frame5.setVisible(False)

        ##################第6个功能设置界面框架()
        self.frame6 = QFrame()
        self.frame6.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame6.setMinimumHeight(290)
        # 框架内布局
        self.frame6Layout = QVBoxLayout(self.frame6)
        self.pus6 = QPushButton()
        self.pus6.setText("确认")
        # 添加控件到布局
        self.frame6Layout.addWidget(self.pus6)
        self.frame6.setVisible(False)

        ##################第7个功能设置界面框架()
        self.frame7 = QFrame()
        self.frame7.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.frame7.setMinimumHeight(290)
        # 框架内布局
        self.frame7Layout = QVBoxLayout(self.frame7)
        self.pus7 = QPushButton()
        self.pus7.setText("确认")
        # 添加控件到布局
        self.frame7Layout.addWidget(self.pus6)
        self.frame7.setVisible(False)

        #变量信息区
        self.framemess = QFrame()
        self.framemess.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.framemess.setMinimumHeight(145)
        # 框架内布局
        self.framemessLayout = QHBoxLayout(self.framemess)
        self.lab3 = QLabel('变量信息区')
        self.lab3.setFixedSize(100, 100)
        self.lab3.setStyleSheet("QLabel{background:yellow;}")
        self.btn9 = QPushButton("测试31")
        self.btn10 = QPushButton("测试32")
        self.btn11 = QPushButton("测试33")
        self.btn12 = QPushButton("测试34")

        #数据显示区
        self.lab4 = QLabel('数据显示区')
        self.lab4.setFixedHeight(20)
        self.lab4.setStyleSheet("QLabel{background:yellow;}")
        #########表单控件切换########
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.taball=QTabWidget()
        self.taball.setMaximumHeight(100)
        self.taball.addTab(self.tab1, "图像文件输入")
        self.taball.addTab(self.tab2, "相机数据输入")
        self.tab1UI()
        self.tab2UI()
        self.btn13 = QPushButton("测试41")
        self.btn14 = QPushButton("测试42")
        self.resultlab=QLabel('结果显示')
        self.resultlab.setFixedHeight(20)
        self.resultlab.setStyleSheet("QLabel{background:yellow;}")
        # 结果数据显示(文本浏览器)
        self.textbrowser = QTextBrowser(self)
        #self.textbrowser.setText('结果显示')  # 设置编辑框初始化时显示的文本
        self.textbrowser.setMinimumHeight(305)



        ####功能字典定义（用于连接功能与方法函数）
        self.dictfun = {'图像载入':'self.getfile'}

        ##########树形流程列表选项#############
        self.tree=QTreeWidget()
        self.tree.setMinimumHeight(280)
        # 设置列数
        self.tree.setColumnCount(1)
        # 设置头的标题
        self.tree.setHeaderLabels(['流程选项'])
        # 第一级节点
        root0 = QTreeWidgetItem(self.tree)
        root0.setText(0, '图像输入')
        # 第二级节点
        zerroot1 = QTreeWidgetItem(root0)
        zerroot1.setText(0, '图像文件输入')
        self.dictfun['图像文件输入']='self.getfile' #插入字典

        # zerroot2 = QTreeWidgetItem(root0)
        # zerroot2.setText(0, '相机输入')
        # zerroot3 = QTreeWidgetItem(root0)
        # zerroot3.setText(0, '相机切换')
        # zerroot4 = QTreeWidgetItem(root0)
        # zerroot4.setText(0, '格式切换')
        #第一级节点
        root1 = QTreeWidgetItem(self.tree)
        root1.setText(0, '图像预处理')
        #第二级节点
        firroot1 = QTreeWidgetItem(root1)
        firroot1.setText(0, '灰度转换')
        self.dictfun['灰度转换'] = 'self.grayImg'  # 插入字典

        firroot2 = QTreeWidgetItem(root1)
        firroot2.setText(0, '全局直方图均衡化')
        self.dictfun['全局直方图均衡化'] = 'self.allhist'  # 插入字典

        firroot3= QTreeWidgetItem(root1)
        firroot3.setText(0, '自适应直方图均衡化')
        self.dictfun['自适应直方图均衡化'] = 'self.limhist'  # 插入字典

        firroot4 = QTreeWidgetItem(root1)
        firroot4.setText(0, '滤波处理')
        self.dictfun['滤波处理'] = 'self.GBlur'  # 插入字典

        # 第一级节点
        root2 = QTreeWidgetItem(self.tree)
        root2.setText(0, '图像分割')
        # 第二级节点
        secroot1 = QTreeWidgetItem(root2)
        secroot1.setText(0,'阈值分割')
        self.dictfun['阈值分割'] = 'self.Threshold'  # 插入字典

        secroot2 = QTreeWidgetItem(root2)
        secroot2.setText(0, '形态学处理')
        self.dictfun['形态学处理'] = 'self.Blackhat'  # 插入字典

        # 第一级节点
        root3 = QTreeWidgetItem(self.tree)
        root3.setText(0, '图像检测')
        # 第二级节点
        thrroot1 = QTreeWidgetItem(root3)
        thrroot1.setText(0, '边缘检测')
        self.dictfun['边缘检测'] = 'self.canny'  # 插入字典

        thrroot2 = QTreeWidgetItem(root3)
        thrroot2.setText(0, '几何形状检测')
        self.dictfun['几何形状检测'] = 'self.houghcircles'  # 插入字典

        # 第一级节点
        root4 = QTreeWidgetItem(self.tree)
        root4.setText(0, '缺陷检测')
        # 第二级节点
        fouroot1 = QTreeWidgetItem(root4)
        fouroot1.setText(0, '轮廓提取')
        self.dictfun['轮廓提取'] = 'self.contoursshow'  # 插入字典

        fouroot2 = QTreeWidgetItem(root4)
        fouroot2.setText(0, '几何特征计算')
        self.dictfun['几何特征计算'] = 'self.caldata'  # 插入字典

        fouroot3 = QTreeWidgetItem(root4)
        fouroot3.setText(0, '缺陷判断')
        self.dictfun['缺陷判断'] = 'self.resultjud'  # 插入字典

        # 第一级节点
        root5 = QTreeWidgetItem(self.tree)
        root5.setText(0, '结果显示保存')
        # 第二级节点
        # fifroot1 = QTreeWidgetItem(root5)
        # fifroot1.setText(0, '判断结果')
        #fifroot2 = QTreeWidgetItem(root5)
        #fifroot2.setText(0, '几何特征保存')

        fifroot3 = QTreeWidgetItem(root5)
        fifroot3.setText(0, '项目保存')
        self.dictfun['项目保存'] = 'self.savegeo'  # 插入字典
        #self.tree.clicked.connect(self.onTreeClicked)
        #允许右键产生子菜单
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.treeWidgetItem_fun)
        #插入列表项目
        self.tree.addTopLevelItem(root0)
        self.tree.addTopLevelItem(root1)
        self.tree.addTopLevelItem(root2)
        self.tree.addTopLevelItem(root3)
        self.tree.addTopLevelItem(root4)
        self.tree.addTopLevelItem(root5)

        #########流程列表编辑##############
        self.listwidget=QListWidget()
        self.listwidget.setMinimumHeight(250)

        #添加Item
        self.listwidget.addItem('图像载入')
        self.listwidget.addItem('灰度转换')
        self.listwidget.addItem('滤波处理')
        self.listwidget.addItem('阈值分割')
        self.listwidget.addItem('边缘检测')
        self.listwidget.addItem('轮廓提取')
        self.listwidget.addItem('几何特征计算')
        self.listwidget.addItem('项目保存')



        # 绑定左键单击响应
        self.listwidget.itemClicked.connect(self.item_click)
        # 允许右键产生子菜单
        self.listwidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listwidget.customContextMenuRequested.connect(self.listwidgetItem_fun)

        #在局部布局中添加控件
        #流程编辑区
        v1_h1_hlayout1.addWidget(self.lab2)
        v1_h1_hlayout1.addWidget(self.btn5)
        v1_h1_hlayout1.addWidget(self.listwidget)
        v1_h1_hlayout1.addWidget(self.btn6)
        v1_h1_hlayout1.addWidget(self.lab_set)
        v1_h1_hlayout1.addWidget(self.frame0)
        v1_h1_hlayout1.addWidget(self.frame1)
        v1_h1_hlayout1.addStretch(0)

        #图像显示区
        v1_h1_hlayout2.addWidget(self.lab1)
        v1_h1_hlayout2.addWidget(self.btn1)
        v1_h1_hlayout2.addWidget(self.btn2)
        v1_h1_hlayout2.addStretch(0)



        #变量信息区
        self.framemessLayout.addWidget(self.lab3)
        self.framemessLayout.addWidget(self.btn9)
        self.framemessLayout.addWidget(self.btn10)
        self.framemessLayout.addWidget(self.btn11)
        self.framemessLayout.addWidget(self.btn12)

        #数据显示区
        vlayout2.addWidget(self.lab4)
        vlayout2.addWidget(self.taball)
        vlayout2.addWidget(self.btn13)
        vlayout2.addWidget(self.btn14)
        vlayout2.addWidget(self.tree)
        vlayout2.addWidget(self.resultlab)
        vlayout2.addWidget(self.textbrowser)
        vlayout2.addStretch(0)

        v1_hlayout1.addLayout(v1_h1_hlayout1)
        v1_hlayout1.addLayout(v1_h1_hlayout2)

        vlayout1.addLayout(v1_hlayout1)
        vlayout1.addWidget(self.framemess)

        #局部布局添加到全局布局
        alllayout.addLayout(vlayout1)
        alllayout.addLayout(vlayout2)


        widget = QWidget()
        widget.setLayout(alllayout)
        self.setCentralWidget(widget)

        ####流程字典定义(用于存放功能操作)
        self.dictall = {'图像载入':'self.getfile'}
        self.dictall['灰度转换'] = 'self.grayImg'  # 插入字典
        self.dictall['滤波处理'] = 'self.GBlur'  # 插入字典
        self.dictall['阈值分割'] = 'self.Threshold'  # 插入字典
        self.dictall['边缘检测'] = 'self.canny'  # 插入字典
        self.dictall['轮廓提取'] = 'self.contoursshow'  # 插入字典
        self.dictall['几何特征计算'] = 'self.caldata'  # 插入字典
        self.dictall['项目保存'] = 'self.savegeo'  # 插入字典

        #图片输入判断标志
        self.test=1
        #lab1图像显示信号连接槽函数
        self.lab1.sendmsg2.connect(self.get2)


    # 自定义lab1响应槽函数
    def get2(self, msg1, msg2,msg3,msg4):
        print(msg1, msg2,msg3,msg4)
        self.xslider.setValue(msg1)
        self.yslider.setValue(msg2)
        self.wslider.setValue(msg3)
        self.hslider.setValue(msg4)


    #设置界面切换
    def btn6_func(self):
        self.frame0.setVisible(False)
        self.frame1.setVisible(True)

    def listwidget_clear(self):
        self.listwidget.clear()
        self.dictall.clear()

    ##点击树列表响应
    def onTreeClicked(self):
        item = self.tree.currentItem()
        print("key=%s" % (item.text(0)))

    # 定义树状列表中item右键界面
    def treeWidgetItem_fun(self, pos):
        item = self.tree.currentItem()
        item1 = self.tree.itemAt(pos)

        if item != None and item1 != None:
            popMenu = QMenu(self.tree)
            Item1=popMenu.addAction('插入开头')
            Item2=popMenu.addAction('插入末尾')
            Item3=popMenu.addAction('索引插入')
            #popMenu.triggered[QAction].connect(self.processtrigger)
            action = popMenu.exec_(self.tree.mapToGlobal(pos))
            if action == Item1:
                if item.text(0) in self.dictfun.keys():
                    # 在列表开头插入Item
                    self.listwidget.insertItem(0, item.text(0))
                else:
                    print(item.text(0))
            elif action == Item2:
                if item.text(0) in self.dictfun.keys():
                    # 在列表尾部插入Item
                    self.listwidget.addItem(item.text(0))
                    self.dictall[item.text(0)] = self.dictfun[item.text(0)]
                    print('功能列表：')
                    print(self.dictall)
                else:
                    print(item.text(0))
            elif action == Item3:
                if item.text(0) in self.dictfun.keys():
                    # print(item.text(0))
                    # 获取选中Item索引
                    index = self.listwidget.currentIndex()
                    num = index.row()
                    self.listwidget.insertItem(num + 1, item.text(0))
                else:
                    print(item.text(0))
            else:
                return
    #列表左键单击响应
    def item_click(self,item):
        print(item.text())
    #列表右键菜单
    def listwidgetItem_fun(self,pos):
        itemlist = self.listwidget.currentItem()
        itemlist1 = self.listwidget.itemAt(pos)
        # 获取选中Item索引
        index = self.listwidget.currentIndex()
        if itemlist != None and itemlist1 != None:
            listMenu = QMenu(self.tree)
            listItem1 = listMenu.addAction('设置')
            listItem2 = listMenu.addAction('运行单项')
            listItem3 = listMenu.addAction('删除项目')
            action = listMenu.exec_(self.listwidget.mapToGlobal(pos))

            if action == listItem1:
                #设置项目
                print(itemlist.text())
            elif action == listItem2:
                #运行项目操作
                print(itemlist.text())
                eval(self.dictall[itemlist.text()])()

            elif action == listItem3:
                num=index.row()
                # 删除选中Item
                self.listwidget.takeItem(num)
            else:
                return

    #表单分页控件设置添加
    def tab1UI(self):
        layout = QFormLayout()
        layout.addRow("图像文件输入", QPushButton('输入图像'))
        layout.addRow("图像格式转换", QPushButton('选择图像'))
        self.taball.setTabText(0, "图像文件输入")
        self.tab1.setLayout(layout)

    def tab2UI(self):
        layout = QFormLayout()
        layout.addRow("相机编辑", QPushButton('相机设置'))
        layout.addRow("文件输入", QPushButton('相机数据输入'))
        self.taball.setTabText(1, "相机数据输入")
        self.tab2.setLayout(layout)

    #主窗口居中
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2)-40)

    # 文件载入菜单响应方法
    def getfile(self):
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open Image', './__data', '*.png *.jpg *.bmp')
        if fileName == '':
            return
        # 采用opencv函数读取数据
        self.img = cv2.imread(fileName,-1)
        if self.img.size == 1:
            return
        else:
            print('图片已输入')
            self.draw_img = self.img.copy()
            self.refreshow()
            self.test+=1

    #统一读入图片通道
    def refreshow(self):            #统一读入图片通道
        num_channel = len(self.img.shape)
        print('图像通道数：',num_channel)
        #三通道图片
        if num_channel == 3:
            height, width, channel = self.img.shape
            bytesPerLine = channel * width
            self.qImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        #二通道灰度图
        else:
            rows, columns = self.img.shape
            bytesPerLine = columns
            # 灰度图是单通道，所以需要用Format_Indexed8
            self.qImg = QImage(self.img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        # 将Qimage显示出来
        #self.lab1.setPixmap(QPixmap.fromImage(self.qImg).scaled(self.lab1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lab1.setPixmap(QPixmap.fromImage(self.qImg))

    # 关闭窗口的时候,触发了QCloseEvent，需要重写closeEvent()事件处理程序，这样就可以弹出是否退出的确认窗口
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "退出程序", "确定退出图像处理？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断按钮的选择
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 退出菜单响应
    def fun_Exit(self):
        response_quit=QApplication.instance()
        response_quit.quit()

    # 预处理设置响应
    def befthre(self):
        pass

    ###############功能方法######################
    # 灰度图
    def grayImg(self):
        # global img_new
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # img_new = self.img.copy()
        # self.refreshow()
        if self.test == 1:
            print('图片未载入')
        else:
            global img_new
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            img_new = self.img.copy()
            self.refreshow()


    # 全局灰度直方图均衡化
    def allhist(self):
        self.img = cv2.equalizeHist(self.img)
        self.refreshow()

    # 限制对比度的自适应直方图均衡化
    def limhist(self):
        limhist1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img = limhist1.apply(self.img)
        self.refreshow()

    # 2D滤波器
    def d2filter(self):
        kernel = np.ones((5, 5), np.float32) / 25   #卷积核
        self.img = cv2.filter2D(self.img, -1, kernel)
        self.refreshow()

    # 高斯滤波(二维离散卷积核)（高斯核的高和宽（奇数））
    def GBlur(self):
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)  # （5,5）表示的是卷积核大小，0表示的是沿x与y方向上的标准差
        self.refreshow()

    # 均值滤波(二维离散卷积核)
    def meanval(self):
        self.img = cv2.blur(self.img, (3, 5))  # 卷积核大小为3*5, 模板的大小是可以设定的
        self.refreshow()

    # 方框滤波，normalize=1时，表示进行归一化处理，此时图片处理效果与均值滤波相同，如果normalize=0时，表示不进行归一化处理，像素值为周围像素之和，图像更多为白色
    def boxfilter(self):
        self.img = cv2.boxFilter(self.img, -1, (5, 5), normalize=1)
        self.refreshow()

    # 中值滤波(统计学)(中值滤波模板就是用卷积框中像素的中值代替中心值，达到去噪声的目的。这个模板一般用于去除椒盐噪声。卷积核的大小也是个奇数。)
    def medBlur(self):
        self.img = cv2.medianBlur(self.img, 5)  # 中值滤波函数
        self.refreshow()

    # 双边滤波（保持边缘清晰）双边滤波同时使用了空间高斯权重和灰度相似性高斯权重，确保了边界不会被模糊掉。
    def doufilter(self):
        # 9表示的是滤波领域直径，后面的两个数字：空间高斯函数标准差，灰度值相似性标准差
        doufilter = cv2.bilateralFilter(self.img, 9, 80, 80)
        self.refreshow()

    #阈值分割
    ##全局阈值分割(ret=阈值)
    # cv2.THRESH_BINARY（黑白二值）
    # cv2.THRESH_BINARY_INV（黑白二值翻转）
    # cv2.THRESH_TRUNC（得到额图像为多像素值）
    # cv2.THRESH_TOZERO（当像素高于阈值时像素设置为自己提供的像素值，低于阈值时不作处理）
    # cv2.THRESH_TOZERO_INV（当像素低于阈值时设置为自己提供的像素值，高于阈值时不作处理）
    def Threshold(self):
        thrvalue=self.gsSlider.value()
        print(thrvalue)
        ret, self.img = cv2.threshold(img_new, thrvalue, 255, cv2.THRESH_BINARY)
        self.refreshow()

    # 自适应阈值
    def autothreshold(self):
        # 第一个参数为原始图像矩阵
        # 第二个参数为像素值上限，
        # 第三个是自适应方法（adaptive method）：cv2.ADAPTIVE_THRESH_MEAN_C:领域内均值
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C:领域内像素点加权和，权重为一个高斯窗口
        # 第四个值的赋值方法：只有cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV
        # 第五个Block size：设定领域大小（一个正方形的领域）
        # 第六个参数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
        self.img= cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        self.refreshow()

    # Otsu's阈值（Otsu's非常适合于图像灰度直方图(只有灰度图像才有)具有双峰的情况）
    def Otsthreshold(self):
        ret, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu滤波
        self.refreshow()

    # K-Means聚类阈值分割
    def Kthreshold(self):
        # 获取图像高度、宽度
        rows, cols = self.img.shape[:]
        # 图像二维像素转换为一维
        data = self.img.reshape((rows * cols, 1))
        data = np.float32(data)
        # 定义中心 (type,max_iter,epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # 设置标签
        flags = cv2.KMEANS_RANDOM_CENTERS
        # K-Means聚类 聚集成4类
        compactness, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, flags)
        # 图像转换回uint8二维类型
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        self.img = res.reshape((self.img.shape))
        self.refreshow()

    # 形态学处理
    # 腐蚀（腐蚀可以使目标区域范围“变小”，其实质造成图像的边界收缩，可以用来消除小且无意义的目标物。）
    def corrode(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核/结构元素(getStructuringElement函数也可以构造)
        self.img = cv2.erode(self.img, kernel, iterations=1)  # 腐蚀(iterations：迭代次数)
        self.refreshow()

    # 膨胀（膨胀会使目标区域范围“变大”，将于目标区域接触的背景点合并到该目标物中，使目标边界向外部扩张。
    # 作用就是可以用来填补目标区域中某些空洞以及消除包含在目标区域中的小颗粒噪声。 膨胀也可以用来连接两个分开的物体。）
    def swelld(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.dilate(self.img, kernel, iterations=1)  # 膨胀
        self.refreshow()

    # 开运算
    # （先腐蚀再膨胀)，它被用来去除噪声。cv2.MORPH_OPEN
    def Opening(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)  # 开运算
        self.refreshow()

    # 闭运算
    # （先膨胀再腐蚀）它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
    def Closing(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)  # 闭运算
        self.refreshow()

    # 形态学梯度
    # 前景物体的轮廓
    def Gradient(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度
        self.refreshow()

    # 礼帽
    # 礼帽图像=原始图像-开运算(cv2.MORPH_TOPHAT)
    def Tophat(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, kernel)  # 礼帽
        self.refreshow()

    # 黑帽
    # 黑帽图像=闭运算-原始图像（cv2.MORPH_BLACKHAT）
    def Blackhat(self):
        kernel = np.ones((5, 5), np.uint8)  # 卷积核
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_BLACKHAT, kernel)  # 黑帽
        self.refreshow()

    # Canny算子
    # Canny方法不容易受噪声干扰，能够检测到真正的弱边缘。
    # 优点在于，使用两种不同的阈值分别检测强边缘和弱边缘，并且当弱边缘和强边缘相连时，才将弱边缘包含在输出图像中。
    def canny(self):
        # 高斯滤波降噪
        #self.img = cv2.GaussianBlur(self.img, (3, 3), 0)
        # Canny算子
        self.img = cv2.Canny(self.img, 50, 150)
        self.refreshow()

    #####轮廓检测
    # （cv2.findContours(image, mode, method[, offset])）
    # 寻找一个二值图像的轮廓。注意黑色表示背景，白色表示物体，即在黑色背景里寻找白色物体的轮廓
    # mode:轮廓检索的方式=====
    # cv2.RETR_EXTERNAL:只检索外部轮廓
    # cv2.RETR_LIST: 检测所有轮廓且不建立层次结构
    # cv2.RETR_CCOMP: 检测所有轮廓，建立两级层次结构
    # cv2.RETR_TREE: 检测所有轮廓，建立完整的层次结构
    # method:轮廓近似的方法 =====
    # cv2.CHAIN_APPROX_NONE:存储所有的轮廓点
    # cv2.CHAIN_APPROX_SIMPLE:压缩水平，垂直和对角线段，只留下端点。 例如矩形轮廓可以用4个点编码。
    # cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS:使用Teh-Chini chain近似算法
    # offset:（可选参数）轮廓点的偏移量，格式为tuple,
    # 如（-10，10）表示轮廓点沿X负方向偏移10个像素点，沿Y正方向偏移10个像素点
    # 返回三个值,分别是img, countours（list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示）, hierarchy（每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号）
    ######绘制轮廓
    # cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]])
    # image:需要绘制轮廓的目标图像，注意会改变原图
    # contours:轮廓点，上述函数cv2.findContours()的第一个返回值
    # contourIdx:轮廓的索引，表示绘制第几个轮廓，-1表示绘制所有的轮廓
    # color:绘制轮廓的颜色
    # thickness:（可选参数）轮廓线的宽度，-1表示填充
    # lineType:（可选参数）轮廓线型，包括cv2.LINE_4,cv2.LINE_8（默认）,cv2.LINE_AA,分别表示4邻域线，8领域线，抗锯齿线（可以更好地显示曲线）
    # hierarchy:（可选参数）层级结构，上述函数cv2.findContours()的第二个返回值，配合maxLevel参数使用
    # maxLevel:（可选参数）等于0表示只绘制指定的轮廓，等于1表示绘制指定轮廓及其下一级子轮廓，等于2表示绘制指定轮廓及其所有子轮廓
    # offset:（可选参数）轮廓点的偏移量
    def contoursshow(self):
        global contours
        contours, hierachy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.img = cv2.drawContours(self.draw_img, contours, -1, (0, 255, 0), 1)
        self.refreshow()



    # 图像特征计算
    def caldata(self):
        text=''
        print('计算面积周长')
        dpi = 96  # 图片的dpi，查看图片属性可得
        ares_list = []
        length_list = []
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
            text = text + '第' + str(i) + '个轮廓面积=' + str(ares) + '\n' + '第' + str(i) + '个轮廓周长=' + str(
                length) + "\n"
        self.textbrowser.append(text)

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

    #结果保存
    def savegeo(self):
        print('保存功能运行')
        savefile= QFileDialog.getSaveFileName(self, "save project", "D:/image_process/image_process","所有文件 (*);;文本文件 (*.txt)")
        print(1)
        filelu=savefile[0]
        if filelu == '':
            return
        else:
            with open(file=filelu, mode='a+', encoding='utf-8') as file:
                file.write(self.textbrowser.toPlainText())
            print('已保存！')

    #图片自适应标签大小
    def lab1reshow(self):
        self.lab1.setScaledContents(True)

    #图片清除
    def clearimg(self):
        self.lab1.setPixmap(QPixmap(""))  # 移除label上的图片
        self.lab1.setScaledContents(False)

    #图像输入，区域设置
    def Pagerange(self):
        xsl_value=self.xslider.value()
        ysl_value = self.yslider.value()
        wsl_value = self.wslider.value()
        hsl_value = self.hslider.value()
        self.gridlab1.setText(str(xsl_value))
        self.gridlab2.setText(str(ysl_value))
        self.gridlab3.setText(str(wsl_value))
        self.gridlab4.setText(str(hsl_value))

    #确认响应函数
    def okbtn_fun(self):
        print('计算区域参数：',self.xslider.value(),self.yslider.value(),self.wslider.value(),self.hslider.value())
        X = self.xslider.value()
        Y = self.yslider.value()
        W = self.wslider.value()
        H = self.hslider.value()
        self.img=self.img[Y:Y+H, X:X+W]
        height, width, channel = self.img.shape
        self.new = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.qimg = QImage(self.new.data, width, height, width * channel, QImage.Format_RGB888)
        self.lab1.setPixmap(QPixmap.fromImage(self.qimg))
        # self.lab1.setPixmap(QPixmap.fromImage(self.qImg).scaled(self.lab1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))




    #重置区域响应函数
    def reshowbtn_fun(self):
        self.img=self.draw_img
        self.refreshow()


class NewWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('判断阈值设置')
        self.resize(400, 100)

        # 分页表格布局
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
        A = self.sl1.value()

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
    demo.show()
    demo.judthre.triggered.connect(newWin.show)
    sys.exit(app.exec_())