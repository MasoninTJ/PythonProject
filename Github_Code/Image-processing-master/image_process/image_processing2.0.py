#图像处理脚本：界面部分和处理部分（基于opencv）
from tkinter import *     #导入tkinter界面模块
from tkinter import ttk
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image,ImageTk     #导入图片相关模块
import cv2
from tkinter.filedialog import askopenfilename    #文件打开
from tkinter.filedialog import asksaveasfilename  #文件保存
import numpy as np
import math


#########################界面部分
#########################界面部分
#########################界面部分
#########################界面部分
#居中处理
def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    # print(size)
    root.geometry(size)

# 根窗口
root = Tk()  # 根窗口
root.title("图像处理界面")  # 窗口标题
center_window(root, 1000, 800) #居中处理
root.resizable(0,0)



################################框架
#框架主题
style1 = ttk.Style()
style1.configure("BW.TLabel", foreground="white", background="black")

#建立分布框架：拟采用上二下一的分布方式
frameUpper0= Frame(root,width=500,height=600,relief=GROOVE,borderwidth=5,style="BW.TLabel")
frameUpper1= Frame(root,width=100,height=200,relief=RAISED,borderwidth=5)
frameUpper2= Frame(root,width=100,height=400,relief=GROOVE,borderwidth=5)
frameDown0 = Frame(root,width=1000,height=200,relief=RIDGE,borderwidth=5)


#框架尺寸固定
frameUpper0.pack_propagate(0)
frameUpper1.pack_propagate(0)
frameUpper2.pack_propagate(0)
frameDown0.pack_propagate(0)
#框架打包
frameDown0.pack(side=BOTTOM,padx=5,pady=5,fill=BOTH,expand=1)
frameUpper0.pack(side=LEFT,padx=5,pady=5,fill=BOTH,expand=1)
frameUpper1.pack(side=TOP,padx=5,pady=5,fill=BOTH,expand=1)
frameUpper2.pack(side=RIGHT,padx=5,pady=5,fill=BOTH,expand=1)


#测试标签
labx=Label(frameUpper0,text='图像显示区',foreground='blue',background='yellow')
laby=Label(frameUpper1,text='输入接口选择区',foreground='blue',background='yellow')
labm=Label(frameUpper2,text='模块区',foreground='blue',background='yellow')
labz=Label(frameDown0,text='消息区',foreground='blue',background='yellow')
labx.pack(fill=X)
laby.pack(fill=X)
labz.pack(fill=X)
labm.pack(fill=X)

#建立相机与图像文件接口
nob=Notebook(frameUpper1)
nob_frame1=Frame()
nob_frame2=Frame()
nob.add(nob_frame1,text='相机接口输入')
nob.add(nob_frame2,text='图像文件输入')
nob.pack(fill=BOTH,expand=1)





#两个转换函数(opencv-->PIL)(PIL-->opencv)
def op1(img_opencv):
    # opencv-->PIL图像测试
    img_PIL = Image.fromarray(cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB))
    op1= ImageTk.PhotoImage(img_PIL)
    return op1
def po1(img_PIL):
    # PIL-->opencv图像测试
    po1 = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return po1

#菜单响应函数
#打开文件
def openFile():
    global original_image, img1, img
    original_image = Image.open(askopenfilename())
    img1 = ImageTk.PhotoImage(original_image)
    img = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)


#打开图像文件按钮
btn_nob_frame2=Button(nob_frame2,text="图像文件载入",width=12,command=openFile)
btn_nob_frame2.pack(side=LEFT,fill=Y)


def saveAsFile():
    global filename
    textContent=text.get("1.0",END)

    #filename="c_2.txt"
    filename = asksaveasfilename(defaultextension=".txt")
    if filename=="":
        return
    with open(filename,"w") as output:
        output.write(textContent)
        root.title(filename)
filename="Untitled"

#编辑阈值设置
# 阈值1：面积判断  thr_val1
# 阈值2：周长判断  thr_val2
# 阈值3：深度判断  thr_val3
# 阈值4：总体判断  thr_all
def Thr_set():
    global thr_val1,thr_val2,thr_val3,thr_all

    window_set = Tk()
    window_set.title("阈值设置界面")  # 窗口标题
    center_window(window_set, 280, 250)  # 居中处理

    # #阈值输入控件
    # slider1=Scale(window_set,from_=0, to=10,)
    # slider1.pack()
    # # slider2=Scale(window_set, from_=0, to=10)
    # # slider2.pack()
    # # slider3=Scale(window_set, from_=0, to=10)
    # # slider3.pack()
    # # slider4=Scale(window_set, from_=0, to=10)
    # # slider4.pack()


    lab_val1 = Label(window_set, text="面积阈值(1~10000)")
    thr_val1=Spinbox(window_set, from_=0, to=10000,increment=1000,command=th_change)
    lab_val2 = Label(window_set, text="周长阈值(1~3000)")
    thr_val2 = Spinbox(window_set, from_=0, to=3000, increment=100,command=th_change)
    lab_val3 = Label(window_set, text="深度阈值(1~10000)")
    thr_val3 = Spinbox(window_set, from_=0, to=10000, increment=1000,command=th_change)
    lab_all = Label(window_set, text="总阈值(1~10000)")
    thr_all = Spinbox(window_set, from_=0, to=10000, increment=1000,command=th_change)
    #控件包装
    lab_val1.pack(side=TOP, padx=5, pady=5)
    thr_val1.pack()
    lab_val2.pack(side=TOP, padx=5, pady=5)
    thr_val2.pack()
    lab_val3.pack(side=TOP, padx=5, pady=5)
    thr_val3.pack()
    lab_all.pack(side=TOP, padx=5, pady=5)
    thr_all.pack()
    btn_Thrset = Button(window_set, text="关闭", width=15, command=window_set.destroy)
    btn_Thrset.pack()
    window_set.mainloop()  # 窗口循环


def th_change():
    global A,L,D,ALL
    A=float(thr_val1.get())
    L=float(thr_val2.get())
    D=float(thr_val3.get())
    ALL=float(thr_all.get())
    print(A,L,D,ALL)


##########################菜单
##########################菜单
#建立最上层菜单
menubar=Menu(root)
filemenu=Menu(menubar,tearoff=False,activebackground='blue')      #文件列，tearoff：无分隔线
editmenu=Menu(menubar,tearoff=False,activebackground='blue')      #编辑列
#文件下拉
menubar.add_cascade(label="文件",menu=filemenu)
#建立文件列表
filemenu.add_command(label="打开文件",command=openFile)
filemenu.add_command(label="保存文件",command=saveAsFile)
filemenu.add_separator()      #分隔线
filemenu.add_command(label="退出",command=root.destroy)
#编辑下拉
menubar.add_cascade(label="编辑",menu=editmenu)
#建立编辑列表
# 判断阈值设置
editmenu.add_command(label="预处理设置")
editmenu.add_command(label="判断阈值设置",command=Thr_set)
#显示菜单对象
root.config(menu=menubar)

###############################建立文本框
#建立文本显示框
text = Text(frameDown0,undo=True,height=10,width=20)
text.pack(fill=BOTH,expand=True)

###############################图片显示与更新函数
def img_change(lab_name,img):
    lab_name["image"]=img


##############################按钮函数
#“显示图像”1
def fun_btn1():
    # if 'img1' in vars().keys():
    global btnclose1, lab1
    lab1 = Label(frameUpper0, text="图像", image=img1, compound="top")
    lab1.pack(side=TOP, padx=5, pady=5)
    #     # btnclose1 = Button(frameUpper1, text="关闭图像1", width=15, command=lab1.destroy)
    #     # btnclose1.pack(side=BOTTOM)
    # else:
    #     # 询问载入图像
    #     messagebox.showwarning("提示",'请载入图像')



def fun_btn2():
    global aussian,aussian1,btnclose2
    #滤波处理
    aussian = cv2.GaussianBlur(img, (3, 3), 1)
    #cv_show("aussian",aussian)
    aussian1=op1(aussian)
    img_change(lab1,aussian1)

    #lab2 = Label(frameUpper1,text="高斯滤波",image=aussian1, compound="top")
    #lab2.pack(side=TOP, padx=5, pady=5)
    #btnclose2 = Button(frameUpper1, text="关闭图像2", width=15, command=lab2.destroy)
    #btnclose2.pack(side=BOTTOM)
#测试按钮
# def testbtn():
#     aussian1=op1(aussian)
#     lab2 = Label(frameUpper1,text="高斯滤波",image=aussian1, compound="top")
#     lab2.pack(side=TOP, padx=5, pady=5)
#     btnclose2 = Button(frameUpper1, text="关闭图像2", width=15, command=lab2.destroy)
#     btnclose2.pack(side=BOTTOM)
#
# btn_test=Button(frameUpper0,text="测试",width=15,command=testbtn)
# btn_test.pack(side=TOP)

def fun_btn3():
    global img_gray,img_gray1,btnclose3
    # 灰度处理
    img_gray = cv2.cvtColor(aussian, cv2.COLOR_BGR2GRAY)
    img_gray1 = op1(img_gray)
    img_change(lab1, img_gray1)
    # lab3 = Label(frameUpper1,text="灰度处理",image=img_gray1, compound="top")
    # lab3.pack(side=TOP, padx=5, pady=5)
    # btnclose3 = Button(frameUpper1, text="关闭图像3", width=15, command=lab3.destroy)
    # btnclose3.pack(side=BOTTOM)

def fun_btn4():
    global res1,contours
    # 边缘检测
    cannypic = cv2.Canny(img_gray, 50, 150)
    # 轮廓提取和绘制
    contours, hierachy = cv2.findContours(cannypic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 传入绘制图像，轮廓轮廓索引，颜色模式，线条厚度 # 注意需要copy,要不原图会变。。。
    draw_img = img.copy()
    res = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 1)
    res1 = op1(res)
    img_change(lab1, res1)
    # lab4 = Label(frameUpper1,text="灰度处理",image=res1, compound="top")
    # lab4.pack(side=TOP, padx=5, pady=5)
    # btnclose4 = Button(frameUpper1, text="关闭图像4", width=15, command=lab4.destroy)
    # btnclose4.pack(side=BOTTOM)



#计算缺陷面积，周长，与最小灰度值
def fun_btn5():

    global dpi
    dpi=96      #图片的dpi，查看图片属性可得
    global ares_list,length_list,depth_list
    ares_list=[]
    length_list=[]
    depth_list=[]
    count = 0  # 轮廓个数
    margin = 0  # 裁剪边距
    ave = 255
    all_ares = 0  # 总面积
    all_length = 0  # 总周长
    h = img.shape[0]  # 图像的高度
    w = img.shape[1]  # 图像的宽度
    img_white = np.zeros([h, w, 1], np.uint8)  # 三维图像
    img_white[:, :, 0] = np.ones([h, w]) * 255  # 空白图像
#画轮廓
    for i, contour in enumerate(contours):
        cv2.drawContours(img, contours, i, (0, 255, 0), 1)  # 在原图中画轮廓
        cv2.drawContours(img_white, contours, i, (0, 0, 0), -1)  # 在空白图中画轮廓
        gray = cv2.bitwise_or(img_gray, img_white)

#正式计算
    for i, contour in enumerate(contours):

        #周长面积
        ares1 = cv2.contourArea(contour)# 计算包围形状的面积
        ares=ares1 / (dpi ** 2) * 25.4 * 25.4
        ares_list.insert(i,ares)
        length1 = cv2.arcLength(contour, True)  # 计算包围形状的周长
        length=length1/ dpi * 25.4
        length_list.insert(i,length)
        print('第' + str(i) + '个轮廓面积=' + str(ares))
        print('第' + str(i) + '个轮廓周长=' + str(length))
        all_ares = all_ares + ares  # 总面积
        all_length = all_length + length  # 总周长

        #灰度值
        rect = cv2.minAreaRect(contour)  # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        box = np.int0(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
        h, w = img.shape[:2]  # 原图像的高和宽
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
        #cv2.imwrite("D:/python-data/{}.jpg".format(count), rotated_canvas)
        # cv_show("rotated_canvas", rotated_canvas)
        #hist = cv2.calcHist([rotated_canvas], [0], None, [256], [0, 256])

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
            depth_list.insert(i,depth)
                        # continue
                    # col = col - 1
                    # row = row - 1
            # print("new=", new)
            # print("new_data=", new_data)
            # print("number=", number)
            # print("zongshu=", new.shape[0] * new.shape[1])

            print('第' + str(i) + '个轮廓深度=' + str(depth))
            text.insert(END, '第' + str(i) + '个轮廓面积=' + str(ares) + "\n")
            text.insert(END, '第' + str(i) + '个轮廓周长=' + str(length) + "\n")
            text.insert(END, '第' + str(i) + '个轮廓深度=' + str(depth) + "\n")


#缺陷度判断
def fun_btn6():
    #判断阈值设置
    # 阈值1：面积判断  thr_val1(面积最小值)
    # 阈值2：周长判断  thr_val2（周长最小值）
    # 阈值3：深度判断  thr_val3（深度最小值）
    # 阈值4：总体判断  thr_all
    print(ares_list,"\n",length_list,"\n",depth_list)    #三个数据列表
    #判断思路：1.先对每一个单一缺陷的单一形状特征值进行判断，超过阈值的直接判断整个图像为缺陷，
            #2.如果所有缺陷的三个形状特征值都在阈值之下，则对每个缺陷的三个形状特征值进行加权处理得到总特征值W_form
           #3.对每个缺陷的总特征值进行判断，超过阈值的直接判断为图像缺陷，否则为图像通过
    #判断加权公式
    #加权系数
    w1=0.35
    w2=0.2
    w3=0.45
    #   f= 0.35 * x1 + 0.2 * x2 + 0.45 * x3
    #总的特征值
    thr_all1=[w1*ares_list[i]+w2*length_list[i]+w3*depth_list[i] for i in range(0, len(ares_list))]
    # if 'A' in locals().keys():
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
    text.insert(END, "判断结果：" + str(a))
    # else:
    #     # 询问设置阈值
    #     messagebox.showwarning("提示",'请设置判断阈值')







#按钮功能
btnall=Button(frameUpper2,text="功能模块导入",width=15)

btn1=Button(frameUpper2,text="显示图像",width=15,command=fun_btn1)
btn2=Button(frameUpper2,text="高斯滤波",width=15,command=fun_btn2)
btn3=Button(frameUpper2,text="灰度处理",width=15,command=fun_btn3)
btn4=Button(frameUpper2,text="轮廓提取",width=15,command=fun_btn4)
btn5=Button(frameUpper2,text="面积周长",width=15,command=fun_btn5)
btn6=Button(frameUpper2,text="缺陷判断",width=15,command=fun_btn6)
btn7=Button(frameUpper2,text="关闭界面",width=15,command=root.destroy)
#包装定位组件
btnall.pack(side=TOP,fill=X)
btn1.pack(side=TOP,fill=X)
btn2.pack(side=TOP,fill=X)
btn3.pack(side=TOP,fill=X)
btn4.pack(side=TOP,fill=X)
btn5.pack(side=TOP,fill=X)
btn6.pack(side=TOP,fill=X)
btn7.pack(side=TOP,fill=X)





# 创建图像显示函数
def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey()
    cv2.destroyWindow()



root.mainloop()  #跟窗口循环