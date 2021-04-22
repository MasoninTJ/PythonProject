######################opencv函数库
######################功能分类编辑

import cv2
from skimage import data,color,filters
import numpy as np
import matplotlib.pyplot as plt
import math

# 创建cv_show函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)


#3.1缩小与平移
#缩小
def lessen(img):
    h,w=img.shape[:2]
    #仿射变换矩阵
    A=np.array([[0.5,0,w/4],[0,0.5,h/4]],np.float32)
    lessen=cv2.warpAffine(img,A,(w,h),borderValue=125)
    return lessen

#旋转
def rotate(img):
    # cv2.ROTATE_180参数顺时针旋转180度
    # cv2.ROTATE_90_COUNTERCLOCKWISE逆时针旋转90度
    #cv2.ROTATE_90_CLOCKWISE顺时针旋转90度
    rotate=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    return rotate

#4.1灰度直方图
def histogram(gray):
    # 直方图绘制
    plt.hist(gray.ravel(), 256)
    #设置坐标轴标签
    plt.ylabel('灰度值数量',fontproperties='SimHei', fontsize=10)
    plt.xlabel('灰度值',fontproperties='SimHei', fontsize=10)
    plt.show()

#全局直方图均衡化
def allhist(gray):
    allhist = cv2.equalizeHist(gray)
    return allhist

#限制对比度的自适应直方图均衡化
def limhist(gray):

    limhist1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    limhist = limhist1.apply(gray)
    return limhist

#5.1图像平滑（低通滤波（LPF）有利于去噪，模糊图像，高通滤波（HPF）有利于找到图像边界）
#2D滤波器
def d2filter(img):
    kernel = np.ones((5, 5), np.float32) / 25   #卷积核
    d2filter = cv2.filter2D(img, -1, kernel)
    return d2filter

#高斯滤波(二维离散卷积核)（高斯核的高和宽（奇数））
def GBlur(img):
    GBlur = cv2.GaussianBlur(img, (5, 5), 0)  #（5,5）表示的是卷积核大小，0表示的是沿x与y方向上的标准差
    return GBlur

#均值滤波(二维离散卷积核)
def meanval(img):
    meanval=cv2.blur(img,(3,5)) # 卷积核大小为3*5, 模板的大小是可以设定的
    return meanval

#方框滤波，normalize=1时，表示进行归一化处理，此时图片处理效果与均值滤波相同，如果normalize=0时，表示不进行归一化处理，像素值为周围像素之和，图像更多为白色
def boxfilter(img):
    boxfilter = cv2.boxFilter(img, -1, (5, 5), normalize=1)
    return boxfilter


#中值滤波(统计学)(中值滤波模板就是用卷积框中像素的中值代替中心值，达到去噪声的目的。这个模板一般用于去除椒盐噪声。卷积核的大小也是个奇数。)
def medBlur(img):
    medBlur = cv2.medianBlur(img, 5)  # 中值滤波函数
    return medBlur

#双边滤波（保持边缘清晰）双边滤波同时使用了空间高斯权重和灰度相似性高斯权重，确保了边界不会被模糊掉。
def doufilter(img):
    #9表示的是滤波领域直径，后面的两个数字：空间高斯函数标准差，灰度值相似性标准差
    doufilter = cv2.bilateralFilter(img, 9, 80, 80)
    return doufilter


#6.1阈值分割
#简单全局阈值
# cv2.THRESH_BINARY（黑白二值）
# cv2.THRESH_BINARY_INV（黑白二值翻转）
# cv2.THRESH_TRUNC（得到额图像为多像素值）
# cv2.THRESH_TOZERO（当像素高于阈值时像素设置为自己提供的像素值，低于阈值时不作处理）
# cv2.THRESH_TOZERO_INV（当像素低于阈值时设置为自己提供的像素值，高于阈值时不作处理）

def Threshold(gray):
    ret, Threshold = cv2.threshold(gray, 117, 255, cv2.THRESH_BINARY)
    return Threshold


#自适应阈值
def autothreshold(gray):
# 第一个参数为原始图像矩阵
# 第二个参数为像素值上限，
# 第三个是自适应方法（adaptive method）：cv2.ADAPTIVE_THRESH_MEAN_C:领域内均值
                                   # cv2.ADAPTIVE_THRESH_GAUSSIAN_C:领域内像素点加权和，权重为一个高斯窗口
# 第四个值的赋值方法：只有cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV
# 第五个Block size：设定领域大小（一个正方形的领域）
# 第六个参数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
    autothreshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    return autothreshold

#Otsu's阈值（Otsu's非常适合于图像灰度直方图(只有灰度图像才有)具有双峰的情况）
def Otsthreshold(gray):
    ret, Otsthreshold = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu滤波
    return Otsthreshold


#K-Means聚类阈值分割
def Kthreshold(gray):
    #获取图像高度、宽度
    rows, cols = gray.shape[:]
    #图像二维像素转换为一维
    data = gray.reshape((rows * cols, 1))
    data = np.float32(data)
    #定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    #K-Means聚类 聚集成4类
    compactness, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, flags)
    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    Kthreshold = res.reshape((gray.shape))
    return Kthreshold

#7.1形态学处理
#腐蚀（腐蚀可以使目标区域范围“变小”，其实质造成图像的边界收缩，可以用来消除小且无意义的目标物。）
def corrode(gray):
    kernel = np.ones((5, 5), np.uint8)  # 卷积核/结构元素(getStructuringElement函数也可以构造)
    corrode = cv2.erode(gray, kernel, iterations=1)  # 腐蚀(iterations：迭代次数)
    return corrode

#膨胀（膨胀会使目标区域范围“变大”，将于目标区域接触的背景点合并到该目标物中，使目标边界向外部扩张。
      #作用就是可以用来填补目标区域中某些空洞以及消除包含在目标区域中的小颗粒噪声。 膨胀也可以用来连接两个分开的物体。）
def swelld(gray):
    kernel = np.ones((5, 5), np.uint8) #卷积核
    swelld = cv2.dilate(gray, kernel, iterations=1)  # 膨胀
    return swelld

#开运算
#（先腐蚀再膨胀)，它被用来去除噪声。cv2.MORPH_OPEN
def Opening(gray):
    kernel = np.ones((5, 5), np.uint8) #卷积核
    Opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # 开运算
    return Opening

#闭运算
# （先膨胀再腐蚀）它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
def Closing(gray):
    kernel = np.ones((5, 5), np.uint8)  # 卷积核
    Closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # 闭运算
    return Closing

#形态学梯度
#前景物体的轮廓
def Gradient(gray):
    kernel = np.ones((5, 5), np.uint8)  # 卷积核
    Gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度
    return Gradient

#礼帽
#礼帽图像=原始图像-开运算(cv2.MORPH_TOPHAT)
def Tophat(gray):
    kernel=np.ones((5, 5), np.uint8) # 卷积核
    Tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)  # 礼帽
    return Tophat

#黑帽
#黑帽图像=闭运算-原始图像（cv2.MORPH_BLACKHAT）
def Blackhat(gray):
    kernel = np.ones((5, 5), np.uint8)  # 卷积核
    Blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # 黑帽
    return Blackhat

#边缘检测
####基于搜索（一阶导数）
#Roberts算子
#交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。
#常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。
#其缺点是对边缘的定位不太准确，提取的边缘线条较粗。
def Roberts(gray):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

#Prewitt算子
#采用33模板对区域内的像素值进行计算，而Robert算子的模板为22，
# 故Prewitt算子的边缘检测结果在水平方向和垂直方向均比Robert算子更加明显。Prewitt算子适合用来识别噪声较多、灰度渐变的图像
def Prewitt(gray):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

#Sobel算子
#结合了高斯平滑和微分求导。该算子用于计算图像明暗程度近似值，
# 根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。
# Sobel算子在Prewitt算子的基础上增加了权重的概念，认为相邻点的距离远近对当前像素点的影响是不同的，距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
#Sobel算子的边缘定位更准确，常用于噪声较多、灰度渐变的图像。
def Sobel(gray):
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

#Scharr算子
# Scharr 算子是对 Sobel 算子差异性的增强，两者之间的在检测图像边缘的原理和使用方式上相同。
# Scharr 算子的主要思路是通过将模版中的权重系数放大来增大像素值间的差异。
# Scharr 算子又称为 Scharr 滤波器，也是计算 x 或 y 方向上的图像差分，在 OpenCV 中主要是配合 Sobel 算子的运算而存在的
def scharr(gray):
    x = cv2.Scharr(gray, cv2.CV_16S, 1, 0)  # X 方向
    y = cv2.Scharr(gray, cv2.CV_16S, 0, 1)  # Y 方向
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return scharr


#Kirsch，Robinson算子,
# Kirsch边缘算子由八个方向的卷积核构成，这8个模板代表8个方向，对图像上的8个特定边缘方向作出最大响应，运算中取最大值作为图像的边缘输出
def Kirsch(gray):
    # 定义Kirsch 卷积模板
    m1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    m2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    m3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    m4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    m5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    m6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    m7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    m8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    # 周围填充一圈
    # 卷积时，必须在原图周围填充一个像素
    graym = cv2.copyMakeBorder(gray, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    temp = list(range(8))
    gray1 = np.zeros(graym.shape)  # 复制空间  此处必须的重新复制一块和原图像矩阵一样大小的矩阵，以保存计算后的结果
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            temp[0] = np.abs((np.dot(np.array([1, 1, 1]), (m1 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            # 利用矩阵的二次型表达，可以计算出矩阵的各个元素之和
            temp[1] = np.abs((np.dot(np.array([1, 1, 1]), (m2 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[2] = np.abs((np.dot(np.array([1, 1, 1]), (m1 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[3] = np.abs((np.dot(np.array([1, 1, 1]), (m3 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[4] = np.abs((np.dot(np.array([1, 1, 1]), (m4 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[5] = np.abs((np.dot(np.array([1, 1, 1]), (m5 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[6] = np.abs((np.dot(np.array([1, 1, 1]), (m6 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            temp[7] = np.abs((np.dot(np.array([1, 1, 1]), (m7 * gray[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]])))
            gray1[i, j] = np.max(temp)
            if gray1[i, j] > 255:# 此处的阈值一般写255，根据实际情况选择0~255之间的值
                gray1[i, j] = 255
            else:
                gray1[i, j] = 0
    #print('Kirsch算子后图片尺寸',gray1.shape)
    gray2= cv2.resize(gray1, (gray.shape[0], gray.shape[1]))
    Kirsch=gray2
    #print('gray2算子后图片尺寸', gray2.shape)
    return Kirsch


#Canny算子
# Canny方法不容易受噪声干扰，能够检测到真正的弱边缘。
# 优点在于，使用两种不同的阈值分别检测强边缘和弱边缘，并且当弱边缘和强边缘相连时，才将弱边缘包含在输出图像中。
def  canny(gray):
    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny算子
    canny = cv2.Canny(gaussian, 50, 180)
    return canny


####基于零交叉（二阶导数）
#Laplacian算子
#n维欧几里德空间中的一个二阶微分算子，常用于图像增强领域和边缘提取
# Laplacian算子其实主要是利用Sobel算子的运算，通过加上Sobel算子运算出的图像x方向和y方向上的导数，得到输入图像的图像锐化结果。
# 同时，在进行Laplacian算子处理之后，还需要调用convertScaleAbs()函数计算绝对值，并将图像转换为8位图进行显示。
def laplacian(gray):
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
    laplacian = cv2.convertScaleAbs(dst)
    return laplacian

#LOG算子#Marr-Hildreth算子
# 根据图像的信噪比来求检测边缘的最优滤波器。
# 该算法首先对图像做高斯滤波，然后再求其拉普拉斯（ Laplacian ）二阶导数，
# 根据二阶导数的过零点来检测图像的边界，即通过检测滤波结果的零交叉（ Zero crossings ）来获得图像或物体的边缘。
# LOG 算子实际上是把 Gauss 滤波和 Laplacian 滤波结合了起来，先平滑掉噪声，再进行边缘检测。
# LOG 算子与视觉生理中的数学模型相似，因此在图像处理领域中得到了广泛的应用。
# 它具有抗干扰能力强，边界定位精度高，边缘连续性好，能有效提取对比度弱的边界等特点。
def log(gray):
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # 再通过拉普拉斯算子做边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    log = cv2.convertScaleAbs(dst)
    return log


#DoG算子
# （1）灰度化图像
# （2）计算方差为2和3.2的两个高斯滤波后的图像
# （3）求两个之差——将二者之差再除以2（归一化）
def DoG(gray):
    gimg1 = filters.gaussian(gray, sigma=2)
    gimg2 = filters.gaussian(gray, sigma=1.6 * 2)
    # 两个高斯运算的差分
    dimg = gimg2 - gimg1
    # 将差归一化
    dimg /= 2
    #二值化边缘
    edge=np.copy(dimg)
    edge[edge>0]=255
    edge[edge <= 0] = 0
    edge=edge.astype(np.uint8)
    #图像边缘抽象化
    asbstraction=-np.copy(dimg)
    asbstraction=asbstraction.astype(np.float32)
    asbstraction[asbstraction>=0]=0.1
    asbstraction[asbstraction<0]=0.1+np.tanh(asbstraction[asbstraction<0])
    DoG=asbstraction
    #return edge
    return DoG


####9.1几何形状检测与拟合
##直线检测
#标准霍夫变换SHT（通过调整边缘检测算子Canny阈值参数和标准霍夫变换阈值参数，来获取较好的检测效果。）
def houghlines(canny):
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 10)  # 霍夫变换返回的就是极坐标系中的两个参数  rho和theta
    if np.shape(lines) == ():
        print('图像无法识别出直线')
    else:
        result = gray.copy()
        for line in lines:
            rho = line[0][0]  # 第一个元素是距离rho
            theta = line[0][1]  # 第二个元素是角度theta
            #print(rho)
            #print(theta)
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                cv2.line(result, pt1, pt2, (255, 255, 255), 1)  # 绘制一条白线
            else:  # 水平直线
                pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(result, pt1, pt2, (255, 255, 255), 1)  # 绘制一条直线
            houghlines = result
            return houghlines


#渐进概率式霍夫变换PPHT
# 标准霍夫变换函数相比，在参数的输入上多了两个参数：
# minLineLengh(线的最短长度，比这个线段短的都忽略掉)
# maxLineGap(两条直线之间的最大间隔，小于此值，就认为是一条直线)。
def houghlinesP(canny):
    result = img.copy()
    lines = cv2.HoughLinesP(canny, 1, np.pi /360, 30, minLineLength=70, maxLineGap=5)
    if np.shape(lines) == ():
        print('图像无法识别出直线')
    else:
        lines = lines[:, 0, :]
        for x1, y1, x2, y2 in lines:
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)
        houghlinesP = result
        return houghlinesP

####霍夫圆检测
def houghcircles(gray):
    circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 18, param1=60, param2=30, minRadius=30, maxRadius=300)
    print(np.shape(circle))
    result = img.copy()
    if circle is None:
        print('图像无法识别出圆')
    else:
        circle = np.uint16(np.around(circle))
        #print(circle)
        for i in circle[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 1)
        houghcircles=result
        return houghcircles

#####轮廓检测
# （cv2.findContours(image, mode, method[, offset])）
#寻找一个二值图像的轮廓。注意黑色表示背景，白色表示物体，即在黑色背景里寻找白色物体的轮廓
#mode:轮廓检索的方式=====
#cv2.RETR_EXTERNAL:只检索外部轮廓
#cv2.RETR_LIST: 检测所有轮廓且不建立层次结构
#cv2.RETR_CCOMP: 检测所有轮廓，建立两级层次结构
#cv2.RETR_TREE: 检测所有轮廓，建立完整的层次结构
#method:轮廓近似的方法 =====
#cv2.CHAIN_APPROX_NONE:存储所有的轮廓点
#cv2.CHAIN_APPROX_SIMPLE:压缩水平，垂直和对角线段，只留下端点。 例如矩形轮廓可以用4个点编码。
#cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS:使用Teh-Chini chain近似算法
#offset:（可选参数）轮廓点的偏移量，格式为tuple,
# 如（-10，10）表示轮廓点沿X负方向偏移10个像素点，沿Y正方向偏移10个像素点
#返回三个值,分别是img, countours（list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示）, hierarchy（每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号）
######绘制轮廓
#cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]])
# image:需要绘制轮廓的目标图像，注意会改变原图
# contours:轮廓点，上述函数cv2.findContours()的第一个返回值
# contourIdx:轮廓的索引，表示绘制第几个轮廓，-1表示绘制所有的轮廓
# color:绘制轮廓的颜色
# thickness:（可选参数）轮廓线的宽度，-1表示填充
# lineType:（可选参数）轮廓线型，包括cv2.LINE_4,cv2.LINE_8（默认）,cv2.LINE_AA,分别表示4邻域线，8领域线，抗锯齿线（可以更好地显示曲线）
# hierarchy:（可选参数）层级结构，上述函数cv2.findContours()的第二个返回值，配合maxLevel参数使用
# maxLevel:（可选参数）等于0表示只绘制指定的轮廓，等于1表示绘制指定轮廓及其下一级子轮廓，等于2表示绘制指定轮廓及其所有子轮廓
# offset:（可选参数）轮廓点的偏移量
def contoursshow(canny):
    contours, hierachy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_img = img.copy()
    contoursshow = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)
    return contoursshow






if __name__=='__main__':
    # 导入图片
    img = cv2.imread('D:/image_process/Lena.png')
    #print('源图片尺寸：',img.shape[:2])

    # 将图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #1缩小平移
    lessen=lessen(img)

    #2旋转
    rotate=rotate(img)
    rotate1 = cv2.resize(rotate, (512, 512))
    #print('旋转操作后图片尺寸：',rotate1.shape[:2])

    #3直方图
    #histogram(img)

    #4全局直方图均衡化
    allhist=allhist(gray)
    #print('全局直方图均衡化后图片尺寸：', allhist.shape[:2])
    #histogram(allhist)
    #cv_show('allhist',allhist)

    #5限制对比度的自适应直方图均衡化
    limhist=limhist(gray)
    #histogram(limhist)

    #62D滤波器
    d2filter=d2filter(img)
    #print('2D滤波器后图片尺寸：', d2filter.shape[:2])

    #7高斯滤波
    GBlur=GBlur(gray)
    cv_show('GBlur',GBlur)

    #8均值滤波
    meanval=meanval(img)

    #9方框滤波
    boxfilter=boxfilter(img)

    #10中值滤波
    medBlur=medBlur(img)

    #11双边滤波
    doufilter=doufilter(img)

    #12全局简单阈值（输入图为单通道灰度图）
    Threshold = Threshold(gray)

    #13自适应阈值
    autothreshold=autothreshold(gray)

    #14Otsu's二值化
    Otsthreshold=Otsthreshold(gray)

    #15基于K-Means聚类的区域分割
    Kthreshold=Kthreshold(gray)
    #cv_show("k",Kthreshold)

    #16腐蚀
    corrode=corrode(Threshold)

    #17膨胀
    swelld=swelld(Threshold)

    #18开运算
    Opening=Opening(Threshold)

    #19闭运算
    Closing=Closing(Threshold)

    #20形态学梯度
    Gradient=Gradient(gray)

    #21礼帽
    Tophat=Tophat(gray)

    #22黑帽
    Blackhat=Blackhat(gray)

    #23Roberts算子
    Roberts=Roberts(gray)
    #cv_show('Roberts',Roberts)

    #24Prewitt算子
    Prewitt=Prewitt(gray)

    #25Sobel算子
    Sobel = Sobel(gray)

    #26Canny算子
    canny=canny(gray)
    #cv_show('canny',canny)


    #27Laplacian算子
    laplacian=laplacian(gray)

    #28scharr算子
    scharr=scharr(gray)

    #29Kirsch算子(尺寸发生变化)(算法自定义函数实现，计算较慢)
    #Kirsch=Kirsch(gray)
    #cv_show('Kirsch',Kirsch)


    #30LoG算子
    log=log(gray)

    #31DoG算子
    DoG=DoG(gray)
    #cv_show('dog',DoG)

    #32标准霍夫变换SHT直线检测
    houghlines=houghlines(canny)
    #cv_show('houghlines',houghlines)

    #33#渐进概率式霍夫变换PPHT检测直线
    houghlinesP=houghlinesP(canny)
    #cv_show('houghlinesP',houghlinesP)

    # #34霍夫圆检测
    houghcircles=houghcircles(gray)
    #cv_show('houghcircles',houghcircles)

    #35轮廓检测
    contoursshow=contoursshow(canny)
    #cv_show('contoursshow',contoursshow)


    ###（高度，宽度，通道数=img.shape）
    # # #对比展示
    #imgs = np.hstack((img,houghcircles))
    #imgs = np.hstack((img,corrode))
    #cv_show("contrast", imgs)









