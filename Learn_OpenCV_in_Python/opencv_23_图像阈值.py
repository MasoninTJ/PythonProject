import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


"""
cv.threshold() 简单阈值接受以下参数
    cv.THRESH_BINARY
    cv.THRESH_BINARY_INV
    cv.THRESH_TRUNC
    cv.THRESH_TOZERO
    cv.THRESH_TOZERO_INV

cv.adaptiveThreshold() 自适应阈值

cv.ADPTIVE_THRESH_MEAN_C    # 阈值取自相邻区域的平均值
cv.ADPTIVE_THRESH_GAUSSIAN_C # 阈值取自相邻区域的加权和，权重为一个高斯窗口
"""
