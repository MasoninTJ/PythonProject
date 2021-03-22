# -*- encoding: utf-8 -*-
"""
使用固定管线方式实现openGL绘制三角形
"""

import glfw
from OpenGL.GL import *
import numpy as np


def glfw_test_github():
    if not glfw.init():
        raise Exception('glfw can not be initialized!')

    window = glfw.create_window(800, 600, "Hello World", None, None)
    if not window:
        glfw.terminate()
        raise Exception('glfw windows can not be created!')

    glfw.set_window_pos(window, 280, 240)
    glfw.make_context_current(window)

    # openGL的固定管线方式
    test_opengl_fixed_pipeline()

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)

        # 使用设置的背景颜色刷新颜色缓冲
        glClear(GL_COLOR_BUFFER_BIT)

        # The glRotatef function multiplies the current matrix by a rotation matrix.
        # glRotatef 函数将当前矩阵乘以旋转矩阵
        # 这里glRotatef 函数使用轴角旋转，第一个参数为角度，后面为旋转轴
        # 每循环一次，则左乘一次旋转矩阵
        glRotatef(2, 0, 1, 0)

        # The glDrawArrays function specifies multiple primitives to render.
        # glDrawArray 函数指定需要渲染的多个基元
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.poll_events()
    glfw.terminate()


def test_opengl_fixed_pipeline():
    # 设置背景颜色
    glClearColor(0.3, 0.5, 0.5, 1)

    # 设置三角形顶点
    vertices = np.array([-0.5, -0.5, 0.0,
                         0.5, -0.5, 0.0,
                         0.0, 0.5, 0.0], dtype=np.float32)

    # 设置颜色
    colors = np.array([1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0], dtype=np.float32)

    # The glEnableClientState and glDisableClientState functions enable and disable arrays respectively.
    # glEnableClientState 和 glDisableClientState 分别启用和禁用数组
    # The glVertexPointer function defines an array of vertex data.
    # glVertexPointer 函数定义了顶点数据的数组
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)

    # The glColorPointer function defines an array of colors.
    # glColorPointer 函数定义颜色数组
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, colors)


if __name__ == "__main__":
    glfw_test_github()
