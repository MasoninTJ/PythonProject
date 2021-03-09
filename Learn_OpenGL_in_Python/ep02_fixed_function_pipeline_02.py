# -*- encoding: utf-8 -*-
"""
使用固定管线方式对三角形进行变换
使用 glScale,glRotatef,glTransfer方式
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

    glfw.set_window_pos(window, 2480, 240)
    glfw.make_context_current(window)

    # openGL的固定管线方式
    test_opengl_fixed_pipeline()

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT)

        # 对三角形进行变换，对当前矩阵进行操作
        test_opengl_transform_fixed_pipeline()

        glDrawArrays(GL_TRIANGLES, 0, 3)
        glfw.poll_events()

    glfw.terminate()


def test_opengl_fixed_pipeline():
    glClearColor(0.3, 0.5, 0.5, 1)

    vertices = np.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
    colors = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)

    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, colors)


def test_opengl_transform_fixed_pipeline():
    # return the elapsed time, since init was called
    ct = glfw.get_time()

    # The glLoadIdentity function replaces the current matrix with the identity matrix.
    # glLoadIdentity 函数使用单位矩阵替换当前矩阵
    glLoadIdentity()
    glScale(abs(np.sin(ct)), abs(np.sin(ct)), 1)
    glRotatef(np.sin(ct) * 45, 0, 0, 1)
    glTranslatef(np.sin(ct), np.cos(ct), 0)


if __name__ == "__main__":
    glfw_test_github()
