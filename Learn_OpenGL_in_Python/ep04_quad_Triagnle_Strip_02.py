# -*- encoding: utf-8 -*-
"""
实现窗口调整大小的回调函数
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np


def window_resize(window, width, height):
    # glViewport —— set the viewport 设置视口
    # x, y —— Specify the lower left corner of the viewport rectangle, in pixels. The initial value is (0,0).
    # width, height —— Specify the width and height of the viewport.
    # When a GL context is first attached to a window, width and height are set to the dimensions of that window.
    glViewport(0, 0, width, height)


vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1)in vec3 a_color;

out vec3 v_color;

void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_color = a_color;
}
"""

fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 0.5);
}
"""


def glfw_test_github():
    if not glfw.init():
        raise Exception('glfw can not be initialized!')

    window = glfw.create_window(800, 600, "Hello World", None, None)
    if not window:
        glfw.terminate()
        raise Exception('glfw windows can not be created!')

    glfw.set_window_pos(window, 2480, 240)
    # This function sets the size callback of the specified window, which is called when the window is resized.
    # 此函数设置指定窗口的大小回调，在调整窗口大小时调用该函数
    # The callback is provided with the size, in screen coordinates, of the content area of the window.
    # 回调函数提供了窗口内容区域的大小（以屏幕坐标为单位）
    glfw.set_window_size_callback(window, window_resize)

    glfw.make_context_current(window)
    test_opengl_programmable_pipeline()

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glfw.poll_events()

    glfw.terminate()


def test_opengl_programmable_pipeline():
    glClearColor(0.3, 0.5, 0.5, 1)

    vertices = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                -0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
                0.5, 0.5, 0.0, 1.0, 1.0, 1.0]
    vertices = np.array(vertices, dtype=np.float32)

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))


if __name__ == "__main__":
    glfw_test_github()
