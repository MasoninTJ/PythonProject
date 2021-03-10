# -*- encoding: utf-8 -*-
"""
使用可编程管线方式绘制红色三角形
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# 顶点着色器
vertex_src = """
# version 330

in vec3 a_position;

void main()
{
    gl_Position = vec4(a_position, 1.0);
}
"""

# 片段着色器
fragment_src = """
# version 330

out vec4 out_color;

void main()
{
    out_color = vec4(1.0, 0.0, 0.0, 1.0);
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
    glfw.make_context_current(window)

    # 现代可编程管线方式画图
    test_opengl_programmable_pipeline()

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawArrays(GL_TRIANGLES, 0, 3)
        glfw.poll_events()

    glfw.terminate()


def test_opengl_programmable_pipeline():
    glClearColor(0.3, 0.5, 0.5, 1)

    vertices = np.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)

    # 编译着色器代码
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    # 创建VBO
    vbo = glGenBuffers(1)
    # 当前的 GL_ARRAY_BUFFER 位置为 vbo 缓冲区
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # 将 vertices 传入 GL_ARRAY_BUFFER 位置
    # vertices.nbytes 返回数组的大小，单位是字节
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 绑定的VBO传入着色器中，指定接收的变量为a_position
    position = glGetAttribLocation(shader, 'a_position')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))


if __name__ == "__main__":
    glfw_test_github()
