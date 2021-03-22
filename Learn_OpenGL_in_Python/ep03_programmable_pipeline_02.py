# -*- encoding: utf-8 -*-
"""
使用可编程管线方式对三角形进行着色
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# 顶点着色器
vertex_src = """
# version 330

in vec3 a_position;
in vec3 a_color;

out vec3 v_color;

void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# 片段着色器
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

    glfw.set_window_pos(window, 280, 240)
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

    # 下面的方法只创建了一个缓冲区，因此临时将顶点和颜色放在同一个数组中
    # 这里使用的方式是 (x,y,z,r,g,b)方式，因此下面将创建的vbo传给着色器时，间隔为24，且color从第12个字节开始
    vertices = np.array([-0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
                         0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
                         0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)


    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, 'a_position')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))  # vertex 从第0个字节开始

    # 由于color和vertex放在同一个数组中，因此不需要创建两个缓冲区
    color = glGetAttribLocation(shader, 'a_color')
    glEnableVertexAttribArray(color)
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))  # color 从第12个字节开始


if __name__ == "__main__":
    glfw_test_github()
