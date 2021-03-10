# -*- encoding: utf-8 -*-
"""
使用三角形带参数(GL_TRIANGLE_STRIP)
"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

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
    glfw.make_context_current(window)
    test_opengl_programmable_pipeline()

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT)

        # 这里的参数改为三角形带(GL_TRIANGLE_STRIP)，就可以同时画两个三角形，而不用指定六个顶点，以及顶点的数量要修改为4
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glfw.poll_events()

    glfw.terminate()


def test_opengl_programmable_pipeline():
    glClearColor(0.3, 0.5, 0.5, 1)

    # 修改并增加一个顶点
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
