# -*- encoding: utf-8 -*-
"""
绘制3D立方体，并旋转
"""

import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


# 传入旋转矩阵，使用uniform
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 rotation;

out vec3 v_color;

void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
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
    glfw.set_window_size_callback(window, window_resize)
    glfw.make_context_current(window)

    glClearColor(0.3, 0.5, 0.5, 1)
    # 深度刷新开
    glEnable(GL_DEPTH_TEST)

    # 立方体的8个顶点
    vertices = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, 0.5, 1.0, 1.0, 1.0,

                -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, -0.5, 1.0, 1.0, 1.0]

    # 每行构成一个面
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               4, 5, 1, 1, 0, 4,
               6, 7, 3, 3, 2, 6,
               5, 6, 2, 2, 1, 5,
               7, 4, 0, 0, 3, 7]

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    # 获得旋转矩阵在GPU中的位置
    rotation_loc = glGetUniformLocation(shader, 'rotation')

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 定义旋转矩阵
        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.5 * glfw.get_time())

        # 传入旋转矩阵，矩阵的乘法有多种形式
        # glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, rot_x * rot_y)
        # glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, rot_x @ rot_y)
        glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, pyrr.matrix44.multiply(rot_x, rot_y))

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    glfw_test_github()
