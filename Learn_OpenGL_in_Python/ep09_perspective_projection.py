# -*- encoding: utf-8 -*-
"""
透视投影(近大远小)
"""

from io import BytesIO

import glfw
import numpy as np
import pyrr
import requests
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image


def window_resize(width, height):
    """
    窗口调整大小
    """
    glViewport(0, 0, width, height)


# 增加model矩阵和projection矩阵
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 model;  // 包含平移和旋转
uniform mat4 projection;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * model * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330

in vec3 v_color;
in vec2 v_texture;

uniform sampler2D s_texture;

out vec4 out_color;

void main()
{
    out_color = texture(s_texture, v_texture) * vec4(v_color, 1.0);
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
    glfw.set_window_size_callback(window, window_resize)
    glfw.make_context_current(window)

    glClearColor(0.3, 0.5, 0.5, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    vertices = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                -0.5, 0.5, -0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                -0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                -0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                -0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,

                0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
                -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0]

    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               8, 9, 10, 10, 11, 8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

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

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    r = requests.get('https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210310113410.jpg')
    image = Image.open(BytesIO(r.content))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert('RGB').tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 3))

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 6))

    # 创建透视投影矩阵
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800 / 600, 0.1, 100)
    proj_loc = glGetUniformLocation(shader, 'projection')

    # 给着色器传递透视投影矩阵
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # 旋转之后平移一下，才能正确观察到立方体，否则只能观察到立方体的内部
    # 这里有一点需要注意的是，为什么是-3
    # 因为平移的是模型，模型向下平移了-3，相当于相机向上平移了3
    translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]))

    model_loc = glGetUniformLocation(shader, 'model')

    # 这里当时有一个疑惑：旋转和平移是以什么样的形式复合的？
    # 通常 Q = R * P + T 其中 R 为旋转，T 为平移
    # 在这里 Q = M_T * M_R * P 其中 M_T 为平移矩阵，M_R 为旋转矩阵
    # 这里着色器代码每次都传递新值，因此 a_position 是不变的
    # 因此下面更新的变量只有 rotation，所以旋转速度恒定，且不会一直平移
    # 那么至于为什么 model = pyrr.matrix44.multiply(rotation,translation) 表达的是先旋转再平移，就看文档
    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(np.pi / 2 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(np.pi / 2 * glfw.get_time())

        # 构造model矩阵
        rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        model = pyrr.matrix44.multiply(rotation, translation)

        # 给着色器传递model矩阵
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    glfw_test_github()
