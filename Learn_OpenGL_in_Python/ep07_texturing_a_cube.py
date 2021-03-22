# -*- encoding: utf-8 -*-
"""
给立方体添加纹理
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


# 修改顶点着色器
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 rotation;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
}
"""

# 修改片段着色器
# 这里out_color使用纹理和颜色的混合，显示为彩色
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
    # 启用透明事件
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # 一行8个值，(x,y,z,r,g,b,u,v)，最后两个表示纹理的位置
    # 如果这里还像原来一样只有8个顶点，则只有上下两面的纹理是正确的，侧面的纹理则是错误的
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

    # 新建纹理的缓冲区，并进行绑定
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # 设置纹理交换参数 u,v 坐标
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # 设置纹理填充参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # 这里做了一点修改，使用网络图片
    r = requests.get('https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210310113410.jpg')
    image = Image.open(BytesIO(r.content))
    # image的起始从左上角开始，但是openGL的坐标从左下角开始
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # 这里使用的素材是jpg格式的，因此使用RGB方式读取
    img_data = image.convert('RGB').tobytes()
    # 传入image数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 3))

    # 传入纹理的坐标
    # 这里 vertices.itemsize 返回每个元素的大小，4 * 8 = 32，8 是间隔
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 6))

    rotation_loc = glGetUniformLocation(shader, 'rotation')

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.5 * glfw.get_time())

        glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, pyrr.matrix44.multiply(rot_x, rot_y))

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    glfw_test_github()
