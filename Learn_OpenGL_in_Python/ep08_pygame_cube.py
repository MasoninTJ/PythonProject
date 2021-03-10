# -*- encoding: utf-8 -*-
"""
使用pygame来代替glfw和PIL
"""

from io import BytesIO

import numpy as np
import pygame
import pyrr
import requests
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

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


def test_pygame():
    # 初始化pygame窗口
    pygame.init()
    pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

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

    glClearColor(0.3, 0.5, 0.5, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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

    # 使用pygame自己的读取图片的方法
    r = requests.get('https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210310113410.jpg')
    image = pygame.image.load(BytesIO(r.content))
    image = pygame.transform.flip(image, False, True)
    image_width, image_height = image.get_rect().size
    img_data = pygame.image.tostring(image, "RGB")
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 3))

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(vertices.itemsize * 6))

    rotation_loc = glGetUniformLocation(shader, 'rotation')

    # 持续渲染pygame窗口
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 关闭窗口
                running = False
            if event.type == pygame.VIDEORESIZE:  # 调整窗口的尺寸
                glViewport(0, 0, event.w, event.h)

        ct = pygame.time.get_ticks() / 1000  # 这里以ms为单位

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * ct)
        rot_y = pyrr.Matrix44.from_y_rotation(0.5 * ct)

        # glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, rot_x * rot_y)
        # glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, rot_x @ rot_y)
        glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, pyrr.matrix44.multiply(rot_x, rot_y))

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        pygame.display.flip()  # 交换缓冲区

    pygame.quit()


if __name__ == "__main__":
    test_pygame()
