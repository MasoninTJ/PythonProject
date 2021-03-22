# -*- encoding: utf-8 -*-
"""
不使用VAO画多顶点对象
"""

from io import BytesIO

import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_color;

uniform mat4 model;  // 包含平移和旋转
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
}
"""

# 增加switcher判断
fragment_src = """
# version 330

in vec3 v_color;
in vec2 v_texture;

uniform sampler2D s_texture;
uniform int switcher;

out vec4 out_color;

void main()
{
    if(switcher == 0)
    {
        out_color = texture(s_texture, v_texture);
    }
    else if (switcher == 1)
    {
        out_color = vec4(v_color, 1.0);
    }
    
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

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    image_path_1 = 'https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210310113410.jpg'
    image_path_2 = 'https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210311143110.jpg'
    image_path_3 = 'https://cdn.jsdelivr.net/gh/sheng962464/PicGo/img/20210311143112.jpg'
    texture = glGenTextures(3)
    load_texture(image_path_1, texture[0])
    load_texture(image_path_2, texture[1])
    load_texture(image_path_3, texture[2])

    # region 构造cube并且设置对应的参数，这里只需要定义如何读取点云即可，而不用传入点云
    cube_vertices = [-0.5, -0.5, 0.5, 0.0, 0.0,
                     0.5, -0.5, 0.5, 1.0, 0.0,
                     0.5, 0.5, 0.5, 1.0, 1.0,
                     -0.5, 0.5, 0.5, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 0.0, 0.0,
                     0.5, -0.5, -0.5, 1.0, 0.0,
                     0.5, 0.5, -0.5, 1.0, 1.0,
                     -0.5, 0.5, -0.5, 0.0, 1.0,

                     0.5, -0.5, -0.5, 0.0, 0.0,
                     0.5, 0.5, -0.5, 1.0, 0.0,
                     0.5, 0.5, 0.5, 1.0, 1.0,
                     0.5, -0.5, 0.5, 0.0, 1.0,

                     -0.5, 0.5, -0.5, 0.0, 0.0,
                     -0.5, -0.5, -0.5, 1.0, 0.0,
                     -0.5, -0.5, 0.5, 1.0, 1.0,
                     -0.5, 0.5, 0.5, 0.0, 1.0,

                     -0.5, -0.5, -0.5, 0.0, 0.0,
                     0.5, -0.5, -0.5, 1.0, 0.0,
                     0.5, -0.5, 0.5, 1.0, 1.0,
                     -0.5, -0.5, 0.5, 0.0, 1.0,

                     0.5, 0.5, -0.5, 0.0, 0.0,
                     -0.5, 0.5, -0.5, 1.0, 0.0,
                     -0.5, 0.5, 0.5, 1.0, 1.0,
                     0.5, 0.5, 0.5, 0.0, 1.0]

    cube_indices = [0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    8, 9, 10, 10, 11, 8,
                    12, 13, 14, 14, 15, 12,
                    16, 17, 18, 18, 19, 16,
                    20, 21, 22, 22, 23, 20]

    cube_vertices = np.array(cube_vertices, dtype=np.float32)
    cube_indices = np.array(cube_indices, dtype=np.uint32)

    cube_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

    cube_ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

    # endregion

    # region 构造quad，并设置对于的参数，这里只需要定义如何读取点云即可，而不用传入点云
    quad_vertices = [-0.5, -0.5, 0, 0.0, 0.0,
                     0.5, -0.5, 0, 1.0, 0.0,
                     0.5, 0.5, 0, 1.0, 1.0,
                     -0.5, 0.5, 0, 0.0, 1.0]

    quad_indices = [0, 1, 2, 2, 3, 0]

    quad_vertices = np.array(quad_vertices, dtype=np.float32)
    quad_indices = np.array(quad_indices, dtype=np.uint32)

    quad_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    quad_ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    # endregion

    # region 构造triangle，并设置对于的参数，这里只需要定义如何读取点云即可，而不用传入点云
    triangle_vertices = [-0.5, -0.5, 0, 1, 0, 0,
                         0.5, -0.5, 0, 0, 1, 0,
                         0.0, 0.5, 0, 0, 0, 1]

    triangle_vertices = np.array(triangle_vertices, dtype=np.float32)

    tri_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tri_vbo)
    glBufferData(GL_ARRAY_BUFFER, triangle_vertices.nbytes, triangle_vertices, GL_STATIC_DRAW)
    # endregion

    # model matrix
    scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([200, 200, 200]))
    cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([400, 300, 0]))
    quad_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([100, 300, 0]))
    triangle_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([700, 300, 0]))
    model_loc = glGetUniformLocation(shader, 'model')

    # view matrix
    view_loc = glGetUniformLocation(shader, 'view')
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # projection matrix
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 800, 0, 600, -1000, 1000)
    proj_loc = glGetUniformLocation(shader, 'projection')
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    switcher_loc = glGetUniformLocation(shader, 'switcher')

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.5 * glfw.get_time())

        rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        model = pyrr.matrix44.multiply(scale, rotation)

        # 切换为纹理代码
        glUniform1i(switcher_loc, 0)

        # region 循环传入点云，并绘制cube
        glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo)

        # 定义cube顶点的读取方式
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube_vertices.itemsize * 5, ctypes.c_void_p(0))

        # 定义cube纹理的读取方式
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cube_vertices.itemsize * 5, ctypes.c_void_p(cube_vertices.itemsize * 3))

        glBindTexture(GL_TEXTURE_2D, texture[0])
        cube_model = pyrr.matrix44.multiply(model, cube_pos)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, cube_model)
        glDrawElements(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None)
        # endregion

        # region 循环传入点云，并绘制quad
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_ebo)

        # 定义quad顶点的读取方式
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(0))

        # 定义quad纹理的读取方式
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(quad_vertices.itemsize * 3))

        glBindTexture(GL_TEXTURE_2D, texture[1])
        quad_model = pyrr.matrix44.multiply(model, quad_pos)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, quad_model)
        glDrawElements(GL_TRIANGLES, len(quad_indices), GL_UNSIGNED_INT, None)
        # endregion

        # 切换为绘制无纹理的代码
        glUniform1i(switcher_loc, 1)

        # region 循环传入点云，并绘制彩色三角形
        glBindBuffer(GL_ARRAY_BUFFER, tri_vbo)

        # 定义tri顶点的读取方式
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, triangle_vertices.itemsize * 6, ctypes.c_void_p(0))

        # 定义tri颜色的读取方式
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, triangle_vertices.itemsize * 6, ctypes.c_void_p(triangle_vertices.itemsize * 3))

        triangle_model = pyrr.matrix44.multiply(model, triangle_pos)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, triangle_model)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        # endregion

        glfw.poll_events()

    glfw.terminate()


def window_resize(width, height):
    """
    窗口调整大小
    """
    glViewport(0, 0, width, height)


def load_texture(path: str, texture):
    """
    读取纹理图片
    """
    if path.startswith('https:'):  # 从网络读取图片
        import requests
        html_image = requests.get(path)
        image = Image.open(BytesIO(html_image.content))
    else:  # 从本地读取图片
        image = Image.open(path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert('RGB').tobytes()

    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)


if __name__ == "__main__":
    glfw_test_github()
