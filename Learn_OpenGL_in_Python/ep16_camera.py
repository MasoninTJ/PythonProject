# -*- encoding: utf-8 -*-
"""
读取3D文件-stl并显示出来
"""

import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pyrr import Vector3, matrix44

from Alogrithm.LoadMesh import load_stl_model
from Class3D import STLModel


class Camera:
    def __init__(self):
        self.camera_pos = Vector3([0, 0, 3])
        self.camera_front = Vector3([0, 0, -1])
        self.camera_up = Vector3([0, 1, 0])
        self.camera_right = Vector3([1, 0, 0])

        self.camera_senitivity = 0.25
        self.jaw = -90
        self.pitch = 0

    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)

    def process_mouse_movement(self, m_x_offset, m_y_offset, constrain_pitch=True):
        m_x_offset *= self.camera_senitivity
        m_y_offset *= self.camera_senitivity

        self.jaw += m_x_offset
        self.pitch += m_y_offset

        if constrain_pitch:
            if self.pitch > 45:
                self.pitch = 45
            if self.pitch < -45:
                self.pitch = 45

        self.update_camera_vectors()

    def update_camera_vectors(self):
        pass


# 对顶点着色器进行修改，顶点 + 向量 + 颜色
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

uniform mat4 model;  // 包含平移和旋转
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# 修改片段着色器，去掉多余的输入与输出
fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
}
"""


def glfw_test_github():
    if not glfw.init():
        raise Exception('glfw can not be initialized!')

    window = glfw.create_window(800, 800, "Hello World", None, None)
    if not window:
        glfw.terminate()
        raise Exception('glfw windows can not be created!')

    glfw.set_window_pos(window, 280, 140)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)
    glfw.set_window_size_callback(window, window_resize)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_key_callback(window, key_callback)

    glfw.make_context_current(window)

    glClearColor(0.3, 0.5, 0.5, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    glUseProgram(shader)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    # 读取stl文件，并转化为可以读取的列表形式
    m_model_path = r'D:\下载\全局标定测试\单层NEY模型.stl'
    m_model: STLModel = load_stl_model(m_model_path)
    m_model_vertices = []
    for m_mesh in m_model:
        # [x,y,z,i,j,k,r,g,b 形式]
        m_model_vertices.extend(m_mesh.vertex.vertex1.to_array())
        m_model_vertices.extend(m_mesh.normal.to_array())
        m_model_vertices.extend([1.0, 1.0, 0.0])
        m_model_vertices.extend(m_mesh.vertex.vertex2.to_array())
        m_model_vertices.extend(m_mesh.normal.to_array())
        m_model_vertices.extend([1.0, 1.0, 0.0])
        m_model_vertices.extend(m_mesh.vertex.vertex3.to_array())
        m_model_vertices.extend(m_mesh.normal.to_array())
        m_model_vertices.extend([1.0, 1.0, 0.0])
    m_model_vertices = np.array(m_model_vertices, dtype=np.float32)
    print(f'共{len(m_model) * 3}个顶点')

    # model VAO
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, m_model_vertices.nbytes, m_model_vertices, GL_STATIC_DRAW)

    # 定义mesh顶点的读取方式
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, m_model_vertices.itemsize * 9, ctypes.c_void_p(0))

    # 定义mesh法向量的读取方式
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, m_model_vertices.itemsize * 9, ctypes.c_void_p(m_model_vertices.itemsize * 3))

    # 定义mesh颜色的读取方式
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, m_model_vertices.itemsize * 9, ctypes.c_void_p(m_model_vertices.itemsize * 6))
    # endregion

    # model matrix
    model_loc = glGetUniformLocation(shader, 'model')
    scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([2, 2, 2]))  # 这里缩放，好像是数字越小，物体也越小
    model_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([400, 400, 0]))

    # view matrix
    view_loc = glGetUniformLocation(shader, 'view')
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # projection matrix
    proj_loc = glGetUniformLocation(shader, 'projection')
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 800, 0, 800, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    while not glfw.window_should_close(window):
        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.5 * glfw.get_time())

        rotation = pyrr.matrix44.multiply(rot_x, rot_y)
        model = pyrr.matrix44.multiply(scale, rotation)

        # 绘制STL模型
        glBindVertexArray(VAO)
        triangle_model = pyrr.matrix44.multiply(model, model_pos)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, triangle_model)
        # glDrawArrays(GL_TRIANGLES, 0, len(m_model) * 3)  # 每个网格三个顶点
        glDrawArrays(GL_LINES, 0, len(m_model) * 3)  # 每个网格三个顶点
        glfw.poll_events()

    glfw.terminate()


def window_resize(windows, width, height):
    """
    窗口调整大小
    """
    glViewport(0, 0, width, height)


# action：动作GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
# mods： 辅助键ALT，CTRL，SHIFT，META等
def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        print(f'鼠标左键按下：{glfw.get_cursor_pos(window)}')
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        print(f'鼠标左键抬起：{glfw.get_cursor_pos(window)}')
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
        print(f'鼠标中键按下：{glfw.get_cursor_pos(window)}')
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
        print(f'鼠标中键抬起：{glfw.get_cursor_pos(window)}')


def key_callback(windows, key, scancode, action, mods):
    if key == glfw.KEY_SPACE and action == glfw.REPEAT:
        print(f'空格键持续按下')  # 自动旋转功能
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        print(f'按下空格键')  # 恢复默认视角
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(windows, True)  # 关闭窗口


if __name__ == "__main__":
    glfw_test_github()
