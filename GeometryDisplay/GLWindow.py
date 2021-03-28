import glfw
import pyrr
from OpenGL.GL.shaders import compileProgram, compileShader

from Alogrithm.LoadMesh import load_stl_model
from Class3D import STLModel
from GeometryDisplay.TrackBall import *

track_ball = TrackBall()
rotate_matrix = pyrr.matrix44.create_identity()
scale_matrix = pyrr.matrix44.create_from_scale(pyrr.Vector3([2, 2, 2]))  # 这里缩放，好像是数字越小，物体也越小
view_matrix = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
projection_matrix = pyrr.matrix44.create_orthogonal_projection_matrix(0, 800, 0, 800, -1000, 1000)


def window_resize(windows, m_width, m_height):
    """
    窗口调整大小回调函数
    :param windows:需要调整尺寸的窗口
    :param m_width:调整后的宽度
    :param m_height:调整后的高度
    :return:
    """
    glViewport(0, 0, m_width, m_height)
    track_ball.resize_track_ball(m_width, m_height)


def mouse_button_callback(window, button, action, mods):
    """
    鼠标点击回调函数
    :param window:鼠标点击窗口
    :param button:鼠标按键
    :param action:鼠标动作
    :param mods:粘滞键
    :return:
    """
    global rotate_matrix
    b_left_button_down = False
    m_begin_point = (0, 0)
    m_end_point= (0,0)
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        # 鼠标左键按下，返回相对于窗口左上角的起始点坐标
        b_left_button_down = True
        m_begin_point = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        # 鼠标左键抬起，返回相对于窗口左上角的结束点坐标
        b_left_button_down = False
        m_end_point = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
        print(f'鼠标中键按下：{glfw.get_cursor_pos(window)}')
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
        print(f'鼠标中键抬起：{glfw.get_cursor_pos(window)}')

    if b_left_button_down:
        print(m_begin_point, m_end_point)
        rotate_matrix = rotate_object(track_ball, m_begin_point, m_end_point)
        rotate_matrix = Matrix4d.from_rotate(rotate_matrix)
        track_ball.update_track_ball(m_rotate_matrix=rotate_matrix)
        m_begin_point = m_end_point


def key_callback(windows, key, scancode, action, mods):
    """
    键盘输入回调函数
    :param windows:键盘输入窗口
    :param key:键盘按键
    :param scancode:
    :param action:键盘动作，包含PRESS，RELEASE，REPEAT
    :param mods:粘滞键
    :return:
    """
    if key == glfw.KEY_SPACE and action == glfw.REPEAT:  # 空格长按自动旋转
        print(f'空格键持续按下')
    if key == glfw.KEY_SPACE and action == glfw.PRESS:  # 空格按一下恢复默认视角
        print(f'按下空格键')
    if key == glfw.KEY_ESCAPE:  # ESC关闭窗口
        glfw.set_window_should_close(windows, True)
        print(f'关闭窗口')


class GLWindow:
    def __init__(self, m_width=800, m_height=800, m_xpos=100, m_ypos=100, auto_size_pos=False):
        if not glfw.init():
            raise Exception('glfw can not be initialized !')

        if auto_size_pos:
            # 获取显示器1的尺寸，并自动调节窗口大小与位置
            m_monitor = glfw.get_monitors()
            glfw_video_mode = glfw.get_video_mode(m_monitor[0])
            m_scene_width, m_scense_height = glfw_video_mode.size
            # 自动调节的窗口长宽均为显示器1长宽的一半，左上角位于显示器长宽1/4位置处
            m_width = m_scene_width // 2
            m_height = m_scense_height // 2
            m_xpos = m_scene_width // 4
            m_ypos = m_scense_height // 4

        self._win = glfw.create_window(m_width, m_height, 'PyOpenGL Display Window', None, None)
        track_ball.resize_track_ball(m_width, m_height)

        if not self._win:
            glfw.terminate()
            raise Exception('glfw window can not be created !')

        # 设置窗口位置
        glfw.set_window_pos(self._win, m_xpos, m_ypos)
        # 设置回调函数
        glfw.set_window_size_callback(self._win, window_resize)
        glfw.set_mouse_button_callback(self._win, mouse_button_callback)
        glfw.set_key_callback(self._win, key_callback)

        # 当前窗口作为上下文
        glfw.make_context_current(self._win)

        # 背景颜色
        glClearColor(0.3, 0.5, 0.5, 1)
        glEnable(GL_DEPTH_TEST)  # 启用深度缓存
        glEnable(GL_BLEND)  # 启用混合
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # 定义混合因子

        # 顶点着色器与片段着色器
        m_vertex_src_path = './OpenGL_Vertex_Src'
        m_fragment_src_path = './OpenGL_Fragment_Src'
        m_vertex_src = open(m_vertex_src_path, 'r', encoding='utf-8').read()
        m_fragment_src = open(m_fragment_src_path, 'r', encoding='utf-8').read()
        shader = compileProgram(compileShader(m_vertex_src, GL_VERTEX_SHADER), compileShader(m_fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(shader)

        # 读取stl文件，并转化为可以读取的列表形式
        m_model_path = r'D:\下载\全局标定测试\单层NEY模型.stl'
        self.m_model: STLModel = load_stl_model(m_model_path)
        m_model_vertices = []
        for m_mesh in self.m_model:
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
        print(f'共{len(self.m_model) * 3}个顶点')

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

        # model VAO
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
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

        # model matrix
        self.model_loc = glGetUniformLocation(shader, 'model')
        self.model_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([400, 400, 0]))  # 模型位置

        # view matrix
        view_loc = glGetUniformLocation(shader, 'view')
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)

        # projection matrix
        proj_loc = glGetUniformLocation(shader, 'projection')
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)

    def main_loop(self):
        while not glfw.window_should_close(self._win):
            # 开始渲染
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            draw_sphere_icon()

            model_matrix = track_ball.result_matrix()  # trackBall 的旋转矩阵
            # 绘制STL模型
            glBindVertexArray(self.VAO)
            model_matrix = pyrr.matrix44.multiply(model_matrix.to_array().reshape((4, 4)), self.model_pos)
            glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model_matrix)
            glDrawArrays(GL_TRIANGLES, 0, len(self.m_model) * 3)  # 每个网格三个顶点

            glfw.swap_buffers(self._win)
            glfw.poll_events()

        glfw.terminate()


if __name__ == '__main__':
    test_window = GLWindow()
    test_window.main_loop()
