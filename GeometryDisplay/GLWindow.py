import glfw

from GeometryDisplay.TrackBall import *

track_ball = TrackBall()
rotate_matrix = None


def window_resize(windows, m_width, m_height):
    """
    窗口调整大小回调函数
    :param windows:需要调整尺寸的窗口
    :param m_width:调整后的宽度
    :param m_height:调整后的高度
    :return:
    """
    glViewport(0, 0, m_width, m_height)


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
    m_begin_point = ()
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        # 鼠标左键按下，返回相对于窗口左上角的起始点坐标
        b_left_button_down = True
        m_begin_point = glfw.get_cursor_pos(window)
        return m_begin_point
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        # 鼠标左键抬起，返回相对于窗口左上角的结束点坐标
        b_left_button_down = False
        m_end_point = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
        print(f'鼠标中键按下：{glfw.get_cursor_pos(window)}')
    if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
        print(f'鼠标中键抬起：{glfw.get_cursor_pos(window)}')

    if b_left_button_down:
        m_end_point = glfw.get_cursor_pos(window)
        rotate_matrix = rotate_object(track_ball, m_begin_point, m_end_point)
        m_begin_point = m_end_point


def mouse_move_callback(window, xpos, ypos):
    pass


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

    def main_loop(self):
        while not glfw.window_should_close(self._win):
            # 开始渲染
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            draw_sphere_icon()

            glfw.swap_buffers(self._win)
            glfw.poll_events()

        glfw.terminate()


if __name__ == '__main__':
    test_window = GLWindow()
    test_window.main_loop()
