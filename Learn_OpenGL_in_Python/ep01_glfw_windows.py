# -*- encoding: utf-8 -*-
"""
glfw窗口初始化，这里使用的是 pyglfw github 主页提供的测试用例
"""

import glfw


def glfw_test_github():
    # Initialize the library
    # 初始化库
    if not glfw.init():
        raise Exception('glfw can not be initialized!')

    # Create a windowed mode window and its OpenGL context
    # 创建窗口
    window = glfw.create_window(640, 480, "Hello World", None, None)
    if not window:
        glfw.terminate()
        raise Exception('glfw windows can not be created!')

    # 设置窗口位置
    glfw.set_window_pos(window, 200, 400)

    # Make the window's context current
    # 使创建的窗口成为当前上下文
    glfw.make_context_current(window)

    # Loop until the user closes the window
    # 循环这个窗口，直到用户关闭
    while not glfw.window_should_close(window):
        # Render here, e.g. using pyOpenGL

        # Swap front and back buffers
        # 交换缓冲区
        glfw.swap_buffers(window)

        # Poll for and process events
        # 处理活动事件
        glfw.poll_events()

    # 关闭窗口
    glfw.terminate()


if __name__ == "__main__":
    glfw_test_github()
