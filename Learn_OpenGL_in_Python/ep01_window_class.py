# -*- encoding: utf-8 -*-
"""
使用面向对象的方式进行glfw窗口的初始化
"""

import glfw


class Window:
    def __init__(self, width, height, xpos, ypos):
        if not glfw.init():
            raise Exception('glfw can not be initialized !')

        self._win = glfw.create_window(width, height, 'My Windows Class', None, None)

        if not self._win:
            glfw.terminate()
            raise Exception('glfw window can not be created !')

        glfw.set_window_pos(self._win, xpos, ypos)
        glfw.make_context_current(self._win)

    def main_loop(self):
        while not glfw.window_should_close(self._win):
            glfw.poll_events()
            glfw.swap_buffers(self._win)
        glfw.terminate()


if __name__ == '__main__':
    test_window = Window(800, 600, 2480, 240)
    test_window.main_loop()
