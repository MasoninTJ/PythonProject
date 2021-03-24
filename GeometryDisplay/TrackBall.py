import numpy as np
from OpenGL.GL import *


class TrackBall:
    def __init__(self):
        pass


def draw_sphere_coin():
    # 绘制轨迹球的动画
    m_sphere_radius = 0.3
    draw_circle(m_sphere_radius, m_color=(1, 0, 0, 0))
    glRotatef(np.pi/2, 1, 0, 0)
    draw_circle(m_sphere_radius, m_color=(0, 1, 0, 0))
    glRotatef(np.pi / 2, 0, 1, 0)
    draw_circle(m_sphere_radius, m_color=(0, 0, 1, 0))
    glLoadIdentity()


def draw_circle(m_radius=1.0, m_point_num=100, m_color=(0, 0, 1, 0)):
    # 在(0,0)原点位置画一个圆
    glColor4f(*m_color)
    glBegin(GL_LINE_LOOP)
    for index in range(m_point_num):
        index_angle = 2 * np.pi * index / m_point_num
        glVertex2f(m_radius * np.cos(index_angle), m_radius * np.sin(index_angle))
    glEnd()
    glFlush()
