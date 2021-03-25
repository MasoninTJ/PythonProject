import numpy as np
from OpenGL.GL import *

from Geometry3D.Class3D import Vector3D, Point3D
from Matrix import Matrix4d


class TrackBall:
    def __init__(self, m_center=Point3D(0, 0, 0), m_radius=1):
        self.center = m_center
        self.radius = m_radius

        self.rotate = Matrix4d.identity()
        self.translate = Matrix4d.from_translation(Vector3D(0, 0, 0))
        self.scale = Matrix4d.from_scale(1)

    def result_matrix(self):
        """
        计算最终的转换矩阵 SCTR(-C)顺序
        Q=S[R(P-C)+T+C]
        C为旋转中心，T为平移量，R为旋转量，S为缩放量，
        P为转换前的点，Q为转换后的点
        :return:
        """
        matrix_minus_center = Matrix4d.from_translation(Point3D(0, 0, 0) - self.center)
        matrix_center = Matrix4d.from_translation(self.center - Point3D(0, 0, 0))
        return self.scale * matrix_center * self.translate * self.rotate * matrix_minus_center


def draw_sphere_coin(m_sphere_radius=0.3):
    # 绘制轨迹球的动画
    glEnable(GL_LINE_SMOOTH)  # 开启平滑

    glPushMatrix()

    red_color = (1, 0, 0, 0)
    draw_circle(m_sphere_radius, m_color=red_color)

    glRotatef(90, 1, 0, 0)
    green_color = (0, 1, 0, 0)
    draw_circle(m_sphere_radius, m_color=green_color)

    glRotatef(90, 0, 1, 0)
    blue_color = (0, 0, 1, 0)
    draw_circle(m_sphere_radius, m_color=blue_color)

    glPopMatrix()


def draw_circle(m_radius=1.0, m_circle_step=100, m_color=(0, 0, 1, 0)):
    """
    在(0,0)原点位置画一个圆
    :param m_radius:圆的半径
    :param m_circle_step:圆的步进数，即分成多少份
    :param m_color:颜色
    :return:
    """
    glColor4f(*m_color)
    glBegin(GL_LINE_LOOP)
    for index in range(m_circle_step):
        index_angle = 2 * np.pi * index / m_circle_step
        glVertex2f(m_radius * np.cos(index_angle), m_radius * np.sin(index_angle))
    glEnd()
