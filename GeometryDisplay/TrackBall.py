from typing import Tuple

import numpy as np
from OpenGL.GL import *

from Alogrithm.BaseAlgorithm import intersection_of_ray_and_sphere, get_rotate_matrix_from_two_vector
from Geometry3D.Class3D import Vector3D, Point3D, Sphere, Ray3D
from Matrix import Matrix4d


class TrackBall:
    def __init__(self, m_center=Point3D(0, 0, 0), m_radius=1):
        self.center = m_center
        self.radius = m_radius

        self.rotate = Matrix4d.identity()
        self.translate = Matrix4d.from_translation(Vector3D(0, 0, 0))
        self.scale = Matrix4d.from_scale(1)

    def update_track_ball(self, m_rotate_matrix, m_translate_matrix, m_scale_matrix):
        self.rotate *= m_rotate_matrix
        self.translate *= m_translate_matrix
        self.scale *= m_scale_matrix
        self.result_matrix()

    def resize_track_ball(self, m_width, m_height):
        """
        根据窗口的大小，调整 track Ball 的大小,确保比窗口尺寸稍微大一点
        """
        m_radius = (m_height ** 2 + m_width ** 2) ** 0.5 + 1
        self.radius = m_radius

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


def draw_sphere_icon(m_sphere_radius=0.3):
    # 绘制轨迹球的动画
    glEnable(GL_LINE_SMOOTH)  # 开启平滑
    glPushMatrix()

    red_color = (1, 0, 0, 1)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red_color)
    draw_circle(m_sphere_radius, m_color=red_color)

    glRotatef(90, 1, 0, 0)
    green_color = (0, 1, 0, 1)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, green_color)
    draw_circle(m_sphere_radius, m_color=green_color)

    glRotatef(90, 0, 1, 0)
    blue_color = (0, 0, 1, 1)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, blue_color)
    draw_circle(m_sphere_radius, m_color=blue_color)

    glPopMatrix()


def draw_circle(m_radius=1.0, m_circle_step=100, m_color=(0, 0, 1, 1)):
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


def rotate_object(m_track_ball: TrackBall, m_begin_point, m_end_point):
    m_sphere = Sphere(m_track_ball.center, m_track_ball.radius)
    m_begin_point = Point3D(*m_begin_point, m_track_ball.radius)
    m_end_point = Point3D(*m_end_point, m_track_ball.radius)
    m_begin_ray = Ray3D(m_begin_point, Vector3D(0, 0, -1))
    m_end_ray = Ray3D(m_end_point, Vector3D(0, 0, -1))
    m_begin_point = intersection_of_ray_and_sphere(m_begin_ray, m_sphere)
    m_end_point = intersection_of_ray_and_sphere(m_end_ray, m_sphere)
    m_begin_vector = Vector3D(*m_begin_point.to_array())
    m_end_vector = Vector3D(*m_end_point.to_array())
    rotate_matrix = get_rotate_matrix_from_two_vector(m_begin_vector, m_end_vector)
    print(rotate_matrix)
    return rotate_matrix
