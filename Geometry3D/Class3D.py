from copy import deepcopy

import numpy as np


class Point2D:
    def __init__(self, mx=0.0, my=0.0):
        self.x = mx
        self.y = my

    def __add__(self, other):
        """
        点加向量表示平移
        @param other:
        @return:
        """
        if isinstance(other, Vector2D):
            return Point2D(self.x + other.i, self.y + other.j)
        else:
            return None

    def __sub__(self, other):
        """
        点减向量表示平移,点点相减表示向量
        @param other:
        @return:
        """
        if isinstance(other, Vector2D):
            return Point2D(self.x - other.i, self.y - other.j)
        elif isinstance(other, Point2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        else:
            return None

    def __str__(self):
        return f'({self.x:.3f},{self.y:.3f})'

    def to_array(self):
        """
        点转化为numpy数组
        """
        return np.array([self.x, self.y])


class Vector2D:
    def __init__(self, mi=0.0, mj=0.0):
        self.i = mi
        self.j = mj

    def __add__(self, other):
        """
        向量相加
        @param other:
        @return:
        """
        if isinstance(other, Vector2D):
            return Vector2D(self.i + other.i, self.j + other.j)
        else:
            return None

    def __sub__(self, other):
        """
        向量相减
        @param other:
        @return:
        """
        if isinstance(other, Vector2D):
            return Vector2D(self.i + other.i, self.j + other.j)
        else:
            return None

    def __mul__(self, other):
        """
        向量缩放
        @param other:
        @return:
        """
        if isinstance(other, (int, float)):
            return Vector2D(self.i * other, self.j * other)
        else:
            return None

    def __truediv__(self, other):
        """
        向量缩放
        @param other:
        @return:
        """
        if isinstance(other, (int, float)) and other:
            return Vector2D(self.i * other, self.j * other)
        else:
            return None

    def __str__(self):
        return f'({self.i:.3f},{self.j:.3f})'

    def to_array(self):
        """
        转化为numpy数组
        @return:
        """
        return np.array([self.i, self.j])

    def length(self):
        """
        向量长度
        @return:
        """
        return np.sqrt(self.i ** 2 + self.j ** 2)

    def normalize(self):
        """
        向量归一化
        @return:
        """
        length = self.length()
        return self / length


class Point3D:
    def __init__(self, mx=0.0, my=0.0, mz=0.0):
        self.x = mx
        self.y = my
        self.z = mz

    def __add__(self, other):
        """
        点加向量表示平移
        @param other:
        @return:
        """
        if isinstance(other, Vector3D):
            return Point3D(self.x + other.i, self.y + other.j, self.z + other.k)
        else:
            return None

    def __sub__(self, other):
        """
        点减向量表示平移,点点相减表示向量
        @param other:
        @return:
        """
        if isinstance(other, Vector3D):
            return Point3D(self.x - other.i, self.y - other.j, self.z - other.k)
        elif isinstance(other, Point3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return None

    def __str__(self):
        return f'({self.x:.3f},{self.y:.3f},{self.z:.3f})'

    def to_array(self):
        """
        点转化为numpy数组
        """
        return np.array([self.x, self.y, self.z])


class Vector3D:
    def __init__(self, mi=0.0, mj=0.0, mk=1.0):
        self.i = mi
        self.j = mj
        self.k = mk

    def __add__(self, other):
        """
        向量相加后仍为一个向量,向量加点后为一个点
        @param other:
        @return:
        """
        if isinstance(other, Vector3D):
            return Vector3D(self.i + other.i, self.j + other.j, self.k + other.k)
        elif isinstance(other, Point3D):
            return Point3D(self.i + other.x, self.j + other.y, self.k + other.z)
        else:
            return None

    def __sub__(self, other):
        """
        向量相减后仍为一个向量
        @param other:
        @return:
        """
        if isinstance(other, Vector3D):
            return Vector3D(self.i - other.i, self.j - other.j, self.k - other.k)
        else:
            return None

    def __mul__(self, other):
        """
        向量缩放
        @param other:
        @return:
        """
        if isinstance(other, (int, float)):
            return Vector3D(self.i * other, self.j * other, self.k * other)
        else:
            return None

    def __truediv__(self, other):
        """
        向量缩放
        @param other:
        @return:
        """
        if isinstance(other, (int, float)) and other:
            return Vector3D(self.i * other, self.j * other, self.k * other)
        else:
            return None

    def __str__(self):
        return f'({self.i:.3f},{self.j:.3f},{self.k:.3f})'

    def to_array(self):
        """
        转换为numpy数组
        @return:
        """
        return np.array([self.i, self.j, self.k])

    def length(self):
        """
        向量长度
        @return:
        """
        return np.sqrt(self.i ** 2 + self.j ** 2 + self.k ** 2)

    def normalize(self):
        """
        向量归一化
        @return:
        """
        length = self.length()
        return self / length

    def check_valid(self):
        return self.to_array().all()

    def check_normal(self):
        return self.length() == 1


class Line2D:
    def __init__(self, m_begin_point=Point2D(0, 0), m_end_point=Point2D(0, 0)):
        self.begin = deepcopy(m_begin_point)
        self.end = deepcopy(m_end_point)

    def __str__(self):
        return f'begin point:{self.begin},end point:{self.end}'

    def direction(self) -> Vector2D:
        """
        直线的方向
        @return:
        """
        return (self.end - self.begin).normalize()

    def get_point_from_t(self, m_t) -> Point2D:
        """
        根据参数值，获取直线上的点
        """
        return self.begin + self.direction() * m_t


class Line3D:
    def __init__(self, m_begin_point=Point3D(0, 0, 0), m_end_point=Point3D(0, 0, 1)):
        self.begin = deepcopy(m_begin_point)
        self.end = deepcopy(m_end_point)

    def __str__(self):
        return f'begin point:{self.begin},end point:{self.end}'

    def direction(self) -> Vector3D:
        """
        直线的方向
        @return:
        """
        return (self.end - self.begin).normalize()

    def get_point_from_t(self, m_t) -> Point3D:
        """
        根据参数值，获取直线上的点
        """
        return self.begin + self.direction() * m_t


class Plane:
    def __init__(self, m_point=Point3D(0, 0, 0), m_vector=Vector3D(0, 0, 1)):
        self.point = deepcopy(m_point)
        self.normal = deepcopy(m_vector.normalize())

    def __str__(self):
        return f'point:{self.point},normal:{self.normal}'


class Sphere:
    def __init__(self, m_center=Point3D(), m_radius=1):
        self.center = deepcopy(m_center)
        self.radius = m_radius

    def __str__(self):
        return f'center:{self.center},radius:{self.radius}'


class Triangle:
    def __init__(self, m_vertex1=Point3D(), m_vertex2=Point3D(), m_vertex3=Point3D()):
        self.vertex1 = deepcopy(m_vertex1)
        self.vertex2 = deepcopy(m_vertex2)
        self.vertex3 = deepcopy(m_vertex3)

    def __str__(self):
        return f'vertex1:{self.vertex1}, vertex2:{self.vertex2}, vertex3:{self.vertex3}'


class Mesh:
    def __init__(self, m_normal: Vector3D, m_vertex: Triangle):
        self.normal = m_normal
        self.vertex = m_vertex

    def __str__(self):
        return f'normal:{self.normal}, {self.vertex}'


class STLModel:
    def __init__(self, m_mesh_list):
        self.mesh_list = m_mesh_list

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, m_index):
        return self.mesh_list[m_index]

    def __str__(self):
        print('三角面片开始显示')
        for i, x in enumerate(self.mesh_list):
            print(i, ':', x)
        return '三角面片显示结束'

class Ray2D:
    def __init__(self, m_origin: Point2D, m_direction: Point2D):
        self.origin = m_origin
        self.direction = m_direction

    def __str__(self):
        return f'origin:{self.origin},direction:{self.direction}'


class Ray3D:
    def __init__(self, m_origin: Point3D, m_direction: Vector3D):
        self.origin = m_origin
        self.direction = m_direction

    def __str__(self):
        return f'origin:{self.origin},direction:{self.direction}'

    def get_point_from_t(self, t):
        if t > 0:
            return self.origin + self.direction * t
        else:
            return None


class Box3D:
    def __init__(self, m_min: Point3D = Point3D(-1, -1, -1), m_max: Point3D = Point3D(1, 1, 1)):
        self.min = m_min
        self.max = m_max

        if self.min.x > self.max.x or self.min.y > self.max.y or self.min.z > self.max.z:
            self.regularization()

    def regularization(self):
        """
        AABB包围盒正则化，使 max 的三个值都大于 min 的三个值
        """
        if self.min.x > self.max.x:
            self.min.x, self.max.x = self.max.x, self.min.x
        if self.min.y > self.max.y:
            self.min.y, self.max.y = self.max.y, self.min.y
        if self.min.z > self.max.z:
            self.min.z, self.max.z = self.max.z, self.min.z
