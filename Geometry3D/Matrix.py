from Class3D import *

import numpy as np


class Matrix3d:
    def __init__(self, value: (list, np.ndarray)):
        shape = (3, 3)
        self._value: np.ndarray = np.array(value).reshape(shape)

    def __mul__(self, other):
        if isinstance(other, Point3D):
            return Point3D(*np.dot(self._value, other.to_array()))
        elif isinstance(other, Vector3D):
            return Vector3D(*np.dot(self._value, other.to_array()))
        elif isinstance(other, Matrix3d):
            return Matrix3d(*np.dot(self._value, other._value))
        elif isinstance(other, np.ndarray):
            return Matrix3d(*np.dot(self._value, other))
        else:
            return None

    def __str__(self):
        return self._value.__str__()

    def to_array(self):
        return self._value.flatten()

    def inv(self):
        return np.linalg.inv(self._value)

    def transpose(self):
        return self._value.transpose()

    @staticmethod
    def identity():
        return np.identity(3)

    @staticmethod
    def zeros():
        return np.zeros(shape=(3, 3))

    @staticmethod
    def from_euler_angle(m_euler_x_rad: (float, int), m_euler_y_rad: (float, int), m_euler_z_rad: (float, int)):
        """
        欧拉角转旋转矩阵，这里使用的旋转方式为 x-y-z 内旋，欧拉角使用弧度表示。
        """
        c1, s1 = np.cos(m_euler_x_rad), np.sin(m_euler_x_rad)
        c2, s2 = np.cos(m_euler_y_rad), np.sin(m_euler_y_rad)
        c3, s3 = np.cos(m_euler_z_rad), np.sin(m_euler_z_rad)

        return Matrix3d([[c2 * c3, -c2 * s3, s2],
                         [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                         [-c1 * c3 * s2 + s1 * s3, c1 * s2 * s3 + c3 * s1, c1 * c2]])

    @staticmethod
    def from_axis_angle(m_axis: Vector3D, m_angle_rad: (int, float)):
        """
        轴角转旋转矩阵，这里的轴必须为单位向量，角度使用弧度表示
        """
        if not m_axis.check_normal():
            m_axis = m_axis.normalize()

        x, y, z = m_axis.to_array()
        c, s = np.cos(m_angle_rad), np.sin(m_angle_rad)
        x2, y2, z2 = x ** 2, y ** 2, z ** 2
        return Matrix3d([[x2 + (1 - x2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                         [x * y * (1 - c) + z * s, y2 + (1 - y2) * c, y * z * (1 - c) - x * s],
                         [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z2 + (1 - z2) * c]])


class Matrix4d:
    def __init__(self, value: (list, np.ndarray)):
        shape = (4, 4)
        self._value: np.ndarray = np.array(value).reshape(shape)

    def __mul__(self, other):
        matrix = self._value
        if isinstance(other, Point3D):  # 这里不想引入齐次形式，就直接写了
            m_x = other.x * matrix[0, 0] + other.y * matrix[0, 1] + other.z * matrix[0, 2] + matrix[0, 3]
            m_y = other.x * matrix[1, 0] + other.y * matrix[1, 1] + other.z * matrix[1, 2] + matrix[1, 3]
            m_z = other.x * matrix[2, 0] + other.y * matrix[2, 1] + other.z * matrix[2, 2] + matrix[2, 3]
            return Point3D(m_x, m_y, m_z)
        elif isinstance(other, Vector3D):  # 这里不想引入齐次形式，就直接写了
            m_i = other.i * matrix[0, 0] + other.j * matrix[0, 1] + other.k * matrix[0, 2]
            m_j = other.i * matrix[1, 0] + other.j * matrix[1, 1] + other.k * matrix[1, 2]
            m_k = other.i * matrix[2, 0] + other.j * matrix[2, 1] + other.k * matrix[2, 2]
            return Vector3D(m_i, m_j, m_k)
        elif isinstance(other, Matrix4d):
            return Matrix4d(*np.dot(self._value, other._value))
        elif isinstance(other, np.ndarray):
            return Matrix4d(*np.dot(self._value, other))
        else:
            return None

    def __str__(self):
        return self._value.__str__()

    def to_array(self):
        return self._value.flatten()

    def inv(self):
        return np.linalg.inv(self._value)

    def transpose(self):
        return self._value.transpose()

    @staticmethod
    def identity():
        return np.identity(4)

    @staticmethod
    def zeros():
        return np.zeros(shape=(4, 4))

    @staticmethod
    def from_rotate(m_rotate_matrix: Matrix3d):
        matrix = Matrix4d.identity()
        matrix[0, :] = *m_rotate_matrix._value[0, :], 0
        matrix[1, :] = *m_rotate_matrix._value[1, :], 0
        matrix[2, :] = *m_rotate_matrix._value[2, :], 0
        matrix[3, :] = 0, 0, 0, 1
        return matrix

    @staticmethod
    def from_translation(m_translation:Vector3D):
        matrix = Matrix4d.identity()
        matrix[0:3,3] = m_translation.to_array()
        return matrix

    @staticmethod
    def from_scale(scale:(int,float)):
        matrix = Matrix4d(np.diag((scale,scale,scale,1)))
        return matrix


if __name__ == '__main__':
    matrix = Matrix3d([[1.000000e+00, 0.000000e+00, 0.000000e+00],
                       [0.000000e+00, 6.123234e-17, 1.000000e+00],
                       [-0.000000e+00, -1.000000e+00, 6.123234e-17]])

    print(Matrix4d.from_rotate(matrix))

    print(Matrix4d.from_translation(Vector3D(1,2,3)))

    print(Matrix4d.from_scale(2))
