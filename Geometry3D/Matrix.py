from Class3D import *

import numpy as np


class Matrix3d():
    def __init__(self, value: (list, np.ndarray)):
        shape = (3, 3)
        self._value: np.ndarray = np.array(value).reshape(shape)

    def __mul__(self, other):
        if isinstance(other, Point3D):
            return Point3D(*np.dot(self._value,other.to_array()))
        elif isinstance(other, Vector3D):
            return Vector3D(*np.dot(self._value,other.to_array()))
        elif isinstance(other, Matrix3d):
            return Matrix3d(*np.dot(self._value,other._value))
        elif isinstance(other, np.ndarray):
            return Matrix3d(*np.dot(self._value,other))
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


if __name__ == '__main__':
    matrix = Matrix3d([[1.000000e+00, 0.000000e+00, 0.000000e+00],
                   [0.000000e+00, 6.123234e-17, 1.000000e+00],
                   [-0.000000e+00, -1.000000e+00, 6.123234e-17]])

    print(matrix.inv())

    point = Point3D(0,0,1)
    print(matrix * point)
