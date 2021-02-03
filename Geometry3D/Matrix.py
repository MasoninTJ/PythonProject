from Class3D import *

import numpy as np


class Matrix3d():
    def __init__(self, value: (list, np.ndarray), shape=(3, 3)):
        self.value: np.ndarray = np.array(value).reshape(shape)

    def __str__(self):
        return self.value.__str__()

    def to_array(self):
        return self.value.flatten()

    def inv(self):
        return np.linalg.inv(self.value)

    def transpose(self):
        return self.value.transpose()

    @staticmethod
    def identity():
        return np.identity(3)

    @staticmethod
    def zeros():
        return np.zeros(shape=(3, 3))


if __name__ == '__main__':
    matrix = Matrix3d([2, 2, 3, 4, 5, 6, 7, 8, 9])

    print(Matrix3d.zeros())
