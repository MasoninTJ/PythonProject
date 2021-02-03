from Class3D import *

import numpy as np


class Matrix():
    def __init__(self, value: (list, np.ndarray), shape=(3, 3)):
        self.value: np.ndarray = np.array(value).reshape(shape)

    def __str__(self):
        return self.value.__str__()

    def to_array(self):
        return self.value.flatten()


if __name__ == '__main__':
    matrix = Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(matrix)
    print(matrix.to_array())
