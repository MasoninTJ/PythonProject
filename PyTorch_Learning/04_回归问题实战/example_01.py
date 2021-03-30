import numpy as np


def gradient_w(curr_w, curr_b, points):
    result = 2 * points[:, 0] * (curr_w * points[:, 0] + curr_b - points[:, 1])
    return np.mean(result)


def gradient_b(curr_w, curr_b, points):
    result = 2 * (curr_w * points[:, 0] + curr_b - points[:, 1])
    return np.mean(result)


def error(curr_w, curr_b, points):
    result = (curr_w * points[:, 0] + curr_b - points[:, 1]) ** 2
    return np.mean(result)


test_points = np.genfromtxt("data.csv", delimiter=",")

init_w = 0
init_b = 0
learning_rate = 0.0001
points_count = len(test_points)

b = init_b
w = init_w
for i in range(1000):
    w -= (gradient_w(w, b, test_points) * learning_rate)
    b -= (gradient_b(w, b, test_points) * learning_rate)
    print(f'w={w}, b={b}, error={error(w, b, test_points)}')
