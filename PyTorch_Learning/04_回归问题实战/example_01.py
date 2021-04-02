import numpy as np


def gradient_w(curr_w, curr_b, points):
    x_array = points[:, 0]
    y_array = points[:, 1]
    result = 2 * x_array * (curr_w * x_array + curr_b - y_array)
    return np.mean(result)


def gradient_b(curr_w, curr_b, points):
    x_array = points[:, 0]
    y_array = points[:, 1]
    result = 2 * (curr_w * x_array + curr_b - y_array)
    return np.mean(result)


def loss(curr_w, curr_b, points):
    x_array = points[:, 0]
    y_array = points[:, 1]
    result = (curr_w * x_array + curr_b - y_array) ** 2
    return np.mean(result)


# 读取数据
test_points = np.genfromtxt("data.csv", delimiter=",")

# 设置初值与学习率
init_w = 0
init_b = 0
learning_rate = 0.0001

b = init_b
w = init_w
for i in range(1000):
    w -= (gradient_w(w, b, test_points) * learning_rate)
    b -= (gradient_b(w, b, test_points) * learning_rate)
    print(f'w={w}, b={b}, error={loss(w, b, test_points)}')
