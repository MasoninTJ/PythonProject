import numpy as np


def loss_to_w(curr_w, curr_b, x, y):
    return 2 * x * (curr_w * x + curr_b - y)


def loss_to_b(curr_w, curr_b, x, y):
    return 2 * (curr_w * x + curr_b - y)


def read_data():
    points = np.genfromtxt("data.csv", delimiter=",")
    return points


def error(w, b, points):
    list_error = []
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        list_error.append((y - w * x - b) ** 2)
    return sum(list_error) / len(points)


def run():
    init_w = 0
    init_b = 0
    learning_rate = 0.0001
    points = read_data()
    points_count = len(points)

    b = init_b
    w = init_w
    for i in range(1000):
        b_gradient_list = []
        w_gradient_list = []
        for j in range(points_count):
            x = points[j, 0]
            y = points[j, 1]
            w_gradient_list.append(loss_to_w(w, b, x, y))
            b_gradient_list.append(loss_to_b(w, b, x, y))
        w_gradient = sum(w_gradient_list) / points_count
        b_gradient = sum(b_gradient_list) / points_count
        w -= (w_gradient * learning_rate)
        b -= (b_gradient * learning_rate)
        print(f'w={w}, b={b}, error={error(w, b, points)}')


if __name__ == '__main__':
    run()