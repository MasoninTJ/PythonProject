import numpy as np

epsilon = 1e-6


def rodrigues(x_vector):
    """
    罗德里格斯变换:绕任意轴旋转某个角度
    x = [nx,ny,nz] = n*[x,y,z]
    [x,y,z]为单位向量，表示旋转轴
    n表示旋转角度，单位：弧度
    计算旋转矩阵

    :return: 变换矩阵
    """
    assert len(x_vector) == 3
    vector = np.array(x_vector)
    theta = np.linalg.norm(vector)

    s = np.sin(theta)
    c = np.cos(theta)
    c1 = 1.0 - c

    if theta < epsilon:
        itheta = 0
    else:
        itheta = 1 / theta
    vector = vector * itheta

    me = np.eye(3)
    rrt = np.dot(np.matrix(vector).T, np.matrix(vector))

    r_x = np.array([[0, -vector[2], vector[1]],
                    [vector[2], 0, -vector[0]],
                    [-vector[1], vector[0], 0]])

    r = me * c + r_x * s + rrt * c1
    return np.array(r)


def euler_angle_to_matrix(x_euler_angle):
    """
    欧拉角（单位：度）转换为旋转矩阵
    """
    assert len(x_euler_angle) == 3
    euler_angle = np.array(x_euler_angle)
    ax, ay, az = map(np.deg2rad, euler_angle)

    mx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    my = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    mz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])

    return np.dot(mz, np.dot(my, mx))


def is_rotate_matrix(x_matrix):
    """
    旋转矩阵为正交矩阵，正交矩阵与正交矩阵的逆的乘积为单位矩阵，行列式的值为1
    """
    matrix = np.array(x_matrix)
    if abs(1 - np.linalg.det(np.dot(matrix, matrix.T))) < epsilon:
        return True
    else:
        return False


def matrix_to_euler_angle(x_matrix):
    """
    旋转矩阵转换为欧拉角
    """
    # 判断是否旋转矩阵
    if is_rotate_matrix(x_matrix):
        temp = np.sqrt(x_matrix[0][0] ** 2 + x_matrix[1][0] ** 2)
        if not temp < epsilon:
            theta_x = np.arctan2(x_matrix[2][1], x_matrix[2][2])
            theta_y = np.arctan2(-x_matrix[2][0], temp)
            theta_z = np.arctan2(x_matrix[1][0], x_matrix[0][0])
        else:
            # 万向锁
            theta_x = np.arctan2(x_matrix[1][2], x_matrix[1][1])
            theta_y = np.arctan2(-x_matrix[2][0], temp)
            theta_z = 0
        return np.rad2deg(theta_x), np.rad2deg(theta_y), np.rad2deg(theta_z)
    else:
        return None


def test_unit():
    test_matrix = euler_angle_to_matrix([90, 90, 0])
    print('测试-旋转矩阵为：\n', test_matrix)
    print('测试-判断是否为旋转矩阵：\n', is_rotate_matrix(test_matrix))
    print('测试-矩阵转为欧拉角\n', matrix_to_euler_angle(test_matrix))
    print('测试-罗德里格斯变换:\n', rodrigues([-2.100418, -2.167796, 0.273330]))
    print('测试-罗德里格斯变换:\n', rodrigues([0, 0, 0]))


if __name__ == '__main__':
    test_unit()
