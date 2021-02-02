import BaseClass
import numpy as np

FLT_EPSILON = 1.192092896e-07
DBL_MAX = 1.7976931348623158e+308


def fit_line_2d_weight_of_dist(x_tuple_of_points, x_count, x_tuple_weights):
    """
    带权重的最小二乘法拟合直线
    @param x_tuple_of_points: 点的集合
    @param x_count: 点的数量
    @param x_tuple_weights: 权重集合
    @return: 返回一个列表，包含一个直线上的点和方向向量
    """
    x, y, x2, y2, xy, w = 0, 0, 0, 0, 0, 0

    for i in range(x_count):
        x_point = x_tuple_of_points[i]
        x_weight = x_tuple_weights[i]
        x += x_weight * x_point.x
        y += x_weight * x_point.y
        x2 += x_weight * x_point.x ** 2
        y2 += x_weight * x_point.y ** 2
        xy += x_weight * x_point.x * x_point.y
        w += x_weight

    x /= w
    y /= w
    x2 /= w
    y2 /= w
    xy /= w

    dx2 = x2 - x ** 2
    dy2 = y2 - y ** 2
    dxy = xy - x * y
    t = np.arctan2(2 * dxy, dx2 - dy2) / 2

    return [np.cos(t), np.sin(t), x, y]


def std_deviation(x_d, x_count):
    """
    计算标准差
    @param x_d: 数集
    @param x_count: 集合的大小
    @return:
    """
    assert x_count > 2
    x_dist = x_d[:]
    x_dist.sort()
    is_odd = (x_count % 2 != 0)
    if is_odd:
        x_median = x_dist[x_count // 2 + 1]
    else:
        x_median = (x_dist[x_count // 2 - 1] + x_dist[x_count // 2]) / 2
    return x_median / 0.6745


def weight_huber(x_d, x_count, x_c):
    """
    huber方法计算权重
    @param x_d:数集
    @param x_count:集合的大小
    @param x_c:常数项参数
    @return:
    """
    x_w = [0.0] * x_count
    std_dev = std_deviation(x_d, x_count)
    c = 1.345 * std_dev if x_c <= 0 else x_c
    for i in range(x_count):
        if x_d[i] < c:
            x_w[i] = 1.0
        else:
            x_w[i] = c / x_d[i]
    return x_w


def weight_tukey(x_d, x_count, x_c):
    """
    tukey方法计算权重
    @param x_d: 数集
    @param x_count: 集合的大小
    @param x_c: 常数项参数
    @return:
    """
    x_w = [0.0] * x_count
    std_dev = std_deviation(x_d, x_count)
    c = 2 * std_dev if x_c <= 0 else x_c
    for i in range(x_count):
        if x_d[i] < c:
            d1 = x_d[i] / c
            d2 = 1 - d1 ** 2
            x_w[i] = d2 ** 2
        else:
            x_w[i] = 0.0
    return x_w


def calc_dist_2d(x_tuple_of_points, x_count, x_line, x_dist):
    """
    计算每个点到直线的距离，不包含权重
    @param x_tuple_of_points:
    @param x_count:
    @param x_line:
    @param x_dist:
    @return:
    """
    px = x_line[2]
    py = x_line[3]
    nx = x_line[1]
    ny = -x_line[0]
    sum_dist = 0

    for i in range(x_count):
        x = x_tuple_of_points[i].x - px
        y = x_tuple_of_points[i].y - py
        x_dist[i] = np.fabs(nx * x + ny * y)
        sum_dist += x_dist[i]

    return sum_dist


def calc_weight_dist_2d(x_tuple_of_points, x_count, x_line, x_weight, x_dist):
    """
    计算每个点到直线的距离，包含权重
    @param x_tuple_of_points:
    @param x_count:
    @param x_line:
    @param x_weight:
    @param x_dist:
    @return:
    """
    px = x_line[2]
    py = x_line[3]
    nx = x_line[1]
    ny = -x_line[0]
    sum_dist = 0

    for i in range(x_count):
        x = x_tuple_of_points[i].x - px
        y = x_tuple_of_points[i].y - py
        x_dist[i] = np.fabs(nx * x + ny * y) * x_weight[i]
        sum_dist += x_dist[i]

    return sum_dist


def fit_line_2d(x_tuple_of_points, x_count, x_dist, x_c, x_reps, x_aeps):
    """
    拟合2D直线
    @param x_tuple_of_points:
    @param x_count:
    @param x_dist:
    @param x_c:
    @param x_reps:
    @param x_aeps:
    @return:
    """
    r_delta = x_reps if x_reps != 0 else 1.0
    a_delta = x_aeps if x_aeps != 0 else 0.01
    min_err, err = DBL_MAX, 0
    _line_prev = [0.0] * 4
    eps = x_count * FLT_EPSILON
    x_w = [0.0] * x_count
    r = [0.0] * x_count
    result_line = None
    # 迭代上限为20次
    for k in range(20):

        is_first = True

        for i in range(min([2, x_count])):
            j = int(np.random.uniform(0, x_count))
            if x_w[j] < FLT_EPSILON:
                x_w[j] = 1.0
        _line = fit_line_2d_weight_of_dist(x_tuple_of_points, x_count, x_w)

        for j in range(30):
            if is_first:
                is_first = False
            else:
                t = _line[0] * _line_prev[0] + _line[1] * _line_prev[1]
                t = t if t > -1 else -1
                t = t if t < 1 else 1
                if np.fabs(np.arccos(t) < a_delta):
                    x = np.fabs(_line[2] - _line_prev[2])
                    y = np.fabs(_line[3] - _line_prev[3])
                    d = x if x > y else y
                    if d < r_delta:
                        break
            calc_dist_2d(x_tuple_of_points,x_count,_line,r)

            x_w = weight_huber(r, x_count, x_c)

            sum_w = np.sum(x_w)

            if np.fabs(sum_w) > FLT_EPSILON:
                sum_w = 1 / sum_w
                for i in range(x_count):
                    x_w[i] = x_w[i] * sum_w
            else:
                for i in range(x_count):
                    x_w[i] = 1.0

            err = calc_weight_dist_2d(x_tuple_of_points, x_count, _line, x_w, r)
            if err < eps:
                break
            _line_prev = _line[:]
            _line = fit_line_2d_weight_of_dist(x_tuple_of_points, x_count, x_w)

        if err < min_err:
            min_err = err
            result_line = _line[:]
            if err < eps:
                break
    return result_line


def read_point(x_point_path):
    x_point_list = []
    with open(x_point_path, 'r') as f:
        for x_point_str in f.readlines():
            x_point = BaseClass.Point3D(*map(float, x_point_str.split(',')[:3]))
            x_point_list.append(x_point)
    return x_point_list


if __name__ == '__main__':
    m_point_list = read_point(r"D:\全局标定测试\拟合直线边缘点.txt")
    m_result_line = fit_line_2d(m_point_list, len(m_point_list), 0, 0, 0, 0)

    x_origin = BaseClass.Point3D(m_result_line[2], m_result_line[3], 0)
    x_direction = BaseClass.Point3D(m_result_line[0], m_result_line[1], 0)
    m_line = BaseClass.Line3D(x_origin, x_direction)
    m_line.save(r"D:\全局标定测试\LineFit3D.txt")
