import BaseClass
import BaseAlgorithm
import numpy as np

FLT_EPSILON = 1.192092896e-07
DBL_MAX = 1.7976931348623158e+308


def fit_sphere_zzz(x_tuple_of_point):
    x_array_of_point = []
    for x in x_tuple_of_point:
        x_array_of_point.append(x.to_array())
    points = np.array(x_array_of_point)  # coor为点云坐标的列表
    points = points.astype(np.uint64)  # 防止溢出
    num_points = points.shape[0]
    print(num_points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_avr = sum(x) / num_points
    y_avr = sum(y) / num_points
    z_avr = sum(z) / num_points
    xx_avr = sum(x * x) / num_points
    yy_avr = sum(y * y) / num_points
    zz_avr = sum(z * z) / num_points
    xy_avr = sum(x * y) / num_points
    xz_avr = sum(x * z) / num_points
    yz_avr = sum(y * z) / num_points
    xxx_avr = sum(x * x * x) / num_points
    xxy_avr = sum(x * x * y) / num_points
    xxz_avr = sum(x * x * z) / num_points
    xyy_avr = sum(x * y * y) / num_points
    xzz_avr = sum(x * z * z) / num_points
    yyy_avr = sum(y * y * y) / num_points
    yyz_avr = sum(y * y * z) / num_points
    yzz_avr = sum(y * z * z) / num_points
    zzz_avr = sum(z * z * z) / num_points

    A = np.array([[xx_avr - x_avr * x_avr, xy_avr - x_avr * y_avr, xz_avr - x_avr * z_avr],
                  [xy_avr - x_avr * y_avr, yy_avr - y_avr * y_avr, yz_avr - y_avr * z_avr],
                  [xz_avr - x_avr * z_avr, yz_avr - y_avr * z_avr, zz_avr - z_avr * z_avr]])
    b = np.array([xxx_avr - x_avr * xx_avr + xyy_avr - x_avr * yy_avr + xzz_avr - x_avr * zz_avr,
                  xxy_avr - y_avr * xx_avr + yyy_avr - y_avr * yy_avr + yzz_avr - y_avr * zz_avr,
                  xxz_avr - z_avr * xx_avr + yyz_avr - z_avr * yy_avr + zzz_avr - z_avr * zz_avr])
    # print(A, b)
    b = b / 2
    center = np.linalg.solve(A, b)
    x0 = center[0]
    y0 = center[1]
    z0 = center[2]
    r2 = xx_avr - 2 * x0 * x_avr + x0 * x0 + yy_avr - 2 * y0 * y_avr + y0 * y0 + zz_avr - 2 * z0 * z_avr + z0 * z0
    r = r2 ** 0.5
    return BaseClass.Sphere(BaseClass.Point3D(*center[0:3]), r)


def fit_sphere_simple(x_tuple_of_points, x_max_iteration):
    x_count = len(x_tuple_of_points)
    sphere_center_init = BaseAlgorithm.get_average_center(x_tuple_of_points)
    x_sphere = BaseClass.Sphere(sphere_center_init)
    for i in range(x_max_iteration):
        current_center = x_sphere.center

        d_der_len_avg_x, d_der_len_avg_y, d_der_len_avg_z = 0, 0, 0
        d_len_avg = 0
        for x_point in x_tuple_of_points:
            d_dif_x = x_point.x - current_center.x
            d_dif_y = x_point.y - current_center.y
            d_dif_z = x_point.z - current_center.z
            d_len = (d_dif_x ** 2 + d_dif_y ** 2 + d_dif_z ** 2) ** (1/2)

            if d_len > 0:
                d_len_avg += d_len
                d_der_len_avg_x -= d_dif_x / d_len
                d_der_len_avg_y -= d_dif_y / d_len
                d_der_len_avg_z -= d_dif_z / d_len

        d_len_avg /= x_count
        d_der_len_avg_x /= x_count
        d_der_len_avg_y /= x_count
        d_der_len_avg_z /= x_count

        x_sphere.center.x = sphere_center_init.x + d_len_avg * d_der_len_avg_x
        x_sphere.center.y = sphere_center_init.y + d_len_avg * d_der_len_avg_y
        x_sphere.center.z = sphere_center_init.z + d_len_avg * d_der_len_avg_z

        x_sphere.r = d_len_avg

        d_diff_x = x_sphere.center.x - current_center.x
        d_diff_y = x_sphere.center.y - current_center.y
        d_diff_z = x_sphere.center.z - current_center.z

        d_max = max([d_diff_x, d_diff_y, d_diff_z])

        if d_max < 1e-5:
            break

    return x_sphere


def read_point(x_point_path):
    x_point_list = []
    with open(x_point_path, 'r') as f:
        for x_point_str in f.readlines():
            x_point = BaseClass.Point3D(*map(float, x_point_str.split(',')[:3]))
            x_point_list.append(x_point)
    return x_point_list


if __name__ == '__main__':
    m_point = read_point(r"D:\全局标定测试\Sphere5.txt")
    m_sphere = fit_sphere_zzz(m_point)
    print(m_sphere)
    m_sphere.save(r"D:\全局标定测试\Sphere_Fit5.txt")