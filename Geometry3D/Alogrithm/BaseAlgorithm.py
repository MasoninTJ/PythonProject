import sys

from Class3D import *
import ConstMember


def cross(m_vec1: (Vector3D, Vector2D), m_vec2: (Vector3D, Vector2D)) -> (Vector3D, float):
    """
    在三维几何中，向量a和向量b的叉乘结果是一个向量，更为熟知的叫法是法向量，该向量垂直于a和b向量构成的平面
    在二维几何中，向量a和向量b的叉乘结果是一个值，该值表示两个向量围成的平行四边形的面积
    """
    if isinstance(m_vec1, Vector3D) and isinstance(m_vec2, Vector3D):
        return Vector3D(*np.cross(m_vec1.to_array(), m_vec2.to_array()))
    elif isinstance(m_vec1, Vector2D) and isinstance(m_vec2, Vector2D):
        return np.cross(m_vec1.to_array(), m_vec2.to_array())
    else:
        return None


def dot(m_vec1: (Vector3D, Vector2D), m_vec2: (Vector3D, Vector2D)) -> np.ndarray:
    """
    点乘的几何意义是可以用来表征或计算两个向量之间的夹角,以及在b向量在a向量方向上的投影
    dot(x,y) = |x| * |y| * cos
    """
    return np.dot(m_vec1.to_array(), m_vec2.to_array())


def intersection_of_line_and_plane(m_line: Line3D, m_plane: Plane) -> (Point3D, None):
    """
    计算直线与面的交点（不判断线在面上，此时有无穷多点）
    """
    if dot(m_line.direction(), m_plane.normal):  # 判断平行,这里使用的是绝对判断，后面可以优化为小于极小值
        temp = dot((m_plane.point - m_line.begin), m_plane.normal) / dot(m_line.direction(), m_plane.normal)
        return m_line.get_point_from_t(temp)
    else:
        return None


def intersection_ray_and_triangle(m_ray: Ray3D, m_triangle: Triangle) -> (Point3D, None):
    """
    射线与三角形的交点，如果没有交点则返回None
    """
    vec_ab = m_triangle.vertex2 - m_triangle.vertex1
    vec_ac = m_triangle.vertex3 - m_triangle.vertex1
    p = cross(m_ray.direction, vec_ac)
    a = dot(p, vec_ab)
    if -ConstMember.epsilon5 < a < ConstMember.epsilon5:  # 判断平行
        return None
    f = 1 / a
    s = m_ray.origin - m_triangle.vertex1
    u = f * dot(s, p)
    if u < 0:  # 点在三角形外
        return None
    q = cross(s, vec_ab)
    v = f * dot(m_ray.direction, q)
    if (v < 0 or u + v > 1):  # 点在三角形外
        return None

    t = f * np.dot(vec_ac, q)
    return m_ray.get_point_from_t(t)


def create_plane_from_3point(m_point1, m_point2, m_point3):
    """
    通过三个点构造面，计算平面的法向量,用点法式构造平面
    """
    x_vector1 = m_point1 - m_point2
    x_vector2 = m_point1 - m_point3
    return Plane(m_point1, cross(x_vector1, x_vector2))


def rotate(m_matrix, geometry, m_center=Point3D()):
    pass


def point_rotate(x_point, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    既是点的旋转，又是向量的旋转
    """
    assert isinstance(x_point, Point3D) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    new_point_array = np.dot(x_matrix, (x_point - x_center).to_array()) + x_center.to_array()
    return Point3D(*new_point_array)


def line_rotate(x_line, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    直线起始点绕旋转中心旋转，直线方向向量绕原点旋转
    """
    assert isinstance(x_line, Line3D) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    new_line_origin = point_rotate(x_line.origin, x_matrix, x_center)
    new_line_direction = point_rotate(x_line.direction, x_matrix)
    return Line3D(new_line_origin, new_line_direction)


def plane_rotate(x_plane, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    面起始点绕旋转中心旋转，直线方向向量绕原点旋转
    """
    assert isinstance(x_plane, Plane) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    new_plane_origin = point_rotate(x_plane.point, x_matrix, x_center)
    new_plane_vector = point_rotate(x_plane.normal, x_matrix)
    return Plane(new_plane_origin, new_plane_vector)


def triangle_rotate(x_triangle, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    三角形三个顶点绕旋转中心旋转
    """
    assert isinstance(x_triangle, Triangle3D) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    new_vertex1 = point_rotate(x_triangle.vertex1, x_matrix, x_center)
    new_vertex2 = point_rotate(x_triangle.vertex2, x_matrix, x_center)
    new_vertex3 = point_rotate(x_triangle.vertex3, x_matrix, x_center)
    return Triangle3D(new_vertex1, new_vertex2, new_vertex3)


def triangle_slice_rotate(x_triangle_slice, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    三角形三个顶点绕旋转中心旋转,法向量绕原点旋转
    """
    assert isinstance(x_triangle_slice, TriangleSlice) and isinstance(x_matrix, np.ndarray) and isinstance(x_center,
                                                                                                           Point3D)
    new_vertex = triangle_rotate(x_triangle_slice.vertex, x_matrix, x_center)
    new_facet = point_rotate(x_triangle_slice.facet, x_matrix)
    return TriangleSlice(new_facet, new_vertex)


def model_rotate(x_model, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    每个三角面片绕旋转中心旋转
    """
    assert isinstance(x_model, STLModel) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    triangle_list = []
    for x_triangle_slice in x_model:
        triangle_list.append(triangle_slice_rotate(x_triangle_slice, x_matrix, x_center))
    return STLModel(triangle_list)


def is_point_in_triangle_3d(x_point, x_triangle):
    pass


def is_point_in_triangle_2d(x_point, x_triangle_2d):
    """
    判断平面上的点是否在三角形内
    算法原理使用向量的叉乘。假设三角形的三个点按照顺时针顺序为A,B,C
    对于某一点P,求出三个向量PA，PB，PC
    t1 = PA * PB
    t2 = PB * PC
    t3 = PC * PA
    如果t1,t2,t3同号，则P在三角形内部，否则在外部
    如果t1*t2*t3 = 0，则表示该点在三角形的边界
    """
    assert isinstance(x_point, Point2D) and isinstance(x_triangle_2d, Triangle2D)

    tx, ty = x_point.x, x_point.y
    t_box = x_triangle_2d.get_box_2d()
    if not (t_box.x_min <= tx <= t_box.x_max and t_box.y_min <= ty <= t_box.y_max):
        return False

    pa = x_triangle_2d.vertex1 - x_point
    pb = x_triangle_2d.vertex2 - x_point
    pc = x_triangle_2d.vertex3 - x_point
    t1 = cross_multiply_2d(pa, pb)
    t2 = cross_multiply_2d(pb, pc)
    t3 = cross_multiply_2d(pc, pa)
    if t1 > 0 and t2 > 0 and t3 > 0 or t1 < 0 and t2 < 0 and t3 < 0:
        return True
    elif t1 == 0 or t2 == 0 or t3 == 0:
        return True
    else:
        return False


def get_rotate_matrix_from_two_vector(x_vector_old, x_vector_new):
    """
    已知旋转前后的两个向量，计算该旋转矩阵
    使用罗德里格斯变换，通过余弦公式计算旋转角度，通过向量叉乘计算旋转轴
    返回的矩阵为由老向量至新向量的矩阵
    """
    assert isinstance(x_vector_old, Point3D) and isinstance(x_vector_new, Point3D)
    x_theta = np.arccos(dot_multiply(x_vector_old, x_vector_new) / (x_vector_old.norm() * x_vector_new.norm()))
    if x_theta <= epsilon5:
        return np.eye(3)
    x_axis = cross_multiply(x_vector_old, x_vector_new)
    return BaseTransfer.Rodrigues((x_axis * x_theta).to_array())


def subsample_in_mesh(x_model):
    """
    在mesh表格上随机采样点
    该函数存在bug，临时先把思路写下来，后续把点的类优化为继承numpy.array类即可
    @param x_model:
    @return:
    """
    x_point_list = []
    for x_triangle_slice in x_model:
        for i in range(10):
            x_triangle = x_triangle_slice.vertex
            a = np.random.uniform()
            b = np.random.uniform()
            c = 1 - a - b
            x_point_list.append(x_triangle.vertex1 * a + x_triangle.vertex2 * b + x_triangle.vertex3 * c)
    return x_point_list


def intersection_of_line_and_model(x_line, x_model):
    """
    先把直线的方向向量转到(0,0,-1),计算该旋转矩阵，为axis_to_z_matrix
    axis_to_z_matrix * model = temp_model
    判断直线与三角面片的交点(Z向投影)
    计算axis_to_z_matrix的逆矩阵，z_to_axis_matrix

    """
    assert isinstance(x_line, Line3D) and isinstance(x_model, STLModel)

    matrix_line_to_z = get_rotate_matrix_from_two_vector(x_line.direction, Point3D(0, 0, -1))
    temp_model = model_rotate(x_model, matrix_line_to_z)

    for x_triangle_slice in temp_model:
        temp = intersection_of_line_and_triangle_slice(x_line, x_triangle_slice)
        if temp:
            # triangle_slice_model = STLModel([x_triangle_slice])
            return temp


def intersection_of_line_and_triangle_slice(x_line, x_triangle_slice):
    """
    直线的方向为(0,0,-1),与三角面片计算交点
    """
    assert isinstance(x_line, Line3D) and isinstance(x_triangle_slice, TriangleSlice)
    if is_point_in_triangle_2d(x_line.origin.to_point_2d(), x_triangle_slice.vertex.to_triangle_2d()):
        x_plane = create_plane_from_3point(x_triangle_slice.vertex.vertex1,
                                           x_triangle_slice.vertex.vertex2,
                                           x_triangle_slice.vertex.vertex3)
        intersection_point = intersection_of_line_and_plane(x_line, x_plane)
        if intersection_point:
            return intersection_point
    else:
        return None


def is_point_2d_in_polygon_2d(x_point, x_polygon):
    """
    判断2D点是否在2D多边形内,返回一个bool值
    """
    assert isinstance(x_point, Point2D) and len(x_polygon) >= 3
    b_ret = False
    j = len(x_polygon) - 1
    for i in range(len(x_polygon)):
        if x_polygon[i].y < x_point.y < x_polygon[j].y or x_polygon[j].y < x_point.y < x_polygon[i].y:
            if x_point.x > (x_point.y - x_polygon[i].y) * (x_polygon[j].x - x_polygon[i].x) / (
                    x_polygon[j].y - x_polygon[i].y) + x_polygon[i].x:
                b_ret = not b_ret
        j = i
    return b_ret


def get_average_center(x_list_of_point):
    sum_point = Point3D(0, 0, 0)
    for x_point in x_list_of_point:
        sum_point += x_point
    count = len(x_list_of_point)
    return Point3D(sum_point.x / count, sum_point.y / count, sum_point.z / count)


def get_intersection_of_two_box_2d(x_box_1: Box2D, x_box_2: Box2D):
    """
    获得两个box的重叠区域
    @param x_box_1:
    @param x_box_2:
    @return:
    """
    new_x_min = max([x_box_1.x_min, x_box_2.x_min])
    new_x_max = min([x_box_1.x_max, x_box_2.x_max])
    new_y_min = max([x_box_1.y_min, x_box_2.y_min])
    new_y_max = min([x_box_1.y_max, x_box_2.y_max])
    if new_x_min >= new_x_max or new_y_min >= new_y_max:
        # 无交集
        return
    else:
        return Box2D(new_x_min, new_x_max, new_y_min, new_y_max)


def is_box_2d_exist(x_box: Box2D):
    if (x_box.x_min == -1.0) and (x_box.x_max == -1.0) and (x_box.y_min == -1.0) and (x_box.y_max == -1.0):
        return False
    return True


def intersection_of_sensor_laser_and_model(x_sensor: Sensor, x_model: STLModel):
    """
    优化计算方式，先提前计算好所有激光的起始点与方向，然后遍历模型中的三角面片，计算三角面片范围中的激光线，求交点
    @return:
    """
    print(f'仿真正在进行中：')
    x_list_intersection = []
    triangle_slice_count = len(x_model)
    # 计算所有的激光的起始点
    x_laser_origin = temp_laser_origin()
    temp_x = [x_index * 0.01 for x_index in
              range(int(x_laser_origin[0][0][0] * 100), int(x_laser_origin[0][-1][0] * 100))]
    temp_y = [y_index * 0.1 for y_index in range(int(x_laser_origin[0][0][1] * 10), int(x_laser_origin[-1][0][1] * 10))]

    for x_triangle_index, x_triangle in enumerate(x_model):
        assert isinstance(x_triangle, TriangleSlice)
        x_box = x_triangle.vertex.to_triangle_2d().get_box_2d()
        # 增加判断
        x_intersection_box = get_intersection_of_two_box_2d(x_box, Box2D(temp_x[0], temp_x[-1], temp_y[0], temp_y[-1]))
        if x_intersection_box:
            first_index_start = int((round(x_intersection_box.x_min, 2) - temp_x[0]) * 100)
            first_index_end = int((round(x_intersection_box.x_max, 2) - temp_x[0]) * 100 + 2)
            second_index_start = int((round(x_intersection_box.y_min, 1) - temp_y[0]) * 10)
            second_index_end = int((round(x_intersection_box.y_max, 1) - temp_y[0]) * 10 + 2)

            for i in range(first_index_start, first_index_end):
                for j in range(second_index_start, second_index_end):

                    m_line = Line3D(xorigin=Point3D(*x_laser_origin[j][i]), xdirection=Point3D(0, 0, -1))
                    temp = intersection_of_line_and_triangle_slice(m_line, x_triangle)
                    if temp:
                        x_list_intersection.append(temp)
                    else:
                        pass
            # 刷新控制台进度条
            sys.stdout.write(f'\r{"█" * round(x_triangle_index / triangle_slice_count * 100)}'
                             f'{x_triangle_index / triangle_slice_count:.2%}结束'
                             f'{"." * (100 - round(x_triangle_index / triangle_slice_count * 100))} '
                             f'已有{len(x_list_intersection)}个点')
        else:
            continue
    return x_list_intersection


def temp_laser_origin():
    """
    用于临时获得激光的起始点，因为传感器代码还没写好
    @return:
    """
    x_laser_origin = []
    x_start = int(-36 / 0.01)
    x_end = int(36 / 0.01)
    y_start = int(-80 / 0.1)
    y_end = int(80 / 0.1)

    for i in range(y_start, y_end):
        x_line_origin = []
        for j in range(x_start, x_end):
            x_line_origin.append([j * 0.01, i * 0.1, 20])
        x_laser_origin.append(x_line_origin)
    return x_laser_origin


def save(x_object, x_file_name):
    if isinstance(x_object, list):
        if isinstance(x_object[0], Point3D):
            with open('D:\\全局标定测试\\' + x_file_name, 'w') as x_f:
                for x_point in x_object:
                    print(f'{x_point.x},{x_point.y},{x_point.z}', end="\n", file=x_f)


def is_point_equal(x_point_1, x_point_2):
    # 判断两个点是否相同
    assert isinstance(x_point_1, Point3D) and isinstance(x_point_2, Point3D)
    delta = x_point_1 - x_point_2
    if delta.x <= epsilon5 and delta.y <= epsilon5 and delta.z < epsilon5:
        return True
    return False


def distance_from_point_to_plane(x_point, x_plane):
    assert isinstance(x_point, Point3D) and isinstance(x_plane, Plane)
    x_normal_vector = x_plane.normal
    x_point_vector = x_point - x_plane.point
    return dot_multiply(x_point_vector, x_normal_vector)


def test_unit():
    # # region 测试point_rotate()
    # old_point = POINT3D(0, 1, 1)
    # m_matrix = BaseTransfer.euler_angle_to_matrix((-90, 0, 0))
    # new_point = point_rotate(old_point, m_matrix,POINT3D(0,1,0))
    # print(new_point)
    # # endregion

    # # region 测试line_rotate()
    # old_line = LINE2D(xorigin=POINT3D(0,0,1),xdirection=POINT3D(0,0,1))
    # m_matrix = BaseTransfer.euler_angle_to_matrix((-90, 0, 0))
    # new_line = line_rotate(old_line,m_matrix,POINT3D(0,0,1))
    # print(new_line)
    # # endregion

    # # region 测试plane_rotate()
    # old_plane = Plane(POINT3D(0,1,0),POINT3D(0,0,1))
    # m_matrix = BaseTransfer.euler_angle_to_matrix((-90, 0, 0))
    # new_plane = plane_rotate(old_plane,m_matrix,POINT3D(0,0,0))
    # print(new_plane)
    # # endregion

    # region 叉乘测试
    m_point1 = Point3D(1, 22, 0)
    m_point2 = Point3D(3, 1, 0)
    print(cross_multiply(m_point1, m_point2))
    print(dot_multiply(m_point1, m_point2))
    # endregion

    # # region 测试get_rotate_matrix_from_two_vector
    # m_vector1 = POINT3D(0, 0, 1)
    # m_vector2 = POINT3D(0, 0, 1)
    # print(get_rotate_matrix_from_two_vector(m_vector1, m_vector2))
    # # endregion

    # # region 测试is_point_in_triangle_2D()
    # start_time = time.time()
    # m_point = POINT3D(0.1, 0.1, 0)
    # m_triangle = Triangle3D(POINT3D(0, 0, 0), POINT3D(1, 0, 0), POINT3D(0, 1, 0))
    # m_point_2D = m_point.to_point_2d()
    # m_triangle_2D = m_triangle.to_triangle_2d()
    # print(m_point_2D)
    # print(m_triangle_2D)
    # print(is_point_in_triangle_2d(m_point_2D, m_triangle_2D))
    # end_time = time.time()
    # print(f'总共耗时：{end_time - start_time}s')
    # # endregion

    # # region 测试intersection_of_line_and_triangle_slice
    # m_line = LINE2D(xorigin=POINT3D(0.000,99.000,20.000), xdirection=POINT3D(0, 0, -1))
    # m_triangle_slice = TriangleSlice(x_facet=(0.000,0.000,0.024),
    #                                   x_vertex=Triangle3D(POINT3D(-0.000,43.307,-4.210),
    #                                                     POINT3D(-0.012,43.282,-4.210),
    #                                                     POINT3D(-0.000,41.300,-4.210)))
    # print(intersection_of_line_and_triangle_slice(m_line,m_triangle_slice))
    # # endregion

    # # region 测试model_rotate
    # m_model = STLModel.read_stl(r'D:\全局标定测试\单层NEY模型.stl')
    # m_matrix = BaseTransfer.euler_angle_to_matrix([0, 0, 90])
    # n_model = model_rotate(m_model, m_matrix)
    # n_model.save(r'D:\全局标定测试\单层NEY模型-111.stl')
    # # endregion

    # # region 测试subsample_in_mesh()
    # t_model = STLModel.read_stl(r'D:\全局标定测试\sphere_R5.stl')
    # list_point = subsample_in_mesh(t_model)
    # save(list_point, 'subsample.txt')
    # # endregion

    # # region 测试distance_from_point_to_plane()
    # t_point = Point3D(2, 2, 2)
    # t_plane = Plane(Point3D(0, 0, 0), Point3D(1, 1, 1))
    # print(distance_from_point_to_plane(t_point, t_plane))
    # # endregion

    pass


def simulation_test():
    m_model = STLModel.read_stl(r'D:\单层NEY模型.stl')
    list_intersection = intersection_of_sensor_laser_and_model(Sensor(), m_model)
    save(list_intersection, 'result-TuMo.txt')


if __name__ == '__main__':
    test_unit()
