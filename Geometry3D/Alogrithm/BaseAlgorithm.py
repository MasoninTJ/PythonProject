from typing import List

import ConstMember
from Class3D import *
from Matrix import Matrix3d


# region 数学计算
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


def rodrigues(m_vector: Vector3D) -> Matrix3d:
    """
    罗德里格斯变换:绕任意轴旋转某个角度
    x = [nx,ny,nz] = n*[x,y,z]
    [x,y,z]为单位向量，表示旋转轴
    n表示旋转角度，单位：弧度
    计算旋转矩阵

    :return: 变换矩阵
    """
    m_theta = m_vector.length()

    if m_theta < ConstMember.epsilon5:
        return Matrix3d.identity()
    else:
        itheta = 1 / m_theta
    m_vector = m_vector * itheta
    return Matrix3d.from_axis_angle(m_vector, m_theta)


def get_rotate_matrix_from_two_vector(m_vector_old: Vector3D, m_vector_new: Vector3D):
    """
    已知旋转前后的两个向量，计算该旋转矩阵
    使用罗德里格斯变换，通过余弦公式计算旋转角度，通过向量叉乘计算旋转轴
    返回的矩阵为由老向量至新向量的矩阵
    """
    m_theta = np.arccos(dot(m_vector_old, m_vector_new) / (m_vector_old.length() * m_vector_new.length()))
    if m_theta <= ConstMember.epsilon5:
        return np.eye(3)
    m_axis = cross(m_vector_old, m_vector_new)
    return Matrix3d.from_axis_angle(m_axis, m_theta)


def distance_from_point_to_plane(m_point: Point3D, m_plane: Plane) -> (float, int):
    m_normal_vector = m_plane.normal
    m_point_vector = m_point - m_plane.point
    return dot(m_point_vector, m_normal_vector)


# endregion

# region 构造几何
def create_plane_from_3point(m_point1: Point3D, m_point2: Point3D, m_point3: Point3D) -> (Plane, None):
    """
    通过三个点构造面，计算平面的法向量,用点法式构造平面
    """
    m_vec1 = m_point2 - m_point1
    m_vec2 = m_point3 - m_point1
    m_normal = cross(m_vec1, m_vec2)
    if m_normal.check_valid():  # 叉乘结果为零向量时三点共线
        return Plane(m_point1, m_normal)
    else:
        return None


def subsample_point_in_mesh(m_model: STLModel, m_density=10):
    """
    在mesh表格上随机采样点
    m_density 表示一个三角面片上采集多少个点
    这是一种不均匀采样，以三角面片为单位
    """
    m_point_list = []
    for m_mesh in m_model:
        for i in range(m_density):
            m_triangle: Triangle = m_mesh.vertex
            a = np.random.uniform()
            b = np.random.uniform()
            c = 1 - a - b
            m_point_list.append(m_triangle.get_point_from_abc(a, b, c))
    return m_point_list


def get_average_center(m_list_of_point: List[Point3D]) -> Point3D:
    sum_x, sum_y, sum_z = 0, 0, 0
    for m_point in m_list_of_point:
        sum_x += m_point.x
        sum_y += m_point.y
        sum_z += m_point.z
    count = len(m_list_of_point)
    return Point3D(sum_x / count, sum_y / count, sum_z / count)


# endregion

# region 旋转
def vector_rotate(m_vector: Vector3D, m_matrix: Matrix3d) -> Vector3D:
    """
    向量旋转
    """
    return m_matrix * m_vector


def point_rotate(m_point: Point3D, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)) -> Point3D:
    """
    点的旋转，包含旋转中心
    """
    if m_center.to_array().all():  # 原点为旋转中心
        return m_matrix * m_point
    else:
        return m_matrix * (m_point - m_center) + m_center


def line_rotate(m_line: Line3D, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)) -> Line3D:
    """
    直线旋转，直线的两个点绕旋转中心旋转
    """
    new_line_begin = point_rotate(m_line.begin, m_matrix, m_center)
    new_line_end = point_rotate(m_line.end, m_matrix, m_center)
    return Line3D(new_line_begin, new_line_end)


def plane_rotate(m_plane: Plane, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)):
    """
    面的旋转，面的点绕旋转中心旋转，法线向量绕原点旋转
    """
    new_plane_point = point_rotate(m_plane.point, m_matrix, m_center)
    new_plane_normal = vector_rotate(m_plane.normal, m_matrix)
    return Plane(new_plane_point, new_plane_normal)


def triangle_rotate(m_triangle: Triangle, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)):
    """
    三角形旋转，三角形三个顶点绕旋转中心旋转
    """
    new_vertex1 = point_rotate(m_triangle.vertex1, m_matrix, m_center)
    new_vertex2 = point_rotate(m_triangle.vertex2, m_matrix, m_center)
    new_vertex3 = point_rotate(m_triangle.vertex3, m_matrix, m_center)
    return Triangle(new_vertex1, new_vertex2, new_vertex3)


def mesh_rotate(m_mesh: Mesh, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)):
    """
    三角面片旋转，三角形三个顶点绕旋转中心旋转,法向量绕原点旋转
    """
    new_vertex = triangle_rotate(m_mesh.vertex, m_matrix, m_center)
    new_normal = vector_rotate(m_mesh.normal, m_matrix)
    return Mesh(new_normal, new_vertex)


def model_rotate(m_model: STLModel, m_matrix: Matrix3d, m_center: Point3D = Point3D(0, 0, 0)):
    """
    每个三角面片绕旋转中心旋转
    """
    m_mesh_list = []
    for m_mesh in m_model:
        m_mesh_list.append(mesh_rotate(m_mesh, m_matrix, m_center))
    return STLModel(m_mesh_list)


# endregion

# region 判断相等、相交或内部
def check_point_equal(m_point_1: Point3D, m_point_2: Point3D) -> bool:
    # 判断两个点是否相同
    delta = m_point_1 - m_point_2
    return delta.i <= ConstMember.epsilon5 and delta.j <= ConstMember.epsilon5 and delta.k < ConstMember.epsilon5


def check_intersect_ray_and_box(m_ray: Ray3D, m_box: Box3D) -> (bool, Point3D, Point3D):
    """
    射线与AABB包围盒的相交检测，如果相交，同时会返回两个交点
    """
    bounds = [m_box.min, m_box.max]
    with np.errstate(divide='ignore'):  # 屏蔽分母都为0的异常
        inv_direction = 1 / m_ray.direction.to_array()
    sign_x = 0 if inv_direction[0] > 0 else 1
    sign_y = 0 if inv_direction[1] > 0 else 1
    sign_z = 0 if inv_direction[2] > 0 else 1

    with np.errstate(divide='ignore', invalid='ignore'):  # 屏蔽分子分母都为0的异常
        t_min = (bounds[sign_x].x - m_ray.origin.x) * inv_direction[0]
        t_max = (bounds[1 - sign_x].x - m_ray.origin.x) * inv_direction[0]
        t_y_min = (bounds[sign_y].y - m_ray.origin.y) * inv_direction[1]
        t_y_max = (bounds[1 - sign_y].y - m_ray.origin.y) * inv_direction[1]

    if (t_min > t_y_max) or (t_y_min > t_max):
        return False
    if t_y_min > t_min or np.isnan(t_min):
        t_min = t_y_min
    if t_y_max < t_max or np.isnan(t_max):
        t_max = t_y_max

    with np.errstate(divide='ignore', invalid='ignore'):  # 屏蔽分子分母都为0的异常
        t_z_min = (bounds[sign_z].z - m_ray.origin.z) * inv_direction[2]
        t_z_max = (bounds[1 - sign_z].z - m_ray.origin.z) * inv_direction[2]

    if (t_min > t_z_max) or (t_z_min > t_max):
        return False
    if t_z_min > t_min or np.isnan(t_min):
        t_min = t_z_min
    if t_z_max < t_max or np.isnan(t_max):
        t_max = t_z_max

    return True, m_ray.get_point_from_t(t_min), m_ray.get_point_from_t(t_max)


def check_point_in_polygon_from_z(m_point: Point3D, m_polygon: List[Point3D]):
    """
    判断在Z方向上，点是否在多边形内,返回一个bool值
    """
    if m_polygon_length := len(m_polygon) < 3:
        return None
    b_check_in = False
    j = m_polygon_length - 1
    for i in range(m_polygon_length):
        if m_polygon[i].y < m_point.y < m_polygon[j].y or m_polygon[j].y < m_point.y < m_polygon[i].y:
            if m_point.x > (m_point.y - m_polygon[i].y) * (m_polygon[j].x - m_polygon[i].x) / (m_polygon[j].y - m_polygon[i].y) + m_polygon[i].x:
                b_check_in = not b_check_in
        j = i
    return b_check_in


def check_point_in_triangle_from_z(m_point: Point3D, m_triangle: Triangle):
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
    m_box = Box3D.from_triangle(m_triangle)
    if not (m_box.min.x <= m_point.x <= m_box.max.x and m_box.min.y <= m_point.y <= m_box.max.y):
        return False

    pa = m_triangle.vertex1 - m_point
    pb = m_triangle.vertex2 - m_point
    pc = m_triangle.vertex3 - m_point
    t1 = cross(pa, pb)
    t2 = cross(pb, pc)
    t3 = cross(pc, pa)
    if t1 > 0 and t2 > 0 and t3 > 0 or t1 < 0 and t2 < 0 and t3 < 0:
        return True
    elif t1 == 0 or t2 == 0 or t3 == 0:
        return True
    else:
        return False


def check_intersect_ray_and_sphere(m_ray: Ray3D, m_sphere: Sphere):
    """
    检测射线与球是否相交，不计算交点
    :param m_ray:
    :param m_sphere:
    :return:
    """
    a = m_sphere.center - m_ray.origin
    a2 = dot(a, a)
    r2 = m_sphere.radius ** 2
    if a2 <= r2:
        return True
    l = dot(m_ray.direction, a)
    if l < 0:
        return False
    m2 = a2 - l ** 2
    if m2 > r2:
        return False
    return True


# endregion

# region 计算相交
def intersection_of_ray_and_plane(m_ray: Ray3D, m_plane: Plane) -> (Point3D, None):
    """
    计算射线与面的交点（不判断线在面上，此时有无穷多点）
    """
    f = dot(m_ray.direction, m_plane.normal)
    if -ConstMember.epsilon5 < f < ConstMember.epsilon5:  # 判断平行,使用小于极小值
        temp = dot((m_plane.point - m_ray.origin), m_plane.normal) / f
        return m_ray.get_point_from_t(temp)
    else:
        return None


def intersection_of_ray_and_triangle(m_ray: Ray3D, m_triangle: Triangle) -> (Point3D, None):
    """
    射线与三角形的交点，如果没有交点则返回None
    """
    vec_ab = m_triangle.vertex2 - m_triangle.vertex1
    vec_ac = m_triangle.vertex3 - m_triangle.vertex1
    p = cross(m_ray.direction, vec_ac)
    a = dot(p, vec_ab)
    if -ConstMember.epsilon5 < a < ConstMember.epsilon5:  # 判断平行,使用小于极小值
        return None
    f = 1 / a
    s = m_ray.origin - m_triangle.vertex1
    u = f * dot(s, p)
    if u < 0:  # 点在三角形外
        return None
    q = cross(s, vec_ab)
    v = f * dot(m_ray.direction, q)
    if v < 0 or u + v > 1:  # 点在三角形外
        return None

    t = f * np.dot(vec_ac, q)
    return m_ray.get_point_from_t(t)


def intersection_of_ray_and_mesh(m_ray, m_mesh):
    """
    射线的方向为(0,0,-1),与三角面片计算交点
    """
    if m_ray.direction == Vector3D(0, 0, -1):
        result_point = intersection_of_ray_and_triangle(m_ray, m_mesh.vertex)
        return result_point
    else:
        minus_z = Vector3D(0, 0, -1)
        matrix_ray_to_z = get_rotate_matrix_from_two_vector(m_ray.direction, minus_z)
        temp_ray = Ray3D(m_ray.origin, minus_z)
        temp_mesh = mesh_rotate(m_mesh, matrix_ray_to_z)
        matrix_z_to_ray = matrix_ray_to_z.inv()  # 计算逆矩阵，后面需要转回来
        temp_point = intersection_of_ray_and_triangle(temp_ray, temp_mesh.vertex)
        result_point = point_rotate(temp_point, matrix_z_to_ray)
        return result_point


def intersection_of_ray_and_model(m_ray: Ray3D, m_model: STLModel):
    """
    计算射线与STL模型的交点，如果无交点，则返回None
    """
    # 先将射线转到(0,0,-1)方向，计算点是否在三角形内
    minus_z = Vector3D(0, 0, -1)
    matrix_ray_to_z = get_rotate_matrix_from_two_vector(m_ray.direction, minus_z)
    # 转换之后的射线与STL模型
    temp_ray = Ray3D(m_ray.origin, minus_z)
    temp_model = model_rotate(m_model, matrix_ray_to_z)
    matrix_z_to_ray = matrix_ray_to_z.inv()  # 计算逆矩阵，后面需要转回来

    for m_mesh in temp_model:
        temp_point = intersection_of_ray_and_mesh(temp_ray, m_mesh)
        if temp_point:
            result_point = point_rotate(temp_point, matrix_z_to_ray)
            return result_point
    return None


def intersection_of_ray_and_sphere(m_ray: Ray3D, m_sphere: Sphere):
    """
    计算射线与球的交点 todo
    :param m_ray:
    :param m_sphere:
    :return:
    """
    e = m_ray.origin - m_sphere.center
    b = dot(m_ray.direction, e)
    c = dot(e, e) - m_sphere.radius ** 2
    k = b ** 2 - c
    if k < 0:
        return None
    elif k == 0:
        a = dot(m_ray.direction, m_ray.direction)
        t = -b / a
        Q = m_ray.get_point_from_t(t)
        return Q
    else:
        k = k ** 0.5
        a = dot(m_ray.direction, m_ray.direction)
        t1 = (-b - k) / a
        t2 = (-b + k) / a
        Q1 = m_ray.get_point_from_t(t1)
        Q2 = m_ray.get_point_from_t(t2)
        return Q1, Q2


def intersection_of_ray_and_sphere_accelerate(m_ray: Ray3D, m_sphere: Sphere):
    """
    计算射线与球的交点，优化算法 todo
    只计算第一个交点
    :param m_ray:
    :param m_sphere:
    :return:
    """
    a = m_sphere.center - m_ray.origin
    l = dot(m_ray.direction, a)
    a2 = dot(a, a)
    r2 = m_sphere.radius ** 2
    if a2 > r2 and l < 0:
        return None
    m2 = a2 - l ** 2
    if m2 > r2:
        return None
    q = (r2 - m2) ** 0.5
    if a2 > r2:
        t = l - q
    else:
        t = l + q
    return m_ray.get_point_from_t(t)


# endregion

if __name__ == '__main__':
    pass
