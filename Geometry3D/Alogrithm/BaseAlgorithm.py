from Class3D import *
import ConstMember
from Alogrithm import BaseTransfer


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


def intersection_of_ray_and_plane(m_Ray: Ray3D, m_plane: Plane) -> (Point3D, None):
    """
    计算射线与面的交点（不判断线在面上，此时有无穷多点）
    """
    f = dot(m_Ray.direction, m_plane.normal)
    if -ConstMember.epsilon5 < f < ConstMember.epsilon5:  # 判断平行,使用小于极小值
        temp = dot((m_plane.point - m_Ray.origin), m_plane.normal) / f
        return m_Ray.get_point_from_t(temp)
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


def create_plane_from_3point(m_point1: Point3D, m_point2: Point3D, m_point3: Point3D) -> (Plane, None):
    """
    通过三个点构造面，计算平面的法向量,用点法式构造平面
    """
    m_vec1 = m_point2 - m_point1
    m_vec2 = m_point3 - m_point1
    m_normal = cross(m_vec1, m_vec2)
    if m_normal.check_valid():  # 叉乘结果全是0时三点共线
        return Plane(m_point1, m_normal)
    else:
        return None


def check_intersect_ray_and_box(m_ray:Ray3D,m_box:Box3D) -> bool:
    """
    射线与AABB包围盒的相交检测
    """
    pass


def vector_rotate(m_vector, m_matrix):
    new_point_array = np.dot(m_matrix, m_vector.to_array())
    return Vector3D(*new_point_array)


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
    new_line_direction = point_rotate(x_line.direction(), x_matrix)
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
    assert isinstance(x_triangle, Triangle) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    new_vertex1 = point_rotate(x_triangle.vertex1, x_matrix, x_center)
    new_vertex2 = point_rotate(x_triangle.vertex2, x_matrix, x_center)
    new_vertex3 = point_rotate(x_triangle.vertex3, x_matrix, x_center)
    return Triangle(new_vertex1, new_vertex2, new_vertex3)


def triangle_slice_rotate(x_triangle_slice, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    三角形三个顶点绕旋转中心旋转,法向量绕原点旋转
    """
    new_vertex = triangle_rotate(x_triangle_slice.vertex, x_matrix, x_center)
    new_facet = vector_rotate(x_triangle_slice.normal, x_matrix)
    return Mesh(new_facet, new_vertex)


def model_rotate(x_model, x_matrix, x_center=Point3D(0, 0, 0)):
    """
    每个三角面片绕旋转中心旋转
    """
    assert isinstance(x_model, STLModel) and isinstance(x_matrix, np.ndarray) and isinstance(x_center, Point3D)
    triangle_list = []
    for x_triangle_slice in x_model:
        triangle_list.append(triangle_slice_rotate(x_triangle_slice, x_matrix, x_center))
    return STLModel(triangle_list)


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

    tx, ty = x_point.x, x_point.y
    t_box = x_triangle_2d.get_box_2d()
    if not (t_box.x_min <= tx <= t_box.x_max and t_box.y_min <= ty <= t_box.y_max):
        return False

    pa = x_triangle_2d.vertex1 - x_point
    pb = x_triangle_2d.vertex2 - x_point
    pc = x_triangle_2d.vertex3 - x_point
    t1 = cross(pa, pb)
    t2 = cross(pb, pc)
    t3 = cross(pc, pa)
    if t1 > 0 and t2 > 0 and t3 > 0 or t1 < 0 and t2 < 0 and t3 < 0:
        return True
    elif t1 == 0 or t2 == 0 or t3 == 0:
        return True
    else:
        return False


def get_rotate_matrix_from_two_vector(x_vector_old: Vector3D, x_vector_new: Vector3D):
    """
    已知旋转前后的两个向量，计算该旋转矩阵
    使用罗德里格斯变换，通过余弦公式计算旋转角度，通过向量叉乘计算旋转轴
    返回的矩阵为由老向量至新向量的矩阵
    """
    assert isinstance(x_vector_old, Point3D) and isinstance(x_vector_new, Point3D)
    x_theta = np.arccos(dot(x_vector_old, x_vector_new) / (x_vector_old.normalize() * x_vector_new.normalize()))
    if x_theta <= ConstMember.epsilon5:
        return np.eye(3)
    x_axis = cross(x_vector_old, x_vector_new)
    return BaseTransfer.rodrigues((x_axis * x_theta).to_array())


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

    matrix_line_to_z = get_rotate_matrix_from_two_vector(x_line.direction(), Vector3D(0, 0, -1))
    temp_model = model_rotate(x_model, matrix_line_to_z)

    for x_triangle_slice in temp_model.mesh_list:
        temp = intersection_of_line_and_triangle_slice(x_line, x_triangle_slice)
        if temp:
            # triangle_slice_model = STLModel([x_triangle_slice])
            return temp


def intersection_of_line_and_triangle_slice(x_line, x_triangle_slice):
    """
    直线的方向为(0,0,-1),与三角面片计算交点
    """
    assert isinstance(x_line, Line3D) and isinstance(x_triangle_slice, Mesh)
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


def is_point_equal(x_point_1, x_point_2):
    # 判断两个点是否相同
    assert isinstance(x_point_1, Point3D) and isinstance(x_point_2, Point3D)
    delta = x_point_1 - x_point_2
    if delta.x <= ConstMember.epsilon5 and delta.y <= ConstMember.epsilon5 and delta.z < ConstMember.epsilon5:
        return True
    return False


def distance_from_point_to_plane(x_point, x_plane):
    assert isinstance(x_point, Point3D) and isinstance(x_plane, Plane)
    x_normal_vector = x_plane.normal
    x_point_vector = x_point - x_plane.point
    return dot(x_point_vector, x_normal_vector)


if __name__ == '__main__':
    pass
