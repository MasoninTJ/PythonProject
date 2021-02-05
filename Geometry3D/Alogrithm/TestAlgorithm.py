import Alogrithm.BaseAlgorithm as Alg
from Class3D import *
from Matrix import Matrix3d, Matrix4d


def test_check_intersect_ray_and_box():
    # 无交点，返回false
    t_box0 = Box3D(m_min=Point3D(-2, -2, -2), m_max=Point3D(2, 2, 2))
    t_ray0 = Ray3D(m_origin=Point3D(-3, -4, 2), m_direction=Vector3D(0, 1, 0))
    test_data0 = (t_ray0, t_box0)

    # 边界条件，线与包围盒的某条边重合，本来应该存在无数个交点，但是这里输出两个点 (-2.000,-2.000,2.000) (-2.000,2.000,2.000)
    t_box1 = Box3D(m_min=Point3D(-2, -2, -2), m_max=Point3D(2, 2, 2))
    t_ray1 = Ray3D(m_origin=Point3D(-2, -4, 2), m_direction=Vector3D(0, 1, 0))
    test_data1 = (t_ray1, t_box1)

    # 边界条件，线在包围盒的某个面上，本来应该存在无数个交点，但是这里输出两个点 (-2.000,-2.000,0.000)(-2.000,2.000,0.000)
    t_box2 = Box3D(m_min=Point3D(-2, -2, -2), m_max=Point3D(2, 2, 2))
    t_ray2 = Ray3D(m_origin=Point3D(-2, -4, 0), m_direction=Vector3D(0, 1, 0))
    test_data2 = (t_ray2, t_box2)

    # 存在两个交点 (0.000,-2.000,0.000)(2.000,0.000,0.000)
    t_box3 = Box3D(m_min=Point3D(-2, -2, -2), m_max=Point3D(2, 2, 2))
    t_ray3 = Ray3D(m_origin=Point3D(-2, -4, 0), m_direction=Vector3D(1, 1, 0))
    test_data3 = (t_ray3, t_box3)

    result = Alg.check_intersect_ray_and_box(*test_data1)
    if result:
        print(result[1])
        print(result[2])
    else:
        print(result)


def test_vector_rotate():
    t_vector0 = Vector3D(1,0,0)
    t_matrix0 = Matrix3d.from_euler_angle(0,0,np.pi/2)
    test_data0 = (t_vector0,t_matrix0)    # 返回 (0,1,0)

    print(Alg.vector_rotate(*test_data0))

def test_point_rotate():
    t_point0 = Point3D(1,0,0)
    t_matrix0 = Matrix3d.from_euler_angle(0,0,np.pi/2)
    test_data0 = (t_point0,t_matrix0)    # 返回 (0,1,0)

    t_point1 = Point3D(1,0,0)
    t_matrix1 = Matrix3d.from_euler_angle(0,0,np.pi/2)
    t_center1 = Point3D(-1,0,0)
    test_data1 = (t_point1,t_matrix1,t_center1)    # 返回 (0,1,0)

    print(Alg.point_rotate(*test_data1))

if __name__ == '__main__':
    test_point_rotate()
