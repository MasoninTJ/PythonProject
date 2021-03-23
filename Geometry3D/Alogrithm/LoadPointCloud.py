import re

from Geometry3D.Class3D import Point3D


def load_point_cloud(m_path: str):
    """
    加载点云文件，通常后缀为 txt 或者 pn
    """
    if m_path.endswith(('.txt', '.pn', 'csv')):
        return load_ascii_scatter_point(m_path)


def load_ascii_scatter_point(m_path: str):
    """
    读取ascii散点文件
    """
    m_split_char = ',| |;|\t'  # 分隔符，支持逗号|分号|空格|Tab
    m_point_list = []
    with open(m_path, 'r') as m_point_file:
        for m_string in m_point_file.readlines():
            m_string_split = re.split(m_split_char, m_string)
            m_point = map(float, m_string_split)
            m_point_list.append(Point3D(*m_point))
    return m_point_list


def load_ascii_sparse_array_point(m_path: str, with_intensity: bool = False):
    """
    读取ascii稀疏阵列文件
    """
    point_length = 4 if with_intensity else 3  # 含有光强值的点长度为4
    m_split_char = ',| |;|\t'  # 分隔符，支持逗号|分号|空格|Tab
    m_point_list = []
    with open(m_path, 'r') as m_point_file:
        for m_string in m_point_file.readlines():
            m_point_list_in_line = []
            m_string_split = re.split(m_split_char, m_string)
            m_float_in_line = list(map(float, m_string_split))
            for index in range(0, len(m_float_in_line), step=point_length):
                m_point = m_float_in_line[index: index + point_length]
                if with_intensity:
                    pass  # todo 光强点的类型
                else:
                    m_point_list_in_line.append(Point3D(*m_point))
            # 以一整行为单位传入数组中
            m_point_list.append(m_point_list_in_line)
    return m_point_list


def load_ascii_dense_array_point(m_path):
    """
    读取ascii密集阵列文件
    """
    m_split_char = ',| |;|\t'  # 分隔符，支持逗号|分号|空格|Tab
    m_point_list = []
    with open(m_path, 'r') as m_point_file:
        for m_string in m_point_file.readlines():
            m_string_split = re.split(m_split_char, m_string)
            m_point = map(float, m_string_split)
            m_point_list.append(Point3D(*m_point))
    return m_point_list


if __name__ == '__main__':
    test_path = r'D:\Line3D.txt'
    test_point_cloud = load_point_cloud(test_path)
    for _ in test_point_cloud:
        print(_)
