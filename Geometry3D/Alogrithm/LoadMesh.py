import codecs
import struct

from Geometry3D.Class3D import STLModel, Mesh, Point3D, Vector3D, Triangle


def load_stl_model(m_path: str) -> STLModel:
    """
    加载 STL 模型
    """
    if is_binary_file(m_path):
        return STLModel(load_binary_mesh(m_path))
    else:
        return STLModel(load_ascii_mesh(m_path))


def is_binary_file(file_path):
    """
    判断是否为二进制文件
    首先检查文件是否以BOM开始，如果不在初始8192字节内查找零字节
    """
    m_bom_list = (
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF32_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF8,
    )

    with open(file_path, 'rb') as file:
        initial_bytes = file.read(8192)
        file.close()
        for bom in m_bom_list:
            if initial_bytes.startswith(bom):
                continue
            else:
                if b'\0' in initial_bytes:
                    return True
    return False


# region 读取Binary的STL模型
def load_binary_mesh(m_path):
    """
    读取 stl 的二进制格式 binary
    """
    m_mesh_list = []
    with open(m_path, 'rb') as m_mesh_file:
        m_mesh_file.read(80)  # 流出80字节，文件名
        m_mesh_num = m_mesh_file.read(4)  # 流出4字节，文件中 mesh 的数量
        m_mesh_num = struct.unpack('I', m_mesh_num)[0]
        for _ in range(m_mesh_num):
            m_temp_mesh = read_one_binary_mesh(m_mesh_file)
            m_mesh_list.append(m_temp_mesh)
    print(f'读取STL文件结束，共{m_mesh_num}个网格')
    return m_mesh_list


def read_one_binary_mesh(m_byte_stream) -> Mesh:
    """
    从字节流中读取一个二进制的网格(mesh)
    """
    m_facet_normal = read_binary_normal(m_byte_stream)
    m_vertex1 = read_binary_vertex(m_byte_stream)
    m_vertex2 = read_binary_vertex(m_byte_stream)
    m_vertex3 = read_binary_vertex(m_byte_stream)
    m_byte_stream.read(2)
    m_mesh = Mesh(m_facet_normal, Triangle(m_vertex1, m_vertex2, m_vertex3))
    return m_mesh


def read_binary_normal(m_byte_stream) -> Vector3D:
    """
    读取 mesh 的法线
    """
    m_normal_i = struct.unpack('f', m_byte_stream.read(4))[0]
    m_normal_j = struct.unpack('f', m_byte_stream.read(4))[0]
    m_normal_k = struct.unpack('f', m_byte_stream.read(4))[0]
    return Vector3D(m_normal_i, m_normal_j, m_normal_k)


def read_binary_vertex(m_byte_stream) -> Point3D:
    """
    读取 mesh 的一个顶点
    """
    m_point_x = struct.unpack('f', m_byte_stream.read(4))[0]
    m_point_y = struct.unpack('f', m_byte_stream.read(4))[0]
    m_point_z = struct.unpack('f', m_byte_stream.read(4))[0]
    return Point3D(m_point_x, m_point_y, m_point_z)


# endregion

# region 读取ASCII的STL模型
def load_ascii_mesh(m_path):
    """
    读取 stl 的ascii格式
    """
    m_mesh_list = []
    with open(m_path, 'r') as m_mesh_file:
        m_mesh_file.readline()  # 'solid Mesh'
        while m_temp_mesh := read_one_ascii_mesh(m_mesh_file):
            m_mesh_list.append(m_temp_mesh)
    print(f'读取STL文件结束，共{len(m_mesh_list)}个网格')
    return m_mesh_list


def read_one_ascii_mesh(m_file) -> (Mesh, None):
    """
    从字节流中读取一个ASCII的网格(mesh)
    """
    if (m_normal_string := m_file.readline()).startswith('facet'):
        m_facet_normal = read_ascii_normal(m_normal_string)
        m_file.readline()  # 'outer loop'
        m_vertex1_string = m_file.readline()
        m_vertex1 = read_ascii_vertex(m_vertex1_string)
        m_vertex2_string = m_file.readline()
        m_vertex2 = read_ascii_vertex(m_vertex2_string)
        m_vertex3_string = m_file.readline()
        m_vertex3 = read_ascii_vertex(m_vertex3_string)
        m_file.readline()  # 'endloop'
        m_file.readline()  # 'endfacet'

        m_mesh = Mesh(m_facet_normal, Triangle(m_vertex1, m_vertex2, m_vertex3))
        return m_mesh
    else:
        return None


def read_ascii_normal(m_string: str) -> Vector3D:
    """
    读取 mesh 的法线
    """
    m_string_split = m_string.split(' ')
    m_normal_i = float(m_string_split[2])
    m_normal_j = float(m_string_split[3])
    m_normal_k = float(m_string_split[4])
    return Vector3D(m_normal_i, m_normal_j, m_normal_k)


def read_ascii_vertex(m_string: str) -> Point3D:
    """
    读取 mesh 的一个顶点
    """
    m_string_split = m_string.split(' ')
    m_point_x = float(m_string_split[1])
    m_point_y = float(m_string_split[2])
    m_point_z = float(m_string_split[3])
    return Point3D(m_point_x, m_point_y, m_point_z)


# endregion


if __name__ == '__main__':
    t_ascii_stl_file_path = r'D:/全局标定测试/单层NEY模型-ASCII格式.stl'
    t_binary_stl_file_path = r'D:/全局标定测试/单层NEY模型-Binary格式.stl'
    t_binary_stl_model = load_stl_model(t_binary_stl_file_path)
    t_ascii_stl_model = load_stl_model(t_ascii_stl_file_path)
