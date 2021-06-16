def affine_fit(from_pts, to_pts):
    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print("原始点和目标点的个数必须相同.")
        return False

    dim = len(q[0])  # 维度
    if len(q) < dim:
        print("至少为9个点.")
        return False

    # 新建一个空的 维度 x (维度+1) 矩阵 并填满
    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # 新建一个空的 (维度+1) x (维度+1) 矩阵 并填满
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    # 判断原始点和目标点是否共线，共线则无解. 耗时计算，如果追求效率可以不用。
    # 其实就是解n个三元一次方程组
    def gauss_jordan(m, eps=1.0 / (10 ** 10)):
        (h, w) = (len(m), len(m[0]))
        for y in range(0, h):
            maxrow = y
            for y2 in range(y + 1, h):
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:
                return False
            for y2 in range(y + 1, h):
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h - 1, 0 - 1, -1):
            c = m[y][y]
            for y2 in range(0, y):
                for x in range(w - 1, y - 1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):
                m[y][x] /= c
        return True

    M = [Q[i] + c[i] for i in range(dim + 1)]
    if not gauss_jordan(M):
        print("错误，原始点和目标点也许是共线的.")

        return False

    class transformation:
        """对象化仿射变换."""

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str += "x%d * %f + " % (i, M[i][j + dim + 1])
                str += "%f" % M[dim][j + dim + 1]
                res += str + "\n"
            return res

        def transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res

    return transformation()


def test():
    from_pt = [[1698.650, 681.653],
               [2371.946, 664.784],
               [3049.477, 698.211],
               [1711.106, 1327.899],
               [2366.661, 1354.134],
               [3034.910, 1386.819],
               [1702.339, 1963.508],
               [2399.918, 2010.131],
               [3095.774, 2038.798]]  # 输入点坐标对
    to_pt = [[-235.000, 200],
             [-255.000, 200],
             [-275.000, 200],
             [-235, 220],
             [-255, 220],
             [-275, 220],
             [-235, 240],
             [-255, 240],
             [-275, 240]]  # 输出点坐标对

    trn = affine_fit(from_pt, to_pt)

    if trn:
        print("转换公式:")
        print(trn.To_Str())

        err = 0.0
        for i in range(len(from_pt)):
            fp = from_pt[i]
            tp = to_pt[i]
            t = trn.transform(fp)
            print("%s => %s ~= %s" % (fp, tuple(t), tp))
            err += ((tp[0] - t[0]) ** 2 + (tp[1] - t[1]) ** 2) ** 0.5

        print(f"拟合误差 = {err/9}")


if __name__ == "__main__":
    print("测试最小二乘法求解九点标定的六参数仿射变换矩阵：")
    test()
