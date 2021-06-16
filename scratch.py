import cv2
import numpy as np

origin_points_set = np.float32([[1698.650, 681.653],
                                [2371.946, 664.784],
                                [3049.477, 698.211],
                                [1711.106, 1327.899],
                                [2366.661, 1354.134],
                                [3034.910, 1386.819],
                                [1702.339, 1963.508],
                                [2399.918, 2010.131],
                                [3095.774, 2038.798]])

target_points_set = np.float32([[-235.000, 200],
                                [-255.000, 200],
                                [-275.000, 200],
                                [-235, 220],
                                [-255, 220],
                                [-275, 220],
                                [-235, 240],
                                [-255, 240],
                                [-275, 240]])
ret, inline = cv2.estimateAffine2D(origin_points_set, target_points_set, False)
print(ret)

test_point1 = np.float32([[[1698.650, 681.653]]])
res_point = cv2.transform(test_point1, ret)
print(res_point)

test_point2 = np.float32([1698.650, 681.653, 1])
print(np.dot(ret, test_point2))
