import math
from scipy import interpolate
import numpy as np
import pylab as pl

out_t = [-3, 0, 0]  # tx，ty,tz
out_angle = [0, 0, 0]  # angle_x，angle_y,angle_z
in_matrix = [960, 540, 3500, 3000]  # cx,cy,fx,fy

car_height = -4
camera_height = 9  # 相机地面高度
camera_angle_a = 5  # 相机光抽和水平线夹角


def convert_to_world(xyz_args):
    """
    其他-世界
    """
    angle_x = out_angle[0]  # 绕x轴的旋转角
    angle_y = out_angle[1]
    angle_z = out_angle[2]
    tx = out_t[0]
    ty = out_t[1]
    tz = out_t[2]
    angle_x = angle_x * math.pi / 180
    angle_y = angle_y * math.pi / 180
    angle_z = angle_z * math.pi / 180
    rotate = np.mat([
        [math.cos(angle_z) * math.cos(angle_y),
         math.cos(angle_z) * math.sin(angle_x) * math.sin(angle_y) - math.cos(angle_x) * math.sin(angle_z),
         math.sin(angle_z) * math.sin(angle_x) + math.cos(angle_z) * math.cos(angle_x) * math.sin(angle_y)],
        [math.cos(angle_y) * math.sin(angle_z),
         math.cos(angle_z) * math.cos(angle_x) + math.sin(angle_z) * math.sin(angle_x) * math.sin(angle_y),
         math.cos(angle_x) * math.sin(angle_z) * math.sin(angle_y) - math.cos(angle_z) * math.sin(angle_x)],
        [- math.sin(angle_y),
         math.cos(angle_y) * math.sin(angle_x),
         math.cos(angle_x) * math.cos(angle_y)]
    ])

    translation = np.mat([tx, ty, tz])
    xyz_world = np.dot(rotate, xyz_args.T) + translation.T
    return xyz_world.T


def convert_to_uv(xyz_in_camera, intrinsics_array=in_matrix):
    """
    相机--uv
    """
    x = xyz_in_camera[0, 0]
    y = xyz_in_camera[0, 1]
    z = xyz_in_camera[0, 2]

    u0 = intrinsics_array[0]
    v0 = intrinsics_array[1]
    fx = intrinsics_array[2]
    fy = intrinsics_array[3]

    y_in = [22, 25, 32, 42, 55, 70, 100, 200, 470, 800]
    y_result = np.linspace(0, 850, 10)
    y_new = np.linspace(22, 800, 779)
    fy = interpolate.interp1d(y_in, y_result, kind="quadratic")
    y_new_result = fy(y_new)

    v = 2 * v0 - y_new_result[int(z)]

    # v = fy * y / z + v0
    u = fx * x / z + u0

    return [u, v]


# def convert_to_uv(xyz_in_camera, intrinsics_array=in_matrix):
#     """
#     相机--uv
#     """
#     s = xyz_in_camera[0, 2]
#     u0 = intrinsics_array[0]
#     v0 = intrinsics_array[1]
#     fx = intrinsics_array[2]
#     fy = intrinsics_array[3]
#     intrinsics_matrix = np.mat([
#         [fx, 0, u0],
#         [0, fy, v0],
#         [0, 0, 1]
#     ])
#     radar_uv = np.dot(intrinsics_matrix, xyz_in_camera.T)
#     return radar_uv / s


# def calculate_depth(rect_roi, intrinsics_array=in_matrix):
#     """
#     计算单目深度信息
#     :param rect_roi:左上右下
#     :param intrinsics_array:内参，intrinsics_matrix = [960, 540,775.9, 776.9]  # cx,cy,fx,fy
#     :return:目标的位置
#     """
#
#     u0 = intrinsics_array[0]
#     v0 = intrinsics_array[1]
#     fx = intrinsics_array[2]
#     fy = intrinsics_array[3]
#
#     pi = math.pi
#
#     # bbox下沿中心坐标
#     y = int(rect_roi[3])
#     x = (rect_roi[0] + rect_roi[2]) // 2
#
#     op_img = y - v0
#     angle_b = math.atan(math.fabs(op_img) / fy)
#     if op_img > 0:
#         angle_c = camera_angle_a * pi / 180 + angle_b
#     else:
#         angle_c = camera_angle_a * pi / 180 - angle_b
#     if angle_c == 0:
#         angle_c = 0.01
#
#     op_camera = round(camera_height / math.tan(angle_c), 1)
#
#     z_in_cam = (camera_height / math.sin(angle_c)) * math.cos(angle_b)
#     x_in_cam = z_in_cam * (x - u0) / fx
#     y_in_cam = z_in_cam * (y - v0) / fy
#
#     # print('op_img', op_img)
#     # print('angle_a', camera_angle_a * pi / 180)
#     # print('angle_b', angle_b)
#     # print('angle_c', angle_c)
#     # print('z_in_cam', z_in_cam)
#
#     # 注意，这里的距离是相机到地面投影的点O距离目标的距离
#     # 如果是相机到目标的距离，还需要考虑H
#     distance = '%.1f' % z_in_cam
#
#     return np.mat([x_in_cam, y_in_cam, z_in_cam]), distance


def calculate_depth(rect_roi, intrinsics_array=in_matrix):
    y_in = np.linspace(0, 900, 10)
    y_result = [22, 25, 32, 42, 55, 70, 100, 200, 470, 800]
    y_new = np.linspace(0, 899, 900)
    fy = interpolate.interp1d(y_in, y_result, kind="quadratic")
    y_new_result = fy(y_new)

    u0 = intrinsics_array[0]
    v0 = intrinsics_array[1]
    fx = intrinsics_array[2]
    fy = intrinsics_array[3]

    # bbox下沿中心坐标
    y = int(rect_roi[3])
    x = (rect_roi[0] + rect_roi[2]) // 2
    if y > 899:
        y = 899

    z_in_cam = y_new_result[2 * v0 - y]
    x_in_cam = z_in_cam * (x - u0) / fx
    y_in_cam = z_in_cam * (y - v0) / fy

    distance = '%.1f' % z_in_cam

    return np.mat([x_in_cam, y_in_cam, z_in_cam]), distance


def nonlinear():
    y = np.linspace(0, 900, 10)
    y_result = [0, 5, 12, 20, 30, 45, 70, 100, 160, 300]
    ynew = np.linspace(0, 899, 900)
    f = interpolate.interp1d(y, y_result, kind="quadratic")
    ynew_result = f(ynew)
    for i in range(len(ynew)):
        print('ynew', ynew[i], '  ', 'ynew_result', ynew_result[i])
    pl.plot(y, y_result, "ro")
    pl.plot(ynew, ynew_result, label=str("quadratic"))
    pl.show()
