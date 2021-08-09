import math

import numpy as np

out_t = [0, 0, 0]  # tx，ty,tz
out_angle = [0, 0, 0]  # angle_x，angle_y,angle_z
in_matrix = [960, 540, 1500, 1500]  # cx,cy,fx,fy
car_height = 0
camera_height = 1  # 相机地面高度
camera_angle_a = -1.5  # 相机光抽和水平线夹角


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
    s = xyz_in_camera[0, 2]
    u0 = intrinsics_array[0]
    v0 = intrinsics_array[1]
    fx = intrinsics_array[2]
    fy = intrinsics_array[3]
    intrinsics_matrix = np.mat([
        [fx, 0, u0],
        [0, fy, v0],
        [0, 0, 1]
    ])
    radar_uv = np.dot(intrinsics_matrix, xyz_in_camera.T)
    return radar_uv / s


def calculate_depth(rect_roi, intrinsics_array=in_matrix):
    """
    计算单目深度信息
    :param rect_roi:左上右下
    :param intrinsics_array:内参，intrinsics_matrix = [960, 540,775.9, 776.9]  # cx,cy,fx,fy
    :return:目标的位置
    """

    u0 = intrinsics_array[0]
    v0 = intrinsics_array[1]
    fx = intrinsics_array[2]
    fy = intrinsics_array[3]

    pi = math.pi

    # bbox下沿中心坐标
    y = int(rect_roi[3])
    x = (rect_roi[0] + rect_roi[2]) // 2

    op_img = math.fabs(y - v0)
    angle_b = math.atan(op_img / fy)
    angle_c = camera_angle_a * pi / 180 + angle_b
    if angle_c == 0:
        angle_c = 0.01

    print('y - v0:', y - v0)
    print('angle_b:', angle_b)
    print('angle_c:', angle_c)

    op_camera = round(camera_height / math.tan(angle_c), 1)

    z_in_cam = (camera_height / math.sin(angle_c)) * math.cos(angle_b)
    x_in_cam = z_in_cam * (x - u0) / fx
    y_in_cam = z_in_cam * (y - v0) / fy

    # 注意，这里的距离是相机到地面投影的点O距离目标的距离
    # 如果是相机到目标的距离，还需要考虑H
    distance = '%.1f' % z_in_cam

    return np.mat([x_in_cam, y_in_cam, z_in_cam]), distance
