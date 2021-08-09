


# 1225_17_in_camera
# # 输入视频数据
# parser.add_argument('--video', default="input/1225_17_3.mp4")
# # 输入雷达数据
# parser.add_argument('--radar_data_path', default="input/data_XY_1225_Case17_frame0-299.mat")
# radar_all_in_world = [my_file[element[0]][:] for element in my_file['data_XY']]
# out_t = [-1, 2, 0]  # tx，ty,tz
# out_angle = [-13, 0, 0]  # angle_x，angle_y,angle_z
# in_matrix = [640, 360, 1500, 400]  # cx,cy,fx,fy
# car_height = 1.5
# camera_height = 2  # 相机地面高度
# camera_angle_a = -1.5  # 相机光抽和水平线夹角

# 1225_17_in_radar
# # 输入视频数据
# parser.add_argument('--video', default="input/1225_17_3.mp4")
# # 输入雷达数据
# parser.add_argument('--radar_data_path', default="input/data_XY_1225_Case17_frame0-299.mat")
# radar_all_in_world = [my_file[element[0]][:] for element in my_file['data_XY']]
# out_t = [2, 0, 0]  # tx，ty,tz
# out_angle = [0, 0, 0]  # angle_x，angle_y,angle_z
# in_matrix = [640, 360, 1000, 3000]  # cx,cy,fx,fy
# car_height = -4
# camera_height = 1  # 相机地面高度
# camera_angle_a = -1.5  # 相机光抽和水平线夹角
