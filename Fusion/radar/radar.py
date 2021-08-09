import h5py
from Fusion.utils.convert import car_height
import numpy as np


class Radar(object):
    def __init__(
            self,
            radar_data_path,
    ):
        self.radar_data_path = radar_data_path

    def read_data(self):
        my_file = h5py.File(self.radar_data_path, 'r')
        radar_data = [my_file[element[0]][:] for element in my_file['mes_xy']]

        # 插入y轴这一列
        for index in range(len(radar_data)):
            radar_frame = radar_data[index]
            y = [car_height for j in range(len(radar_frame))]
            radar_frame = np.insert(radar_frame, 1, values=y, axis=1)
            radar_data[index] = radar_frame

        return radar_data
