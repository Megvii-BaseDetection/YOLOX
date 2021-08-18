import numpy as np
import math
from Fusion.utils.visualize import draw_in_radar
from Fusion.utils.visualize import _CLASS_SIZE
from Fusion.fusion_in_radar.filter import filter_two_line


def get_euclidean_distance(x, y):
    d = np.sqrt(np.sum(np.square(x - y)))
    return d


def association(camera_frame, camera_class, radar_frame, mode):
    euclidean_threshold = 5
    euclidean_distance = np.zeros((len(camera_frame), len(radar_frame)))
    related_data = np.zeros((4, len(camera_frame), len(radar_frame)))  # x,z,v,e

    for c in range(len(camera_frame)):
        x = camera_frame[c, 0]
        z = camera_frame[c, 2]
        class_name = camera_class[c]

        if class_name in _CLASS_SIZE:
            width = _CLASS_SIZE.get(class_name)[0]
            height = _CLASS_SIZE.get(class_name)[1]
            if z < 60:
                width, height = height, width
            for r in range(len(radar_frame)):
                radar_x = radar_frame[r, 0]
                radar_z = radar_frame[r, 2]
                radar_v = radar_frame[r, 3]
                radar_e = radar_frame[r, 4]
                euclidean_distance[c][r] = get_euclidean_distance(
                    np.array([x, z]),
                    np.array([radar_x, radar_z])
                )
                # we
                if mode == 0:
                    if math.fabs(x - radar_x) < width and math.fabs(z - radar_z) < height:
                        related_data[0][c][r] = radar_x
                        related_data[1][c][r] = radar_z
                        related_data[2][c][r] = radar_v
                        related_data[3][c][r] = radar_e

                # euclidean_distance
                elif mode == 1:
                    dis = euclidean_distance[c][r]
                    if 0 < dis < euclidean_threshold:
                        related_data[0][c][r] = radar_x
                        related_data[1][c][r] = radar_z
                        related_data[2][c][r] = radar_v
                        related_data[3][c][r] = radar_e

    return related_data


def fusion(related_data, camera_class, camera_scores, camera_frame):
    fusion_frame = []
    fusion_class = []
    for c in range(len(related_data[0])):
        x = camera_frame[c, 0]
        z = camera_frame[c, 2]
        class_name = camera_class[c]

        related_x, related_z, related_v, related_e = [], [], [], []
        for r in range(len(related_data[0][c])):
            if related_data[0][c][r] != 0 or related_data[1][c][r] > 0:
                related_x.append(related_data[0][c][r])
                related_z.append(related_data[1][c][r])
                related_v.append(related_data[2][c][r])
                related_e.append(related_data[3][c][r])

        if len(related_x) <= 0:
            continue
        else:
            fusion_frame.append([(np.average(related_x) + x) / 2,
                                 0,
                                 (np.average(related_z) + z) / 2,
                                 np.average(related_v),
                                 np.average(related_e)]
                                )
            fusion_class.append(class_name)

    return np.array(fusion_frame), np.array(fusion_class)


def process_fusion(camera_frame, camera_scores, camera_class, radar_frame):
    # filter,only save two car line
    camera_frame, camera_scores, camera_class, radar_frame, roi = filter_two_line(
        camera_frame, camera_scores, camera_class, radar_frame)

    # association
    related_data = association(camera_frame, camera_class, radar_frame, 0)

    # fusion
    fusion_frame, fusion_class = fusion(related_data, camera_class, camera_scores, camera_frame)

    # draw
    draw_in_radar(camera_frame, camera_class, radar_frame, fusion_frame, fusion_class, roi)
