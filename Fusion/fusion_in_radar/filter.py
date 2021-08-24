import numpy as np

roi = [0, 160, 8, 60]


def is_in_rectangle(x, z):
    x0, z0 = roi[0], roi[1]
    x1, z1 = roi[2], roi[3]
    if x < x0 or x > x1:
        return False
    if z > z0 or z < z1:
        return False
    return True


def filter_car_line(camera_frame, camera_scores, camera_class, radar_frame):
    c_frame, c_scores, c_class, r_frame = [], [], [], []
    for c in range(len(camera_frame)):
        x = camera_frame[c, 0]
        z = camera_frame[c, 2]
        if is_in_rectangle(x, z):
            c_frame.append(camera_frame[c])
            c_scores.append(camera_scores[c])
            c_class.append(camera_class[c])

    for r in range(len(radar_frame)):
        x = radar_frame[r, 0]
        z = radar_frame[r, 2]
        if is_in_rectangle(x, z):
            r_frame.append(radar_frame[r])

    return np.array(c_frame), np.array(c_scores), np.array(c_class), np.array(r_frame), roi
