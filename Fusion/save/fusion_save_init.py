import os
import time
import csv
from loguru import logger


def fusion_save_init(current_time, mode):
    data_save_folder = "./YOLOX_outputs/"
    save_folder = os.path.join(
        data_save_folder, time.strftime("fusion_result/%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if mode == 0:
        save_path = os.path.join(save_folder, "mode_0.csv")
    elif mode == 1:
        save_path = os.path.join(save_folder, "mode_1.csv")
    else:
        save_path = os.path.join(save_folder, "mode_x.csv")
    logger.info(f"video save_path is {save_path}")

    csv_f = open(save_path, 'w', encoding='utf-8', newline='' "")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["cnt", "x", "y", "z", "v", "e", "class"])
    return csv_f, csv_writer



