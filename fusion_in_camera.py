from Fusion.init.args_init import make_parser, args_analysis
from Fusion.yolox.predictor import Predictor
from Fusion.radar.radar import Radar
from Fusion.utils.convert import convert_to_uv, convert_to_world
from YOLOX.yolox.data.datasets import COCO_CLASSES
import cv2
import time
import os
import numpy as np
from loguru import logger


def process_camera(predictor, frame):
    img_outputs, img_info = predictor.inference(frame)
    img_result, _ = predictor.visual(img_outputs[0], img_info, predictor.confthre)
    return img_outputs, img_info, img_result


def process_radar(read_data, time_factor, cnt):
    radar_cnt = int(cnt / time_factor)
    radar_frame = read_data[radar_cnt]

    radar_frame_uv = []
    if len(radar_frame) > 0:
        for index in range(len(radar_frame)):
            radar_xyz = np.mat(radar_frame[index, 0:3])
            # print('radar_xyz: ')
            # print(radar_xyz)

            # get world
            redar_xyz_in_world = convert_to_world(radar_xyz)
            # print('redar_xyz_in_world: ')
            # print(redar_xyz_in_world)

            # get uv
            radar_uv = convert_to_uv(redar_xyz_in_world)
            # print('radar_uv: ')
            # print(radar_uv)
            radar_frame_uv.append(radar_uv)

    return np.array(radar_frame_uv)


def get_time_factor(radar_len, video_len):
    logger.info(f"radar_len: {radar_len}")
    logger.info(f"video_len: {video_len}")
    if radar_len == 0 or video_len == 0:
        return 0
    factor = (video_len + 1) / radar_len
    return factor


def fusion(outputs, img_info, camera_frame, radar_frame_uv):
    # TODO

    # draw
    width = img_info["width"]
    height = img_info["height"]
    if len(radar_frame_uv) > 0:
        for index in range(len(radar_frame_uv)):
            cv2.circle(camera_frame,
                       (int(radar_frame_uv[index, 0] % width),
                        int(radar_frame_uv[index, 1] % height)),
                       4, (0, 0, 255), thickness=-1)
    return camera_frame


def fusion_in_camera(predictor, radar, vis_folder, args):
    # cap
    current_time = time.localtime()
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    # save
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")

    # VideoWriter
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # radar
    read_data = radar.read_data()

    # time_factor
    time_factor = get_time_factor(len(read_data), cap.get(7))
    if time_factor == 0:
        logger.info("camera_data or radar_data is empty!")

    # start
    cnt = -1
    while True:
        cnt += 1
        ret_val, frame = cap.read()
        if ret_val:
            img_outputs, img_info, img_result = process_camera(predictor, frame)

            radar_frame_uv = process_radar(read_data, time_factor, cnt)

            img = fusion(img_outputs, img_info, img_result, radar_frame_uv)

            cv2.imshow("Fusion", img)
            if args.save_result:
                vid_writer.write(img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            logger.info("Processing done!")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = make_parser().parse_args()
    model, exp, decoder, vis_folder = args_analysis(args)
    predictor = Predictor(model, exp, COCO_CLASSES, decoder, args.device)
    radar = Radar(args.radar_data_path)
    fusion_in_camera(predictor, radar, vis_folder, args)
