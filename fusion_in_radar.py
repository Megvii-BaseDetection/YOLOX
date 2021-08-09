from Fusion.init.args_init import make_parser, args_analysis
from Fusion.yolox.predictor import Predictor
from Fusion.radar.radar import Radar
from Fusion.utils.convert import calculate_depth, convert_to_world
from YOLOX.yolox.data.datasets import COCO_CLASSES
import matplotlib.pyplot as plt
import cv2
import time
import os
import numpy as np
from loguru import logger


def draw_distance(im, left, top, right, bottom, distance):
    # 绘制bbox下沿中心坐标
    y = int(bottom)
    x = (left + right) // 2
    cv2.circle(im, (x, y), 4, (255, 178, 50), thickness=-1)

    # 绘制竖直线
    cv2.line(im, (960, 0), (960, 1080), (0, 0, 255), thickness=1)

    # 绘制distance
    cv2.putText(im, distance + 'm', (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def process_camera(predictor, frame):
    img_outputs, img_info = predictor.inference(frame)
    img_result, bboxes = predictor.visual(img_outputs[0], img_info, predictor.confthre)
    camera_frame = []
    if len(bboxes) > 0:
        for index in range(len(bboxes)):
            left = int(bboxes[index, 0])
            top = int(bboxes[index, 1])
            right = int(bboxes[index, 2])
            bottom = int(bboxes[index, 3])

            # get camera
            rect_roi = [left, top, right, bottom]
            camera_xyz, distance = calculate_depth(rect_roi)
            # print('camera_xyz: ')
            # print(camera_xyz)

            # get world
            camera_xyz_in_world = convert_to_world(camera_xyz)
            # print('camera_xyz_in_world: ')
            # print(camera_xyz_in_world)

            camera_frame.append([camera_xyz_in_world[0, 0],
                                 camera_xyz_in_world[0, 1],
                                 camera_xyz_in_world[0, 2]])
            draw_distance(img_result, left, top, right, bottom, distance)

    return img_outputs, img_info, img_result, np.array(camera_frame)


def process_radar(read_data, time_factor, cnt):
    radar_cnt = int(cnt / time_factor)
    radar_frame = read_data[radar_cnt]
    return radar_frame


def get_time_factor(radar_len, video_len):
    logger.info(f"radar_len: {radar_len}")
    logger.info(f"video_len: {video_len}")
    if radar_len == 0 or video_len == 0:
        return 0
    factor = (video_len + 1) / radar_len
    return factor


def fusion(img_outputs, img_info, img_result, camera_frame, radar_frame):
    # TODO

    # draw
    plt.cla()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim(xmax=50, xmin=-50)
    plt.ylim(ymax=300, ymin=0)
    plt.title('Fusion In Radar Coordinate', fontsize='large', fontweight='bold', verticalalignment='center')
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    if len(camera_frame) > 0:
        plt.scatter(camera_frame[:, 0], camera_frame[:, 2], s=area, c=colors2, alpha=0.4, label='camera')
    if len(radar_frame) > 0:
        plt.scatter(radar_frame[:, 0], radar_frame[:, 2], s=area, c=colors1, alpha=0.4, label='radar')
    plt.legend()
    plt.pause(0.0001)


def fusion_in_radar(predictor, radar, vis_folder, args):
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
    plt.ion()
    while True:
        cnt += 1
        ret_val, frame = cap.read()
        if ret_val:
            img_outputs, img_info, img_result, camera_frame = process_camera(predictor, frame)

            radar_frame = process_radar(read_data, time_factor, cnt)

            fusion(img_outputs, img_info, img_result, camera_frame, radar_frame)

            cv2.imshow("Fusion", img_result)
            if args.save_result:
                vid_writer.write(img_result)
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
    fusion_in_radar(predictor, radar, vis_folder, args)
