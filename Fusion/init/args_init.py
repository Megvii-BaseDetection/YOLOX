import sys

sys.path.insert(0, './YOLOX')

import argparse
import os
import torch
from loguru import logger
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import get_model_info


def make_parser():
    model = 'yolox-s'
    ckpt = 'model/yolox_s.pth'
    video_or_image = "video"
    video_path = "input/8th_20210615/radar_20210615_144541.mp4"
    conf = 0.5
    nms = 0.2
    tsize = 640
    device = "gpu"
    save_result = True
    radar_data_path = "input/8th_20210615/radar_20210615_144541_XY_2.mat"

    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("--demo", default=video_or_image, help="demo type, eg. image video")
    parser.add_argument("-n", "--name", type=str, default=model, help="model name")
    parser.add_argument("--path", default=video_path, help="path to images or video")
    parser.add_argument("--save_result",
                        default=save_result,
                        action="store_true",
                        help="whether to save the inference result of image/video",
                        )
    parser.add_argument('--radar_data_path', default=radar_data_path)

    # exp file
    parser.add_argument("-c", "--ckpt", default=ckpt, type=str, help="ckpt for eval")
    parser.add_argument("--device", default=device, type=str, help="can either be cpu or gpu", )
    parser.add_argument("--conf", default=conf, type=float, help="test conf")
    parser.add_argument("--nms", default=nms, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=tsize, type=int, help="test img size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    return parser


def args_analysis(args):
    exp = get_exp(args.exp_file, args.name)

    args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    decoder = None
    return model, exp, decoder, vis_folder
