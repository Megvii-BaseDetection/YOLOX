import argparse
import os
import time
import cv2
import torch
from loguru import logger
import eventlet
import socketio

from yolox.data.data_augment import ValTransform
#FIXME
from yolox.data.datasets import COCO_CLASSES as PLAYERS_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
from yolox.utils.datasets import LoadStreams

HOST = ''
PORT = 1234

sio = socketio.Server()
app = socketio.WSGIApp(sio)

predictor = None


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Server!")
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default=None,
        type=str,
        help="ckpt for eval"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


class Predictor(object):
    def __init__(self, model, exp,
                 cls_names=PLAYERS_CLASSES,
                 trt_file=None,
                 decoder=None,
                 device="cpu",
                 fp16=False,
                 legacy=False,
                 ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Inference time: {:.4f}s".format(time.time() - t0))

        return outputs, img_info


def buildPredictor(exp, args):
    file_name = os.path.join(exp.output_dir, exp.exp_name)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.trt:
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    return Predictor(model, exp,
                     PLAYERS_CLASSES,
                     trt_file, decoder,
                     args.device, args.fp16,
                     )


def init(exp, args):
    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    logger.info("Exp:\n{}".format(exp))

    # build predictor
    global predictor
    predictor = buildPredictor(exp, args)


@sio.event
def connect(sid, environ):
    logger.info(f'connect {sid}')


@sio.event
def disconnect(sid):
    logger.info(f'disconnect {sid}')


@sio.event
def run_detection(sid, messageData):
    logger.info(f'Received Data: {messageData}')

    # dataset = LoadStreams(messageData['rtmp_url'])
    #
    # for _, img, ts in dataset:
    #     outputs, img_info = predictor.inference(img)
    #     outputs = outputs[0]
    #
    #     predData = []
    #     if outputs is not None:
    #         output = outputs.cpu()
    #
    #         bboxes = output[:, 0:4]
    #         bboxes /= img_info["ratio"]  # preprocessing: resize
    #         cls_ids = output[:, 6]
    #         scores = output[:, 4] * output[:, 5]
    #
    #         for i in range(len(bboxes)):
    #             box = bboxes[i]
    #             cls_id = int(cls_ids[i])
    #             score = scores[i]
    #             if score < predictor.confthre:
    #                 continue
    #
    #             d = {'playerId': PLAYERS_CLASSES[cls_id],
    #                  'rect': torch.tensor(box).tolist(),
    #                  'conf': float(score)
    #                  }
    #             predData.append(d)
    #
    #     # send predictions
    #     logger.info(f'Sending prediction...')
    #     detectionResult = {'timestamp': ts, 'bboxes': predData}
    #     sio.emit('detection_result', detectionResult)
    #     sio.sleep(0.001)

    cap = cv2.VideoCapture(messageData['rtmp_url'])

    while True:
        ret_val, frame = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            outputs = outputs[0]

            predData = []
            if outputs is not None:
                output = outputs.cpu()

                bboxes = output[:, 0:4]
                bboxes /= img_info["ratio"]     # preprocessing: resize
                cls_ids = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                for i in range(len(bboxes)):
                    box = bboxes[i]
                    cls_id = int(cls_ids[i])
                    score = scores[i]
                    if score < predictor.confthre:
                        continue

                    d = {'playerId': PLAYERS_CLASSES[cls_id],
                         'rect': torch.tensor(box).tolist(),
                         'conf': float(score)
                         }
                    predData.append(d)

            # send predictions
            logger.info(f'Sending prediction...')
            detectionResult = {'timestamp': ts, 'bboxes': predData}
            sio.emit('detection_result', detectionResult)
            sio.sleep(0.001)
        else:
            break


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    init(exp, args)
    eventlet.wsgi.server(eventlet.listen((HOST, PORT)), app)
