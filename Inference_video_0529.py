import argparse
import os

import cv2
import numpy as np
import torch

from loguru import logger
import time
import onnxruntime
import torchvision.transforms as transforms

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from PIL import Image

CLASSES = (
    'people','car'
)

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/whoami/Documents/Hanvon/yoloxs0413_320.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="video",
        help="mode type, eg. image, video and webcam.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default='/home/whoami/Documents/Hanvon/13-2/20230101_000613_vflip.MP4',
        help="Path to your input image/video/webcam.",
    )
    parser.add_argument(
        "--camid", 
        type=int, 
        default=0, 
        help="webcam demo camera id",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs_videos',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.5,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def inference(args, origin_img):
    # t0 = time.time()
    input_shape = tuple(map(int, args.input_shape.split(',')))

    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    # ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    # print(output[0].shape) # (1, 2100, 7)
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    return predictions, ratio

def image_process(args):
    origin_img = cv2.imread(args.input_path)
    origin_img = inference(args, origin_img)
    mkdir(args.output_path)
    output_path = os.path.join(args.output_path, args.input_path.split("/")[-1])
    logger.info("Saving detection result in {}".format(output_path))
    cv2.imwrite(output_path, origin_img)

def imageflow_demo(args):
    cap = cv2.VideoCapture(args.input_path if args.mode == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    mkdir(args.output_path)
    current_time = time.localtime()
    save_folder = os.path.join(
        args.output_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.mode == "video":
        save_path = os.path.join(save_folder, args.input_path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")

    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()

        # 定义切片参数
        slice_size = (int(width), int(height))
        print("slice size is ", slice_size)
        overlap = 0

        # 对图像进行切片
        stride = int(slice_size[0] * (1 - overlap))
        slices = []
        for y in range(0, int(height), stride):
            for x in range(0, int(width), stride):
                box = (x, y, x + slice_size[0], y + slice_size[1])
                slice_img = frame[y:y + slice_size[1], x:x + slice_size[0]]
                
                crop_height_actual, crop_width_actual = slice_img.shape[:2]
                if crop_height_actual < slice_size[0] or crop_width_actual < slice_size[1]:
                    continue
                
                slices.append((slice_img, box))  # 保存切片及其在原图中的位置信息
        
        # 对每个切片进行目标检测
        if ret_val:
            t0 = time.time()
            results = []
            for slice_img, box in slices:
                # slice_tensor = transform(slice_img).unsqueeze(0).cuda()
                pred, r = inference(args, slice_img)
                results.append((pred, box))

            # 合并检测结果并映射回原始图像的坐标空间
            for preds, box in results:
                if preds is not None:
                    # preds[:, :4] += torch.tensor([box[0], box[1], box[0], box[1]], device='cuda')  
                    
                    boxes = preds[:, :4]
                    scores = preds[:, 4:5] * preds[:, 5:]

                    boxes_xyxy = np.ones_like(boxes)
                    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
                    boxes_xyxy /= r
                    boxes_xyxy[:, :4] += [box[0], box[1], box[0], box[1]] # 映射回原始图像坐标
                    
                    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.45)
                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                        conf=args.score_thr, class_names=CLASSES)
                    
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

            cv2.imshow('frame',frame)
            # 写入拼接后的图像帧
            vid_writer.write(frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.mode == "image":
        image_process(args)
    elif args.mode == "video" or args.mode == "webcam":
        imageflow_demo(args)


    










# import os, sys

# sys.path.append(os.getcwd())
# import onnxruntime
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # 自定义的数据增强
# def get_test_transform(): 
#     return transforms.Compose([
#         transforms.Resize([320, 320]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

# # 推理的图片路径
# image = Image.open('assets/car_and_person.png').convert('RGB')

# img = get_test_transform()(image)
# img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224
# # 模型加载
# onnx_model_path = "/home/whoami/Documents/Hanvon/yoloxs0413_320.onnx"
# resnet_session = onnxruntime.InferenceSession(onnx_model_path)
# inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
# outs = resnet_session.run(None, inputs)[0]
# print("onnx weights", outs)
# print("onnx prediction", outs.argmax(axis=1)[0])




# import torch
# from torchviz import make_dot
# import onnx
# import torchvision.transforms as transforms
# from PIL import Image

# # 自定义的数据增强
# def get_test_transform(): 
#     return transforms.Compose([
#         transforms.Resize([320, 320]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

# # 推理的图片路径
# image = Image.open('assets/car_and_person.png').convert('RGB')

# img = get_test_transform()(image)
# img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224

# # 加载 ONNX 模型
# onnx_model = onnx.load("/home/whoami/Documents/Hanvon/yoloxs0413_320.onnx")

# # 使用 torch.onnx 中的函数将 ONNX 模型导入为 PyTorch 模型
# # 注意：这里需要确保你的 PyTorch 版本支持 ONNX 模型的导入
# pytorch_model = torch.onnx.imports(model=onnx_model, opset_version=11)

# # 可视化 PyTorch 模型的计算图
# # 这里需要提供一个示例输入以生成计算图
# # 你需要替换这个示例输入为你模型接受的实际输入
# # example_input = torch.randn(1, 3, 224, 224)  # 这里假设模型接受大小为(1, 3, 224, 224)的输入
# dot = make_dot(pytorch_model(img), params=dict(pytorch_model.named_parameters()))
# dot.render('pytorch_model', format='png')  # 可以选择将计算图保存为图片




# import torch
# import torch.onnx
# import os
# import torchvision.transforms as transforms
# from PIL import Image
# dependencies = ["torch"]
# from yolox.models import(
#     yolox_tiny,
#     yolox_nano,
#     yolox_s,
#     yolox_m,
#     yolox_l,
#     yolox_x,
#     yolov3,
#     yolox_custom
# )

# # 自定义的数据增强
# def get_test_transform(): 
#     return transforms.Compose([
#         transforms.Resize([320, 320]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

# # 推理的图片路径
# image = Image.open('assets/car_and_person.png').convert('RGB')

# img = get_test_transform()(image)
# img = img.unsqueeze_(0)  # -> NCHW, 1,3,320,320

# def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
#     if not onnx_path.endswith('.onnx'):
#         print('Warning! The onnx model name is not correct,\
#               please give a name that ends with \'.onnx\'!')
#         return 0

#     model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s") #导入模型
#     model.load_state_dict(torch.load(checkpoint)["model"]) #初始化权重
#     model.eval()
#     # model.to(device)
    
#     torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
#     print("Exporting .pth model to onnx model has been successful!")

# if __name__ == '__main__':
    
#     checkpoint = '/home/whoami/Downloads/yolox_s.pth'
#     onnx_path = '/home/whoami/Downloads/yolox_s.onnx'
#     # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
#     pth_to_onnx(img.cuda(), checkpoint, onnx_path)
