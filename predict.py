import json

import cv2
import torch
from typing import Optional

from tools.demo import get_exp
from tools.demo import Predictor as YoloPredictor
from yolox.data import COCO_CLASSES

from cog import BasePredictor, Path, Input, BaseModel

YOLO_VERSIONS = ["yolox-s", "yolox-m", "yolox-l", "yolox-x"]


class Output(BaseModel):
    img: Optional[Path]
    json_str: Optional[str]


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        input_image: Path = Input(description="Path to an image"),
        model_name: str = Input(
            description="Model name", default="yolox-s", choices=YOLO_VERSIONS
        ),
        conf: float = Input(description="Confidence threshold: Only detections with confidence higher are kept",
                            default=0.3, ge=0., le=1.),
        nms: float = Input(description="NMS threshold: NMS removes redundant detections. Detections with overlap "
                                       "percentage (IOU) above this threshold are consider redundant to each other and "
                                       "only one of them will be kept", default=0.3, ge=0., le=1.),
        tsize: int = Input(description="Resize image to this size", default=640),

        return_json: bool = Input(
        description="Return results in json format", default=False
        ),
    ) -> Output:
        input_image = str(input_image)
        model_name = str(model_name)
        tsize = int(tsize)

        # Load model
        exp = get_exp(None, model_name)
        exp.test_conf = float(conf)
        exp.nmsthre = float(nms)
        exp.test_size = (tsize, tsize)
        model = exp.get_model().cuda()
        model.eval()

        ckpt = f'models/{model_name.replace("-", "_")}.pth'
        ckpt = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        predictor = YoloPredictor(
            model, exp, COCO_CLASSES, None, None, "gpu", None, False
        )

        # inference
        outputs, img_info = predictor.inference(input_image)

        if bool(return_json):
            return Output(
                json_str=get_output_str(
                    outputs[0], img_info["ratio"], predictor.cls_names
                )
            )
        else:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            output_path = f"outputs.png"
            cv2.imwrite(output_path, result_image)
            return Output(img=Path(output_path))


def get_output_str(outputs, ratio, class_names, cls_conf=0.35):
    if outputs is None:
        return json.dumps("")

    output = outputs.cpu().numpy()

    bboxes = output[:, 0:4]
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    output_dict = dict()
    for i in range(len(bboxes)):
        if scores[i] > cls_conf:
            box = bboxes[i]
            output_dict[f"Det-{i}"] = {
                "x0": box[0],
                "y0": box[1],
                "x1": box[2],
                "y1": box[3],
                "score": scores[i],
                "cls": class_names[int(cls[i])],
            }

    return json.dumps(str(output_dict))
