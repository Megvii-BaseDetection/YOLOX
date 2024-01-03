import numpy as np
import torch
from loguru import logger

from yolox.exp import get_exp
from yolox_torchscript import DeployModule


@logger.catch
def export(exp_file, ckpt_path, output_path):
    exp = get_exp(exp_file)
    exp.merge([])

    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = True
    model_deploy = DeployModule()
    model_deploy.yolox = model
    model.eval()
    logger.info("loading checkpoint done.")

    image = torch.tensor(np.zeros((1, 416, 416, 3)).tolist())
    zone = torch.tensor([(400, 10), (400, 400), (250, 400), (250, 10), (250, 20), (250, 30)])

    torchscript_model = torch.jit.trace(model_deploy, {"inputs": {"input": image, "zone": zone}})
    torchscript_model.save(output_path)
    logger.info("generated torchscript model named {}".format(output_path))


if __name__ == "__main__":
    exp_file = r'N:\ComputerVision\object_detection\models\SoS2.6_NDS_extended2\yolox\SoS_2.6_NDS_extended2_s_416\yolox_s.py'
    ckpt_path = r'N:\ComputerVision\object_detection\models\SoS2.6_NDS_extended2\yolox\SoS_2.6_NDS_extended2_s_416\best_ckpt.pth'
    output_path = r'N:\ComputerVision\projects\Sos_2\deployment\DSVA_1888\yolox_torchscript.pt'
    export(exp_file, ckpt_path, output_path)
