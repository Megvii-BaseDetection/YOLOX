import torch
import cv2
import sys
import os
from nebullvm.api.functions import optimize_model

sys.path.append(os.path.join(sys.path[0], '../..'))
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform


# Get YOLO model
exp = get_exp(None, 'yolox-s')
model = exp.get_model()
model.cuda()
model.eval()
ckpt = torch.load('../../models/yolox_s.pth', map_location="cpu")
model.load_state_dict(ckpt["model"])

print('Optimizing')
# Run optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_data =  [((torch.randn(1, 3, 640, 640).to(device), ), 0) for i in range(100)]
optimized_model = optimize_model(model, input_data=input_data, optimization_time="unconstrained") 
# Check optimized model
print('Optimized model: ', optimized_model)
