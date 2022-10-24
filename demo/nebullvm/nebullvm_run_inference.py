import torch
import cv2
import sys
import os
import time
from nebullvm.api.functions import optimize_model

sys.path.append(os.path.join(sys.path[0], '../..'))
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform



# Get and process image
exp = get_exp(None, 'yolox-s')
preproc = ValTransform(legacy=False)
img = cv2.imread('../../assets/dog.jpg')
img, _ = preproc(img, None, exp.test_size)
img = torch.from_numpy(img).unsqueeze(0)
img = img.float()
img = img.cuda()

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
model_opt = optimize_model(model, input_data=input_data, optimization_time="unconstrained")
print('Optimization complete')

# Check perfomance
warmup_iters = 30
num_iters = 100

print('Benchmarking')
# Unptimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    o = model(img)

    start = time.time()
    for i in range(num_iters):
      o = model(img)
stop = time.time()
print(f"Average inference time of unoptimized YOLOX: {(stop - start)}")

# Optimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    res = model_opt(img)

    start = time.time()
    for i in range(num_iters):
      res = model_opt(img)
stop = time.time()
print(f"Average inference time of YOLOX otpimized with nebullvm: {(stop - start)}")

