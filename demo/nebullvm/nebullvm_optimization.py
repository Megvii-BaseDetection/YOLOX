import torch
import sys
import os
import time
import cv2
from nebullvm.api.functions import optimize_model # Install DL compilers
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform

# Get YOLO model
exp = get_exp(None, 'yolox-s') # select model name
model = exp.get_model()
model.cuda()
model.eval()

## Load YOLOX weights
# weight_path = '../../models/yolox_s.pth'
# ckpt = torch.load(weight_path, map_location="cpu")
# model.load_state_dict(ckpt["model"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy data for the optimizer
input_data =  [((torch.randn(1, 3, 640, 640).to(device), ), 0) for i in range(100)] 

# ---------- Optimization ---------- 
optimized_model = optimize_model(model, input_data=input_data, optimization_time="constrained")  # Optimization without performance loss


# ---------- Benchmarks ---------- 
# Select image to test the latency of the optimized model

# Option 1: Create dummy image
img = torch.randn(1, 3, 640, 640).to(device)

## Option 2: Get image from yolox repository
# exp = get_exp(None, 'yolox-s') # select model name
# preproc = ValTransform(legacy=False)
# img = cv2.imread('../../assets/dog.jpg')
# from pathlib import Path
# print(Path('../../assets/dog.jpg').exists())
# img, _ = preproc(img, None, exp.test_size)
# img = torch.from_numpy(img).unsqueeze(0)
# img = img.float()
# img = img.cuda()

# Check perfomance
warmup_iters = 30
num_iters = 100

# Unptimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    o = model(img)

    start = time.time()
    for i in range(num_iters):
      o = model(img)
stop = time.time()
print(f"Average inference time of unoptimized YOLOX: {(stop - start)/num_iters*1000} ms")

# Optimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    res = optimized_model(img)

    start = time.time()
    for i in range(num_iters):
      res = optimized_model(img)
stop = time.time()
print(f"Average inference time of YOLOX otpimized with nebullvm: {(stop - start)/num_iters*1000} ms")
