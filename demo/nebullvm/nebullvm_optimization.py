import torch
import time
from nebullvm.api.functions import optimize_model # Install DL compilers
from yolox.exp import get_exp

# Get YOLO model
exp = get_exp(None, 'yolox-s') # select model name
model = exp.get_model()
model.cuda()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy data for the optimizer
input_data =  [((torch.randn(1, 3, 640, 640).to(device), ), 0) for i in range(100)] 

# ---------- Optimization ---------- 
optimized_model = optimize_model(model, input_data=input_data, optimization_time="constrained")  # Optimization without performance loss


# ---------- Benchmarks ---------- 
# Select image to test the latency of the optimized model

# Create dummy image
img = torch.randn(1, 3, 640, 640).to(device)

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
