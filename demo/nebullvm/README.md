# **Accelerate and deploy YOLOX with with nebullvm in Python**

## Steps to integrate YOLOX with nebullvm
1. Download the YOLOX model
Download model from orginal [README](../../README.md)
2. Optimize model
```python
# get YOLO model
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform

exp = get_exp(None, 'yolox-s') # type here your model name
model = exp.get_model()
model.cuda()
model.eval()
ckpt = torch.load('../../models/yolox_s.pth', map_location="cpu") # type path to the your model
model.load_state_dict(ckpt["model"])

# run optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_data =  [((torch.randn(1, 3, 640, 640).to(device), ), 0) for i in range(100)]
optimized_model = optimize_model(model, input_data=input_data, optimization_time="unconstrained") 
```
More details [here](nebullvm_optimize.py)

3. Run inference and compare latency of the optimized model with that of the original model
```python
# check perfomance
warmup_iters = 30
num_iters = 100

# Non optimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    o = model(img)

    start = time.time()
    for i in range(num_iters):
      o = model(img)
stop = time.time()
print(f"Took unoptimized YOLO: {(stop - start)}")

# Optimized model perfomance
with torch.no_grad():
  for i in range(warmup_iters):
    res = model_opt(img)

    start = time.time()
    for i in range(num_iters):
      res = model_opt(img)
stop = time.time()
print(f"Took optimized YOLO: {(stop - start)}")
```
More details [here](nebullvm_run_inference.py)