# **Accelerate YOLOX inference with nebullvm in Python**

This document shows how to accelerate YOLOX inference time with nebullvm.

[nebullvm](https://github.com/nebuly-ai/nebullvm) is an open-source library designed to accelerate AI inference of deep learning models in a few lines of code. nebullvm leverages state-of-the-art model optimization techniques such as deep learning compilers (TensorRT, Openvino, ONNX Runtime, TVM, TF Lite, DeepSparse, etc.), various quantization and compression strategies to achieve the maximum physically possible acceleration on the user's hardware.

## Benchmarks
Following are the results of the nebullvm optimization on YOLOX without loss of accuracy.
For each model-hardware pairing, response time was evaluated as the average over 100 predictions. The test was run on Nvidia Tesla T4 (g4dn.xlarge) and Intel XEON Scalable (m6i.24xlarge and c6i.12xlarge) on AWS.

| Model   | Hardware     | Unoptimized (ms)| Nebullvm optimized (ms) | Speedup |
|---------|--------------|-----------------|-------------------------|---------|
| YOLOX-s | g4dn.xlarge  |       13.6      |           9.0           |   1.5x  |
| YOLOX-s | m6i.24xlarge |       32.7      |           8.8           |   3.7x  |
| YOLOX-s | c6i.12xlarge |       34.4      |           12.4          |   2.8x  |
| YOLOX-m | g4dn.xlarge  |       24.2      |           22.4          |   1.1x  |
| YOLOX-m | m6i.24xlarge |       55.1      |           36.0          |   2.3x  |
| YOLOX-m | c6i.12xlarge |       62.5      |           26.9          |   2.6x  |
| YOLOX-l | g4dn.xlarge  |       84.4      |           80.5          |   1.5x  |
| YOLOX-l | m6i.24xlarge |       88.0      |           33.7          |   2.6x  |
| YOLOX-l | c6i.12xlarge |      102.8      |           54.2          |   1.9x  |
| YOLOX-x | g4dn.xlarge  |       87.3      |           34.0          |   2.6x  |
| YOLOX-x | m6i.24xlarge |      134.5      |           56.6          |   2.4x  |
| YOLOX-x | c6i.12xlarge |      162.0      |           95.4          |   1.7x  |

## Steps to accelerate YOLOX with nebullvm
1. Download a YOLOX model from the original [readme](https://github.com/Megvii-BaseDetection/YOLOX)
2. Optimize YOLOX with nebullvm
3. Perform inference and compare the latency of the optimized model with that of the original model

[Here](nebullvm_optimization.py) you can find a demo in python.


First, let's install nebullvm. The simplest way is by using pip.
```
pip install nebullvm
```
Now, let's download one of YOLOX models and optimize it with nebullvm.

```python
# Import YOLOX model
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform

exp = get_exp(None, 'yolox-s') # select model name
model = exp.get_model()
model.cuda()
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_data =  [((torch.randn(1, 3, 640, 640).to(device), ), 0) for i in range(100)]

# Run nebullvm optimization without performance loss
optimized_model = optimize_model(model, input_data=input_data, optimization_time="constrained")
```
Find [here](nebullvm_optimize.py) the complete script in python with more details.

In this example, we optimized YOLOX without any loss in accuracy. To further speed up the model by means of more aggressive optimization techniques, proceed as follows:
- Set *optimization_time="unconstrained"*. With the unconstrained option, nebullvm will test time-consuming techniques such as pruning and quantization-aware training (QAT).
- Set the *metric_drop_ths* parameter to be greater than zero (by default, *metric_drop_ths=0*). In this way, we will allow nebullvm to test optimization techniques that involve a tradeoff of some trade-off of a certain metric. For example, to test maximum acceleration with a minimum loss of accuracy of 3%, set *metric_drop_ths=0.03* and *metric="accuracy"*.
For more information about nebullvm API, see [nebullvm documentation](https://github.com/nebuly-ai/nebullvm).


Let's now compare the latency of the optimized model with that of the original model. 
Note that before testing latency of the optimized model, it is necessary to perform some warmup runs, as some optimizers fine-tune certain internal parameters during the first few inferences after optimization.

```python
# Check perfomance
warmup_iters = 30
num_iters = 100

# Unoptimized model perfomance
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
    res = model_opt(img)

    start = time.time()
    for i in range(num_iters):
      res = model_opt(img)
stop = time.time()
print(f"Average inference time of YOLOX otpimized with nebullvm: {(stop - start)/num_iters*1000} ms")
```
Find [here](nebullvm_optimization.py) the complete script in python with more details.
