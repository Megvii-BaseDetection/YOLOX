## YOLOX-executorch in Python

This doc introduces how to convert your pytorch model into executorch, and how to run an executorch runtime demo to verify your convertion.

### Step1: Install executorch

run the following command to install onnxruntime:
```shell
pip install executorch
```

#### Convert Your Model to ONNX

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```
Then, you can:

1. Convert a standard YOLOX model by -n:
```shell
python3 tools/export_executorch.py --output-name yolox_s.pte -n yolox-s -c yolox_s.pth
```
Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nano, yolox-tiny, yolov3]
* -c: the model you have trained
* To customize an input shape for onnx model,  modify the following code in tools/export_executorch.py:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

1. Convert a standard YOLOX model by -f. When using -f, the above command is equivalent to:

```shell
python3 tools/export_executorch.py --output-name yolox_s.pte -f exps/default/yolox_s.py -c yolox_s.pth
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_executorch.py --output-name your_yolox.pte -f exps/your_dir/your_yolox.py -c your_yolox.pth
```

### Step3: Executorch Runtime Demo

Step1.
```shell
cd <YOLOX_HOME>/demo/executorch
```

Step2. 
```shell
python3 executorch_inference.py -m <EXECUTORCH_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted pte model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for executorch convertion.
