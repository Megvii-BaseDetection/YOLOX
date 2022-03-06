# Freeze module

This page guide users to freeze module in YOLOX.  
Exp controls everything in YOLOX, so let's start from creating an Exp object.

## 1. Create your own expermiment object

We take an example of YOLOX-S model on COCO dataset to give a more clear guide.

Import the config you want (or write your own Exp object inherit from `yolox.exp.BaseExp`).
```python
from yolox.exp.default.yolox_s import Exp as MyExp
```

## 2. Override `get_model` method

Here is a simple code to freeze backbone (FPN not included) of module.
```python
class Exp(MyExp):

    def get_model(self):
        from yolox.utils import freeze_module
        model = super().get_model()
        freeze_module(model.backbone.backbone)
        return model
```
if you only want to freeze FPN, `freeze_module(model.backbone)` might help.

## 3. Train
Suppose that the path of your Exp  is `/path/to/my_exp.py`, use the following command to train your model.
```bash
python3 -m yolox.tools.train -f /path/to/my_exp.py
```
For more details of training, run the following command.
```bash
python3 -m yolox.tools.train --help
```
