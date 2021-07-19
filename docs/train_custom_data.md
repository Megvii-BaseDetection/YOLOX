# Train Custom Data.
This page explains how to train your own custom data with YOLOX.

We take an example of finetuing YOLOX-S model on VOC dataset to give a more clear guide.

## 0. Before you start
Clone this repo and follow the [README](../README.md) to install YOLOX.

## 1. Create your own dataset
**Step 1** Prepare your own dataset with images and labels first. For labeling images, you may use a tool like [Labelme](https://github.com/wkentaro/labelme) or [CVAT](https://github.com/openvinotoolkit/cvat).

**Step 2** Then, you should write the corresponding Dataset Class which can load images and labels through "\_\_getitem\_\_" method. We currently support COCO format and VOC format.  

You can also write the Dataset by you own. Let's take the [VOC](../yolox/data/datasets/voc.py#L151) Dataset file for example:
```python
    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

```

One more thing worth noting is that you should also implement "[pull_item](../yolox/data/datasets/voc.py#L129)" and "[load_anno](../yolox/data/datasets/voc.py#L121)" method for the Mosiac and MixUp augmentation.

**Step 3** Prepare the evaluator. We currently have [COCO evaluator](../yolox/evaluators/coco_evaluator.py) and [VOC evaluator](../yolox/evaluators/voc_evaluator.py). 
If you have your own format data or evaluation metric, you may write your own evaluator.

## 2. Create your Exp file to control everything
We put everything involved in a model to one single Exp file, including model setting, training setting, and testing setting. 

A complete Exp file is at [yolox_base.py](../yolox/exp/yolox_base.py). It may be too long to write for every exp, but you can inherit the base Exp file and only overwrite the changed part.

Let's still take the [VOC Exp file](../exps/example/yolox_voc/yolox_voc_s.py) for an example.

We select YOLOX-S model here, so we should change the network depth and width. VOC has only 20 classes, so we should also change the num_classes.

These configs are changed in the inti() methd:
```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

Besides, you should also overwrite the dataset and evaluator preprared before to training the model on your own data. 

Please see "[get_data_loader](../exps/example/yolox_voc/yolox_voc_s.py#L20)", "[get_eval_loader](../exps/example/yolox_voc/yolox_voc_s.py#L82)", and "[get_evaluator](../exps/example/yolox_voc/yolox_voc_s.py#L113)" for more details.

## 3. Train
Except special cases, we always recommend to use our [COCO pretrained weights](../README.md) for initializing. 

Once you get the Exp file and the COCO pretrained weights we provided, you can train your own model by the following command:
```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/the/pretrained/weights
```

or take the YOLOX-S VOC training for example: 
```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth.tar
```

(Don't worry for the different shape of detection head between the pretrained weights and your own model, we will handle it)

## 4. Tips for Best Training Results

As YOLOX is an anchor-free detector with only several hyper-parameters, most of the time good results can be obtained with no changes to the models or training settings. 
We thus always recommend you first train with all default training settings. 

If at first you don't get good results, there are steps you could considier to take to improve. 

**Model Selection** We provide YOLOX-Nano, YOLOX-Tiny, and YOLOX-S for mobile deployments, while YOLOX-M/L/X for cloud or high performance GPU deployments. 

If your deployment meets some trouble of compatibility. we recommand YOLOX-DarkNet53.  

**Training Configs** If your training overfits early, then you can reduce max\_epochs or decrease the base\_lr and min\_lr\_ratio in your Exp file:
```python
# --------------  training config --------------------- #
    self.warmup_epochs = 5
    self.max_epoch = 300
    self.warmup_lr = 0
    self.basic_lr_per_img = 0.01 / 64.0
    self.scheduler = "yoloxwarmcos"
    self.no_aug_epochs = 15
    self.min_lr_ratio = 0.05
    self.ema = True

    self.weight_decay = 5e-4
    self.momentum = 0.9
```

**Aug Configs** You may also change the degree of the augmentations. 

Generally, for small models, you should weak the aug, while for large models or small size of dataset, you may enchance the aug in your Exp file:
```python
# --------------- transform config ----------------- #
    self.degrees = 10.0
    self.translate = 0.1
    self.scale = (0.1, 2)
    self.mscale = (0.8, 1.6)
    self.shear = 2.0
    self.perspective = 0.0
    self.enable_mixup = True
```

**Design your own detector** You may refer to our [Arxiv]() paper for details and suggestions for designing your own detector.






