# Train Custom Data

This page explains how to train your own custom data with YOLOX.

We take an example of fine-tuning YOLOX-S model on VOC dataset to give a more clear guide.

## 0. Before you start
Clone this repo and follow the [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md) to install YOLOX.

## 1. Create your own dataset
**Step 1** Prepare your own dataset with images and labels first. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme) or [CVAT](https://github.com/openvinotoolkit/cvat).

**Step 2** Then, you should write the corresponding Dataset Class which can load images and labels through `__getitem__` method. We currently support COCO format and VOC format.

You can also write the Dataset by your own. Let's take the [VOC](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/voc.py#L151) Dataset file for example:
```python
    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
```

One more thing worth noting is that you should also implement [pull_item](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/voc.py#L129) and [load_anno](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/voc.py#L121) method for the `Mosiac` and `MixUp` augmentations.

**Step 3** Prepare the evaluator. We currently have [COCO evaluator](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/evaluators/coco_evaluator.py) and [VOC evaluator](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/evaluators/voc_evaluator.py).
If you have your own format data or evaluation metric, you can write your own evaluator.

**Step 4** Put your dataset under `$YOLOX_DIR/datasets`, for VOC:

```shell
ln -s /path/to/your/VOCdevkit ./datasets/VOCdevkit
```
* The path "VOCdevkit" will be used in your exp file described in next section. Specifically, in `get_data_loader` and `get_eval_loader` function.

✧✧✧ You can download the mini-coco128 dataset by the [link](https://drive.google.com/file/d/16N3u36ycNd70m23IM7vMuRQXejAJY9Fs/view?usp=sharing), and then unzip it to the `datasets` directory. The dataset has been converted from YOLO format to COCO format, and can be used directly as a dataset for testing whether the train environment can be runned successfully.

## 2. Create your Exp file to control everything
We put everything involved in a model to one single Exp file, including model setting, training setting, and testing setting.

**A complete Exp file is at [yolox_base.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/exp/base_exp.py).** It may be too long to write for every exp, but you can inherit the base Exp file and only overwrite the changed part.

Let's take the [VOC Exp file](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/example/yolox_voc/yolox_voc_s.py) as an example.

We select `YOLOX-S` model here, so we should change the network depth and width. VOC has only 20 classes, so we should also change the `num_classes`.

These configs are changed in the `init()` method:
```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

Besides, you should also overwrite the `dataset` and `evaluator`, prepared before training the model on your own data.

Please see [get_data_loader](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/example/yolox_voc/yolox_voc_s.py#L20), [get_eval_loader](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/example/yolox_voc/yolox_voc_s.py#L82), and [get_evaluator](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/example/yolox_voc/yolox_voc_s.py#L113) for more details.

✧✧✧ You can also see the `exps/example/custom` directory for more details.

## 3. Train
Except special cases, we always recommend to use our [COCO pretrained weights](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md) for initializing the model.

Once you get the Exp file and the COCO pretrained weights we provided, you can train your own model by the following below command:
```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/the/pretrained/weights [--cache]
```
* --cache: we now support RAM caching to speed up training! Make sure you have enough system RAM when adopting it. 

or take the `YOLOX-S` VOC training for example:
```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth [--cache]
```

✧✧✧ For example:
- If you download the [mini-coco128](https://drive.google.com/file/d/16N3u36ycNd70m23IM7vMuRQXejAJY9Fs/view?usp=sharing) and unzip it to the `datasets`, you can direct run the following training code.
    ```bash
    python tools/train.py -f exps/example/custom/yolox_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth
    ```

(Don't worry for the different shape of detection head between the pretrained weights and your own model, we will handle it)

## 4. Tips for Best Training Results

As **YOLOX** is an anchor-free detector with only several hyper-parameters, most of the time good results can be obtained with no changes to the models or training settings.
We thus always recommend you first train with all default training settings.

If at first you don't get good results, there are steps you could consider to improve the model.

**Model Selection** We provide `YOLOX-Nano`, `YOLOX-Tiny`, and `YOLOX-S` for mobile deployments, while `YOLOX-M`/`L`/`X` for cloud or high performance GPU deployments.

If your deployment meets any compatibility issues. we recommend `YOLOX-DarkNet53`.

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
    self.mosaic_scale = (0.8, 1.6)
    self.shear = 2.0
    self.perspective = 0.0
    self.enable_mixup = True
```

**Design your own detector** You may refer to our [Arxiv](https://arxiv.org/abs/2107.08430) paper for details and suggestions for designing your own detector.
