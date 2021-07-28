# 自定义数据训练.
本也介绍如何使用YOLOX训练你自己的数据集.

我们使用VOC 数据集来微调 YOLOX-S 模型，以给出更加清晰的指导。

## 0. 开始之前
克隆这个仓库 [README](../README.md) 并安装YOLOX.

## 1. 创建自己的数据集
**Step 1** 首先准备您自己的带有图像和标签的数据集。对于标记图像，您可以使用 [Labelme](https://github.com/wkentaro/labelme) 或者 [CVAT](https://github.com/openvinotoolkit/cvat).

**Step 2** 然后，编写对应的Dataset Class，可以通过`__getitem__`方法加载图片和标签。我们目前支持 COCO 格式和 VOC 格式。您也可以自己编写数据集。我们以 [VOC]数据集文件为例(../yolox/data/datasets/voc.py#L151)
```python
    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
```
 还有一点值得注意的是你应该实现[pull_item](../yolox/data/datasets/voc.py#L129) 和 [load_anno](../yolox/data/datasets/voc.py#L121) 方法来实现 `Mosiac` and `MixUp` 增强.

**Step 3** 准备评估器。我们目前有[COCO evaluator](../yolox/evaluators/coco_evaluator.py) 和 [VOC evaluator](../yolox/evaluators/voc_evaluator.py).
如果您有自己的格式数据或评估指标，则可以编写自己的评估器

**Step 4** 将您的数据集放在$YOLOX_DIR/datasets, 对于 VOC：

```shell
ln -s /path/to/your/VOCdevkit ./datasets/VOCdevkit
```
* 路径“VOCdevkit”将在下一节描述的 exp 文件中使用。具体来说，在get_data_loader和get_eval_loader功能

## 2. 创建你的Exp文件来控制一切

我们将模型中涉及的所有内容都放在一个单独的 Exp 文件中，包括模型设置、训练设置和测试设置。
**完整的Exp文件位于[yolox_base.py](../yolox/exp/yolox_base.py).** 可能每个exp都写太长，但是你可以继承基础exp文件，只重写改变的部分.

我们以[VOC Exp file](../exps/example/yolox_voc/yolox_voc_s.py) 为例：

我们选择`YOLOX-S` 模型, 所以我们应该改变网络深度和宽度. VOC 只有20个类 ，所以我们也要改变 `num_classes`，如果你的数据集只有10个类，你也应该改为相应的类别数。

这些配置在`init()`方法中更改
```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

此外，在使用您自己的数据训练模型之前，您还应该重写dataset和evaluator。

有关更多详细信息，请参阅[get_data_loader](../exps/example/yolox_voc/yolox_voc_s.py#L20), [get_eval_loader](../exps/example/yolox_voc/yolox_voc_s.py#L82), and [get_evaluator](../exps/example/yolox_voc/yolox_voc_s.py#L113)

## 3.训练
除特殊情况外，我们始终建议使用我们的[COCO pretrained weights](../README.md) 预训练权重来初始化模型。

获得我们提供的 Exp 文件和 COCO 预训练权重后，您可以通过以下命令训练自己的模型：:
```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/the/pretrained/weights
```

或者以`YOLOX-S` VOC 训练为例：
```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth.tar
```

(不用担心预训练权重和您自己的模型之间检测头的形状不同，我们会处理好的)

## 4.获取最佳训练结果的技巧

由于YOLOX是一个只有几个超参数的无锚检测器，大多数情况下可以在不改变模型或训练设置的情况下获得良好的结果。因此，我们始终建议您首先使用所有默认训练设置进行训练。

如果一开始你没有得到好的结果，你可以考虑采取一些步骤来改进模型。

**模型选择** 我们提供`YOLOX-Nano`, `YOLOX-Tiny`和`YOLOX-S`用于移动端部署，而`YOLOX-M`/`L`/`X`用于云或高性能GPU部署

如果您的部署遇到兼容性问题。我们推荐使用`YOLOX-DarkNet53`.

**训练配置** 如果您的训练过早过拟合，那么您可以减少 max_epochs 或减少您的 Exp 文件中的 base_lr 和 min_lr_ratio：
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

**增强配置** 您还可以更改增强的程度。

一般来说，对于小模型，你应该弱化aug，而对于大模型或小数据集，你可以在exp文件中增强aug：
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

**设计你自己的检测器** 你可以参考我们的 [Arxiv](https://arxiv.org/abs/2107.08430) 论文，希望我们的相关信息和建议可以帮助你设计您自己的检测器。
