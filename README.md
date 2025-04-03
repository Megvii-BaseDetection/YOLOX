<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/logo.png"
     alt="YoloX" width="350"></div>
<br>

`pixeltable-yolox` is a lightweight, Apache-licensed object detection library built on pytorch. It is a fork of the
[MegVii YOLOX package](https://github.com/Megvii-BaseDetection/YOLOX) originally authored by Zheng Ge et al,
modernized for recent versions of Python and refactored to be more easily usable as a Python library.

`pixeltable-yolox` is still a work in progress! Some features of YoloX have not been ported yet.

## Usage

### Installation

```bash
pip install pixeltable-yolox
```

### Inference

```python
import requests
from PIL import Image
from yolox.models import Yolox, YoloxProcessor

url = "https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/tests/data/000000000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Yolox.from_pretrained("yolox_s")
result = model([image])  # Inputs can be PIL images or filenames
```

This yields the following output:

```python
[{'bboxes': [
   (272.36126708984375, 3.5648040771484375, 640.4871826171875, 223.2653350830078),
   (26.643890380859375, 118.68254089355469, 459.80706787109375, 315.089111328125),
   (259.41485595703125, 152.3223114013672, 295.37054443359375, 230.41783142089844)],
  'scores': [0.9417160943584335, 0.8170979975670818, 0.8095869439224117],
  'labels': [7, 2, 12]}]
```

The labels are COCO category indices.

```python
from yolox.data.datasets import COCO_CLASSES

COCO_CLASSES[7]
```

```python
'truck'
```

### Training

First unpack a [COCO dataset](https://cocodataset.org) into `./datasets/COCO`:

```text
COCO/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
    # image files
  val2017/
    # image files
```

Then on the command line:

```bash
yolox train -c yolox-s -d 8 -b 64 --fp16 -o
```

For help:

```bash
yolox train -h
```

### Separate Module/Processor Steps

To separate out the Pytorch module from image pre- and post-processing during inference (as is typical in the Hugging
Face transformers API):

```python
module = YoloxModule.from_pretrained("yolox_s")
processor = YoloxProcessor("yolox_s")
tensor = processor([image])
output = module(tensor)
result = processor.postprocess([image], output)
```

## Background

The original YOLOX implementation, while powerful, has been updated only sporadically since 2022 and now faces
compatibility issues with current Python environments, dependencies, and platforms like Google Colab. This fork aims
to provide a reliable, up-to-date, and easy-to-use version of YOLOX that maintains its Apache license, ensuring it
remains accessible for academic and commercial use.

## Status

`pixeltable-yolox` is a work in progress. So far, it contains the following changes to the base YOLOX repo:

- `pip install`able with all versions of Python (3.9+)
- New `YoloxProcessor` class to simplify inference
- Refactored CLI for training and evaluation
- Improved test coverage

The following are planned:

- CI with regular testing and updates
- Typed for use with `mypy`

## Scope

Further improvements such as model enhancements, training tooling beyond what already exists, etc., are not planned;
our goal is to take the existing feature set and make it more easily usable. However, we are happy to shepherd community
contributions in those areas and provide engineering infrastructure such as CI and regular releases. We intend to
publish a contributors’ guide once the initial release is available.

Thanks for your interest! For any questions or feedback, please contact us at `contact@pixeltable.com`.

## Who are we and why are we doing this?

Pixeltable, Inc. is a venture-backed AI infrastructure startup. Our core product is
[pixeltable](https://github.com/pixeltable/pixeltable), a database and orchestration system purpose-built for
multimodal AI workloads.

Pixeltable integrates with numerous AI services and open source technologies. In the course of integrating with YOLOX,
it became clear that there is a strong need in the vision community for a lightweight object detection library with an
untainted open source license. It also became clear that while YOLOX is an excellent foundation, it would benefit
greatly from code modernization and more regular updates.

We chose to build upon YOLOX both to simplify our own integration, and also to give something back to the community
that will (hopefully) prove useful. The Pixeltable team has decades of collective experience in open source development.
Our backgrounds include companies such as Google, Cloudera, Twitter, Amazon, and Airbnb, that have a strong commitment
to open source development and collaboration.

## Contributing

We welcome contributions from the community! If you're interested in helping maintain and improve `pixeltable-yolox`,
check out the [contributors' guide](https://github.com/pixeltable/pixeltable-yolox/blob/main/CONTRIBUTING.md).

## In memory of Dr. Jian Sun

Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have
been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our
remembrance and condolences to Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand
cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/sunjian.png"
     alt="Dr. Jian Sun" width="200">
</div>
