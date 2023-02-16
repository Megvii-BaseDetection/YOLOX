# Cache Custom Data

The caching feature is specifically tailored for users with ample memory resources. However, we still offer the option to cache data to disk, but disk performance can vary and may not guarantee optimal user experience. Implementing custom dataset RAM caching is also more straightforward and user-friendly compared to disk caching. With a few simple modifications, users can expect to see a significant increase in training speed, with speeds nearly double that of non-cached datasets.

This page explains how to cache your own custom data with YOLOX.

## 0. Before you start

**Step1** Clone this repo and follow the [README](../README.md) to install YOLOX.

**Stpe2** Read the [Training on custom data](./train_custom_data.md) tutorial to understand how to prepare your custom data.

## 1. Inheirit from `CacheDataset`


**Step1** Create a custom dataset that inherits from the `CacheDataset` class. Note that whether inheriting from `Dataset` or `CacheDataset `, the `__init__()` method of your custom dataset should take the following keyword arguments: `input_dimension`, `cache`, and `cache_type`. Also, call `super().__init__()` and pass in `input_dimension`, `num_imgs`, `cache`, and `cache_type` as input, where `num_imgs` is the size of the dataset.

**Step2** Implement the abstract function `read_img(self, index, use_cache=True)` of parent class and decorate it with `@cache_read_img`.  This function takes an `index` as input and returns an `image`, and the returned image will be used for caching. It is recommended to put all repetitive and fixed post-processing operations on the image in this function to reduce the post-processing time of the image during training.

```python
# CustomDataset.py
from yolox.data.datasets import CacheDataset, cache_read_img

class CustomDataset(CacheDataset):
    def __init__(self, input_dimension, cache, cache_type, *args, **kwargs):
        # Get the required keyword arguments of super().__init__()
        super().__init__(
            input_dimension=input_dimension,
            num_imgs=num_imgs,
            cache=cache,
            cache_type=cache_type
        )
        # ...

    @cache_read_img
    def read_img(self, index, use_cache=True):
        # get image ...
        # (optional) repetitive and fixed post-processing operations for image
        return image
```

## 2. Create your Exp file and return your custom dataset

**Step1** Create a new class that inherits from the `Exp` class provided by the `yolox_base.py`. Override the `get_dataset()` and `get_eval_dataset()` method to return an instance of your custom dataset.

**Step2** Implement your own `get_evaluator` method to return an instance of your custom evaluator.

```python
# CustomeExp.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def get_dataset(self, cache, cache_type: str = "ram"):
        return CustomDataset(
            input_dimension=self.input_size,
            cache=cache,
            cache_type=cache_type
        )

    def get_eval_dataset(self):
        return CustomDataset(
            input_dimension=self.input_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return CustomEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
```

**(Optional)** `get_data_loader` and `get_eval_loader` are now a default behavior in `yolox_base.py` and generally do not need to be changed. If you have to change `get_data_loader`, you need to add the following code at the beginning.

```python
# CustomeExp.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        # ...

```

## 3. Cache to Disk
It's important to note that the `cache_type` can be `"ram"` or `"disk"`, depending on where you want to cache your dataset. If you choose `"disk"`, you need to pass in additional parameters to `super().__init__()` of `CustomDataset`: `data_dir`, `cache_dir_name`, `path_filename`.

- `data_dir`: the root directory of the dataset, e.g. `/path/to/COCO`.
- `cache_dir_name`: the name of the directory to cache to disk, for example `"custom_cache"`, then the files cached to disk will be saved under `/path/to/COCO/custom_cache`.
- `path_filename`: a list of paths to the data relative to the `data_dir`, e.g. if you have data `/path/to/COCO/train/1.jpg`, `/path/to/COCO/train/2.jpg`, then `path_filename = ['train/1.jpg', ' train/2.jpg']`.
