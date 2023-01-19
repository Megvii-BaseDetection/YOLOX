# YOLOX Cache

缓存功能主要为拥有大内存的用户设计。尽管我们保留了向磁盘缓存的选项，但由于磁盘性能的差异，我们无法保证磁盘缓存能够获得更好的用户体验。自定义数据集 RAM 缓存的实现也比磁盘缓存更简单方便，用户仅需下面几步简单的修改就可以获得近乎快一倍的训练速度。

实现支持向 RAM 缓存的最简步骤：

- 创建一个继承自 `CacheDataset` 类的自定义数据集。注意，无论是继承自 `Dataset` 还是 `CacheDataset`，自定义数据集的 `__init__()` 方法都应传入以下关键字参数：`input_dimension`、`cache` 和 `cache_type`。同时 `super().__init__()` 传入 `input_dimension`、`num_imgs`、`cache` 和 `cache_type` 作为输入，其中 num_imgs 是数据集的大小。
- 实现父类的抽象函数 `read_img(self, index, use_cache=True)`，并用 `@CacheDataset.cache_read_img` 来装饰它。这个函数根据 `index` 返回一个 `image`，返回的图片将被用于缓存。所以建议把对图片重复且固定的后处理操作都放在该函数，以减少训练时对图像的后处理时间。
- 创建一个继承自 `yolox_base.py` 提供的 `Exp` 类的新类。覆写 `get_dataset()` 方法以返回自定义数据集的实例。

下面是一个 `CustomDataset` 和 `Exp` 类的例子：

```python
# CustomDataset.py
from yolox.data.datasets import CacheDataset

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
        
    @CacheDataset.cache_read_img
    def read_img(self, index, use_cache=True):
        # get image ...
				file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
				
				# repetitive and fixed post-processing operations
				r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

# CustomeExp.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def get_dataset(self, cache, cache_type: str = "ram"):
        return CustomDataset(
            input_dimension=self.input_size,
            cache=cache,
            cache_type=cache_type
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        # ...
```

注意 `cache_type` 可以是 “ram” 或 “disk”，具体取决于要缓存数据的位置。如果选择 “disk”，`CustomDataset` 中的 `super().__init__()` 则需要传入额外的参数：`data_dir`、`cache_dir_name`、`path_filename`。

- `data_dir`：数据集的根目录，例如 `/path/to/COCO`
- `cache_dir_name`：缓存到磁盘的目录名，例如 `“custom_cache”`，那么缓存到磁盘的文件将会保存在 `/path/to/COCO/custom_cache` 下
- `path_filename`：数据相对于 data_dir 的路径的列表，例如有数据 `/path/to/COCO/train/1.jpg`、`/path/to/COCO/train/2.jpg`，那么 `path_filename = ['train/1.jpg', 'train/2.jpg']`
