import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        self._yolox_size = 'yolox_s'

        if self._yolox_size == 'yolox_s':
          self.depth = 0.33
          self.width = 0.50
        elif self._yolox_size == 'yolox_m':
          self.depth = 0.67
          self.width = 0.75
        else: # yolox_l
          self.depth = 1.
          self.width = 1.

        # Define yourself dataset path
        self.data_dir = "/workspace/mnt/data"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.num_classes = 4

        self.max_epoch = 600
        self.data_num_workers = 8#1
        self.eval_interval = 1

        self.input_size = (416, 416)
        self.test_size = (416, 416)

        self.mosaic_prob = 0 # this config means tiling multiple images into one image, having this does not work with albumentations bboxes
        self.enable_mixup = False # this config means blending multiple images into one image with different transparencies
        self.multiscale_range = 0 # this does not resize the image as an augmentation, we will rely on albumentations BBoxSafeRandomCrop instead, since this is not safe for whole fish bbox

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # For MFT-PG only
        self.category_id_to_name_json = '{"0": "HIGH", "1": "LOW", "2": "MEDIUM", "3": "PARTIAL"}' 

