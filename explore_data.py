import numpy as np
import pandas as pd
import os
import json

import fiftyone as fo
import fiftyone.utils.coco as fouc

from tqdm import tqdm

ROOT_DIR = "/workspace/mnt/"
TRAIN_SAVE_PATH = f"{ROOT_DIR}data/train2017/"
VAL_SAVE_PATH = f"{ROOT_DIR}data/val2017/"

ANNOTATION_SAVE_PATH = f"{ROOT_DIR}data/annotations/"

# load the validation data
val_data = json.load(open(f"{ANNOTATION_SAVE_PATH}val.json"))
df_image_data = pd.json_normalize(val_data["images"])

# Load the prediction results
results = json.load(open(f"{ANNOTATION_SAVE_PATH}inference_result.json"))
predictions = results["annotations"]


# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=VAL_SAVE_PATH,
    labels_path=f"{ANNOTATION_SAVE_PATH}val.json",
    include_id=True,
)
coco_dataset.add_sample_field("source_name", fo.StringField)


# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]

# Add COCO predictions to `predictions` field of dataset
classes = coco_dataset.default_classes
fouc.add_coco_labels(coco_dataset, "predictions", predictions, classes)

# loop through the coco_dataset and tag each with the source_name
for sample in tqdm(coco_dataset.iter_samples(), total=len(coco_dataset)):
    # find the row in df_image_data that matches the sample's coco_id
    row = df_image_data[df_image_data["id"] == sample["coco_id"]].iloc[0]
    sample["source_name"] = row["source_name"]
    sample["remote_path"] = row["remote_path"]
    if sample.detections is not None:
        for detection in sample.detections.detections:
            bbox_area = detection.bounding_box[2] * detection.bounding_box[3]
            detection.set_attribute_value("bbox_area", bbox_area)
    sample.save()


session = fo.launch_app(coco_dataset, port=5151, address="0.0.0.0")

# Blocks execution until the App is closed
session.wait()