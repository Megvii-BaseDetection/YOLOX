# %%
import boto3
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from io import BytesIO
import ast
import random
import PIL.Image as Image
import time

from functools import partial
from aquabyte.lib.db.snowflake import snowflake_query_to_df
SNOWFLAKE_DSN = '/dsn/snowflake/mochi'
snowflake_query_to_df = partial(snowflake_query_to_df, ssm_name=SNOWFLAKE_DSN)

import os

# %%
ROOT_DIR = "/workspace/mnt/"
TRAIN_SAVE_PATH = f"{ROOT_DIR}data/train2017/"
VAL_SAVE_PATH = f"{ROOT_DIR}data/val2017/"
ANNOTATION_SAVE_PATH = f"{ROOT_DIR}data/annotations/"
VAL_FRAC = 0.1
os.makedirs(TRAIN_SAVE_PATH, exist_ok=True)
os.makedirs(VAL_SAVE_PATH, exist_ok=True)
os.makedirs(ANNOTATION_SAVE_PATH, exist_ok=True)

# latest plali queues
PLALI_QUEUE_NAMES = [
    "fish_detector_visibility_belsvik_and_ras_f3r",
    "fish_bbox_fish_bbox_laksefjord_smolt_ras",
    "fish_bbox_smolt_ras_gtsf_thumbnails_2024_01_19",
    "fish_bbox__smolt_ras_gtsf_thumbnails__2024_01_19",
    "fish_bbox_imr_austevoll_jellyfish",
    "fish_bbox_fish_bbox_100day_sampled",
    "fish_bbox_novasea_slaughter_line",
    "fish_bbox_synthetic_images",
    "fish_bbox_toy_fish",
    "fish_bbox_penflix_plali_samples",
    "fish_bbox_in_air_gtsf",
    "fish_bbox_highrecall_ice_fish_farm_feedingcam",
]

# these constants are used to load the dataset used to train the previous model
OLD_BUCKET = "s3://aquabyte-frames-resized-inbound/"
NEW_BUCKET = "s3://aquabyte-research/pwais/mft-pg/datasets_s3/high_recall_fish1/images/"
S3_CSV_PATH = "s3://aquabyte-research/pwais/mft-pg/datasets_s3/high_recall_fish1/hrf_with_keypoint_visibility_dataset.csv"

category_id_to_name_json = '{"0": "HIGH", "1": "LOW", "2": "MEDIUM", "3": "PARTIAL"}' 
id_to_category = json.loads(category_id_to_name_json)
category_to_id = dict((c, i) for i, c in id_to_category.items())

s3 = boto3.client('s3')

# %% [markdown]
# ## load annotations from PLALI

# %%
# make the additional sql using PLALI_QUEUE_NAMES
selection_criteria = " or ".join([f"startswith(pw.name, '{q}')" for q in PLALI_QUEUE_NAMES])

sql=f"""   
select
    --load json and select first element of pi.images
    pi.images[0]::varchar as images,
    pa.annotation:annotations as annotation,
    -- find if the annotation is a skip by checking if skipReasons exist as a key
    pa.annotation:skipReasons is not null as is_skip,
    pa.plali_image_id,
    --pa.annotator_email,
    --pa.annotation_time,
    pw.name
from
    prod.plali_workflows as pw
    join prod.plali_images as pi on pi.workflow_id = pw.id
    left join prod.plali_annotations as pa on pa.plali_image_id = pi.id
where true
    and is_skip = false
    and ({selection_criteria})
"""

df_annotations = snowflake_query_to_df(sql)
df_annotations['annotation'] = df_annotations['annotation'].apply(lambda x: json.loads(x) if x is not None else [])
# change the name by removing the suffix, anything that comes after the last underscore, and drop the last character if it is an underscore
df_annotations['name'] = df_annotations['name'].apply(lambda x: x.rsplit('_q', 1)[0].rstrip('_'))

# %% [markdown]
# ## load the original high recall fish detector training set

# %%
# # use boto to load a csv file from s3
bucket, key = S3_CSV_PATH[5:].split('/', 1)
obj = s3.get_object(Bucket=bucket, Key=key)
df_hr_dataset = pd.read_csv(BytesIO(obj['Body'].read()))
# load the literals using ast for the following columns: images, metadata, label_set, original_annotation
df_hr_dataset['images'] = df_hr_dataset['images'].apply(lambda x: ast.literal_eval(x)[0])
df_hr_dataset['metadata'] = df_hr_dataset['metadata'].apply(lambda x: ast.literal_eval(x))
df_hr_dataset['label_set'] = df_hr_dataset['label_set'].apply(lambda x: ast.literal_eval(x))
df_hr_dataset['original_annotation'] = df_hr_dataset['original_annotation'].apply(lambda x: ast.literal_eval(x))
df_hr_dataset['annotation'] = df_hr_dataset['annotation'].apply(lambda x: ast.literal_eval(x))
df_hr_dataset['pen_id'] = df_hr_dataset['images'].apply(lambda x: x.split('/pen-id=')[1].split('/')[0])
df_hr_dataset['captured_at'] = df_hr_dataset['images'].apply(lambda x: x.split('/at=')[1].split('/')[0])

# replace the image path with the new bucket in images
df_hr_dataset['images'] = df_hr_dataset['images'].apply(lambda x: x.replace(OLD_BUCKET, NEW_BUCKET))
df_hr_dataset['is_skip'] = False

#   drop columns starting with "Unnamed"
df_hr_dataset = df_hr_dataset.loc[:, ~df_hr_dataset.columns.str.contains('^Unnamed')]

# %%
# #check if original annotation is the same as annotation, which is indeed the case
# df_hr_dataset['is_same'] = df_hr_dataset.apply(lambda x: x['original_annotation'].get('annotations', []) == x['annotation'], axis=1)

# # get subset of the dataset where original_annotation is not the same as annotation
# df_hr_dataset_not_same = df_hr_dataset[~df_hr_dataset['is_same']]
# df_hr_dataset_not_same

# %% [markdown]
# ## merge datasets, download all images, create coco jsons

# %%
# concat the two dataframes, keeping only the columns that are in df_annotations
df_hr_dataset = df_hr_dataset[df_annotations.columns]
df_hr_dataset = pd.concat([df_hr_dataset, df_annotations], ignore_index=True)
# sort by plali_image_id
df_hr_dataset = df_hr_dataset.sort_values(by='plali_image_id').reset_index(drop=True)

# %%
# download the images from images column to the local machine, save them in the IMAGE_SAVE_PATH
local_paths = []
not_found = []
is_val = []
rand = random.Random(1337)

for i, row in tqdm(df_hr_dataset.iterrows(), total=df_hr_dataset.shape[0]):
    is_val_image = rand.random() < VAL_FRAC
    remote_path = row['images']
    file_extension = remote_path.split('.')[-1]
    if is_val_image:
        local_path = f"{VAL_SAVE_PATH}{row.plali_image_id}.{file_extension}"
        is_val.append(True)
    else:
        local_path = f"{TRAIN_SAVE_PATH}{row.plali_image_id}.{file_extension}"
        is_val.append(False)
    bucket = remote_path.split('/')[2]
    key = '/'.join(remote_path.split('/')[3:])
    local_paths.append(local_path)
    # check if the file already exists
    if os.path.exists(local_path):
        continue
    # download the file, catch if the file is not found
    try:
        s3.download_file(bucket, key, local_path)
    except Exception as e:
        not_found.append(row.plali_image_id)
        print(f"{remote_path}\n{e}")
        print(f"{bucket}/{key}")
df_hr_dataset['local_path'] = local_paths
df_hr_dataset['is_val_image'] = is_val
# drop the rows where the image was not found
df_hr_dataset_found = df_hr_dataset[~df_hr_dataset['plali_image_id'].isin(not_found)]
df_hr_dataset_found = df_hr_dataset_found.reset_index(drop=True).copy()


# %%
# loop through the images and get the image height and width
image_heights = []
image_widths = []
for i, row in tqdm(df_hr_dataset_found.iterrows(), total=df_hr_dataset_found.shape[0]):
    local_path = row['local_path']
    im = Image.open(local_path)
    image_heights.append(im.height)
    image_widths.append(im.width)
df_hr_dataset_found['image_height'] = image_heights
df_hr_dataset_found['image_width'] = image_widths


# %%
# unfortunately, the data is not as clean as we'd like. Clean the dataset in this section

# Quantigo did not label toy fish for a few images, drop these images if they do not have any annotation
# The remote path of the images starts with with the following strings
TOY_FISH_PATHS = [
    "s3://aquabyte-datasets-images/aquabyte-frames-resized-inbound/environment=production/site-id=55/pen-id=117/date=2021-09-24/",
    "s3://aquabyte-datasets-images/aquabyte-frames-resized-inbound/environment=production/site-id=204/pen-id=880/date=2022-08-09/",
    "s3://aquabyte-datasets-images/aquabyte-frames-resized-inbound/environment=production/site-id=204/pen-id=880/date=2023-09-21/",
    "s3://aquabyte-datasets-images/aquabyte-frames-resized-inbound/environment=production/site-id=204/pen-id=880/date=2023-09-25/",
    "s3://aquabyte-datasets-images/aquabyte-frames-resized-inbound/environment=production/site-id=204/pen-id=880/date=2023-11-03/",
]

# drop the rows where the image is a toy fish image and the annotation is empty
indecies_to_drop = []
for i, row in df_hr_dataset_found.iterrows():
    for path in TOY_FISH_PATHS:
        if row['images'].startswith(path) and len(row['annotation']) == 0:
            indecies_to_drop.append(i)
            break
print(f"dropping {len(indecies_to_drop)} toy fish images")
df_hr_dataset_found = df_hr_dataset_found.drop(indecies_to_drop).reset_index(drop=True)

# loop through df_hr_dataset_found.annotations and check each annotation has a label, width, height, xCrop, and yCrop, if there is a missing value, drop the dict
clean_count = 0
def clean_annotations(annotations:list):
    global clean_count
    for annotation in annotations:
        if any([annotation.get('label') is None, annotation.get('width') is None, annotation.get('height') is None, annotation.get('xCrop') is None, annotation.get('yCrop') is None]):
            annotations.remove(annotation)
            clean_count += 1
    return annotations

df_hr_dataset_found['annotation'] = df_hr_dataset_found['annotation'].apply(clean_annotations)
print(f"cleaned {clean_count} annotations that were missing label, width, height, xCrop, or yCrop")

# for select datasets with smolt, Quantigo labeled fish as LOW when they should have been labeled as MEDIUM. 
# Change the label from LOW to MEDIUM for any fish with bounding box area greater than 0.2 of the image area
AFFECTED_SOURCE_NAMES = [
    "fish_bbox_fish_bbox_laksefjord_smolt_ras",
    "fish_bbox_smolt_ras_gtsf_thumbnails_2024_01_19",
    "fish_bbox__smolt_ras_gtsf_thumbnails__2024_01_19",
]

edited_count = 0
def low_to_medium(annotations:list, image_area:int):
    global edited_count
    for annotation in annotations:
        bbox_area = annotation['width'] * annotation['height']
        if bbox_area / image_area > 0.15 and annotation['label'] == 'LOW':
            annotation['label'] = 'MEDIUM'
            edited_count += 1
    return annotations
df_hr_dataset_found['annotation'] = df_hr_dataset_found.apply(lambda x: x['annotation'] if x['name'] not in AFFECTED_SOURCE_NAMES else low_to_medium(x['annotation'], x['image_height']*x['image_width']), axis=1)
    
print(f"edited {edited_count} annotations from LOW to MEDIUM")        

#save the dataframe to a csv file
df_hr_dataset_found.to_csv(f"{ROOT_DIR}hrf_with_keypoint_visibility_dataset.csv", index=False)

# %%
# adapted from https://github.com/aquabyte-new/research-exploration/blob/master/pwais/mft-pg/mft_utils/coco_dataset.py
annos_val = []
images_val = []
annotations = []
images = []
val_images = []

for _, row in df_hr_dataset_found.iterrows():
    is_val_image = row.is_val_image
    # coco image index is often 1 indexed https://docs.voxel51.com/api/fiftyone.utils.coco.html#fiftyone.utils.coco.add_coco_labels
    img_idx = len(val_images)+1 if is_val_image else len(images)+1
    for bbox in row.annotation:
        anno_id = len(annos_val)+1 if is_val_image else len(annotations)+1
        category_id = int(category_to_id[bbox['label']])
        bbox_x, bbox_y, bbox_w, bbox_h = bbox['xCrop'], bbox['yCrop'], bbox['width'], bbox['height']
        anno = {
            "id": anno_id,
            "image_id": img_idx,
            "category_id": category_id,
            "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
            # "keypoints": [],
            # "num_keypoints": 0,
            "score": -1,
            "area": bbox_w * bbox_h,
            "iscrowd": 0,
        }
        if is_val_image:
            annos_val.append(anno)
        else:
            annotations.append(anno)

    img_path = row.local_path
    img_fname = os.path.basename(img_path)

    image = {
        "id": img_idx,
        "file_name": img_fname,
        "height": row.image_height,
        "width": row.image_width,
        "source_name": row["name"],
        "remote_path": row["images"],
    }
    if is_val_image:
        val_images.append(image)
    else:
        images.append(image)
            
categories = [
{'id': int(id_), 'name': name}
for name, id_ in category_to_id.items()
]

train_json = {
'categories': categories,
'images': images,
'annotations': annotations,
}

val_json = {
'categories': categories,
'images': val_images,
'annotations': annos_val,
}

with open(f"{ANNOTATION_SAVE_PATH}train.json", "w") as f:
    json.dump(train_json, f)

with open(f"{ANNOTATION_SAVE_PATH}val.json", "w") as f:
    json.dump(val_json, f)


