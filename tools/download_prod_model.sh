#!/bin/bash

set -euo pipefail

# Get a file from S3 with some caching
#   Usage: get_from_s3 <s3-path> <local-path>
function get_from_s3 {
  if [ -f "$2" ] ; then
    echo "File $2 is already downloaded. Skip download."
  else
    printf "Fetching $1... "
    aws s3 cp "$1" "$2"
    echo "Done"
  fi
}


# yolox not in registry format, specify file downloads
s3_yolox_base="s3://aquabyte-research/nuske/models/yolox_s_416__translate_model_to_pytorch1.4"
local_yolox_base="object-detection-yolox_pytorch1.4"
yolo_files=(
	yolox.NVIDIA.Tegra.X2.fp16.pth
	mft_pg_yolox_exp.py
	YOLOX_outputs/mft_pg_yolox_exp/best_ckpt.pth
)
for yolo_file in "${yolo_files[@]}"
do
	get_from_s3 $s3_yolox_base/$yolo_file $local_yolox_base/$yolo_file
done
