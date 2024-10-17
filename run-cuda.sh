export OMP_NUM_THREADS=16 
export LOGURU_LEVEL="INFO"
export YOLOX_DATADIR=/home/ubuntu/efs/datasets
torchrun --standalone --nproc_per_node=8 tools/train.py -b 64 -n yolox-s
