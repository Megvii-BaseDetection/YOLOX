export OMP_NUM_THREADS=16 
export LOGURU_LEVEL="INFO"
export YOLOX_DATADIR=/datasets
export YOLOX_OUPUT_DIR="./YOLOX_cuda_outputs"
torchrun --standalone --nproc_per_node=8 tools/train.py -b 32 -n yolox-s
