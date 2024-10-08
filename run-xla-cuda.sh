export OMP_NUM_THREADS=16 
export YOLOX_DATADIR=/datasets
torchrun --standalone --nproc_per_node=8 /app/tools/train.py -b 64 -n yolox-s 1>run-xla-cuda.out 2>&1 &
