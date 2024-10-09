export YOLOX_DATADIR=/datasets
export XLA_IR_DEBUG=0
export XLA_HLO_DEBUG=0
export PT_XLA_DEBUG=1
export PT_XLA_DEBUG_FILE=./pt_xla_debug.txt
torchrun --standalone --nproc_per_node=8 tools/train.py -b 64 -n yolox-s
