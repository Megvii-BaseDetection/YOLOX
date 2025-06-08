export YOLOX_DATADIR=/datasets
export YOLOX_OUPUT_DIR="./YOLOX_neuron_outputs"
export OMP_NUM_THREADS=16
export LOGURU_LEVEL="INFO"
export NEURON_CC_FLAGS="--cache_dir=/cache --model-type=generic"
export NEURON_RT_STOCHASTIC_ROUNDING_EN="1"
export XLA_IR_SHAPE_CACHE_SIZE="20480"
export XLA_IR_DEBUG=0
export XLA_HLO_DEBUG=0
export PT_XLA_DEBUG=0
export PT_XLA_DEBUG_FILE=./pt_xla_debug.txt
torchrun --standalone --nproc_per_node=32 tools/train.py -b 128 -n yolox-s
