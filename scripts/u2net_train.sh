# Sample Usage: scripts/u2net.sh [CONFIG] [BATCH_SIZE] [EPOCHS]

export PYTHONPATH="${PYTHONPATH}:/cvps23"

CONFIG= "configs/u2net/u2net-lite_scratch_1xb8-1k_knc-512x512.json"
RESUME = None 
BS = 2
EP = 1 

python tools/train.py \
    --config $CONFIG \
    --resume $RESUME \
    --batch_size $BS \
    --ep $EP