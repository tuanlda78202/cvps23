# Sample Usage: scripts/u2net.sh [CONFIG] [BATCH_SIZE] [EPOCHS]

#export PYTHONPATH="${PYTHONPATH}:/CVP"

PYTHONPATH=. python tools/train.py \
    --config ${1:-"configs/u2net/u2net-lite_scratch_1xb8-1k_knc-512x512.json"} \
    --resume ${2:-"None"}
    --bs ${2:-1} \
    --ep ${3:-20}