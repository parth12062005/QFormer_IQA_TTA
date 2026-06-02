#!/bin/bash
set -e

LOG_FILE="queue_finetune.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "Starting queue at $(date)"
echo "Waiting for experiment1_enhanced_tta.py to finish..."

while pgrep -f "experiment1_enhanced_tta.py" > /dev/null; do
    sleep 30
done

echo "Experiment 1 has finished! $(date)"
echo "Starting finetuning with layernorm unfreezed..."

PYTHON_BIN="/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/anaconda3/envs/tta/bin/python"

$PYTHON_BIN finetune_and_eval.py \
    --dataset a20k \
    --fraction 0.20 \
    --unfreeze_strategy layernorm \
    --output_dir results/finetune_layernorm_only \
    --epochs 15 \
    --lr 1e-4 \
    --batch_size 16

echo "=================================================="
echo "Finetuning finished at $(date)"
echo "=================================================="
