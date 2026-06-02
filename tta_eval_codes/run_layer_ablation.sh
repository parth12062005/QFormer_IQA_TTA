#!/bin/bash
set -e
set -x

CKPT="../checkpoints/evalmi_baseline_qf.pth"
DATASET="a20k"
FRACTION="0.20"
PYTHON_BIN="/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/anaconda3/envs/tta/bin/python"
LOG_FILE="ablation_sweep.log"

echo "Starting ablation sweep at $(date). Logging to $LOG_FILE" > "$LOG_FILE"

# Clean up previous failed run results just in case
rm -rf results/ablation_${DATASET}_layer_* || true

for LAYER in {0..11}; do
    echo "=================================================" | tee -a "$LOG_FILE"
    echo "Running ablation on Layer $LAYER at $(date)" | tee -a "$LOG_FILE"
    echo "=================================================" | tee -a "$LOG_FILE"
    
    # We use PYTHONUNBUFFERED=1 to ensure crash stack traces are instantly flushed to the log file
    PYTHONUNBUFFERED=1 $PYTHON_BIN finetune_and_eval.py \
        --dataset $DATASET \
        --fraction $FRACTION \
        --unfreeze_layer $LAYER \
        --checkpoint $CKPT \
        --output_dir results/ablation_${DATASET}_layer_${LAYER} 2>&1 | tee -a "$LOG_FILE"
done

echo "=================================================" | tee -a "$LOG_FILE"
echo "Layer ablation sweep complete at $(date)." | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

# Aggregate results
PYTHONUNBUFFERED=1 $PYTHON_BIN aggregate_ablation.py 2>&1 | tee -a "$LOG_FILE"
