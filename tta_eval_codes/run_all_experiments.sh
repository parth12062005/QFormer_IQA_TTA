#!/bin/bash

# Master script to run all combinations of TTA experiments
# 3 Datasets × 6 Loss Configs × 3 Unfreeze Strategies × 2 ProjHead modes

LOG_FILE="tta_sweep_$(date +%Y%m%d_%H%M%S).log"

DATASETS=("a3k" "a20k" "evalmi")
# Loss combinations: GC, Rank, GC+Rank, FAGC, ARL, FAGC+ARL
LOSSES=("gc" "rank" "gc rank" "fagc" "adaptive_rank" "fagc adaptive_rank")
UNFREEZES=("query" "layernorm" "both")
PROJ_MODES=("--freeze_proj_head" "--update_proj_head")

echo "==========================================================" | tee -a "$LOG_FILE"
echo " Starting Full TTA Sweep " | tee -a "$LOG_FILE"
echo " Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# 1. Run Baselines (no TTA)
echo "" | tee -a "$LOG_FILE"
echo "--- RUNNING BASELINES ---" | tee -a "$LOG_FILE"
for ds in "${DATASETS[@]}"; do
    echo "Running baseline for $ds..." | tee -a "$LOG_FILE"
    python evaluate_tta.py --dataset "$ds" --unfreeze none >> "$LOG_FILE" 2>&1
done

# 2. Run TTA Combinations
echo "" | tee -a "$LOG_FILE"
echo "--- RUNNING TTA COMBINATIONS ---" | tee -a "$LOG_FILE"

total_runs=$(( ${#DATASETS[@]} * ${#LOSSES[@]} * ${#UNFREEZES[@]} * ${#PROJ_MODES[@]} ))
current_run=0

for ds in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        for uf in "${UNFREEZES[@]}"; do
            for pm in "${PROJ_MODES[@]}"; do
                ((current_run++))
                echo "[$current_run/$total_runs] Dataset: $ds | Loss: [$loss] | Unfreeze: $uf | ProjHead: $pm" | tee -a "$LOG_FILE"
                
                # We use eval to handle the multi-word loss string correctly
                eval "python evaluate_tta.py --dataset $ds --losses $loss --unfreeze $uf $pm" >> "$LOG_FILE" 2>&1
            done
        done
    done
done

echo "==========================================================" | tee -a "$LOG_FILE"
echo " All experiments finished!" | tee -a "$LOG_FILE"
echo " Check $LOG_FILE for full output." | tee -a "$LOG_FILE"
