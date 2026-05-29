#!/bin/bash

# Master script to run all combinations of TTA experiments in parallel across GPUs
# Supports resumability: skips already completed experiments.

LOG_DIR="logs_tta_sweep_master"
mkdir -p "$LOG_DIR"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected via nvidia-smi. Defaulting to 1."
    NUM_GPUS=1
fi

echo "=========================================================="
echo " Starting Full TTA Sweep (Resumable)"
echo " Detected GPUs: $NUM_GPUS"
echo " Logging to directory: $LOG_DIR"
echo "=========================================================="

DATASETS=("qval")
LOSSES=("gc" "rank" "gc rank" "fagc" "adaptive_rank" "fagc adaptive_rank")
UNFREEZES=("query" "layernorm" "both")
PROJ_MODES=("--freeze_proj_head" "--update_proj_head")

# Stabilization strategy arrays
PROJ_INITS=("random" "identity")
WARMUP_STEPS=(0 3)
EMA_DECAYS=(0.0 0.99)

# Generate all commands
COMMANDS=()
LOG_FILES=()

# 1. Baselines
for ds in "${DATASETS[@]}"; do
    COMMANDS+=("python evaluate_tta.py --dataset $ds --unfreeze none")
    LOG_FILES+=("$LOG_DIR/${ds}_baseline.log")
done

# 2. TTA runs
for ds in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        for uf in "${UNFREEZES[@]}"; do
            for pm in "${PROJ_MODES[@]}"; do
                for init in "${PROJ_INITS[@]}"; do
                    for warmup in "${WARMUP_STEPS[@]}"; do
                        for ema in "${EMA_DECAYS[@]}"; do
                            
                            # Base loss name
                            loss_name=$(echo "$loss" | tr ' ' '+')
                            # If EMA is active, we append ema_consistency loss
                            active_loss="$loss"
                            if [ $(echo "$ema > 0.0" | bc -l) -eq 1 ]; then
                                active_loss="$loss ema_consistency"
                                loss_name="${loss_name}+ema"
                            fi

                            pm_name=$(echo "$pm" | sed 's/--//')
                            
                            # Construct command
                            cmd="python evaluate_tta.py --dataset $ds --losses $active_loss --unfreeze $uf $pm --proj_init $init --warmup_steps $warmup --ema_decay $ema"
                            
                            # Construct log file name
                            log_file="$LOG_DIR/${ds}_${loss_name}_${uf}_${pm_name}_init-${init}_warmup-${warmup}_ema-${ema}.log"
                            
                            COMMANDS+=("$cmd")
                            LOG_FILES+=("$log_file")
                        done
                    done
                done
            done
        done
    done
done

TOTAL_CMDS=${#COMMANDS[@]}
echo "Total experiments to check/run: $TOTAL_CMDS"
echo "----------------------------------------------------------"

declare -A pids
gpu_queue=($(seq 0 $((NUM_GPUS - 1))))

for (( i=0; i<$TOTAL_CMDS; i++ )); do
    cmd="${COMMANDS[$i]}"
    log_file="${LOG_FILES[$i]}"
    
    # --- RESUMABILITY CHECK ---
    base_log=$(basename "$log_file")
    skip=false
    
    # 1. Check in current master log dir
    if [ -f "$log_file" ] && grep -q "\[Saved\] Per-image CSVs" "$log_file"; then
        skip=true
    fi
    
    # 2. Check in older timestamped log dirs
    if [ "$skip" = false ]; then
        for old_dir in logs_tta_sweep_*; do
            if [ -d "$old_dir" ] && [ "$old_dir" != "$LOG_DIR" ]; then
                old_log="$old_dir/$base_log"
                if [ -f "$old_log" ] && grep -q "\[Saved\] Per-image CSVs" "$old_log"; then
                    # Copy the successful log to the master directory so everything is in one place
                    cp "$old_log" "$log_file"
                    skip=true
                    break
                fi
            fi
        done
    fi
    
    if [ "$skip" = true ]; then
        echo "[*] Skipping [$((i+1))/$TOTAL_CMDS] - Already completed: $base_log"
        continue
    fi
    
    # Wait until a GPU is available
    while [ ${#gpu_queue[@]} -eq 0 ]; do
        for pid in "${!pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, free its GPU
                gpu_id=${pids[$pid]}
                gpu_queue+=($gpu_id)
                unset pids[$pid]
                echo "[-] Job finished. GPU $gpu_id is now free."
            fi
        done
        sleep 2
    done

    # Pop a GPU from queue
    gpu_id=${gpu_queue[0]}
    gpu_queue=("${gpu_queue[@]:1}")

    echo "[+] Starting [$((i+1))/$TOTAL_CMDS] on GPU $gpu_id: $cmd"
    echo "    Log: $log_file"
    
    # Run in background
    CUDA_VISIBLE_DEVICES=$gpu_id eval "$cmd > \"$log_file\" 2>&1" &
    job_pid=$!
    pids[$job_pid]=$gpu_id
done

echo "----------------------------------------------------------"
echo "All jobs dispatched. Waiting for the remaining background tasks to finish..."
wait

echo "=========================================================="
echo " All experiments finished!"
echo " Logs are saved in $LOG_DIR"
