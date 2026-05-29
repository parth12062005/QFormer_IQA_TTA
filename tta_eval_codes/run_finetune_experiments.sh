#!/bin/bash
# ============================================================================
# Run all fine-tune + evaluate experiments for Q-Former baseline
#
# Protocol: Pre-trained on EvalMI -> Fine-tune on X% of target -> Eval on test
#
# Grid:
#   2 checkpoints  x  2 datasets  x  3 fractions  =  12 experiments
#
# Each experiment is skipped if its summary.csv already exists (resumable).
#
# Usage:
#   bash run_finetune_experiments.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/anaconda3/envs/tta/bin/python"

# ── Checkpoints ──
CKPT1="${PROJECT_DIR}/checkpoints/evalmi_baseline_qf.pth"      # 1-layer linear regressor
CKPT2="${PROJECT_DIR}/checkpoints/evalmi_baseline_qf_2.pth"    # 2-layer MLP regressor

# ── Datasets ──
DATASETS=("a20k" "a3k")

# ── Fractions ──
FRACTIONS=("0.0" "0.05" "0.10" "0.20")

# ── Training params ──
EPOCHS=15
BATCH_SIZE=16
EVAL_BATCH_SIZE=256
LR="1e-4"
SEED=1234

# ── Results collection file ──
RESULTS_FILE="${SCRIPT_DIR}/results/finetune_all_results.csv"
mkdir -p "$(dirname "$RESULTS_FILE")"

# Write header if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "dataset,checkpoint,fraction_pct,test_srcc,test_plcc,best_epoch" > "$RESULTS_FILE"
fi

echo "========================================================================"
echo "  FINE-TUNE EXPERIMENT SWEEP"
echo "  Checkpoints: 2"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Fractions:   ${FRACTIONS[*]}"
echo "  Total runs:  $((2 * ${#DATASETS[@]} * ${#FRACTIONS[@]}))"
echo "========================================================================"

run_experiment() {
    local ckpt="$1"
    local dataset="$2"
    local fraction="$3"
    local ckpt_basename
    ckpt_basename=$(basename "$ckpt" .pth)
    local frac_pct
    frac_pct=$(echo "$fraction * 100" | bc | cut -d. -f1)
    
    local out_dir="${SCRIPT_DIR}/results/finetune_${dataset}_${ckpt_basename}_${frac_pct}pct"
    local summary="${out_dir}/summary.csv"
    local log_file="${out_dir}/training.log"

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Dataset: ${dataset}  |  Checkpoint: ${ckpt_basename}"
    echo "  Fraction: ${frac_pct}%  |  Output: ${out_dir}"
    echo "────────────────────────────────────────────────────────────────────"

    # Skip if already completed
    if [ -f "$summary" ]; then
        echo "  ⏭  SKIPPING — summary.csv already exists"
        # Extract results for the collection
        local srcc plcc epoch
        srcc=$(tail -1 "$summary" | cut -d',' -f13)
        plcc=$(tail -1 "$summary" | cut -d',' -f14  2>/dev/null || echo "N/A")
        echo "  Previous result: SRCC=${srcc} PLCC=${plcc}"
        return 0
    fi

    mkdir -p "$out_dir"

    # Run the experiment
    $PYTHON "${SCRIPT_DIR}/finetune_and_eval.py" \
        --dataset "$dataset" \
        --fraction "$fraction" \
        --checkpoint "$ckpt" \
        --output_dir "$out_dir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$log_file"

    # Extract result from log
    local result_line
    result_line=$(grep "^\[RESULT\]" "$log_file" || true)
    if [ -n "$result_line" ]; then
        local srcc plcc
        srcc=$(echo "$result_line" | grep -oP 'SRCC=\K[0-9.-]+')
        plcc=$(echo "$result_line" | grep -oP 'PLCC=\K[0-9.-]+')
        local epoch
        epoch=$(tail -1 "$summary" | cut -d',' -f10 2>/dev/null || echo "?")
        echo "${dataset},${ckpt_basename},${frac_pct},${srcc},${plcc},${epoch}" >> "$RESULTS_FILE"
        echo "  ✓ Completed: SRCC=${srcc}  PLCC=${plcc}"
    else
        echo "  ✗ WARNING: No [RESULT] line found in log!"
    fi
}

# ── Run all experiments ──
for ckpt in "$CKPT1" "$CKPT2"; do
    for dataset in "${DATASETS[@]}"; do
        for fraction in "${FRACTIONS[@]}"; do
            run_experiment "$ckpt" "$dataset" "$fraction"
        done
    done
done

# ── Print final summary table ──
echo ""
echo "========================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Collected results:"
if [ -f "$RESULTS_FILE" ]; then
    column -t -s',' "$RESULTS_FILE"
fi
echo ""
echo "Full results in: $RESULTS_FILE"
echo "Per-experiment details in: ${SCRIPT_DIR}/results/finetune_*/"
