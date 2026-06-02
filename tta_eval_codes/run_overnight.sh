#!/bin/bash
set -x

PYTHON_BIN="/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/anaconda3/envs/tta/bin/python"
LOG="overnight_experiments.log"

echo "========================================" | tee "$LOG"
echo "Starting overnight experiments at $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── Experiment 2: CLIP-Cluster Contrastive TTA ──
echo "" | tee -a "$LOG"
echo ">>> EXPERIMENT 2: CLIP-Cluster Contrastive TTA" | tee -a "$LOG"
echo ">>> Started at $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 $PYTHON_BIN experiment2_clip_cluster_tta.py 2>&1 | tee -a "$LOG"
echo ">>> Experiment 2 finished at $(date)" | tee -a "$LOG"

# ── Wait for paraphrases to complete ──
echo "" | tee -a "$LOG"
echo ">>> Waiting for paraphrase generation to complete..." | tee -a "$LOG"
while kill -0 $(cat paraphrase.pid 2>/dev/null) 2>/dev/null; do
    DONE=$(wc -l < paraphrased_prompts.json 2>/dev/null || echo 0)
    echo "  Paraphrases in progress... ($(date))" | tee -a "$LOG"
    sleep 30
done
echo ">>> Paraphrase generation complete at $(date)" | tee -a "$LOG"

# ── Experiment 1: Enhanced TTA ──
echo "" | tee -a "$LOG"
echo ">>> EXPERIMENT 1: Enhanced TTA (GC+Rank+Consistency+Prompt)" | tee -a "$LOG"
echo ">>> Started at $(date)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 $PYTHON_BIN experiment1_enhanced_tta.py 2>&1 | tee -a "$LOG"
echo ">>> Experiment 1 finished at $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "All overnight experiments complete at $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
