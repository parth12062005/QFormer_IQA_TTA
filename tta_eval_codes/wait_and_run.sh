#!/bin/bash
echo "Waiting for experiment1_enhanced_tta.py to finish..."
while pgrep -f experiment1_enhanced_tta.py > /dev/null; do
    sleep 30
done
echo "Experiment 1 finished! Starting strict layer ablation..."
nohup bash run_layer_ablation_strict.sh > run_strict_ablation_master.log 2>&1 &
