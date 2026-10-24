#!/bin/bash
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "start $(date -Is)"
while read -r line; do [ -z "$line" ] && continue; echo "=== $line"; $PY scripts/sr/plan_with_pose.py $line || echo "FAILED: $line"; done < results/sr/queues/plan_pose.txt
$PY scripts/sr/evaluate_lm.py --out results/sr/metrics.json || echo "FAILED: evaluate"
$PY scripts/sr/rate_eval.py --runs "msp_b*_d32_s*" --out results/sr/rate.json || echo "FAILED: rate"
echo "end $(date -Is)"
