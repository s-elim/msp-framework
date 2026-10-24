#!/bin/bash
# After training: pose-then-plan re-simulation, metrics and rate evaluation (CPU only).
#   sbatch --job-name=post_v2 -c 20 --mem=32G --time=03:00:00 scripts/sr/post_train.sbatch results/sr/queues/v2_plan_pose.txt
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "start $(date -Is)"
$PY scripts/sr/fp_lm_poses.py --merge 2>/dev/null || baselines/fp-env/bin/python scripts/sr/fp_lm_poses.py --merge || echo "FAILED: fp merge"
while read -r line; do [ -z "$line" ] && continue; echo "=== $line"; $PY scripts/sr/plan_with_pose.py $line || echo "FAILED: $line"; done < "${1:-results/sr/queues/plan_pose.txt}"
$PY scripts/sr/evaluate_lm.py --out results/sr/metrics.json || echo "FAILED: evaluate"
$PY scripts/sr/rate_eval.py --runs "msp_b*_d*_s*" --out results/sr/rate.json || echo "FAILED: rate msp"
$PY scripts/sr/rate_eval.py --runs "direct_s[0-4]" --out results/sr/rate_direct.json || echo "FAILED: rate direct"
echo "end $(date -Is)"
