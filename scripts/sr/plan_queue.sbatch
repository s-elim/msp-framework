#!/bin/bash
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "host $(hostname) start $(date -Is)"
while read -r line; do [ -z "$line" ] && continue; echo "=== $(date -Is) $line"; $PY scripts/sr/plan_with_pose.py $line || echo "FAILED: $line"; done < "$1"
echo "end $(date -Is)"
