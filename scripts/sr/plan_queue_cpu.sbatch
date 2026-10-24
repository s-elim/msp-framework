#!/bin/bash
# Run one plan_with_pose.py argument line per queue-file line (CPU only), then the val selection.
#   sbatch --job-name=s4a -c 32 --mem=16G --time=02:00:00 scripts/sr/plan_queue_cpu.sbatch results/sr/queues/study4_val_a.txt
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "host $(hostname) start $(date -Is)"
while read -r line; do [ -z "$line" ] && continue; echo "=== $(date -Is) $line"; $PY scripts/sr/plan_with_pose.py $line || echo "FAILED: $line"; done < "$1"
echo "end $(date -Is)"
