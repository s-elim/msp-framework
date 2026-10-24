#!/bin/bash
# After FoundationPose train/val poses: merge all splits, then plan train and val (CPU only).
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "start $(date -Is)"
$PY scripts/sr/fp_lm_poses.py --merge || echo "FAILED: merge"
while read -r line; do [ -z "$line" ] && continue; echo "=== $line"; $PY scripts/sr/plan_with_pose.py $line || echo "FAILED: $line"; done < results/sr/queues/fp_plan_trainval.txt
echo "end $(date -Is)"
