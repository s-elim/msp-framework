#!/bin/bash
# Run one train_lm.py argument line per queue-file line, sequentially, on one GPU.
#   sbatch --job-name=q_main --gres=gpu:1 -c 8 --mem=48G --time=08:00:00 scripts/sr/run_queue.sbatch results/sr/queues/main.txt
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework
export HOME=/nas/Selim/selim_sarowar
source scripts/sr/env.sh
echo "host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none} start $(date -Is)"
while read -r line; do
  [ -z "$line" ] && continue
  echo "=== $(date -Is) $line"
  $PY scripts/sr/train_lm.py $line --threads 8 || echo "FAILED: $line"
done < "$1"
echo "end $(date -Is)"
