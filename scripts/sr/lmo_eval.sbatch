#!/bin/bash
# LM-O transfer check: all instances on one node, then merge.
#   sbatch --dependency=afterok:<fp job> --job-name=lmo_eval -w gpu2 -c 36 --mem=32G --time=02:00:00 scripts/sr/lmo_eval.sbatch
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "host $(hostname) start $(date -Is)"
"$PY" scripts/sr/lmo_eval.py --shard 0 --n-shards 1 --workers 34 || echo "FAILED: eval"
"$PY" scripts/sr/lmo_eval.py --merge || echo "FAILED: merge"
echo "end $(date -Is)"
