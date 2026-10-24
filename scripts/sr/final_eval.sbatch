#!/bin/bash
# The single test pass of session 4: every LM-Grasp metric, then the study-5 trigger on test.
#   sbatch --job-name=final_eval -c 4 --mem=32G --time=01:00:00 scripts/sr/final_eval.sbatch
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "start $(date -Is)"
"$PY" scripts/sr/evaluate_lm.py --out results/sr/metrics_session4.json || echo "FAILED: evaluate"
"$PY" scripts/sr/second_view.py --evaluate --split test || echo "FAILED: second view"
echo "end $(date -Is)"
