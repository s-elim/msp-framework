#!/bin/bash
# Generic LM-Grasp Slurm wrapper. --mem is REQUIRED on this cluster: omitting it requests the
# whole node and the job pends forever with reason InvalidAccount.
#   sbatch --job-name=NAME -c 8 --mem=32G [--gres=gpu:1] scripts/sr/slurm_job.sbatch scripts/sr/X.py ARGS
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
set -uo pipefail
cd /nas/Selim/selim_sarowar/msp_framework
export HOME=/nas/Selim/selim_sarowar
source scripts/sr/env.sh
echo "host $(hostname)  start $(date -Is)  gpu ${CUDA_VISIBLE_DEVICES:-none}"
"$PY" "$@"
echo "exit $?  end $(date -Is)"
