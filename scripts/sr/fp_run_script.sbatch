#!/bin/bash
# Run one script in the FoundationPose env on one GPU.
#   sbatch --job-name=fp_lmo --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 scripts/sr/fp_run_script.sbatch scripts/sr/fp_lmo.py
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
R=/nas/Selim/selim_sarowar/msp_framework
ENV=$R/baselines/fp-env
export HOME=/nas/Selim/selim_sarowar PATH=$ENV/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $R
nvidia-smi --query-gpu=name --format=csv,noheader
echo "start $(date -Is) $*"
$ENV/bin/python "$@" || echo "FAILED: $*"
echo "end $(date -Is)"
