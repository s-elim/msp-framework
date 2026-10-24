#!/bin/bash
# FoundationPose poses for LM-Grasp scenes, one GPU, FoundationPose env.
#   sbatch --job-name=fp_test0 --gres=gpu:1 -c 8 --mem=32G --time=04:00:00 scripts/sr/fp_run.sbatch --split test --shard 0 --n-shards 2
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
$ENV/bin/python scripts/sr/fp_lm_poses.py "$@" || echo "FAILED: $*"
echo "end $(date -Is)"
