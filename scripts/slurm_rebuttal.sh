#!/bin/bash
#SBATCH --job-name=msp_rebut
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=results/rebuttal/run.%j.log
set -uo pipefail
REPO=/nas/Selim/selim_sarowar/msp_framework
cd "$REPO"
export HOME=/nas/Selim/selim_sarowar
export MUJOCO_GL=egl
export PYTHONPATH="$REPO/src:$REPO"
export MPLCONFIGDIR=/tmp/mpl_cache
# Compute nodes have no outbound network, so torchvision cannot fetch the ImageNet
# weights; they are staged on the shared filesystem instead.
export TORCH_HOME=/nas/Selim/selim_sarowar/.cache/torch
mkdir -p results/rebuttal
"$REPO/../AGI/.venv/bin/python" scripts/paper/run_rebuttal.py "$@"
