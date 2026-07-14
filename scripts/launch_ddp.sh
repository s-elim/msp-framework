#!/usr/bin/env bash
# Multi-GPU training. The trainer wraps encoder+head in DistributedDataParallel and
# all-reduces the validation metrics; without this launcher it silently runs on one GPU.
#
#   ./scripts/launch_ddp.sh                       # all visible GPUs
#   ./scripts/launch_ddp.sh 2 train.epochs=100    # 2 GPUs, with Hydra overrides
set -euo pipefail

NGPU="${1:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
shift || true

echo "launching on ${NGPU} GPU(s)"
torchrun --standalone --nproc_per_node="${NGPU}" scripts/train.py "$@"
