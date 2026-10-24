#!/bin/bash
# Dimension-reduction payload sweep (rate_eval.py --scheme dims) for one run glob (CPU only).
#   sbatch --job-name=rdims_msp -c 16 --mem=16G --time=03:00:00 scripts/sr/rate_dims.sbatch "msp_b*_d*_s*" results/sr/rate_dims_msp.json
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
export OMP_NUM_THREADS=16
echo "start $(date -Is)"
$PY scripts/sr/rate_eval.py --scheme dims --runs "$1" --out "$2" || echo "FAILED: $1"
echo "end $(date -Is)"
