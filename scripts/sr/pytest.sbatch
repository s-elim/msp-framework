#!/bin/bash
# Full test suite on a compute node.
#   sbatch --job-name=sr_pytest --gres=gpu:1 -c 8 --mem=24G --time=01:00:00 scripts/sr/pytest.sbatch
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
cd /nas/Selim/selim_sarowar/msp_framework; export HOME=/nas/Selim/selim_sarowar; source scripts/sr/env.sh
echo "host $(hostname) start $(date -Is)"
"$PY" -m pytest -q -p no:cacheprovider tests "$@"
echo "exit $? end $(date -Is)"
