#!/bin/bash
#SBATCH --job-name=gn_build
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=baselines/build_graspnet.%j.log
# The CUDA toolkit lives on the COMPUTE nodes (/usr/local/cuda-12.9), not on the login
# node, so the pointnet2 / knn extensions have to be compiled inside an allocation.
set -uo pipefail
ROOT=/nas/Selim/selim_sarowar/msp_framework/baselines
VENV=$ROOT/gn-venv
PY="$VENV/bin/python"

export HOME=/nas/Selim/selim_sarowar
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
# The compute node's default gcc is 12 but only gcc-13 ships cc1plus there, so nvcc's
# host compiler must be pinned or every .cu fails with "cannot execute cc1plus".
export CC=gcc-13 CXX=g++-13
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"

echo "host $(hostname)"; nvcc --version | tail -2
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

for ext in pointnet2 knn; do
  echo "=== building $ext"
  (cd "$ROOT/graspnet-baseline/$ext" && rm -rf build && "$PY" setup.py install 2>&1 | tail -5) \
    || echo "FAILED: $ext"
done

echo "=== import check"
"$PY" - <<'PYCODE'
for m in ("pointnet2_cuda", "knn_pytorch"):
    try:
        __import__(m); print(m, "OK")
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, e)
PYCODE
