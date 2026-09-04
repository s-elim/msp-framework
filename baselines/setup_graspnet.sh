#!/usr/bin/env bash
# Build GraspNet-baseline in its OWN venv.
#
# The cluster has no CUDA toolkit (no nvcc, CUDA_HOME unset) and the GPUs are Blackwell
# (sm_120), so the pointnet2 / knn extensions cannot be built against the system stack.
# This venv is self-consistent instead: its own torch built for CUDA 12.8, a userspace
# nvcc 12.9 wheel, and TORCH_CUDA_ARCH_LIST pinned to sm_120. Nothing here touches the
# msp environment.
set -uo pipefail
ROOT=/nas/Selim/selim_sarowar/msp_framework/baselines
VENV=$ROOT/gn-venv
LOG=$ROOT/setup_graspnet.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) starting"

# The system python has no ensurepip (python3.12-venv is not installed and we have no
# root), so the venv comes from uv, which bootstraps pip itself.
# The system CPython ships no development headers (no Python.h, no root to install
# python3.12-dev), so the extensions cannot compile against it. uv's managed CPython is
# a full build that includes the headers.
uv python install 3.12
uv venv --python cpython-3.12 --seed "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip wheel setuptools ninja

echo "=== torch (cu128) + userspace nvcc"
"$PY" -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q nvidia-cuda-nvcc-cu12==12.9.86 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
"$PY" -m pip install -q numpy scipy pillow open3d graspnetAPI || echo "WARN: graspnetAPI/open3d install issue"

NVCC_DIR=$("$PY" -c "import nvidia.cuda_nvcc, pathlib; print(pathlib.Path(nvidia.cuda_nvcc.__file__).parent)")
export CUDA_HOME="$NVCC_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -2

echo "=== clone graspnet-baseline"
[ -d "$ROOT/graspnet-baseline/.git" ] || git clone -q https://github.com/graspnet/graspnet-baseline.git "$ROOT/graspnet-baseline"

export TORCH_CUDA_ARCH_LIST="12.0"
for ext in pointnet2 knn; do
  echo "=== building $ext"
  (cd "$ROOT/graspnet-baseline/$ext" && "$PY" setup.py install) || echo "FAILED: $ext"
done

echo "=== import check"
"$PY" - <<'PYCODE'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import pointnet2_cuda; print("pointnet2_cuda OK")
except Exception as e:
    print("pointnet2_cuda FAILED:", type(e).__name__, e)
PYCODE
echo "=== $(date -Is) done"
