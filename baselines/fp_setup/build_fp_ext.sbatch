#!/bin/bash
# Second FoundationPose build attempt (session 2). The first attempt (setup_fp.sh, head node)
# failed at nvdiffrast (exit 14) for two reasons seen in results/sr_logs/fp_setup.log:
#   1. the head node's /usr/bin/gcc has no cc1plus, so nvcc could not preprocess host code;
#   2. conda's cuda-toolkit keeps cuda_runtime.h under targets/x86_64-linux/include.
# Fix: compile on a GPU node with the env's own conda gcc/g++ 13.4 (nvcc 12.8 supports it)
# against torch 2.7.1+cu128 in the same env, with the CUDA include/lib paths exported.
# Sources are pre-cloned to baselines/fp_setup/src (compute nodes may lack internet).
#   sbatch --job-name=fp_build --gres=gpu:1 -c 16 --mem=32G --time=02:00:00 baselines/fp_setup/build_fp_ext.sbatch
#SBATCH -p gpu
#SBATCH -o /nas/Selim/selim_sarowar/msp_framework/results/sr_logs/%x.%j.log
set -x
R=/nas/Selim/selim_sarowar/msp_framework
ENV=$R/baselines/fp-env
SRC=$R/baselines/fp_setup/src
export HOME=/nas/Selim/selim_sarowar TMPDIR=$R/baselines/fp_setup/tmp
mkdir -p $TMPDIR
export CUDA_HOME=$ENV PATH=$ENV/bin:/usr/bin:/bin
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export CPATH=$ENV/targets/x86_64-linux/include:$ENV/include
export LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/targets/x86_64-linux/lib/stubs:$ENV/lib
export LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib
export TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 MAX_JOBS=16
PY=$ENV/bin/python
nvidia-smi --query-gpu=name,driver_version --format=csv
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version | tail -2
date
$PY -m pip install --no-build-isolation --no-deps $SRC/nvdiffrast || exit 14
date
$PY -c "import nvdiffrast.torch as dr, torch; g=dr.RasterizeCudaContext(); print('nvdiffrast OK')" || exit 14
$PY -m pip install --no-build-isolation --no-deps $SRC/pytorch3d || exit 15
date
$PY -c "import pytorch3d, pytorch3d.transforms; print('pytorch3d', pytorch3d.__version__)" || exit 15
cd $R/FoundationPose/mycpp && rm -rf build && mkdir -p build && cd build && \
  cmake .. -DPYTHON_EXECUTABLE=$PY -DCMAKE_PREFIX_PATH=$ENV -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX && make -j16 || exit 16
date
echo BUILD_OK
