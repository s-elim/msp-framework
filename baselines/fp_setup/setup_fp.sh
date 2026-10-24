#!/bin/bash
# Timeboxed FoundationPose environment build for the LM-Grasp pose-then-plan baseline.
set -x
FP=/nas/Selim/selim_sarowar/msp_framework/FoundationPose
ENV=/nas/Selim/selim_sarowar/msp_framework/baselines/fp-env
MM=/home/selim/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/nas/Selim/selim_sarowar/msp_framework/baselines/mamba-root
date
$MM create -y -p $ENV -c nvidia/label/cuda-12.8.1 -c conda-forge python=3.11 cuda-toolkit boost eigen pybind11 cmake ninja gdown || exit 11
PY=$ENV/bin/python
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 MAX_JOBS=16
$PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 || exit 12
$PY -m pip install numpy scipy scikit-learn h5py joblib PyYAML ruamel.yaml opencv-python-headless imageio open3d trimesh transformations matplotlib pandas Pillow pyrender PyOpenGL kornia omegaconf psutil tqdm warp-lang || exit 13
date
$PY -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git || exit 14
date
$PY -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" || exit 15
date
cd $FP/mycpp && rm -rf build && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$PY -DCMAKE_PREFIX_PATH=$ENV && make -j16 || exit 16
cd $FP && mkdir -p weights && $ENV/bin/gdown --folder "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" -O weights || exit 17
ls -R weights | head
date
echo SETUP_OK
