# Source before running LM-Grasp scripts. Head node (no NVIDIA driver): OSMesa. GPU nodes: EGL.
REPO=/nas/Selim/selim_sarowar/msp_framework
export PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO/src:$REPO
export TORCH_HOME=/nas/Selim/selim_sarowar/.cache/torch
export MPLCONFIGDIR=/tmp/mpl_cache_$USER
if nvidia-smi >/dev/null 2>&1; then
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
else
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export LD_LIBRARY_PATH=/nas/Selim/selim_sarowar/AFFORD-X/outputs/runtime/mesa24-osmesa/lib:${LD_LIBRARY_PATH:-}
fi
