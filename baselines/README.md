# Baselines

The paper's model-based baseline is **FoundationPose** (Wen et al., CVPR 2024), run with its official
pretrained weights (refiner `2023-10-28-18-33-37`, scorer `2024-01-11-20-02-45`) from the clone in
`FoundationPose/` (commit `a1b694b`). Third-party code and environments are git-ignored and rebuilt
from the scripts here.

| Path | What it is |
|---|---|
| `fp_setup/setup_fp.sh` | creates `baselines/fp-env` (micromamba, CUDA 12.8, torch 2.7.1) and fetches the weights |
| `fp_setup/build_fp_ext.sbatch` | compiles nvdiffrast, pytorch3d and `FoundationPose/mycpp` on a GPU node (the head node's gcc cannot build CUDA host code) |
| `fp-env/`, `mamba-root/` | the resulting environment (ignored) |

FoundationPose poses for every LM-Grasp scene: `scripts/sr/fp_lm_poses.py` through
`scripts/sr/fp_run.sbatch`, merged into `results/sr/runs/fpose/scores.pt`.
