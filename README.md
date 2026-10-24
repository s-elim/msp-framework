# Pose-posterior grasp planning on LM-Grasp

Code and experiment record for the Scientific Reports manuscript in `scientific_reports_paper/`.
A model-based pose estimator (FoundationPose) returns one pose and no uncertainty; this repository
plans grasps under a pose posterior built from the estimator's own residuals, and evaluates every
method on **LM-Grasp**: real LINEMOD RGB-D test images paired with simulated grasp outcomes.

## Layout

| Path | Contents |
|---|---|
| `src/msp/data/linemod.py` | LM-Grasp scenes: crops, observation-only grasp proposals, block splits |
| `src/msp/oracle/linemod_sim.py` | MuJoCo parallel-jaw rollouts on the CoACD-decomposed LINEMOD meshes |
| `src/msp/pose/` | SE(3) helpers, observable pose-fit features, residual-bootstrap pose posterior |
| `src/msp/grasp/` | candidate scoring under pose samples, rejection reasons, camera-base-flange frames and checks |
| `src/msp/models/` | RGB-D outcome scorer (ResNet-18 + FiLM head), LM baselines, point-cloud branch |
| `src/msp/math/`, `belief/`, `engine/` | rate-regularised bottleneck (ablation), conformal certificates, trainer |
| `scripts/sr/` | corpus build, training, FoundationPose poses, planners, evaluation, figures support |
| `scientific_reports_paper/` | manuscript, figure and table scripts, experiment log (`TODO_2026_10_24.md`) |
| `baselines/` | FoundationPose environment setup |
| `tests/` | pytest suite (geometry, posterior, planner equivalence, frames, models, metrics) |

`data/` (LINEMOD, LM-Grasp corpus) and `results/` are git-ignored.

## Environment

```bash
source scripts/sr/env.sh        # sets $PY (repo .venv), PYTHONPATH, MuJoCo GL backend
```

GPU work runs through Slurm on the lab cluster; every job passes `--mem` (see the sbatch headers in
`scripts/sr/`). FoundationPose runs in its own environment (`baselines/README.md`).

## Reproduce

The order of the studies, the pre-declared selection rules, every command and every val/test
number are recorded in `scientific_reports_paper/TODO_2026_10_24.md`. Main entry points:

```bash
$PY scripts/download_lm.py                                   # BOP LINEMOD
$PY scripts/sr/build_lm_grasp.py --out data/lm_grasp         # LM-Grasp corpus
$PY scripts/sr/plan_with_pose.py --pose gt --split test      # annotated-pose planner
$PY scripts/sr/pose_features.py && $PY scripts/sr/pose_posterior.py --mode knn --splits test
$PY scripts/sr/plan_with_pose.py --pose results/sr/runs/fpose/scores.pt \
    --pose-samples results/sr/posterior/knn_s0_test.npz --split test --tag fposeU1_s0_p4
$PY scripts/sr/evaluate_lm.py --out results/sr/metrics_session4.json
sbatch --gres=gpu:1 -c 8 --mem=24G --time=01:00:00 scripts/sr/pytest.sbatch
```
