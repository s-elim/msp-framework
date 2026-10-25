# Pose-posterior grasp planning

Code and experiment record for **Planning on a residual-based pose posterior closes the
grasp-success gap between estimated and annotated object poses** (Md Selim Sarowar and Sungho Kim,
Advanced Visual Intelligence Lab, Yeungnam University). The manuscript is in preparation for
Scientific Reports; its sources are in `scientific_reports_paper/`.

Project page: <https://physical-agi.github.io/MSP-FRAMEWORK/>

A pose estimator such as FoundationPose returns one pose and no uncertainty, so a planner treats
the pose as exact. This repository builds a pose posterior from the estimator's own errors on
annotated training frames (a nearest-neighbour residual bootstrap on observable fit features) and
plans every candidate grasp under its samples and the physics prior. Methods are evaluated on
**LM-Grasp**: 15,800 real LINEMOD RGB-D images, 16 candidate grasps each, with outcomes simulated in
MuJoCo at the annotated pose, and on occluded LM-O scenes never used for tuning.

| Top-1 grasp success | Point estimate | Posterior (ours) | Annotated pose |
|---|---|---|---|
| LM-Grasp test (2,579 scenes) | 0.791 | 0.859 | 0.856 |
| LM-O (1,474 objects) | 0.683 | 0.750 | 0.784 |
| Precision of certified grasps, LM-Grasp, α = 0.1 | 0.904 | 0.972 | 0.974 |

Every number in the manuscript comes from `scientific_reports_paper/numbers.tex`, which
`make_tables.py` writes from the results files.

## Layout

| Path | Contents |
|---|---|
| `src/msp/data/linemod.py` | LM-Grasp scenes: crops, observation-only grasp proposals, block splits |
| `src/msp/oracle/linemod_sim.py` | MuJoCo parallel-jaw rollouts on the CoACD-decomposed LINEMOD meshes; optional state traces for figures and videos |
| `src/msp/pose/` | SE(3) helpers, observable pose-fit features, residual-bootstrap pose posterior |
| `src/msp/grasp/` | candidate scoring under pose samples, rejection reasons, camera-base-flange frames and checks |
| `src/msp/models/` | RGB-D outcome scorer (ResNet-18 + FiLM head), learned baselines, point-cloud branch |
| `src/msp/math/`, `belief/`, `engine/` | conformal certificates, rate-regularised bottleneck (ablation), trainer |
| `scripts/sr/` | corpus build, training, FoundationPose poses, planners, LM-O transfer, evaluation |
| `scientific_reports_paper/` | manuscript (`main.tex`), tables (`make_tables.py`), figures (`figures/`), experiment log (`TODO_2026_10_24.md`, `TODO_2026_10_25_lmo.md`) |
| `baselines/` | FoundationPose environment setup |
| `tests/` | pytest suite (geometry, posterior, planner equivalence, rollout traces, frames, models, metrics) |

`data/` (LINEMOD, LM-O, LM-Grasp corpus) and `results/` are git-ignored.

## Environment

```bash
source scripts/sr/env.sh        # sets $PY (repo .venv), PYTHONPATH, MuJoCo GL backend
```

GPU work runs through Slurm on the lab cluster; every job passes `--mem` (see the sbatch headers
in `scripts/sr/`). FoundationPose runs in its own environment (`baselines/README.md`).

## Reproduce

The order of the studies, the pre-declared selection rules, every command and every val/test
number are recorded in `scientific_reports_paper/TODO_2026_10_24.md` (LM-Grasp) and
`TODO_2026_10_25_lmo.md` (LM-O). Main entry points:

```bash
# data and corpus
$PY scripts/download_lm.py                                   # BOP LINEMOD
$PY scripts/sr/build_lm_grasp.py --out data/lm_grasp         # LM-Grasp corpus

# planners
$PY scripts/sr/plan_with_pose.py --pose gt --split test      # annotated-pose planner
$PY scripts/sr/pose_features.py && $PY scripts/sr/pose_posterior.py --mode knn --splits test
$PY scripts/sr/plan_with_pose.py --pose results/sr/runs/fpose/scores.pt \
    --pose-samples results/sr/posterior/knn_s0_test.npz --split test --tag fposeU1_s0_p4
$PY scripts/sr/evaluate_lm.py --out results/sr/metrics_session4.json

# LM-O transfer: FoundationPose on LM-O (GPU), then the three planners on CPU
sbatch --job-name=fp_lmo --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 \
    scripts/sr/fp_run_script.sbatch scripts/sr/fp_lmo.py
sbatch --dependency=afterok:<fp job> --job-name=lmo_eval -c 36 --mem=32G --time=02:00:00 \
    scripts/sr/lmo_eval.sbatch

# manuscript
$PY scientific_reports_paper/make_tables.py                  # numbers.tex, table_*.tex
$PY scientific_reports_paper/figures/make_figures.py         # fig1-fig6 (PDF + PNG)
bash scientific_reports_paper/make_overleaf_folder.sh        # standalone main.tex folder for Overleaf

# tests
sbatch --gres=gpu:1 -c 8 --mem=24G --time=01:00:00 scripts/sr/pytest.sbatch
```

The figure scripts recompute the MuJoCo rollouts they show and check them against the stored
planner scores and outcomes; `make_overleaf_folder.sh` checks that the flattened manuscript has the
same text as the build from the split sources.
