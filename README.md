# Pose-posterior grasp planning

Project page: <https://physical-agi.github.io/MSP-FRAMEWORK/>

## Environment

```bash
source scripts/sr/env.sh
```

## Reproduce

```bash
# data and corpus
$PY scripts/download_lm.py
$PY scripts/sr/build_lm_grasp.py --out data/lm_grasp

# planners
$PY scripts/sr/plan_with_pose.py --pose gt --split test
$PY scripts/sr/pose_features.py && $PY scripts/sr/pose_posterior.py --mode knn --splits test
$PY scripts/sr/plan_with_pose.py --pose results/sr/runs/fpose/scores.pt \
    --pose-samples results/sr/posterior/knn_s0_test.npz --split test --tag fposeU1_s0_p4
$PY scripts/sr/evaluate_lm.py --out results/sr/metrics_session4.json

# LM-O transfer
sbatch --job-name=fp_lmo --gres=gpu:1 -c 8 --mem=32G --time=02:00:00 \
    scripts/sr/fp_run_script.sbatch scripts/sr/fp_lmo.py
sbatch --dependency=afterok:<fp job> --job-name=lmo_eval -c 36 --mem=32G --time=02:00:00 \
    scripts/sr/lmo_eval.sbatch

# tests
sbatch --gres=gpu:1 -c 8 --mem=24G --time=01:00:00 scripts/sr/pytest.sbatch
```
