# MSP Cookbook

This cookbook provides practical instructions on how to use the Manipulation-Sufficient Perception (MSP) framework. It covers environment setup, training, evaluation, and test-time adaptation (TTA).

---

## 1. Installation & Setup

Before running any scripts, ensure your environment is set up correctly.

```bash
# Clone the repository (assuming it's hosted)
git clone https://github.com/s-elim/msp-framework.git
cd msp-framework

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (PyTorch, Hydra, WandB, PyBullet)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install hydra-core wandb numpy h5py scipy
```

---

## 2. Training the Model

Training is handled by Hydra, which allows you to easily compose configurations. 

### Basic Training
To start a training run with the default configuration (which uses the `sim_ycb` dataset and `resnet` backbone):
```bash
python scripts/train.py
```

### Overriding Configurations
You can override any configuration parameter directly from the command line:
```bash
# Change the backbone to PointNet and increase the batch size
python scripts/train.py model.backbone=pointnet2 training.batch_size=32

# Change the VIB beta tradeoff parameter
python scripts/train.py training.beta=0.1
```

### What Happens During Training?
1. The `Trainer` will initialize the `BeliefEncoder` and `OutcomeHead`.
2. It will query the `BasePhysicsOracle` to get ground-truth outcomes.
3. The VIB loss (Rate and Distortion) is minimized.
4. Checkpoints are automatically saved to `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`.
5. Metrics are logged to Weights & Biases (if configured).

---

## 3. Conformal Calibration

After the model is trained, you must calibrate the conformal prediction sets to guarantee success bounds. This is typically done automatically at the end of `train.py`, but can be run offline.

```bash
# Run calibration on a held-out dataset using a saved checkpoint
python scripts/evaluate.py mode=calibrate model.checkpoint_path="outputs/2026/checkpoints/best.pth"
```
This script will compute the nonconformity scores and save the $\hat{q}$ quantile to a `calibration.json` file.

---

## 4. Evaluation (Offline)

To rigorously evaluate the model's expected success rate and the tightness of the conformal bounds on an offline test set:

```bash
python scripts/evaluate.py mode=test model.checkpoint_path="outputs/.../best.pth" dataset=test_set
```
This will log the empirical coverage rate. If calibration was successful, the coverage rate should tightly bound $1 - \alpha$.

---

## 5. Deployment & Active Perception

To run the interactive inference loop (which includes Active Perception and Test-Time Adaptation), use the deployment script.

```bash
python scripts/deploy.py model.checkpoint_path="outputs/.../best.pth"
```

### The Inference Loop (`scripts/deploy.py`):
1. **Observe:** The camera captures RGB-D.
2. **Encode:** The `BeliefEncoder` produces $Z$.
3. **Ambiguity Check:** The system checks epistemic variance. If the scene is highly occluded, the `ActivePerceptionManager` will trigger the robot to move the camera to a new viewpoint.
4. **Action Selection:** The model proposes actions and filters them through the certified conformal set.
5. **Execution:** The robot executes the best action.
6. **Test-Time Adaptation (TTA):** If the grasp fails or slips unexpectedly, the `TTAOptimizer` will use trust-region gradient descent to update the belief given the physical outcome, adapting to the domain shift on the fly.

---

## 6. Extending the Framework

### Adding a New Backbone
1. Create a new file in `msp/models/backbones/`.
2. Inherit from `BaseBackbone`.
3. Decorate your class with `@BACKBONE_REGISTRY.register()`.
4. Create a corresponding YAML file in `configs/model/backbone/my_backbone.yaml`.

### Adding a New Physics Oracle
1. Create a new file in `msp/physics/`.
2. Inherit from `BasePhysicsOracle`.
3. Implement the `query_outcome(scene, action)` method.
4. Register it with `@ORACLE_REGISTRY.register()`.
