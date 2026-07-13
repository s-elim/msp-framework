# MSP V5 - Complete Implementation Blueprint

This document specifies the complete file-by-file blueprint for the Manipulation-Sufficient Perception (MSP V5) repository. Modeled after enterprise-grade research frameworks like Detectron2 and OpenVLA, this blueprint enforces a strict registry pattern, clear abstraction boundaries, and a scalable hierarchy.

---

## `msp/__init__.py`
*   **Purpose:** Top-level package initialization and versioning.
*   **Classes:** None.
*   **Functions:** None.
*   **Dependencies:** None.
*   **Expected Lines:** ~10

---

## 1. Data Module (`msp/data/`)

### `msp/data/__init__.py`
*   **Purpose:** Expose dataset and transform registries.

### `msp/data/dataset.py`
*   **Purpose:** Data loading and batching for scenes, actions, and physics outcomes.
*   **Classes:** `BaseDataset(Dataset)`, `MSPOfflineDataset(BaseDataset)`.
*   **Functions:** `build_dataset`, `msp_collate_fn`.
*   **Dependencies:** `torch.utils.data`, `numpy`, `h5py` (or `zarr`), `msp.utils.registry`.
*   **Expected Lines:** ~250

### `msp/data/transforms.py`
*   **Purpose:** Observation augmentation (color jitter, depth noise, synthetic occlusion).
*   **Classes:** `Compose`, `RandomColorJitter`, `RandomDepthNoise`, `DropPixels`.
*   **Functions:** `build_transforms`.
*   **Dependencies:** `torchvision.transforms`, `torch`.
*   **Expected Lines:** ~150

---

## 2. Model Module (`msp/models/`)

### `msp/models/__init__.py`
*   **Purpose:** Expose network builders and module registries.

### `msp/models/backbones/base.py`
*   **Purpose:** Interface contract for observation feature extractors.
*   **Classes:** `BaseBackbone(nn.Module)`.
*   **Functions:** None.
*   **Dependencies:** `torch.nn`.
*   **Expected Lines:** ~30

### `msp/models/backbones/vision.py`
*   **Purpose:** Concrete implementations of image/depth backbones.
*   **Classes:** `ResNetBackbone(BaseBackbone)`, `PointNetBackbone(BaseBackbone)`.
*   **Functions:** `build_backbone`.
*   **Dependencies:** `torchvision.models`, `torch`, `msp.utils.registry`.
*   **Expected Lines:** ~200

### `msp/models/encoders.py`
*   **Purpose:** Belief encoder mapping observation features to the sufficient statistic distribution $Z$.
*   **Classes:** `BaseEncoder(nn.Module)`, `GaussianBeliefEncoder(BaseEncoder)`.
*   **Functions:** `build_encoder`.
*   **Dependencies:** `torch.nn`, `msp.utils.registry`.
*   **Expected Lines:** ~100

### `msp/models/heads.py`
*   **Purpose:** Decodes $(z, a)$ into physical outcomes and estimates the Value of Information.
*   **Classes:** `BaseOutcomeHead(nn.Module)`, `MLPOutcomeHead(BaseOutcomeHead)`, `AcquisitionNet(nn.Module)`.
*   **Functions:** `build_heads`.
*   **Dependencies:** `torch.nn`, `msp.utils.registry`.
*   **Expected Lines:** ~150

### `msp/models/readouts.py`
*   **Purpose:** Optional shape/pose decoder for "honest" metric evaluation.
*   **Classes:** `BaseReadout(nn.Module)`, `PoseReadout(BaseReadout)`.
*   **Functions:** `build_readout`.
*   **Dependencies:** `torch.nn`, `msp.utils.registry`.
*   **Expected Lines:** ~100

---

## 3. Physics & Oracle Module (`msp/physics/`)

### `msp/physics/__init__.py`
*   **Purpose:** Expose oracle and sampler registries.

### `msp/physics/oracle.py`
*   **Purpose:** Interface to query ground-truth physical outcomes ($Y$) from actions.
*   **Classes:** `BasePhysicsOracle`, `PyBulletOracle(BasePhysicsOracle)`, `MuJoCoOracle(BasePhysicsOracle)`.
*   **Functions:** `build_oracle`.
*   **Dependencies:** `pybullet`, `mujoco` (optional), `numpy`, `msp.utils.registry`.
*   **Expected Lines:** ~300

### `msp/physics/samplers.py`
*   **Purpose:** Implementations of the action proposal measure $\rho$.
*   **Classes:** `BaseActionSampler`, `HeuristicGraspSampler(BaseActionSampler)`, `BoundaryFocusedSampler(BaseActionSampler)`.
*   **Functions:** `build_action_sampler`.
*   **Dependencies:** `torch`, `numpy`, `msp.utils.registry`.
*   **Expected Lines:** ~200

---

## 4. Core Algorithms Module (`msp/core/`)

### `msp/core/__init__.py`
*   **Purpose:** Core mathematical logic.

### `msp/core/losses.py`
*   **Purpose:** The Variational Information Bottleneck (VIB) objective (Eq 10, 11).
*   **Classes:** `VIBLoss(nn.Module)`.
*   **Functions:** `compute_distortion`, `compute_rate`.
*   **Dependencies:** `torch.nn`, `torch.distributions`.
*   **Expected Lines:** ~150

### `msp/core/inference.py`
*   **Purpose:** The decision rule orchestrator (Algorithm 2).
*   **Classes:** `InferenceEngine`.
*   **Functions:** `compute_marginal_success`, `compute_epistemic_variance`.
*   **Dependencies:** `torch`, `msp.core.calibration`.
*   **Expected Lines:** ~150

### `msp/core/active.py`
*   **Purpose:** Value of Information (VoI) and active perception ambiguity calculation (Eq 16, 17, 18).
*   **Classes:** `ActivePerceptionManager`.
*   **Functions:** `compute_ambiguity`, `evaluate_candidate_views`.
*   **Dependencies:** `torch`.
*   **Expected Lines:** ~120

### `msp/core/tta.py`
*   **Purpose:** Test-time adaptation via trust-region belief optimization (Eq 20, 21).
*   **Classes:** `TTAOptimizer`.
*   **Functions:** `optimize_static_belief`, `optimize_perturbed_belief`.
*   **Dependencies:** `torch.optim`.
*   **Expected Lines:** ~150

### `msp/core/calibration.py`
*   **Purpose:** Conformal prediction sets and Adaptive Conformal Inference (Eq 22, 23, 24).
*   **Classes:** `BaseCalibrator`, `ConformalCalibrator(BaseCalibrator)`.
*   **Functions:** `compute_nonconformity`, `update_aci_alpha`.
*   **Dependencies:** `torch`, `numpy`.
*   **Expected Lines:** ~120

---

## 5. Execution Engine (`msp/engine/`)

### `msp/engine/__init__.py`
*   **Purpose:** Expose Trainer/Evaluator.

### `msp/engine/trainer.py`
*   **Purpose:** Standardized training loop, backward passes, metric logging, and DDP management.
*   **Classes:** `Trainer`.
*   **Functions:** `train_epoch`, `validate_epoch`, `save_checkpoint`.
*   **Dependencies:** `torch.optim`, `msp.core.losses`, `msp.utils.logger`.
*   **Expected Lines:** ~350

### `msp/engine/evaluator.py`
*   **Purpose:** Full offline evaluation loop (computing coverage, average success, rate/distortion curves).
*   **Classes:** `Evaluator`.
*   **Functions:** `evaluate_dataset`, `compute_metrics`.
*   **Dependencies:** `torch`, `msp.core.inference`.
*   **Expected Lines:** ~250

---

## 6. Utilities (`msp/utils/`)

### `msp/utils/registry.py`
*   **Purpose:** String-based class registration (e.g., `@BACKBONE_REGISTRY.register()`).
*   **Classes:** `Registry`.
*   **Functions:** None.
*   **Dependencies:** None.
*   **Expected Lines:** ~80

### `msp/utils/logger.py`
*   **Purpose:** Unified logging interface (Console + Weights & Biases + Tensorboard).
*   **Classes:** `LoggerManager`.
*   **Functions:** `setup_logger`, `log_metrics`, `log_artifact`.
*   **Dependencies:** `logging`, `wandb`.
*   **Expected Lines:** ~120

### `msp/utils/geometry.py`
*   **Purpose:** Shared math for SE(3) transformations and quaternions.
*   **Classes:** None.
*   **Functions:** `quat_to_matrix`, `matrix_to_quat`, `transform_points`, `compute_iou`.
*   **Dependencies:** `torch`.
*   **Expected Lines:** ~150

### `msp/utils/seed.py`
*   **Purpose:** Ensures strict deterministic behavior across the entire framework.
*   **Classes:** None.
*   **Functions:** `seed_everything`.
*   **Dependencies:** `torch`, `numpy`, `random`.
*   **Expected Lines:** ~40

---

## 7. Entry Points (`scripts/`)

### `scripts/train.py`
*   **Purpose:** Main training entry point. Composes Hydra config to instantiate Trainer.
*   **Classes:** None.
*   **Functions:** `main`.
*   **Dependencies:** `hydra`, `omegaconf`, `msp.engine.trainer`.
*   **Expected Lines:** ~60

### `scripts/evaluate.py`
*   **Purpose:** Main evaluation entry point to test a checkpoint against an offline dataset.
*   **Classes:** None.
*   **Functions:** `main`.
*   **Dependencies:** `hydra`, `omegaconf`, `msp.engine.evaluator`.
*   **Expected Lines:** ~60

### `scripts/deploy.py`
*   **Purpose:** Example interactive inference loop connecting Inference, Active Perception, and TTA.
*   **Classes:** None.
*   **Functions:** `main`, `run_deployment_loop`.
*   **Dependencies:** `hydra`, `msp.core.*`, `msp.physics.*`.
*   **Expected Lines:** ~120
