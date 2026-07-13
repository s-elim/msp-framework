# Manipulation-Sufficient Perception (MSP V5) - Software Architecture Design

This document outlines the software architecture for the MSP V5 research framework. The design prioritizes modularity, scalability, and reusability, allowing future researchers to swap backbones, outcome parameterizations, physics simulators, and calibration methods without touching the core logic.

---

## STEP 1: Logical Components

Based on the MSP V5 formalization, the system breaks down into the following logical components:

**Data & Simulation**
*   **Dataset / Dataloader:** Handles loading offline observations, actions, and physics outcomes.
*   **Physics Oracle (M):** Differentiable physics simulator or rigid-body engine to query ground truth outcomes ($Y$).
*   **Action Sampler ($\rho$):** Generates candidate grasps, heavily focused near decision boundaries.

**Neural Architectures (Models)**
*   **Observation Backbone:** Feature extractor for RGB-D/Language (e.g., symmetry-equivariant networks).
*   **Belief Encoder ($q_\theta$):** Maps backbone features to the latent belief distribution ($\mu, \log\sigma^2$).
*   **Outcome Head ($p_\psi$):** Decodes $(z, a, g)$ into outcome parameters (success, margin, slip).
*   **Acquisition Network ($\alpha_\omega$):** Amortized estimator for the Value of Information (VoI).
*   **Pose Readout (Optional):** Honest shape/pose decoder $g_\xi(z)$ mapping $z$ back to the identifiable quotient space.

**Core Algorithms**
*   **Variational Information Bottleneck (VIB) Loss:** Calculates distortion (outcome NLL) and rate (KL divergence).
*   **Inference Engine:** Handles $K$-shot posterior sampling, computes $s(o, a)$ and $v(o, a)$, and risk-averse selection.
*   **Active Perception:** Evaluates $U(o)$ and triggers the acquisition net to find the Next-Best-View ($b^*$).
*   **Conformal Calibration:** Computes nonconformity scores on a calibration fold to output $\hat{q}$ and manages Adaptive Conformal Inference (ACI) drift.
*   **Test-Time Adaptation (TTA):** Trust-region optimizer for updating belief given a physical probe outcome.

**Infrastructure**
*   **Trainer & Evaluator:** Orchestrates the training and evaluation loops.
*   **Logger & Config Manager:** Manages Hydra configs and Weights & Biases reporting.

---

## STEP 2: Directory Structure

```text
msp_framework/
├── configs/                # Hydra YAML configurations
│   ├── dataset/            # e.g., sim_ycb.yaml, real_franka.yaml
│   ├── model/              # e.g., resnet_encoder.yaml, mlp_head.yaml
│   └── experiment/         # e.g., active_perception_ablation.yaml
├── msp/
│   ├── data/               # Data loading, augmentations, and samplers
│   ├── models/             
│   │   ├── backbones/      # Interchangeable feature extractors (ResNet, PointNet)
│   │   ├── encoders.py     # Belief encoder (q_theta)
│   │   ├── heads.py        # Outcome head (p_psi) and Acquisition net
│   │   └── readouts.py     # Optional pose/shape decoders
│   ├── physics/            # Simulation APIs (PyBullet, MuJoCo interfaces)
│   │   ├── oracle.py       # Handles querying M(y | x, a)
│   │   └── samplers.py     # Action measure (rho) implementations
│   ├── core/               # Mathematical formulations of MSP
│   │   ├── losses.py       # VIB Objective (Eq 10, 11)
│   │   ├── active.py       # VoI and Ambiguity (Eq 16, 17, 18)
│   │   ├── tta.py          # Belief update optimization (Eq 20, 21)
│   │   └── calibration.py  # Split conformal and ACI (Eq 22, 23, 24)
│   ├── engine/             # Execution pipelines
│   │   ├── trainer.py      # Main training loop
│   │   └── inference.py    # Deployment loop (Algorithm 2)
│   └── utils/              # Metrics, logging, visualization, geometry utils
├── scripts/                # Entry points
│   ├── train.py            
│   └── evaluate.py         
└── tests/                  # Unit and Integration tests
```

**Why this structure?** It heavily enforces the Separation of Concerns. Researchers investigating new conformal techniques only touch `core/calibration.py`. Researchers designing new vision architectures only touch `models/backbones/`.

---

## STEP 3: Python Class Design

### `BeliefEncoder`
*   **Purpose:** Maps observations to the sufficient statistic distribution $Z$.
*   **Input:** $O$ (Tensor `[B, 4, H, W]`).
*   **Output:** $\mu$ `[B, d]`, $\log\sigma^2$ `[B, d]`.
*   **Dependencies:** `BaseBackbone`.
*   **Methods:** `forward(self, o)` - passes $O$ through backbone, applies an MLP to emit mean and variance.

### `OutcomeHead`
*   **Purpose:** Approximates $M(y | x, a)$ from the latent code.
*   **Input:** $Z$ `[B, K, d]`, $A$ `[B, N_a, 7]`.
*   **Output:** `y_hat` dict containing `succ_logits`, `margin_mu`, `margin_logvar`, `slip_params`.
*   **Dependencies:** None.
*   **Methods:** `forward(self, z, a)` - concats $z$ and $a$, returns parameter distributions.

### `VIBLoss`
*   **Purpose:** Computes the Lagrangian of the rate and distortion (Eq 10).
*   **Input:** `y_hat`, `y_true`, $\mu$, $\log\sigma^2$, $\beta$.
*   **Output:** Scalar `loss`, dict `metrics`.
*   **Methods:** `_compute_distortion()`, `_compute_rate()`, `forward()`.

### `ActivePerceptionManager`
*   **Purpose:** Computes ambiguity $U$ and evaluates views.
*   **Input:** Belief $(\mu, \log\sigma^2)$, Observation $O$.
*   **Output:** Next-Best-View $b^*$, boolean `should_sense`.
*   **Methods:** `compute_ambiguity(s_variance)`, `get_next_view(o)`.

### `ConformalCalibrator`
*   **Purpose:** Maintains prediction sets and ACI drift.
*   **Input:** Calibration set `{(s_i, y_i)}`.
*   **Output:** $\hat{q}$ (quantile).
*   **Methods:** `fit(scores)`, `predict_set(s_test)`, `update_aci(err_t)`.

---

## STEP 4: Interfaces (Abstract Base Classes)

To ensure the codebase survives years of research, we define strict interfaces:

*   **`BaseEncoder(nn.Module)`**: Guarantees a `forward` method returning a parameterized distribution, abstracting away deterministic vs stochastic beliefs.
*   **`BaseOutcomeHead(nn.Module)`**: Guarantees inputs `(z, a)` and returns a standardized outcome dictionary. Future researchers can swap a Gaussian parameterization for a Normalizing Flow without breaking the loss.
*   **`BasePhysicsOracle`**: Abstract methods `query_outcome(x, a)`. Implementations: `PyBulletOracle`, `MuJoCoOracle`, `RealWorldLogger`. Ensures the training loop doesn't care where the physics data comes from.
*   **`BaseActionSampler`**: Abstract method `sample(scene_context, n_samples)`.
*   **`BaseCalibrator`**: Abstract methods `fit`, `predict`.

---

## STEP 5: Training Pipeline (Data Flow)

1.  **Dataset:** Yields physical scene $x^{(b)}$ `[B]`.
2.  **Physics Oracle & Sampler:** Renders $O$ `[B, 4, H, W]`. Samples actions $A$ `[B, N_a, 7]`. Queries true outcomes $Y_{true}$ `[B, N_a, 3]`.
3.  **Backbone & Belief Encoder:** $O \rightarrow \mu, \log\sigma^2$ `[B, d]`.
4.  **Reparameterization:** $z = \mu + e^{0.5 \log\sigma^2} \odot \epsilon \rightarrow Z$ `[B, d]`.
5.  **Outcome Head:** $Z, A \rightarrow \hat{Y}$ `[B, N_a, ...]`.
6.  **Loss:** Evaluates distortion (BCE for success, NLL for margin/slip) and rate (KL against $\mathcal{N}(0,I)$) scaled by $\beta$.
7.  **Optimizer:** Computes gradients $\nabla_{\theta, \psi} L$, updates parameters.
8.  **Validation & Checkpoint:** Evaluates held-out conformal scores, saves weights.

---

## STEP 6: Inference Pipeline (Data Flow)

1.  **Observe:** Retrieve RGB-D $O$.
2.  **Belief Encoding:** Encode $O \rightarrow (\mu, \log\sigma^2)$. Sample $Z$ `[K, d]` ($K=32$).
3.  **Action Proposal:** Sample candidate actions $A$ `[N_a, 7]`.
4.  **Outcome Prediction:** Predict $s(z_k, a)$. Marginalize to compute mean success $s(a)$ and variance $v(a)$.
5.  **Active Perception Check:** Compute ambiguity $U = \text{mean}_a(v(a))$. If $U > \tau_U$, query `AcquisitionNet`, move camera, goto 1.
6.  **Calibration & Abstention:** Compute certified set $A_{cert} = \{a \mid s(a) \geq 1 - \hat{q}\}$. If empty, ABSTAIN.
7.  **Selection:** Execute $a^* = \arg\max_{a \in A_{cert}} (s(a) - \lambda v(a))$.
8.  **Test-Time Adaptation:** Observe real outcome $y_p$. Optimize $\mu, \log\sigma^2$ using trust-region gradient descent against $\hat{Y}_{pred} \leftrightarrow y_p$.

---

## STEP 7: Configuration System (Hydra)

Hydra handles compositional configuration.
*   **`dataset.yaml`**: defines `name`, `oracle_backend`, `resolution`.
*   **`model.yaml`**: defines `backbone.type`, `encoder.latent_dim`, `head.hidden_layers`.
*   **`training.yaml`**: defines `beta` (VIB tradeoff), `lr`, `batch_size`, `K` (samples).
*   **`calibration.yaml`**: defines $\alpha$ (error rate), $\gamma$ (ACI step).

*Why Hydra?* A researcher can run `python train.py model.backbone=resnet50 training.beta=0.5` without altering code. It supports multirun grid searches natively.

---

## STEP 8: Experiment Management

*   **Weights & Biases (W&B):** Primary logger. Logs $L_{dist}$, $L_{rate}$, conformal coverage rates, and active perception trigger frequency.
*   **Versioning:** Hydra automatically hashes the configuration and Git commit. Artifacts (checkpoints) are saved locally and synced to W&B.
*   **Seed Control:** A global `utils.seed_everything(seed)` locks PyTorch, NumPy, and Python random to ensure bit-level reproducibility of the $K$ samples and Action Sampler.
*   **Ablation Management:** Managed entirely via Hydra overrides (e.g., turning off TTA or Active Perception by overriding a boolean flag in the inference config).

---

## STEP 9: Unit & Integration Tests

*   **Shape & Determinism Tests:** `pytest` suites to verify that `BeliefEncoder` strictly returns `[B, d]`, and that fixed seeds yield exact tensor outputs.
*   **Physics Mocks:** A mocked `BasePhysicsOracle` that returns deterministic $y = 1$ if $a_{z} > 0$ else $0$. Ensures the outcome head learns simple distributions perfectly.
*   **Conformal Correctness:** A test passing dummy $s_{test}$ through the `ConformalCalibrator` to assert that exactly $1-\alpha$ of the points fall in the certified set.
*   **Integration Tests:** A miniature `train.py` run on a synthetic dataset of 10 samples for 2 epochs. Validates that the loss decreases and backward passes don't crash due to detached graphs (crucial for reparameterization trick).

---

## STEP 10: Implementation Roadmap

*   **Stage 1: Minimal Prototype (Week 1-2)**
    *   Build abstract base classes.
    *   Implement `BeliefEncoder`, `OutcomeHead`, and `VIBLoss`.
    *   Test on a mocked dataset.
*   **Stage 2: Simulation Integration & Training (Week 3-5)**
    *   Implement PyBullet/MuJoCo `PhysicsOracle`.
    *   Build `BaseActionSampler`.
    *   Implement and scale the training loop using DDP.
*   **Stage 3: Inference & Calibration (Week 6-8)**
    *   Implement Algorithm 2.
    *   Build `ConformalCalibrator` and `TestTimeAdaptation` optimizers.
*   **Stage 4: Active Perception (Week 9-10)**
    *   Implement `AcquisitionNet` and ambiguity logic.
*   **Stage 5: Real Robot Deployment (Week 11-12+)**
    *   Write `RealWorldOracle` for ROS/Franka integration.
    *   Finalize journal experiments and baselines.

*Difficulty:* Moderate to High. The neural networks are standard, but the dynamic inference loop (TTA + Active Perception + Abstention) requires precise graph management in PyTorch.
