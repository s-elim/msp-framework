# Manipulation-Sufficient Perception (MSP)

Perception for manipulation, formulated as estimating a calibrated belief over the **minimal
statistic that preserves all and only the information that changes action outcomes** — not an
accurate pose.

Two world states are equivalent when every admissible action produces the same outcome
distribution. Geometry is recoverable only up to that equivalence, and accuracy beyond the
sufficiency resolution is both unrecoverable and unnecessary. Grasp selection, distribution-free
abstention, active perception, and test-time adaptation all fall out of **two learned modules** as
inference procedures. No reconstruction, no dynamics rollout, no RL.

---

## Status

Rebuilt from the mathematics up. The previous implementation was audited equation by equation;
of the 23 equations carrying an implementation obligation, 7 were correct. Six were *incorrect* —
present, running, producing plausible numbers, silently violating the theorem they instantiated.

This tree keeps the formalization, the hypothesis, and the algorithms. Everything else is new.

**70 tests pass.** Every audited defect has a named regression test that fails against the old
code and passes against this one.

---

## The experiment: RGB-D grasping

The observation is a **rendered RGB-D image**, the labels come from **MuJoCo rigid-body rollouts**,
and the grasp quality comes from a **differentiable Ferrari-Canny** wrench-space metric.

```bash
MUJOCO_GL=egl python scripts/train.py data=rgbd model=resnet
```

Corpus generation is cached: ~2 ms/frame to render, ~10 ms/grasp to roll out, so a 20k-scene corpus
is a few minutes and is never regenerated.

**What the camera can and cannot see.** It sees the object's pose and size. It does **not** see
friction, mass, or the centre of mass -- and those three decide slip and torque failure. Two scenes
differing only in them render *identically* and yet grasp *differently*
(`tests/data/test_rgbd.py::test_the_camera_cannot_see_friction_mass_or_centre_of_mass`).

That is the entire point. A pose estimator cannot even represent that ignorance. MSP's belief can,
its certificate abstains on it, and active touch (Eq 20/21) is what resolves it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sim,log]"

pytest                                          # 70 tests, ~10s, no GPU needed
python scripts/train.py                         # train + calibrate + evaluate
python scripts/train.py train.beta.succ=50 model.latent_dim=32
python scripts/deploy.py checkpoint=outputs/.../best.pth   # Algorithm 2, closed loop
```

Reproduce every number and figure in the paper:

```bash
python scripts/paper/run_experiments.py --out results/      # E1-E5
python scripts/paper/make_figures.py                        # -> paper/figures/*.pdf
python scripts/paper/make_tables.py                         # -> paper/tables/*.tex
```

---

## What the experiments currently show

Run on `SyntheticOracle`, a physics operator whose indistinguishability class is known in
**closed form** — so a diagnostic can be validated against ground truth before it is ever pointed
at a simulator. Numbers below are from a short (4-epoch) smoke run; treat them as pipeline
evidence, not as paper results.

| Claim | Result | Status |
|---|---|---|
| **Theorem 4** — residual ambiguity = `ker J(x)` | mean principal angle **0.01°**, max **0.06°**, over 50 states | **Verified** |
| **Theorem 3** — β → ∞ ⇒ sufficiency | β: 0.5→100 traces R: 0.003→14.2, D: 0.69→−0.96 | **Verified** |
| **Theorem 7** — marginal coverage ≥ 1−α | 0.979/0.949/0.901/0.802/0.700 vs targets 0.98/0.95/0.90/0.80/0.70 | **Verified** |
| *z*-ablation — the head actually uses *z* | D = −0.730 (full) vs **+0.656** (ablated) | **Verified** |
| Abstention buys success | 0.949 when acting (abstained 9/400) vs 0.935 always-acting | Real but **modest** |
| Hypothesis: "lower-dimensional" | distortion is **not monotone in d** and does **not** saturate at d = rank = 3 | **Not supported yet** |

The last row is reported because it is true. On a 4-epoch run the latent-dimension sweep is noisy
and does not demonstrate the saturation the hypothesis predicts. It needs a real training budget
and error bars before it can be claimed. Do not put it in a paper until it does.

### The tier gap: the analytic prior is a poor predictor of real lift outcome

Measured on 320 grasps, analytic Ferrari-Canny prior vs. the MuJoCo rigid-body simulator on the
**same** `(x, a)`:

| | |
|---|---|
| Success agreement | **0.42** |
| Analytic success rate | 0.68 |
| Simulated success rate | 0.25 |
| **Analytic false-positive rate** | **0.74** |
| Slip correlation | −0.02 |

The analytic tier says "force-closed, good grasp" and the object falls out of the hand **three
times in four**. Its quasi-static slip model is essentially uncorrelated with dynamic slip.

This is the in-simulation analogue of the sim-to-real sufficiency gap, and it is the most important
number the oracle produces: it says quantitatively that **training the encoder on the analytic tier
alone would teach it to be confidently wrong.** It is exactly why Section 11 demands the composition
rather than either tier alone, and exactly the concern behind reviewer attacks 3 and 11 ("force
closure computed on estimated geometry is not grounded in outcome"). `CompositeOracle.tier_gap()`
computes it. (Caveat: the simulator itself is not yet validated — see limitations.)

---

## Repository layout

```
src/msp/
├── types.py          Domain types. Decision = ActionChoice | Abstain (a SUM type, so
│                     fail-open is a type error). PredictionSet carries BOTH label
│                     memberships, so the Eq-24 singleton test cannot be skipped.
├── math/             Pure functions, one per equation. No nn.Module, no device, no I/O.
│   ├── divergences   Eq 11
│   ├── bottleneck    Eq 7-10   ** beta MULTIPLIES distortion. See the module docstring. **
│   ├── decision      Eq 13-16
│   └── conformal     Eq 22-24 + ACI
├── belief/           The posterior q(z|o). Its ONLY route to the head is rsample(), which
│                     is what makes the frozen-variance bug unwritable.
├── oracle/           M — the object that DEFINES the estimand. SyntheticOracle has an
│                     analytically known ker J(x), which turns Theorem 4 into a unit test.
├── models/           The two learned modules: BeliefEncoder (A), OutcomeHead (B).
├── inference/        Algorithm 2: engine (abstention), calibrator, TTA.
├── engine/           Trainer (AMP/DDP/cosine/best-ckpt), Evaluator (MARGINAL coverage).
├── diagnostics/      J(x), null space, principal angles. Contribution C2.
└── data/             Scenes carry the state x (J(x) needs it) and importance weights.

tests/                70 tests. 8+ are named regression tests for audited defects.
configs/              Hydra. Every documented override actually works.
scripts/paper/        run_experiments -> results/*.json -> figures + LaTeX tables.
```

---

## Three invariants this codebase enforces structurally

Each one is a bug the previous implementation shipped, made *unwritable* rather than merely fixed.

**1. The belief cannot be collapsed to its mean.** `Belief` exposes `rsample()` and no `.mean()`.
The old TTA evaluated the head at the mean, so `∂NLL/∂logvar ≡ 0` and the belief's variance never
moved — which made contribution C3 false, since TTA cannot reduce an ambiguity that is a functional
of a variance it cannot change.

**2. Acting without a certificate is a type error.** `select()` returns `ActionChoice | Abstain`.
The old engine masked uncertified scores with `-inf` and let `argmax` return index 0, silently
executing an uncertified grasp when the certified set was empty.

**3. β cannot be inverted.** `L = Σ_j β_j·D_j + R`. Larger β buys **sufficiency** (Theorem 3). The
old code computed `L = Σ_j D_j/β_j + R`, so larger β bought *compression* — the exact opposite —
and any β sweep would have traced the frontier backwards.

---

## Known limitations, stated up front

- **`ResNetBackbone` global-average-pools**, so `z` is one code for the whole scene. The theory's
  own physical reading — contact regions matter *because dM/dx is large there* — is spatial and
  per-object. This is the right inductive bias for the V1 single-object scope and the wrong one for
  the full claim.
- **`DiagonalGaussianBelief` is unimodal.** Theorem 5 requires the belief on a symmetric object to
  be supported on a *group orbit*. A mixture or flow posterior is needed there; the `Belief` ABC
  exists so it can be dropped in without touching anything else.
- **Tier 3 does not exist.** A residual fit to a few thousand *real* robot grasp outcomes is the
  tier the reviewers say carries the paper. It is deliberately absent rather than stubbed, because a
  stub would let someone believe the grounding exists. **Nothing here transfers to a real robot yet.**
- **No real object meshes.** Boxes and cylinders only. YCB / GraspNet-1Billion is the next step.
- **No baselines.** AnyGrasp, Contact-GraspNet, FoundationPose+sampling are all unimplemented.
- **No pose readout.** Per the thesis you should not *estimate* pose, but the honest max-entropy
  readout (§10) is what would demonstrate that readout error concentrates in `ker J(x)` -- the
  Corollary, and the claim that makes "ADD-S misranks methods" land. It is not built.
- **Active perception is not wired.** Eq 17/18 need a rendered look-ahead for `IG_true`.
- **Theorem 4 is vacuous for a generic box.** `rank J(x) = d_X`, so nothing is continuously
  unidentifiable: pose, size, friction, mass and COM all change some outcome. The theorem has content
  only for objects with a genuine outcome-invariance — a **cylinder**, where rotation about the
  symmetry axis gives `dim ker J = 1`. This limits the theorem's reach and belongs in the paper.
- **Active perception is not wired.** Eq 17/18 need a rendered look-ahead to produce `IG_true`, and
  there is no renderer. The `AcquisitionNet` was **removed** rather than shipped untrainable — the
  audited repo kept one and took an argmax over its random weights to steer a camera.

## License

MIT.
