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
- **`SyntheticOracle` is not physics.** It is a world with a *known answer*, for validating the
  diagnostics. A real Ferrari-Canny + rigid-body + real-residual operator is the next build, and
  it slots in behind `PhysicsOracle` without changing a line above it.
- **Active perception is not wired.** `AcquisitionNet` exists but Eq 17/18 (the look-ahead target)
  need the renderer. The net refuses to run until `fitted` is set, rather than steering a camera
  with random weights as the old one did.

## License

MIT.
