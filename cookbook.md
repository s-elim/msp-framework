# MSP Cookbook

Practical recipes for the Manipulation-Sufficient Perception framework.

Every command here has been executed against this tree. If a recipe does not work, that is a bug —
open an issue rather than working around it. The previous cookbook documented a Hydra interface,
checkpoint paths, and a calibration step that did not exist, and it cost a reader a day to find out.

**Contents**

1. [Setup](#1-setup)
2. [The 60-second tour](#2-the-60-second-tour)
3. [Training](#3-training)
4. [Multi-GPU](#4-multi-gpu)
5. [Calibration and the certificate](#5-calibration-and-the-certificate)
6. [Deployment: Algorithm 2](#6-deployment-algorithm-2)
7. [Test-time adaptation](#7-test-time-adaptation)
8. [Identifiability: measuring ker J(x)](#8-identifiability-measuring-ker-jx)
9. [Reproducing the paper](#9-reproducing-the-paper)
10. [Extending the framework](#10-extending-the-framework)
11. [Gotchas that will cost you a day](#11-gotchas-that-will-cost-you-a-day)

---

## 1. Setup

```bash
git clone https://github.com/s-elim/msp-framework.git && cd msp-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sim,log]"

pytest                      # 70+ tests, ~10 s, CPU only
```

If `pytest` is green, the mathematics is verified: every numbered equation of the formalization has
a test, and every defect found in the audit of the previous implementation has a named regression
test that pins it.

---

## 2. The 60-second tour

```bash
python scripts/train.py train.epochs=5 data.n_train=4000
```

That trains the two learned modules, calibrates a conformal certificate on a **held-out** fold, and
evaluates coverage. You should see something like:

```
q_hat = 0.6918
coverage        = 0.9011  (target 0.90)  OK
cert. precision = 0.9491
abstention      = 0.0990
rate / distortion = 4.2115 / 0.1742 +/- 0.0322
```

Read those five numbers as follows.

| Number | Meaning |
|---|---|
| `q_hat` | The conformal threshold (Eq 23). |
| `coverage` | **Theorem 7.** `P(succ ∈ C(o,a))`, over *all* test points. This is the guarantee. |
| `cert. precision` | `P(succ=1 \| a ∈ A_cert)`. The *operational* number. **A different estimand** — do not conflate them. |
| `abstention` | Fraction of scenes where nothing certified and MSP declined to act (Eq 24). |
| `rate / distortion` | One point on the rate–distortion frontier, with its standard error. |

---

## 3. Training

Hydra composes the config. Every override below actually works.

```bash
# Change the sufficiency budget (see the warning in §11 — the direction matters)
python scripts/train.py train.beta_uniform=50

# Change the latent dimension d
python scripts/train.py model.latent_dim=32

# Swap to the RGB-D backbone
python scripts/train.py model=resnet

# Longer run, bigger batch
python scripts/train.py train.epochs=200 data.batch_size=512
```

Outputs land in `outputs/YYYY-MM-DD/HH-MM-SS/`:

```
best.pth            # lowest validation loss
last.pth            # for resuming
calibration.json    # q_hat, alpha, and the ACI state
report.json         # the evaluation above, machine-readable
```

Checkpoints carry the **RNG state and the git SHA**, so a resumed run is bit-exact and every number
is traceable to the commit that produced it.

---

## 4. Multi-GPU

```bash
./scripts/launch_ddp.sh                        # all visible GPUs
./scripts/launch_ddp.sh 2 train.epochs=100     # 2 GPUs, with overrides
```

The trainer wraps the encoder and head in `DistributedDataParallel`, shards the training fold with a
`DistributedSampler`, and **all-reduces the validation metrics** so rank 0 does not report only its
own shard. Without the launcher, `scripts/train.py` runs single-GPU — the same script works both ways.

---

## 5. Calibration and the certificate

Calibration happens automatically at the end of `train.py`. To do it standalone:

```bash
python scripts/evaluate.py checkpoint=outputs/.../best.pth
```

**The calibration fold must be disjoint from the training fold.** This is Assumption A5, and it is not
advisory: calibrating on training data produces a certificate that is invalid *but still prints a
plausible number*. `ConformalCalibrator.fit` fingerprints the fold and raises if it collides.

To tighten or loosen the guarantee:

```bash
python scripts/train.py calibration.alpha=0.05   # 95% coverage; more abstention
python scripts/train.py calibration.alpha=0.20   # 80% coverage; less abstention
```

Under distribution shift, enable Adaptive Conformal Inference:

```bash
python scripts/train.py calibration.gamma=0.05   # 0 disables adaptation
```

ACI maintains long-run coverage **without** exchangeability, by adjusting α online against a *fixed*
target. It is the right tool when the deployment distribution drifts away from the calibration fold.

---

## 6. Deployment: Algorithm 2

```bash
python scripts/deploy.py checkpoint=outputs/.../best.pth episodes=50
```

The loop is:

1. **Observe** → encode to a belief `q(z|o)`.
2. **Score** K posterior samples against Nₐ candidate actions → `s(o,a)`, `v(o,a)` (Eq 13–14).
3. **Sense?** If ambiguity `U(o) = E_a[v]` exceeds `τ_U`, acquire a new view (Eq 16).
4. **Certify** → build `A_cert` (Eq 24).
5. **Act, or ABSTAIN** if `A_cert` is empty.
6. **Adapt** on failure (Eq 19–20).

In code, the decision is a **sum type**, and this is the single most important API in the framework:

```python
from msp.types import Abstain, ActionChoice

decision = engine.select(scored)

if isinstance(decision, Abstain):
    robot.stop()                 # nothing was certified. Do NOT act.
else:
    assert isinstance(decision, ActionChoice)
    robot.execute(decision.action)
```

You cannot get an action index out of an abstention. That is deliberate: the previous implementation
masked uncertified scores with `-inf`, let `argmax` return index 0, and **silently executed an
uncertified grasp** whenever the certified set was empty.

---

## 7. Test-time adaptation

After a probe returns an outcome, fold it into the belief:

```python
from msp.inference import TTAConfig, adapt_belief

belief = adapt_belief(
    belief, head, probe_action, probe_outcome,
    TTAConfig(steps=20, num_samples=32, trust_radius=1.0),
)
```

Three things to know.

- **`trust_radius` is a hard bound on `‖μ' − μ‖`**, projected after every step. It is a *constraint*,
  not a penalty weight.
- **`kl_weight` must stay at 1.0.** The stationary point of `min E_q[−log p] + λ·KL(q‖q₀)` is
  `q* ∝ q₀·p^(1/λ)`. Only λ = 1 gives the Bayes update of Eq 19. Setting λ = 0.1 does not "restrain"
  the update — it raises the probe likelihood to the *tenth power*.
- **The full outcome is assimilated**, not just success. Margin and slip carry two thirds of the
  probe's information.

---

## 8. Identifiability: measuring ker J(x)

This is contribution C2, and it is the result the T-RO review identifies as decisive.

```python
from msp.diagnostics import analyze
from msp.oracle import AnalyticGraspOracle

oracle = AnalyticGraspOracle(shape="cylinder")
state = oracle.sample_states(1)[0]
actions = oracle.sample_actions(state.unsqueeze(0), 64)

report = analyze(oracle, state, actions)
print(report.null_dim)          # dim of the manipulation-indistinguishable subspace
print(report.max_angle_deg)     # predicted ker J vs the MEASURED invariant subspace
```

`measure_invariant_subspace` uses **no gradients** — it recovers `ker J` by fitting the quadratic
form `‖Ju‖² = uᵀGu` from black-box outcome queries. That independence is what makes the comparison
evidence rather than a tautology.

Try `shape="box"` and you will find `null_dim = 0`: for a generic box, **every state direction
changes some outcome, so Theorem 4 is vacuous.** Use a cylinder, where rotation about the symmetry
axis leaves every grasp outcome invariant and `null_dim = 1`. This is worth internalizing — the
theorem has content only for objects with a genuine outcome-invariance.

---

## 9. Producing the paper: which script makes which figure and table

**Everything the manuscript quotes is generated. Nothing is typed by hand.** `run_libero.py` writes
JSON (and the raw scores); the figure and table scripts read *only* those files. So a figure can be
restyled without retraining, and the paper cannot silently drift from the experiments.

### The whole pipeline

```bash
# 0. GPU-1 is often busy. Check first, and pin to a free card.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 1. Run the experiments. Writes results/libero/*.json  (hours; trains one model per experiment)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/paper/run_libero.py --out results/libero

# 2. LaTeX tables -> paper/tables/*.tex   (seconds, CPU, no GPU)
python scripts/paper/make_tables.py  --results results/libero --out paper/tables

# 3. PDF figures  -> paper/figures/*.pdf (seconds, CPU, no GPU)
python scripts/paper/make_figures.py --results results/libero --out paper/figures
```

`MUJOCO_GL=egl` is **required** (headless box). Steps 2 and 3 need `pip install -e ".[plot]"`.

Run a single experiment with `--only`, e.g. `--only l1 l6`. Smoke-test the whole thing in ~2 minutes
with `--quick` (tiny corpus, 3 epochs — the numbers are meaningless, the plumbing is not).

### What each experiment produces

| Experiment | Writes | Feeds |
|---|---|---|
| **L1** the decisive one | `l1_proxy_vs_msp.json`, `l1_scores.pt` | `tab_proxy_vs_msp`, `tab_geometry_split`, `tab_selective`, `fig_within_scene`, `fig_geometry_split`, `fig_risk_coverage`, `fig_per_object` |
| **L2** coverage / abstention | `l2_coverage.json` | `fig_coverage` |
| **L3** identifiability (Thm 4) | `l3_identifiability.json` | `tab_identifiability`, `fig_identifiability` |
| **L4** active perception | `l4_active.json` | `tab_active` |
| **L5** rate–distortion frontier | `l5_frontier.json` | `tab_frontier`, `fig_frontier` |
| **L6** ablations | `l6_ablations.json` | `tab_libero_ablations`, `fig_ablations` |

`l1_scores.pt` carries the raw per-grasp scores, so every L1 figure can be regenerated **without
retraining**. If you restyle a plot, you do not re-run a sweep.

### The four figures to actually put in the paper

| Figure | What it shows | Why it is the one |
|---|---|---|
| `fig_risk_coverage` | act rate vs. fraction of executed grasps that lifted | **Lead with this.** MSP starts at 1.00 and declines; Ferrari-Canny starts *below* the random-grasp line and only crosses it near a 0.68 act rate. Its confidence is anti-correlated with success. |
| `fig_within_scene` | within-scene AUC, with chance and the oracle ceiling drawn | The honest headline. Ranking grasps *of one object*, which object recognition cannot fake. |
| `fig_geometry_split` | box-shaped vs curved objects | The falsifiable form of the claim, with a built-in control group. |
| `fig_per_object` | per-object AUC over the base rates | Makes the pooled-AUC trap visible: base rates run 0.04 → 0.999. |

### Tables

`tab_proxy_vs_msp` (main), `tab_geometry_split` (control group), `tab_selective` (deployment),
`tab_libero_ablations`. All use `booktabs`, so put `\usepackage{booktabs}` in the Overleaf preamble
and `\input{tables/tab_proxy_vs_msp}`.

### Ablations (L6)

Every ablation is refereed by **within-scene AUC**, not pooled — see §11. They are: `full`,
`uniform beta` (the headline: what happens with no sufficiency budget), `z ablated` (no perception at
all — reviewer attack 21), `K=1 posterior sample`, and `latent dim 16 / 32`.

---

## 10. Extending the framework

### A new physics oracle

`M` defines the estimand: "manipulation-sufficient" means sufficient *for M*. Subclass `PhysicsOracle`:

```python
class MyOracle(PhysicsOracle):
    differentiable = True          # can outcome_params() be autodiffed in x?

    def query(self, state, actions) -> Outcome: ...
    def outcome_params(self, state, actions) -> Tensor: ...   # Phi(x), Def 9.1
    @property
    def state_dim(self) -> int: ...
```

Make `outcome_params` **differentiable in the state** if you possibly can. `J(x)` is obtained by
autodiff through it, and the finite-difference fallback in float32 is barely better than noise.

### A new posterior family

`DiagonalGaussianBelief` is *unimodal*, and Theorem 5 requires the belief on a symmetric object to be
supported on a **group orbit**. Subclass `Belief` to fix that. Note the interface exposes `rsample()`
and deliberately **no `.mean()`** — evaluating the outcome head at the belief's mean is the bug the
abstraction exists to prevent.

### A new backbone

Subclass `Backbone` with the contract `(B, ...) -> (B, output_dim)` and register it in
`msp.build.build_models`. There is no string registry: the previous one had six registries that were
all *empty at import*, so config-driven instantiation raised `KeyError`.

---

## 11. Gotchas that will cost you a day

**Never report a pooled AUC on its own — report the within-scene AUC.** The per-object base success
rates on this corpus run from **0.043** (macaroni) to **0.999** (bbq_sauce), so a model that has
learned nothing except *"which grocery is this?"* scores a pooled AUC of ~0.72 while being unable to
rank two grasps of the same object. MSP did exactly that. The tell was **Simpson's paradox**: the
pooled figure (0.716) *exceeded both strata it was computed from* (0.691 box, 0.400 curved). Whenever
the pooled number beats every subgroup, the between-group variation is doing the work. Use
`diagnostics.within_scene_auc`; the ceiling is **0.685** (an MLP handed the *true* state).

**`BetaSchedule.uniform()` will quietly break the model.** `D_succ`, `D_margin` and `D_slip` are
log-likelihoods on incommensurate scales (a BCE ≈ 0.6 nats; a zero-inflated log-normal ≈ 3.9; a
Gaussian NLL). An equal β is *not* an equal budget — capacity goes to whichever likelihood is
accidentally largest, and `succ`, the only outcome Eq 13 marginalises and Eq 24 certifies, is starved.
The head then stops attending to the action entirely (prediction std across a scene's grasps: 0.002,
against 0.415 in the truth). Worse, on LIBERO `margin` **is** the Ferrari-Canny ε on the bounding box
— the proxy the paper discredits — so a uniform β spends the belief's capacity becoming sufficient
for exactly the quantity we prove is uninformative. Use `BetaSchedule.sufficiency_for_success`.

**An AUC needs a scene with both a success and a failure.** Three of the thirteen objects
(bbq_sauce and the near-saturated bottles) succeed ~99% of the time, so almost no scene of theirs is
*rankable*. An unguarded within-scene AUC over the two or three that remain is noise, and it plots as
a near-zero point that reads as catastrophic failure. `fig_per_object` requires ≥15 rankable scenes
and labels the rest "no rankable scenes", which is the honest thing to draw.

**The action is a grasp point in the WORLD frame.** `sample_actions` returns `t_obj + lift`. The
simulator must not add the object's position again. It did, once: the gripper was displaced by the
object's entire world position, the nominal grasp succeeded 6% of the time instead of 99.6%, and the
analytic tier — which plans correctly — was scoring one grasp while the simulator executed another.
That is precisely how a grasp metric measures as "uninformative" when it is nothing of the kind.
Cheapest detector: **success must RISE as the proposal tightens** (`spread` → 0). If it falls, the
grasp is being aimed at the wrong place.

**The corpus cache key hashes the physics source files.** Change `libero_sim.py`, `analytic.py` or
`libero_assets.py` and every cached corpus is invalidated automatically. It used to key on the spec
alone, which meant a physics fix would silently reload the corpus built under the *old* physics and
reproduce the old numbers perfectly.

**Report active perception as 17.5%, not 43%.** The naive max-over-viewpoints number is the winner's
curse. See `diagnostics/active_eval.py`.

**β multiplies distortion.** `L = Σ_j β_j·D_j + R`.

```
beta -> inf  =>  sufficiency   (Theorem 3)
beta -> 0    =>  compression
```

The audited implementation computed `L = Σ_j D_j/β_j + R`, which inverts the meaning of the paper's
central hyperparameter. If you "fix" it back, every frontier you plot will be the mirror image of the
truth. There is a test that will stop you: `test_regression_beta_direction_matches_theorem_3`.

**Coverage ≠ certified precision.** Theorem 7 is about `P(succ ∈ C(o,a))` over *all* test points.
`P(succ=1 | certified)` conditions on selection, destroys exchangeability, and has no reason to
converge to `1−α`. Both are reported, under their own names. Do not put the second one in a table
labelled "coverage".

**The certified set is a singleton test.** `C(o,a) == {1}` requires label 1 *in* and label 0 *out*.
Testing only `s ≥ 1−q̂` certifies the ambiguous set `{0,1}` — at a realistic `q̂ ≈ 0.68` that is
every action from `s = 0.32` to `s = 0.68`, a coin flip included.

**The loss is computed in fp32 even when the network runs in bf16.** The rate and distortion are
plotted on the frontier, and bf16 carries 2–3 decimal digits. Training in reduced precision is fine;
*measuring* in it is not.

**`SyntheticOracle` is not physics.** It is a world with a *known answer*, for validating the
diagnostics before they are trusted. Use `AnalyticGraspOracle` for grasp mechanics.

**A parallel-jaw grasp with point contacts is not force-closed in 6D.** You need torsional friction
(the soft-finger model). Without it, ε is negative for essentially every grasp, the success label
collapses to a constant 0, and Assumption A8 fails. `AnalyticGraspOracle` includes it.
