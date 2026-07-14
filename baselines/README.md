# Baselines

Third-party code is **cloned, never vendored** (`baselines/*/` is git-ignored). Two reasons:

1. **Licence.** AnyDexGrasp is CC BY-NC 4.0 (non-commercial). Committing it into this tree would
   pull a non-commercial restriction into a repository we intend to publish next to the paper.
2. **Reproducibility beats vendoring.** `fetch.sh` pins the exact commit, which is what a reviewer
   actually needs.

```bash
./baselines/fetch.sh          # clone every baseline at its pinned commit
```

---

## What the paper needs a baseline to DO

The claim under test is narrow, and it decides which baselines are relevant:

> An analytic grasp score computed on a **reconstructed geometry** is uninformative about whether
> the object is actually lifted. A belief trained on **outcomes** is informative.

So a useful baseline is *any grasp scorer whose supervision is geometry rather than outcome*. It
does not have to be a dexterous-hand system, and it does not have to be state of the art. It has to
sit on the other side of that one line.

That yields a ladder, cheapest first:

| Baseline | Supervision | Status | Cost |
|---|---|---|---|
| **B0** Ferrari-Canny epsilon on the OBB | analytic, geometric | **done** (`AnalyticGraspOracle`) | free |
| **B1** Same net, trained on epsilon instead of outcomes | learned, geometric | **recommended next** | ~1 GPU-hour |
| **B2** Contact-GraspNet / GraspNet-baseline | learned, geometric (sim labels) | feasible | days |
| **B3** AnyDexGrasp | learned, geometric | see below | high risk |

**B1 is the controlled experiment and it is the one I would run.** Take the *identical* encoder,
the *identical* head, the *identical* data, and change exactly one thing: regress Ferrari-Canny
epsilon instead of the observed outcome. Then score both against the real lift. Same capacity, same
optimiser, same images. The only difference is the supervision signal, so any gap is attributable to
the signal and to nothing else. B2 and B3 confound the comparison with architecture, training set,
input modality and resolution all at once, and a reviewer will say so.

---

## AnyDexGrasp — read this before investing time

`https://github.com/graspnet/AnyDexGrasp` @ `c9c4a43`. CC BY-NC 4.0.

It is a strong system and a genuinely useful **code reference**: the representation-model /
decision-model split (a GraspNet-style geometric grasp proposer, then a learned module that maps a
proposal to a hand configuration) is a clean architecture, and `models/minkowski_graspnet*.py` is
worth reading.

As an **executable baseline for this paper**, be aware of four obstacles, in descending severity:

1. **MinkowskiEngine will very likely not build on your GPUs.** It requires MinkowskiEngine v0.5,
   PyTorch 1.13, CUDA 11.7, Python 3.8. It is effectively unmaintained (last release 2021) and does
   not support recent architectures. Your machine has **RTX Pro 6000 Blackwell (sm_120)**. Expect the
   CUDA extension compile to fail outright. This is the blocker to test first, before anything else
   — an afternoon, not a week.
2. **Wrong end effector.** AnyDexGrasp outputs *multifinger dexterous* grasps (Inspire / DH3 /
   Allegro). Our oracle executes a **parallel-jaw** grasp with a 7-D action. Only the upstream
   *representation* model is parallel-jaw; the dexterous decision model has no counterpart in our
   action space and would have to be discarded, which is most of the paper.
3. **Data.** Needs GraspNet-1Billion (hundreds of GB) plus weights from a Google Drive folder.
4. **Domain.** Trained on real RealSense point clouds. We render MuJoCo depth at 96x96. A fair
   comparison needs depth re-rendered at realistic resolution with correct intrinsics, and the
   sim-to-real depth gap then confounds the very thing we are trying to measure.

**Recommendation.** Keep it as a code reference and cite it. If you want a *learned* geometric
baseline in the paper, B1 gives you a rigorous one this week; B2 (Contact-GraspNet, ordinary PyTorch,
parallel-jaw) is the sane external option if a reviewer demands a published method. Reach for B3
only if a reviewer names it specifically, and test obstacle (1) before promising anything.
