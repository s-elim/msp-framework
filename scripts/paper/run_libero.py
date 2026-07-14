"""THE PAPER'S EXPERIMENTS, on real LIBERO objects. Run this; it writes results/*.json.

    MUJOCO_GL=egl python scripts/paper/run_libero.py --out results/libero
    python scripts/paper/make_tables.py  --results results/libero
    python scripts/paper/make_figures.py --results results/libero

Every number the manuscript quotes comes out of this script, into JSON, and from there into the
LaTeX. Nothing is typed by hand, so the paper and the experiments cannot silently drift apart.

THE EXPERIMENTS, and what each one is for.

  L1  THE DECISIVE ONE.  Does Ferrari-Canny epsilon, computed on a RECONSTRUCTED geometry (an
      oriented bounding box, roughly what a pose-and-shape pipeline delivers), predict whether the
      object is actually lifted -- and does MSP, trained on OUTCOMES, do better?

      This replaces the sim-to-real study. A simulation-only paper cannot claim "my M resembles
      reality", and does not need to: with no hardware, M IS the ground truth and sufficiency with
      respect to it is true by construction. The gap that remains, and that the paper is actually
      about, is between the geometry a perception system RECONSTRUCTS and the physics that happens.
      On a parametric box that gap is identically zero. On a scanned ketchup bottle it is not.

  L2  Calibrated abstention (Theorem 7) on real objects.
  L3  Identifiability (Theorem 4): ker J(x) vs the measured grasp-invariant subspace.
  L4  Active perception (Eq 16-18), BIAS-CORRECTED -- see diagnostics/active_eval.py. The naive
      max-over-viewpoints number is inflated by the winner's curse by ~25 points.
  L5  Rate-distortion frontier over beta (the minimality evidence).
  L6  Ablations: z-ablation, and what abstention actually buys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MUJOCO_GL", "egl")

from msp.data import LiberoCorpusSpec, LiberoGraspDataset, collate, generate_libero_corpus  # noqa: E402
from msp.diagnostics import analyze, compare_predictors, evaluate_active_perception  # noqa: E402
from msp.engine import Evaluator, TrainConfig, Trainer  # noqa: E402
from msp.inference import ConformalCalibrator, InferenceConfig, InferenceEngine  # noqa: E402
from msp.math.bottleneck import BetaSchedule  # noqa: E402
from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.oracle import AnalyticGraspOracle, LiberoGraspOracle  # noqa: E402
from msp.utils import seed_everything  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("libero")

ACTION_DIM = 7


# ======================================================================================
# shared
# ======================================================================================


def build_corpora(out: Path, n_train: int, n_views: int, img: int) -> dict[str, LiberoGraspDataset]:
    cache = out / "corpus"
    sizes = {"train": n_train, "val": max(500, n_train // 8), "calib": max(1000, n_train // 4),
             "test": max(1000, n_train // 4)}
    seeds = {"train": 1, "val": 2, "calib": 3, "test": 4}
    ds = {}
    for k, n in sizes.items():
        p = generate_libero_corpus(
            LiberoCorpusSpec(n_scenes=n, n_actions=8, image_size=img, n_views=n_views,
                             seed=seeds[k]),
            cache,
        )
        ds[k] = LiberoGraspDataset(p, max_train_views=3 if (k == "train" and n_views > 1) else 1)
    return ds


def loaders(ds: dict[str, LiberoGraspDataset], bs: int = 128) -> dict[str, DataLoader]:
    return {
        k: DataLoader(v, batch_size=bs, shuffle=(k == "train"), collate_fn=collate)
        for k, v in ds.items()
    }


def train_model(ds, dls, device, epochs: int, beta: float, latent: int, out: Path):
    seed_everything(0)
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=latent)
    head = OutcomeHead(latent_dim=latent, action_dim=ACTION_DIM)
    cfg = TrainConfig(
        epochs=epochs, lr=5e-4, warmup_epochs=3,
        amp_dtype="bf16" if device.type == "cuda" else "off",
        beta=BetaSchedule.uniform(beta), out_dir=str(out),
    )
    Trainer(enc, head, dls["train"], dls["val"], cfg, device).fit()
    enc.eval()
    head.eval()
    return enc, head


# ======================================================================================
# L1 -- THE DECISIVE EXPERIMENT
# ======================================================================================


def l1_proxy_vs_msp(ds, dls, device, out: Path, epochs: int) -> dict:
    """Does the analytic proxy predict real lift outcomes? Does MSP?"""
    log.info("=== L1: analytic proxy vs MSP, on the REAL lift outcome ===")
    enc, head = train_model(ds, dls, device, epochs, beta=30.0, latent=64, out=out / "l1")

    test: LiberoGraspDataset = ds["test"]
    n = len(test)

    # MSP's own prediction: s(o, a), the belief-averaged success probability of Eq 13.
    s_all = []
    with torch.no_grad():
        for i in range(0, n, 128):
            obs = test.obs[i : i + 128].to(device)
            acts = test.actions[i : i + 128].to(device)
            belief = enc(obs)
            probs = head.success_probs(belief.rsample(32), acts)  # (b, K, Na)
            s_all.append(probs.mean(dim=1).cpu())
    s_msp = torch.cat(s_all)  # (N, Na)

    # THE NATURAL EXPERIMENT. Six of the thirteen groceries have a SINGLE-box collision hull --
    # they simply ARE boxes, so the bounding-box reconstruction is exact and there is no
    # hallucinated surface. The rest are cans, bottles and cartons. The thesis predicts the proxy
    # works on the former and fails on the latter; stratifying is what makes that falsifiable.
    sim = LiberoGraspOracle()
    boxlike_obj = torch.tensor([o.n_boxes == 1 for o in sim.objects])
    boxlike = boxlike_obj[test.object_index].unsqueeze(1).expand_as(test.succ.squeeze(-1))

    rep = compare_predictors(
        analytic_score=test.margin.squeeze(-1),  # Ferrari-Canny on the RECONSTRUCTION
        true_success=test.succ.squeeze(-1),  # the simulator, on the TRUE mesh
        executable=test.executable,
        msp_score=s_msp,
        boxlike=boxlike,
    )
    log.info("\n%s", rep.summary())
    return asdict(rep)


# ======================================================================================
# L2 -- calibrated abstention on real objects
# ======================================================================================


def l2_coverage(ds, dls, device, out: Path, epochs: int) -> list[dict]:
    log.info("=== L2: coverage and abstention on real objects (Theorem 7) ===")
    rows = []
    for alpha in (0.05, 0.10, 0.20):
        enc, head = train_model(ds, dls, device, epochs, 30.0, 64, out / f"l2_a{alpha}")
        cal = ConformalCalibrator(alpha=alpha, gamma=0.0)
        eng = InferenceEngine(enc, head, cal, InferenceConfig(num_samples=32))
        ev = Evaluator(eng, device)
        ev.calibrate(dls["calib"], cal)
        r = ev.evaluate(dls["test"])
        log.info("  alpha=%.2f target=%.2f coverage=%.4f abstain=%.3f  %s",
                 alpha, 1 - alpha, r.coverage, r.abstention_rate,
                 "OK" if r.coverage_holds() else "VIOLATED")
        rows.append(asdict(r) | {"alpha": alpha})
    return rows


# ======================================================================================
# L3 -- identifiability on real objects
# ======================================================================================


def l3_identifiability(out: Path) -> dict:
    """Theorem 4 on the analytic tier's geometry, per object.

    NOTE FOR THE PAPER. Phi -- and therefore J(x) -- comes from the ANALYTIC tier, because you
    cannot autodiff through mj_step. So the identifiability result is a statement about the
    analytic outcome map. L1 is what tells you how much that map has to do with reality, and the
    two must be reported together.
    """
    log.info("=== L3: identifiability (Theorem 4), per object ===")
    sim = LiberoGraspOracle()
    an = AnalyticGraspOracle(shape="box", noise=False)
    g = torch.Generator().manual_seed(0)

    rows = []
    for k in range(sim.n_objects):
        # J(x) is evaluated at ONE state, so the bounding box must be installed for exactly that
        # one scene -- not for a batch. (The oracle now raises a clear error if they disagree.)
        oi = torch.tensor([k])
        _, x = sim.sample_scenes(1, generator=g)
        an.set_base_half(sim.base_half(oi))
        a = an.sample_actions(x, 48, generator=g)
        rep = analyze(an, x[0], a, measure_empirical=True)
        rows.append({
            "object": sim.object_names()[k],
            "rank": rep.numerical_rank,
            "null_dim": rep.null_dim,
            "max_angle_deg": rep.max_angle_deg,
        })
        log.info("  %-22s rank J = %2d   dim ker J = %d   max angle = %s",
                 sim.object_names()[k], rep.numerical_rank, rep.null_dim,
                 f"{rep.max_angle_deg:.2f} deg" if rep.max_angle_deg is not None else "n/a")
    return {"state_dim": 14, "per_object": rows}


# ======================================================================================
# L4 -- active perception, bias-corrected
# ======================================================================================


def l4_active(ds, dls, device, out: Path, epochs: int) -> dict:
    log.info("=== L4: active perception, bias-corrected ===")
    if ds["train"].n_views < 2:
        log.warning("corpus has 1 view; regenerate with n_views=8. skipping L4.")
        return {}
    enc, head = train_model(ds, dls, device, epochs, 30.0, 64, out / "l4")
    test = ds["test"]
    m = min(512, len(test))
    idx = torch.arange(m)
    rep = evaluate_active_perception(
        enc, head, test.all_views(idx).to(device), test.actions[idx].to(device)
    )
    log.info("\n%s", rep.summary())
    return asdict(rep)


# ======================================================================================
# L5 / L6
# ======================================================================================


def l5_frontier(ds, dls, device, out: Path, epochs: int) -> list[dict]:
    log.info("=== L5: rate-distortion frontier over beta ===")
    rows = []
    for beta in (1.0, 3.0, 10.0, 30.0, 100.0):
        enc, head = train_model(ds, dls, device, epochs, beta, 64, out / f"l5_b{beta}")
        cal = ConformalCalibrator(alpha=0.1, gamma=0.0)
        eng = InferenceEngine(enc, head, cal, InferenceConfig(num_samples=32))
        ev = Evaluator(eng, device)
        ev.calibrate(dls["calib"], cal)
        r = ev.evaluate(dls["test"], beta=BetaSchedule.uniform(beta))
        log.info("  beta=%6.1f  rate=%7.3f  distortion=%8.4f +/- %.4f",
                 beta, r.rate, r.distortion, r.distortion_stderr)
        rows.append(asdict(r) | {"beta": beta})
    return rows


def l6_ablations(ds, dls, device, out: Path, epochs: int) -> list[dict]:
    """z-ablation (reviewer attack 21): if the head ignores z and memorizes an action prior, the
    framework is vacuous."""
    log.info("=== L6: ablations ===")
    from msp.belief import DiagonalGaussianBelief

    rows = []
    enc, head = train_model(ds, dls, device, epochs, 30.0, 64, out / "l6_full")
    cal = ConformalCalibrator(alpha=0.1, gamma=0.0)
    eng = InferenceEngine(enc, head, cal, InferenceConfig(num_samples=32))
    ev = Evaluator(eng, device)
    ev.calibrate(dls["calib"], cal)
    full = asdict(ev.evaluate(dls["test"]))
    rows.append(full | {"ablation": "full"})
    log.info("  full       D=%.4f  coverage=%.3f", full["distortion"], full["coverage"])

    class _ZAblated(BeliefEncoder):
        """Emits a belief independent of the observation: z carries zero information."""

        def forward(self, obs):  # type: ignore[override]
            b = super().forward(obs)
            return DiagonalGaussianBelief(torch.zeros_like(b.mu), torch.zeros_like(b.logvar))

    seed_everything(0)
    enc_z = _ZAblated(ResNetBackbone(output_dim=256, pretrained=True), latent_dim=64)
    head_z = OutcomeHead(latent_dim=64, action_dim=ACTION_DIM)
    cfg = TrainConfig(epochs=epochs, lr=5e-4, warmup_epochs=3,
                      amp_dtype="bf16" if device.type == "cuda" else "off",
                      beta=BetaSchedule.uniform(30.0), out_dir=str(out / "l6_z"))
    Trainer(enc_z, head_z, dls["train"], dls["val"], cfg, device).fit()
    enc_z.eval(); head_z.eval()
    cal_z = ConformalCalibrator(alpha=0.1, gamma=0.0)
    eng_z = InferenceEngine(enc_z, head_z, cal_z, InferenceConfig(num_samples=32))
    ev_z = Evaluator(eng_z, device)
    ev_z.calibrate(dls["calib"], cal_z)
    zab = asdict(ev_z.evaluate(dls["test"]))
    rows.append(zab | {"ablation": "z_ablated"})
    log.info("  z-ablated  D=%.4f  coverage=%.3f   (MUST be worse than full)",
             zab["distortion"], zab["coverage"])
    return rows


# ======================================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/libero"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--quick", action="store_true", help="tiny run to smoke-test the pipeline")
    ap.add_argument("--only", nargs="*", default=None, help="l1 l2 l3 l4 l5 l6")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.n_train, args.views = 6, 800, 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("device=%s  n_train=%d  views=%d  epochs=%d",
             device, args.n_train, args.views, args.epochs)

    seed_everything(0)
    ds = build_corpora(args.out, args.n_train, args.views, args.image_size)
    dls = loaders(ds)
    log.info("corpus: %d train | sim success %.3f | executable %.3f",
             len(ds["train"]), float(ds["train"].succ.mean()),
             float(ds["train"].executable.float().mean()))

    want = set(args.only or ["l1", "l2", "l3", "l4", "l5", "l6"])
    W = lambda name, obj: (args.out / name).write_text(json.dumps(obj, indent=2))  # noqa: E731

    if "l3" in want:  # cheapest, and needs no training
        W("l3_identifiability.json", l3_identifiability(args.out))
    if "l1" in want:  # the decisive one
        W("l1_proxy_vs_msp.json", l1_proxy_vs_msp(ds, dls, device, args.out, args.epochs))
    if "l2" in want:
        W("l2_coverage.json", l2_coverage(ds, dls, device, args.out, args.epochs))
    if "l4" in want:
        W("l4_active.json", l4_active(ds, dls, device, args.out, args.epochs))
    if "l5" in want:
        W("l5_frontier.json", l5_frontier(ds, dls, device, args.out, args.epochs))
    if "l6" in want:
        W("l6_ablations.json", l6_ablations(ds, dls, device, args.out, args.epochs))

    log.info("results -> %s", args.out)


if __name__ == "__main__":
    main()
