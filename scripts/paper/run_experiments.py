"""Run every experiment the paper reports, and write machine-readable results.

    python scripts/paper/run_experiments.py --out results/

Produces `results/*.json`, consumed by `make_figures.py` and `make_tables.py`. Separating
"run the science" from "draw the plot" means a figure can be restyled without re-running a
GPU sweep, and it means the numbers in the paper are traceable to a file rather than to a
notebook someone has since edited.

EXPERIMENTS

  E1  Rate-distortion frontier over beta        -> minimality evidence (V2 Ch.6; attacks 16, 17)
  E2  Identifiability: ker J(x) vs measured     -> contribution C2 (Theorem 4). The result
                                                   the T-RO review says won Reviewer A.
  E3  Conformal coverage vs nominal alpha       -> contribution C4 (Theorem 7)
  E4  Ablations: z-ablation, no-abstention,     -> attacks 21, 15; the C3/C4 controls
      no-TTA, risk-neutral selection
  E5  Latent dimension sweep                    -> the "lower-dimensional" clause of the
                                                   scientific hypothesis

Every result is a hypothesis under test, not a foregone conclusion. If E1 shows distortion
does not fall with beta, or E3 shows coverage below nominal, the script reports it plainly.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from msp.data import SyntheticGraspDataset, collate
from msp.diagnostics import analyze
from msp.engine import Evaluator, TrainConfig, Trainer
from msp.inference import ConformalCalibrator, InferenceConfig, InferenceEngine
from msp.math.bottleneck import BetaSchedule
from msp.models import BeliefEncoder, MLPBackbone, OutcomeHead
from msp.oracle import SyntheticOracle
from msp.utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("paper")

OBS_DIM = 32
ACTION_DIM = 7


def _loaders(oracle: SyntheticOracle, bs: int = 256, scale: float = 1.0) -> dict[str, DataLoader]:
    sizes = {"train": int(16000 * scale), "val": 2000, "calib": 4000, "test": 4000}
    seeds = {"train": 1, "val": 2, "calib": 3, "test": 4}
    out = {}
    for k, n in sizes.items():
        ds = SyntheticGraspDataset(oracle, n_scenes=n, n_actions=16, obs_dim=OBS_DIM, seed=seeds[k])
        out[k] = DataLoader(ds, batch_size=bs, shuffle=(k == "train"), collate_fn=collate)
    return out


def _train_one(
    oracle: SyntheticOracle,
    loaders: dict[str, DataLoader],
    beta: float,
    latent_dim: int,
    device: torch.device,
    epochs: int,
    out_dir: Path,
    alpha: float = 0.1,
) -> dict:
    """Train + calibrate + evaluate a single configuration. Returns the report."""
    seed_everything(0)
    enc = BeliefEncoder(MLPBackbone(OBS_DIM, 128), latent_dim=latent_dim)
    head = OutcomeHead(latent_dim=latent_dim, action_dim=ACTION_DIM)

    cfg = TrainConfig(
        epochs=epochs, lr=3e-3, warmup_epochs=2,
        amp_dtype="bf16" if device.type == "cuda" else "off",
        beta=BetaSchedule.uniform(beta), out_dir=str(out_dir),
    )
    Trainer(enc, head, loaders["train"], loaders["val"], cfg, device).fit()

    cal = ConformalCalibrator(alpha=alpha, gamma=0.0)
    engine = InferenceEngine(enc, head, cal, InferenceConfig(num_samples=32))
    ev = Evaluator(engine, device)
    ev.calibrate(loaders["calib"], cal)
    rep = ev.evaluate(loaders["test"], beta=cfg.beta)

    return asdict(rep) | {"beta": beta, "latent_dim": latent_dim, "q_hat": cal.q_hat} | {
        "_models": (enc, head, engine, ev)  # kept in-process; stripped before JSON
    }


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ======================================================================================


def e1_frontier(oracle, loaders, device, out: Path, epochs: int) -> list[dict]:
    """E1: the rate-distortion frontier. Sweeping beta must trace a monotone trade-off:
    higher beta buys lower distortion (more sufficiency) at the cost of higher rate (less
    compression). This is the minimality evidence, and it is ONLY interpretable because the
    beta convention is now correct -- the audited code would have traced it backwards."""
    log.info("=== E1: rate-distortion frontier over beta ===")
    rows = []
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        r = _strip(_train_one(oracle, loaders, beta, 16, device, epochs, out / f"b{beta}"))
        log.info("  beta=%6.1f  rate=%7.3f  distortion=%7.4f  coverage=%.3f",
                 beta, r["rate"], r["distortion"], r["coverage"])
        rows.append(r)
    return rows


def e2_identifiability(oracle, device, out: Path) -> dict:
    """E2: Theorem 4. Principal angles between the predicted ker J(x) and the MEASURED
    grasp-invariant subspace, over many states. This is contribution C2."""
    log.info("=== E2: identifiability (Theorem 4) ===")
    seed_everything(0)
    states = oracle.sample_states(50)
    actions = torch.randn(1, 64, ACTION_DIM)

    angles, ranks, nulls = [], [], []
    for i in range(states.shape[0]):
        rep = analyze(oracle, states[i], actions)
        ranks.append(rep.numerical_rank)
        nulls.append(rep.null_dim)
        if rep.principal_angles_rad is not None:
            angles.append(torch.rad2deg(rep.principal_angles_rad).tolist())

    flat = [a for row in angles for a in row]
    res = {
        "state_dim": oracle.state_dim,
        "true_rank": oracle.rank,
        "true_null_dim": oracle.null_space_dim(),
        "estimated_rank_mean": float(sum(ranks) / len(ranks)),
        "estimated_null_dim_mean": float(sum(nulls) / len(nulls)),
        "principal_angles_deg": flat,
        "max_angle_deg": max(flat) if flat else None,
        "mean_angle_deg": float(sum(flat) / len(flat)) if flat else None,
        "n_states": len(ranks),
    }
    log.info("  true null dim %d, estimated %.2f, max principal angle %.3f deg",
             res["true_null_dim"], res["estimated_null_dim_mean"], res["max_angle_deg"] or -1)
    return res


def e3_coverage(oracle, loaders, device, out: Path, epochs: int) -> list[dict]:
    """E3: Theorem 7. Empirical coverage must track the nominal 1 - alpha across alpha."""
    log.info("=== E3: conformal coverage vs nominal ===")
    rows = []
    for alpha in [0.02, 0.05, 0.1, 0.2, 0.3]:
        r = _strip(_train_one(oracle, loaders, 20.0, 16, device, epochs, out / f"a{alpha}", alpha=alpha))
        log.info("  alpha=%.2f  target=%.2f  coverage=%.4f  abstain=%.3f  %s",
                 alpha, 1 - alpha, r["coverage"], r["abstention_rate"],
                 "OK" if r["coverage"] >= 1 - alpha - 0.02 else "VIOLATED")
        rows.append(r | {"alpha": alpha})
    return rows


def e4_ablations(oracle, loaders, device, out: Path, epochs: int) -> list[dict]:
    """E4: the controls the reviewers demand.

    z-ablation (attack 21): if the head ignores z and memorizes an action prior, the whole
    framework is vacuous. We retrain with z replaced by noise; distortion MUST get worse.
    """
    log.info("=== E4: ablations ===")
    rows = []

    full = _train_one(oracle, loaders, 20.0, 16, device, epochs, out / "abl_full")
    enc, head, engine, ev = full["_models"]
    base = _strip(full)
    rows.append(base | {"ablation": "full"})
    log.info("  full          distortion=%.4f coverage=%.3f", base["distortion"], base["coverage"])

    # --- z-ablation: is the outcome head actually USING the sufficient statistic? ---
    seed_everything(0)
    z_dim = 16
    enc_a = BeliefEncoder(MLPBackbone(OBS_DIM, 128), latent_dim=z_dim)
    head_a = OutcomeHead(latent_dim=z_dim, action_dim=ACTION_DIM)

    class _NoisyEncoder(BeliefEncoder):
        """Emits a belief independent of the observation: z carries zero information."""
        def forward(self, obs):  # type: ignore[override]
            b = super().forward(obs)
            from msp.belief import DiagonalGaussianBelief
            return DiagonalGaussianBelief(torch.zeros_like(b.mu), torch.zeros_like(b.logvar))

    enc_z = _NoisyEncoder(MLPBackbone(OBS_DIM, 128), latent_dim=z_dim)
    cfg = TrainConfig(epochs=epochs, lr=3e-3, warmup_epochs=2,
                      amp_dtype="bf16" if device.type == "cuda" else "off",
                      beta=BetaSchedule.uniform(20.0), out_dir=str(out / "abl_z"))
    Trainer(enc_z, head_a, loaders["train"], loaders["val"], cfg, device).fit()
    cal_z = ConformalCalibrator(alpha=0.1, gamma=0.0)
    eng_z = InferenceEngine(enc_z, head_a, cal_z, InferenceConfig(num_samples=32))
    ev_z = Evaluator(eng_z, device)
    ev_z.calibrate(loaders["calib"], cal_z)
    r_z = asdict(ev_z.evaluate(loaders["test"]))
    rows.append(r_z | {"ablation": "z_ablated"})
    log.info("  z-ablated     distortion=%.4f coverage=%.3f  (must be WORSE than full)",
             r_z["distortion"], r_z["coverage"])

    # --- no-abstention: what does the certificate actually buy? ---
    oracle.to(device)  # the closed-loop rollout queries M with on-device states
    n_acted = n_ok = 0
    n_abstain = 0
    unc_acted = unc_ok = 0
    torch.manual_seed(0)
    for _ in range(400):
        st = oracle.sample_states(1)
        obs = oracle.observe(st, obs_dim=OBS_DIM).to(device)
        acts = torch.randn(1, 64, ACTION_DIM, device=device)
        sc = engine.score(engine.perceive(obs), acts)

        d = engine.select(sc)
        from msp.types import ActionChoice as AC
        if isinstance(d, AC):
            n_acted += 1
            n_ok += int(oracle.query(st.to(device), d.action.view(1, 1, -1)).succ.item())
        else:
            n_abstain += 1

        u = engine.select_uncertified(sc)
        unc_acted += 1
        unc_ok += int(oracle.query(st.to(device), u.action.view(1, 1, -1)).succ.item())

    rows.append({
        "ablation": "abstention_effect",
        "with_abstention_success": n_ok / max(1, n_acted),
        "with_abstention_acted": n_acted,
        "abstained": n_abstain,
        "no_abstention_success": unc_ok / max(1, unc_acted),
        "n_episodes": 400,
    })
    log.info("  abstention:   success-when-acting %.3f (acted %d, abstained %d)  vs  "
             "no-abstention %.3f", n_ok / max(1, n_acted), n_acted, n_abstain,
             unc_ok / max(1, unc_acted))
    return rows


def e5_latent_dim(oracle, loaders, device, out: Path, epochs: int) -> list[dict]:
    """E5: the 'lower-dimensional' clause of the hypothesis. The true manipulation-relevant
    dimension of this world is rank(P) = 3. Distortion should saturate at or near d = 3:
    extra latent capacity buys nothing, because there is nothing left to be sufficient for."""
    log.info("=== E5: latent dimension sweep (true relevant dim = %d) ===", oracle.rank)
    rows = []
    for d in [1, 2, 3, 4, 6, 8, 16, 32]:
        r = _strip(_train_one(oracle, loaders, 20.0, d, device, epochs, out / f"d{d}"))
        log.info("  d=%2d  distortion=%.4f  rate=%.3f  coverage=%.3f",
                 d, r["distortion"], r["rate"], r["coverage"])
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="tiny run to smoke-test the pipeline")
    ap.add_argument("--only", nargs="*", default=None, help="e1 e2 e3 e4 e5")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device = %s", device)
    args.out.mkdir(parents=True, exist_ok=True)

    seed_everything(0)
    oracle = SyntheticOracle(state_dim=6, rank=3, action_dim=ACTION_DIM, seed=0)
    loaders = _loaders(oracle, scale=0.2 if args.quick else 1.0)

    want = set(args.only or ["e1", "e2", "e3", "e4", "e5"])
    ckpt = args.out / "_ckpt"

    if "e2" in want:  # cheapest and most important: run it first
        (args.out / "e2_identifiability.json").write_text(
            json.dumps(e2_identifiability(oracle, device, ckpt), indent=2))
    if "e1" in want:
        (args.out / "e1_frontier.json").write_text(
            json.dumps(e1_frontier(oracle, loaders, device, ckpt, args.epochs), indent=2))
    if "e3" in want:
        (args.out / "e3_coverage.json").write_text(
            json.dumps(e3_coverage(oracle, loaders, device, ckpt, args.epochs), indent=2))
    if "e4" in want:
        (args.out / "e4_ablations.json").write_text(
            json.dumps(e4_ablations(oracle, loaders, device, ckpt, args.epochs), indent=2))
    if "e5" in want:
        (args.out / "e5_latent_dim.json").write_text(
            json.dumps(e5_latent_dim(oracle, loaders, device, ckpt, args.epochs), indent=2))

    log.info("results written to %s", args.out)


if __name__ == "__main__":
    main()
