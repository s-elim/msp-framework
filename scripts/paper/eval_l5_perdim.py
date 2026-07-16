"""Annotate l5_frontier.json with PER-OUTCOME test distortions from the saved L5 checkpoints.

    MUJOCO_GL=egl python scripts/paper/eval_l5_perdim.py --results results/libero

Why this exists. The frontier's y-axis, the unweighted total distortion sum_j D_j, is
dominated by the margin/slip negative log-likelihoods, and the sufficiency-for-success
budget (beta_succ = beta >> beta_margin = beta_slip) deliberately sacrifices exactly those
two. So as beta rises the total distortion RISES, and the rate-distortion figure reads as an
anti-frontier. The minimality-for-success evidence is D_succ falling as the rate rises:
D_succ is the only outcome Eq 13 marginalizes and Eq 24 certifies.

This script measures D_succ / D_margin / D_slip on the test fold for every saved l5_b*
checkpoint, with the same K = 32 posterior samples the decision rule uses, and writes the
numbers back into l5_frontier.json. make_figures.py and make_tables.py pick them up when
present. Everything comes from checkpoints a run actually produced; nothing is invented.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from torch.utils.data import DataLoader  # noqa: E402

from msp.data import LiberoCorpusSpec, LiberoGraspDataset, collate, generate_libero_corpus  # noqa: E402
from msp.math.bottleneck import BetaSchedule, vib_objective  # noqa: E402
from msp.models import BeliefEncoder, OutcomeHead, ResNetBackbone  # noqa: E402
from msp.types import Outcome  # noqa: E402

ACTION_DIM = 7
K = 32  # same K the decision rule and the evaluator use


def test_loader(results: Path, n_train: int = 8000, views: int = 8, img: int = 96) -> DataLoader:
    """The SAME test corpus run_libero.py evaluated on (cache hit; nothing regenerates)."""
    p = generate_libero_corpus(
        LiberoCorpusSpec(n_scenes=max(1000, n_train // 4), n_actions=8, image_size=img,
                         n_views=views, seed=4),
        results / "corpus",
    )
    return DataLoader(LiberoGraspDataset(p, max_train_views=1), batch_size=128,
                      shuffle=False, collate_fn=collate)


@torch.no_grad()
def perdim_distortion(ckpt: Path, loader: DataLoader, beta: float, device) -> dict[str, float]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc = BeliefEncoder(ResNetBackbone(output_dim=256, pretrained=False), latent_dim=64)
    head = OutcomeHead(latent_dim=64, action_dim=ACTION_DIM)
    enc.load_state_dict(blob["encoder"])
    head.load_state_dict(blob["head"])
    enc.to(device).eval()
    head.to(device).eval()
    schedule = BetaSchedule.sufficiency_for_success(beta)

    succ, margin, slip = [], [], []
    for batch in loader:
        obs = batch["observation"].to(device)
        actions = batch["actions"].to(device)
        y = Outcome(succ=batch["succ"].to(device), margin=batch["margin"].to(device),
                    slip=batch["slip"].to(device))
        belief = enc(obs)
        pred = head(belief.rsample(K), actions)
        terms = vib_objective(pred.float(), y.expand_to(K), belief.mu.float(),
                              belief.logvar.float(), schedule)
        succ.append(float(terms.distortion.succ))
        margin.append(float(terms.distortion.margin))
        slip.append(float(terms.distortion.slip))

    nb = len(succ)
    mean = lambda v: sum(v) / nb  # noqa: E731
    m = mean(succ)
    var = sum((x - m) ** 2 for x in succ) / max(1, nb - 1) if nb > 1 else 0.0
    return {
        "distortion_succ": m,
        "distortion_succ_stderr": (var / nb) ** 0.5,
        "distortion_margin": mean(margin),
        "distortion_slip": mean(slip),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/libero"))
    args = ap.parse_args()

    frontier = args.results / "l5_frontier.json"
    rows = json.loads(frontier.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = test_loader(args.results)

    for row in rows:
        ckpt = args.results / f"l5_b{row['beta']}" / "best.pth"
        if not ckpt.exists():
            print(f"SKIP beta={row['beta']} ({ckpt} missing)")
            continue
        row.update(perdim_distortion(ckpt, loader, row["beta"], device))
        print(f"beta={row['beta']:6.1f}  D_succ={row['distortion_succ']:.4f} "
              f"+/- {row['distortion_succ_stderr']:.4f}  "
              f"D_margin={row['distortion_margin']:.4f}  D_slip={row['distortion_slip']:.4f}")

    frontier.write_text(json.dumps(rows, indent=2))
    print(f"annotated {frontier}")


if __name__ == "__main__":
    main()
