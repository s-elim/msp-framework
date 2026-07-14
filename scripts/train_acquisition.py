"""Train the amortized acquisition network. Formalization Eq 18.

    MUJOCO_GL=egl python scripts/train_acquisition.py checkpoint=outputs/.../best.pth

THE STEP THE AUDITED REPOSITORY SKIPPED. It shipped an `AcquisitionNet` with no loss, no target and
no look-ahead, then took an argmax over its randomly-initialized output to steer a camera. This is
the loss that was missing, and it is what makes active perception mean anything.

HOW IT WORKS. IG(b) is an expectation over an observation you have not taken yet, so computing it
honestly needs a generative model of the future observation -- exactly the world model MSP is built
to avoid. In SIMULATION you need no such thing: the state x is known, so o_b is simply RENDERED,
U(o ∪ o_b) is evaluated exactly, and IG_true falls out with no predictive model at all (Eq 17).
Eq 18 then regresses alpha_omega onto those exact values. At deployment, where nothing can be
rendered, one forward pass of alpha_omega replaces the whole look-ahead.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from msp.build import build_all
from msp.data import RGBDGraspDataset
from msp.inference.active import compute_true_information_gain
from msp.math.voi import acquisition_loss
from msp.models import AcquisitionNet
from msp.utils import seed_everything

log = logging.getLogger("msp.acq")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = Path(cfg.get("checkpoint", ""))
    if not ckpt.exists():
        raise SystemExit("pass checkpoint=/path/to/best.pth -- the encoder and head must be trained first")

    b = build_all(cfg)
    st = torch.load(ckpt, map_location=b.device, weights_only=False)
    b.encoder.load_state_dict(st["encoder"]); b.encoder.to(b.device).eval()
    b.head.load_state_dict(st["head"]); b.head.to(b.device).eval()

    ds: RGBDGraspDataset = b.loaders["train"].dataset  # type: ignore[assignment]
    if ds.n_views < 2:
        raise SystemExit(
            f"the corpus has {ds.n_views} view(s). Eq 17 needs the look-ahead rendered from every "
            "candidate viewpoint: regenerate with data.n_views=8."
        )

    acq = AcquisitionNet(b.encoder.backbone.output_dim, n_views=ds.n_views).to(b.device)
    opt = torch.optim.AdamW(acq.parameters(), lr=1e-3, weight_decay=1e-4)

    n = len(ds)
    bs = 64
    epochs = int(cfg.get("acq_epochs", 30))
    view_ids = torch.arange(ds.n_views, device=b.device)

    for ep in range(epochs):
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            views = ds.all_views(idx).to(b.device)  # (bs, V, 4, H, W)
            actions = ds.actions[idx].to(b.device)  # (bs, Na, 7)

            # --- Eq 17: the rendered look-ahead. Only possible because x is known. ---
            ig_true, _ = compute_true_information_gain(
                b.encoder, b.head,
                current_views=[views[:, 0]],
                candidate_views=views,
                actions=actions,
            )

            # --- Eq 18: regress onto it ---
            with torch.no_grad():
                feats = b.encoder.backbone(views[:, 0])
            pred = acq(feats, view_ids.unsqueeze(0).expand(idx.numel(), -1))
            loss = acquisition_loss(pred, ig_true.detach())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss); nb += 1

        if ep % 5 == 0 or ep == epochs - 1:
            log.info("epoch %3d   L_acq = %.6f", ep, total / max(1, nb))

    acq.fitted.fill_(True)  # only now may ActivePerception use it
    out = ckpt.parent / "acquisition.pth"
    torch.save({"acquisition": acq.state_dict(), "n_views": ds.n_views}, out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
