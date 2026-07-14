"""Algorithm 2, closed loop: perceive, score, SENSE, certify, act or ABSTAIN, adapt.

    MUJOCO_GL=egl python scripts/deploy.py checkpoint=outputs/.../best.pth

The audited deploy.py was 35 lines of comments. This one runs, and it runs the whole of Algorithm 2
including the sensing branch that the old repository never implemented.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from msp.build import build_all
from msp.inference import (
    ActiveConfig,
    ActivePerception,
    ConformalCalibrator,
    InferenceEngine,
    TTAConfig,
    adapt_belief,
)
from msp.models import AcquisitionNet
from msp.types import Abstain, ActionChoice
from msp.utils import seed_everything

log = logging.getLogger("msp.deploy")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = Path(cfg.get("checkpoint", ""))
    if not ckpt.exists():
        raise SystemExit("pass checkpoint=/path/to/best.pth")

    b = build_all(cfg)
    st = torch.load(ckpt, map_location=b.device, weights_only=False)
    b.encoder.load_state_dict(st["encoder"]); b.encoder.to(b.device).eval()
    b.head.load_state_dict(st["head"]); b.head.to(b.device).eval()
    cal = ConformalCalibrator.load(ckpt.parent / "calibration.json")
    engine = InferenceEngine(b.encoder, b.head, cal, b.infer_cfg)

    # Active perception is OPTIONAL and is only enabled if a fitted acquisition net exists.
    # It is never enabled with random weights -- see AcquisitionNet's `fitted` buffer.
    active = None
    acq_path = ckpt.parent / "acquisition.pth"
    if acq_path.exists():
        blob = torch.load(acq_path, map_location=b.device, weights_only=False)
        acq = AcquisitionNet(b.encoder.backbone.output_dim, n_views=blob["n_views"]).to(b.device)
        acq.load_state_dict(blob["acquisition"])
        active = ActivePerception(b.encoder, acq, ActiveConfig(
            tau_ambiguity=float(cfg.get("tau_ambiguity", 0.004)),
            sensing_budget=int(cfg.get("sensing_budget", 2)),
        ))
        log.info("active perception ENABLED (|B| = %d viewpoints)", blob["n_views"])
    else:
        log.info("active perception disabled (no acquisition.pth) -- single view, passive")

    from msp.oracle.mujoco_sim import MuJoCoOracle
    sim = MuJoCoOracle(shape=cfg.data.get("shape", "box"))
    oracle = b.oracle
    img = int(cfg.data.get("image_size", 96))

    episodes = int(cfg.get("episodes", 50))
    acted = abstained = succeeded = 0
    sensed = 0

    for ep in range(episodes):
        state = oracle.sample_states(1)
        actions = oracle.sample_actions(state, 64).to(b.device)

        views = [sim.render(state, height=img, width=img, view=0).to(b.device)]
        taken = [0]

        # ---- Algorithm 2: sense until the ambiguity is tolerable or the budget runs out ----
        for _ in range(active.cfg.sensing_budget if active else 0):
            belief = engine.perceive(torch.stack(views, dim=1))
            scored = engine.score(belief, actions)
            if not bool(active.should_sense(scored.ambiguity, len(views) - 1)[0]):
                break
            cand = torch.tensor(
                [[v for v in range(sim.N_VIEWS) if v not in taken]], device=b.device
            )
            if cand.numel() == 0:
                break
            nxt = int(active.select_view(views[0], cand)[0])
            views.append(sim.render(state, height=img, width=img, view=nxt).to(b.device))
            taken.append(nxt)
            sensed += 1
            log.info("ep %02d: U=%.5f > tau -> looked from view %d", ep, float(scored.ambiguity[0]), nxt)

        belief = engine.perceive(torch.stack(views, dim=1))
        scored = engine.score(belief, actions)

        # ---- certify, then act or ABSTAIN (Eq 24) ----
        decision = engine.select(scored)
        if isinstance(decision, Abstain):
            abstained += 1
            log.info("ep %02d: ABSTAIN  (U=%.5f, %d view(s))", ep, float(scored.ambiguity[0]), len(views))
            continue

        assert isinstance(decision, ActionChoice)
        acted += 1
        y = sim.query(state, decision.action.view(1, 1, -1).cpu())
        ok = bool(y.succ.item())
        succeeded += ok
        log.info("ep %02d: ACT idx=%2d s=%.3f v=%.5f -> %s",
                 ep, decision.index, decision.success_prob, decision.epistemic_var,
                 "SUCCESS" if ok else "FAIL")

        # ---- TTA on failure: assimilate the outcome (Eq 19/20) ----
        if not ok:
            before = float(scored.ambiguity[0])
            belief = adapt_belief(belief, b.head, decision.action.view(1, 1, -1),
                                  y.to(b.device), TTAConfig(steps=20, num_samples=32))
            after = float(engine.score(belief, actions).ambiguity[0])
            log.info("        TTA: U %.5f -> %.5f", before, after)

    print(f"\nepisodes {episodes}   sensed {sensed}   acted {acted}   abstained {abstained}")
    print(f"success when acting: {succeeded}/{max(1, acted)} = {succeeded / max(1, acted):.1%}")
    print(f"abstention rate    : {abstained / episodes:.1%}")


if __name__ == "__main__":
    main()
