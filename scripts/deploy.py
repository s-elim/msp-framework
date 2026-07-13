"""Algorithm 2 in a closed loop: perceive, score, sense, certify, act or ABSTAIN, adapt.

The audited deploy.py was 35 lines of comments. This one runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from msp.build import build_all
from msp.inference import ConformalCalibrator, InferenceEngine, TTAConfig, adapt_belief
from msp.types import Abstain, ActionChoice
from msp.utils.seed import seed_everything

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
    oracle = b.oracle

    n_episodes = int(cfg.get("episodes", 20))
    acted = abstained = succeeded = 0

    for ep in range(n_episodes):
        state = oracle.sample_states(1)
        obs = oracle.observe(state, obs_dim=cfg.data.obs_dim).to(b.device)
        actions = torch.randn(1, 128, oracle.action_dim, device=b.device)

        belief = engine.perceive(obs)
        scored = engine.score(belief, actions)

        # --- active perception trigger (Eq 16) ---
        if bool(engine.should_sense(scored)[0]):
            log.info("ep %02d: U=%.4f > tau -> would acquire a new view", ep, float(scored.ambiguity[0]))

        # --- certify, then act or abstain (Eq 24) ---
        decision = engine.select(scored)

        if isinstance(decision, Abstain):
            abstained += 1
            log.info("ep %02d: ABSTAIN (no action certified)  U=%.4f", ep, float(scored.ambiguity[0]))
            continue

        assert isinstance(decision, ActionChoice)
        acted += 1
        y = oracle.query(state.to(b.device), decision.action.view(1, 1, -1))
        ok = bool(y.succ.item())
        succeeded += ok
        log.info("ep %02d: act idx=%3d  s=%.3f v=%.4f -> %s",
                 ep, decision.index, decision.success_prob, decision.epistemic_var,
                 "SUCCESS" if ok else "FAIL")

        # --- TTA on failure: assimilate the outcome (Eq 19/20) ---
        if not ok:
            before = float(scored.ambiguity[0])
            belief = adapt_belief(belief, b.head, decision.action.view(1, 1, -1), y,
                                  TTAConfig(steps=20, num_samples=32))
            after = float(engine.score(belief, actions).ambiguity[0])
            log.info("        TTA: U %.4f -> %.4f", before, after)

    print(f"\nacted {acted}  abstained {abstained}  "
          f"success-when-acting {succeeded}/{max(1, acted)} = {succeeded / max(1, acted):.1%}")


if __name__ == "__main__":
    main()
