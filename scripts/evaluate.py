"""Evaluate a checkpoint: coverage (Theorem 7), abstention, and the frontier point."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from msp.build import build_all
from msp.engine import Evaluator
from msp.inference import ConformalCalibrator, InferenceEngine
from msp.utils.seed import seed_everything

log = logging.getLogger("msp.eval")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = Path(cfg.get("checkpoint", "")) 
    if not ckpt.exists():
        raise SystemExit(f"pass checkpoint=/path/to/best.pth  (got {ckpt!r})")

    b = build_all(cfg)
    state = torch.load(ckpt, map_location=b.device, weights_only=False)
    b.encoder.load_state_dict(state["encoder"]); b.encoder.to(b.device)
    b.head.load_state_dict(state["head"]); b.head.to(b.device)

    cal_path = ckpt.parent / "calibration.json"
    cal = ConformalCalibrator.load(cal_path) if cal_path.exists() else b.calibrator

    engine = InferenceEngine(b.encoder, b.head, cal, b.infer_cfg)
    ev = Evaluator(engine, b.device)
    if cal.q_hat is None:
        ev.calibrate(b.loaders["calib"], cal)

    r = ev.evaluate(b.loaders["test"])
    print(json.dumps(r.__dict__, indent=2))
    log.info("coverage %.4f vs target %.2f -> %s", r.coverage, r.target_coverage,
             "HOLDS" if r.coverage_holds() else "VIOLATED")


if __name__ == "__main__":
    main()
