"""Train the two learned modules, then calibrate on a held-out fold.

    python scripts/train.py
    python scripts/train.py train.beta.succ=50 model.latent_dim=32
    python scripts/train.py --multirun +experiment=frontier      # the beta sweep

Unlike the audited train.py -- in which every functional line was commented out -- this one
actually trains, validates, checkpoints, calibrates, and writes a report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from msp.build import build_all
from msp.engine import Evaluator, Trainer
from msp.inference import InferenceEngine
from msp.utils.seed import seed_everything

log = logging.getLogger("msp.train")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed)
    log.info("\n%s", OmegaConf.to_yaml(cfg))

    b = build_all(cfg)
    log.info("device=%s  latent_dim=%d  beta=%s", b.device, cfg.model.latent_dim, cfg.train.beta)

    trainer = Trainer(b.encoder, b.head, b.loaders["train"], b.loaders["val"], b.train_cfg, b.device)
    metrics = trainer.fit()
    log.info("training done: %s", {k: round(v, 4) for k, v in metrics.items()})

    # Calibrate on the HELD-OUT fold, then evaluate. Both are part of training, not an
    # afterthought: a checkpoint without a certificate cannot be deployed.
    engine = InferenceEngine(b.encoder, b.head, b.calibrator, b.infer_cfg)
    ev = Evaluator(engine, b.device)
    q_hat = ev.calibrate(b.loaders["calib"], b.calibrator)
    report = ev.evaluate(b.loaders["test"], beta=b.train_cfg.beta)

    out = Path(cfg.out_dir)
    b.calibrator.save(out / "calibration.json")
    (out / "report.json").write_text(json.dumps(report.__dict__, indent=2))

    log.info("q_hat = %.4f", q_hat)
    log.info("coverage      = %.4f  (target %.2f)  %s", report.coverage, report.target_coverage,
             "OK" if report.coverage_holds() else "*** VIOLATED ***")
    log.info("cert. precision = %.4f", report.certified_precision)
    log.info("abstention    = %.4f", report.abstention_rate)
    log.info("rate / distortion = %.4f / %.4f", report.rate, report.distortion)
    return report.distortion


if __name__ == "__main__":
    main()
