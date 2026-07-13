"""Config -> objects. The single place where a config becomes a running system.

Deliberately a set of plain functions rather than a registry of stringly-typed names. The
audited repo had six `Registry` objects that were all EMPTY at import, because the packages
defined the registries but never imported the modules carrying the `@register` decorators --
so `BACKBONE_REGISTRY.get("ResNetBackbone")` raised KeyError and the entire advertised
extension mechanism was inert. Explicit construction cannot fail that way, and mypy checks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from msp.data import SyntheticGraspDataset, collate
from msp.engine.trainer import TrainConfig
from msp.inference import ConformalCalibrator, InferenceConfig
from msp.math.bottleneck import BetaSchedule
from msp.models import BeliefEncoder, MLPBackbone, OutcomeHead, ResNetBackbone
from msp.oracle import SyntheticOracle

__all__ = ["Bundle", "build_oracle", "build_models", "build_loaders", "build_all"]


@dataclass
class Bundle:
    oracle: SyntheticOracle
    encoder: BeliefEncoder
    head: OutcomeHead
    loaders: dict[str, DataLoader[Any]]
    train_cfg: TrainConfig
    infer_cfg: InferenceConfig
    calibrator: ConformalCalibrator
    device: torch.device


def build_oracle(cfg: Any) -> SyntheticOracle:
    return SyntheticOracle(
        state_dim=cfg.oracle.state_dim,
        rank=cfg.oracle.rank,
        action_dim=cfg.oracle.action_dim,
        seed=cfg.oracle.seed,
        noise=cfg.oracle.noise,
    )


def build_models(cfg: Any, obs_dim: int) -> tuple[BeliefEncoder, OutcomeHead]:
    if cfg.model.backbone == "mlp":
        backbone = MLPBackbone(obs_dim, output_dim=cfg.model.hidden)
    elif cfg.model.backbone == "resnet":
        backbone = ResNetBackbone(
            output_dim=cfg.model.hidden, pretrained=cfg.model.get("pretrained", True)
        )
    else:
        raise ValueError(f"unknown backbone {cfg.model.backbone!r}; expected 'mlp' or 'resnet'")

    encoder = BeliefEncoder(backbone, latent_dim=cfg.model.latent_dim)
    head = OutcomeHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.oracle.action_dim,
        gripper_dim=cfg.model.get("gripper_dim", 0),
    )
    return encoder, head


def build_loaders(cfg: Any, oracle: SyntheticOracle) -> dict[str, DataLoader[Any]]:
    """Four DISJOINT folds. The seeds differ, which is what makes the calibration fold
    genuinely held out (Assumption A5) -- the calibrator will refuse a fold that collides
    with training."""
    d = cfg.data
    specs = {
        "train": (d.n_train, 1),
        "val": (d.n_val, 2),
        "calib": (d.n_calib, 3),
        "test": (d.n_test, 4),
    }
    out = {}
    for name, (n, seed) in specs.items():
        ds = SyntheticGraspDataset(
            oracle,
            n_scenes=n,
            n_actions=d.n_actions,
            obs_dim=d.obs_dim,
            boundary_focus=d.boundary_focus,
            seed=seed,
        )
        out[name] = DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=(name == "train"),
            num_workers=d.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=(name == "train"),
            persistent_workers=d.num_workers > 0,
        )
    return out


def build_all(cfg: Any) -> Bundle:
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    oracle = build_oracle(cfg)
    encoder, head = build_models(cfg, cfg.data.obs_dim)
    loaders = build_loaders(cfg, oracle)

    train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_epochs=cfg.train.warmup_epochs,
        grad_clip=cfg.train.grad_clip,
        amp_dtype=cfg.train.amp_dtype,
        beta=BetaSchedule(
            succ=cfg.train.beta.succ,
            margin=cfg.train.beta.margin,
            slip=cfg.train.beta.slip,
        ),
        out_dir=cfg.out_dir,
        seed=cfg.seed,
    )
    return Bundle(
        oracle=oracle,
        encoder=encoder,
        head=head,
        loaders=loaders,
        train_cfg=train_cfg,
        infer_cfg=InferenceConfig(),
        calibrator=ConformalCalibrator(
            alpha=cfg.calibration.alpha, gamma=cfg.calibration.gamma
        ),
        device=device,
    )
