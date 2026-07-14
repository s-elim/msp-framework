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
from torch.utils.data.distributed import DistributedSampler

from msp.data import SyntheticGraspDataset, collate
from msp.engine.trainer import TrainConfig, is_distributed
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
    if cfg.data.get("kind", "synthetic") in ("rgbd", "libero") and cfg.model.backbone == "mlp":
        raise ValueError(
            f"data={cfg.data.kind} yields (4, H, W) images, but model=mlp expects a vector. Use "
            "model=resnet. "
            "(The audited README advertised the RGB-D backbone while the dataset produced 32-dim "
            "vectors; `model=resnet` crashed inside conv2d. Fail loudly here instead.)"
        )
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


def build_loaders(cfg: Any, oracle: Any) -> dict[str, DataLoader[Any]]:
    """Four DISJOINT folds. The seeds differ, which is what makes the calibration fold
    genuinely held out (Assumption A5) -- the calibrator will refuse a fold that collides
    with training."""
    kind = cfg.data.get("kind", "synthetic")
    if kind == "libero":
        return _build_libero_loaders(cfg)
    if kind == "rgbd":
        return _build_rgbd_loaders(cfg)
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
        # Under DDP the training fold must be SHARDED, not replicated, or every rank sees
        # the same data and the effective batch is unchanged.
        sampler = None
        if is_distributed() and name == "train":
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)

        out[name] = DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=(name == "train" and sampler is None),
            sampler=sampler,
            num_workers=d.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=(name == "train"),
            persistent_workers=d.num_workers > 0,
            worker_init_fn=_seed_worker,  # workers were previously unseeded => irreproducible
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return out


def _build_rgbd_loaders(cfg: Any) -> dict[str, DataLoader[Any]]:
    """The real experiment: RGB-D images rendered from MuJoCo, labelled by rigid-body rollouts.

    The corpus is generated ONCE and cached. Rendering is ~2 ms/frame and a rollout ~10 ms/grasp,
    so a 20k-scene corpus is a few minutes -- but only if it is not regenerated every epoch.
    """
    from pathlib import Path

    from msp.data import CorpusSpec, RGBDGraspDataset, generate_corpus

    d = cfg.data
    cache = Path(d.cache_dir)
    specs = {
        "train": (d.n_train, 1),
        "val": (d.n_val, 2),
        "calib": (d.n_calib, 3),
        "test": (d.n_test, 4),
    }
    out = {}
    for name, (n, seed) in specs.items():
        path = generate_corpus(
            CorpusSpec(
                n_scenes=n,
                n_actions=d.n_actions,
                shape=d.shape,
                image_size=d.image_size,
                seed=seed,
                use_simulator=d.use_simulator,
                n_views=d.get("n_views", 1),
            ),
            cache,
        )
        # Only the TRAINING fold sees a random view count -- evaluation must be a fixed,
        # reproducible protocol, and the look-ahead supplies its own views explicitly.
        ds = RGBDGraspDataset(
            path,
            max_train_views=d.get("max_train_views", 1) if name == "train" else 1,
        )
        sampler = None
        if is_distributed() and name == "train":
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        out[name] = DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=(name == "train" and sampler is None),
            sampler=sampler,
            num_workers=d.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=(name == "train"),
            persistent_workers=d.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return out


def _build_libero_loaders(cfg: Any) -> dict[str, DataLoader[Any]]:
    """The paper's dataset: real LIBERO groceries, RGB-D, MuJoCo grasp outcomes."""
    from pathlib import Path

    from msp.data import LiberoCorpusSpec, LiberoGraspDataset, generate_libero_corpus

    d = cfg.data
    cache = Path(d.cache_dir)
    specs = {"train": (d.n_train, 1), "val": (d.n_val, 2), "calib": (d.n_calib, 3),
             "test": (d.n_test, 4)}
    out = {}
    for name, (n, seed) in specs.items():
        path = generate_libero_corpus(
            LiberoCorpusSpec(
                n_scenes=n,
                n_actions=d.n_actions,
                image_size=d.image_size,
                n_views=d.get("n_views", 1),
                seed=seed,
                objects=tuple(d.get("objects", []) or ()),
            ),
            cache,
            asset_root=d.get("asset_root", None),
        )
        # Only the TRAINING fold gets a random view count. Evaluation must be a fixed,
        # reproducible protocol.
        ds = LiberoGraspDataset(
            path, max_train_views=d.get("max_train_views", 1) if name == "train" else 1
        )
        sampler = None
        if is_distributed() and name == "train":
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        out[name] = DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=(name == "train" and sampler is None),
            sampler=sampler,
            num_workers=d.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=(name == "train"),
            persistent_workers=d.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return out


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker. Without this, anything random inside a worker is not
    reproducible across runs, which quietly breaks the reproducibility claim."""
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_all(cfg: Any, device: torch.device | None = None) -> Bundle:
    if device is None:
        device = torch.device(
            cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        )
    oracle = build_oracle(cfg)
    encoder, head = build_models(cfg, cfg.data.get('obs_dim', 32))
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
        compile=cfg.train.get("compile", False),
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
