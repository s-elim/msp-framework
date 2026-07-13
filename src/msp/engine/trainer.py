"""Training loop for the two learned modules.

Engineering choices worth stating, since the audited trainer had none of them:

* AMP (bf16) with a GradScaler only when fp16 is selected. bf16 needs no scaler and has the
  dynamic range that the Gaussian NLL wants -- an fp16 exp(-logvar) is a good way to meet
  inf. Default is bf16 on Ampere+.
* Gradient clipping on the JOINT parameter set, not on encoder and head separately. Clipping
  two groups to max_norm=1.0 each permits a joint norm of 2.0, which is not what
  "clip to 1.0" means.
* A real validation pass, a real best-checkpoint, and a real resume. The audited trainer
  stored `val_loader` and never used it, wrote `checkpoint_epoch_N.pth` into the CWD, and
  had no `best.pth` -- while the cookbook instructed users to load one.
* DDP-aware: rank-0-only logging and checkpointing, and the sampler's epoch is set so that
  shuffling actually differs across epochs (a classic silent bug).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from msp.math.bottleneck import BetaSchedule, vib_objective
from msp.models.nets import BeliefEncoder, OutcomeHead
from msp.types import Outcome

__all__ = ["TrainConfig", "Trainer"]


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "off"
    beta: BetaSchedule = field(default_factory=BetaSchedule)
    out_dir: str = "outputs/run"
    log_every: int = 50
    seed: int = 0


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


class Trainer:
    def __init__(
        self,
        encoder: BeliefEncoder,
        head: OutcomeHead,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.encoder, self.head = encoder.to(device), head.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.cfg, self.device = cfg, device

        self.params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)
        self.use_amp = self.amp_dtype is not None and device.type == "cuda"
        # A GradScaler is required for fp16 and HARMFUL/unnecessary for bf16.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16
        )

        self.out_dir = Path(cfg.out_dir)
        if _rank() == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_val = math.inf
        self.epoch = 0

    # -- schedule ------------------------------------------------------------

    def _lr_at(self, epoch: int) -> float:
        """Linear warmup then cosine decay to 1% of peak."""
        w = self.cfg.warmup_epochs
        if epoch < w:
            return self.cfg.lr * (epoch + 1) / max(1, w)
        t = (epoch - w) / max(1, self.cfg.epochs - w)
        return self.cfg.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # -- one step ------------------------------------------------------------

    def _forward(self, batch: dict[str, Any]) -> Any:
        obs = batch["observation"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        y = Outcome(
            succ=batch["succ"].to(self.device, non_blocking=True),
            margin=batch["margin"].to(self.device, non_blocking=True),
            slip=batch["slip"].to(self.device, non_blocking=True),
        )
        weights = batch.get("weights")
        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)

        belief = self.encoder(obs)
        z = belief.rsample(1).squeeze(1)  # one sample per scene, as Alg 1 prescribes
        pred = self.head(z, actions)
        return vib_objective(
            pred, y, belief.mu, belief.logvar, self.cfg.beta, weights=weights
        )

    def train_epoch(self) -> dict[str, float]:
        self.encoder.train()
        self.head.train()

        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)  # without this, DDP shuffles identically every epoch

        for g in self.opt.param_groups:
            g["lr"] = self._lr_at(self.epoch)

        totals: dict[str, float] = {}
        n = 0
        for batch in self.train_loader:
            self.opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                terms = self._forward(batch)

            self.scaler.scale(terms.loss).backward()
            self.scaler.unscale_(self.opt)
            # Clip the JOINT parameter set.
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            for k, v in terms.to_metrics().items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        out = {k: v / max(1, n) for k, v in totals.items()}
        out["lr"] = self._lr_at(self.epoch)
        return out

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.encoder.eval()
        self.head.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                terms = self._forward(batch)
            for k, v in terms.to_metrics().items():
                totals[f"val/{k}"] = totals.get(f"val/{k}", 0.0) + v
            n += 1
        return {k: v / max(1, n) for k, v in totals.items()}

    # -- checkpoints ---------------------------------------------------------

    def save(self, name: str, extra: dict[str, Any] | None = None) -> Path | None:
        if _rank() != 0:
            return None
        path = self.out_dir / name
        torch.save(
            {
                "epoch": self.epoch,
                "encoder": _unwrap(self.encoder).state_dict(),
                "head": _unwrap(self.head).state_dict(),
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val": self.best_val,
                "config": self.cfg.__dict__ | {"beta": self.cfg.beta.__dict__},
                **(extra or {}),
            },
            path,
        )
        return path

    def load(self, path: str | Path, *, weights_only: bool = False) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.encoder).load_state_dict(ck["encoder"])
        _unwrap(self.head).load_state_dict(ck["head"])
        if not weights_only:
            self.opt.load_state_dict(ck["optimizer"])
            self.scaler.load_state_dict(ck["scaler"])
            self.epoch = ck["epoch"] + 1
            self.best_val = ck.get("best_val", math.inf)

    # -- driver --------------------------------------------------------------

    def fit(self, logger: Any = None) -> dict[str, float]:
        metrics: dict[str, float] = {}
        while self.epoch < self.cfg.epochs:
            metrics = self.train_epoch()
            metrics |= self.validate()

            if _rank() == 0:
                if logger is not None:
                    logger.log(metrics, step=self.epoch)
                key = metrics.get("val/loss/total", metrics.get("loss/total", math.inf))
                if key < self.best_val:
                    self.best_val = key
                    self.save("best.pth")
                self.save("last.pth")

            self.epoch += 1
        return metrics


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m
