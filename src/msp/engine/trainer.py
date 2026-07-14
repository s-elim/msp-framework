"""Training loop for the two learned modules.

Three engineering commitments, each of which the audited trainer got wrong or omitted:

* THE NETWORK RUNS IN BF16; THE LOSS DOES NOT. `autocast` wraps only the encoder and head.
  The rate and distortion are then computed in fp32, because they are not merely training
  signals -- they are the coordinates plotted on the paper's rate-distortion frontier, and
  bf16 carries ~8 mantissa bits (2-3 decimal digits). Training in reduced precision is
  correct; *measuring* in reduced precision is not. A GradScaler is used only for fp16;
  bf16 has the dynamic range and needs none.

* GRADIENTS ARE CLIPPED ON THE JOINT PARAMETER SET. Clipping encoder and head separately to
  max_norm=1.0 each permits a joint norm of 2.0, which is not what "clip to 1.0" means.

* DDP IS ACTUALLY WIRED. `setup_distributed()` initializes the process group and the modules
  are wrapped in `DistributedDataParallel`; validation metrics are all-reduced so rank 0 does
  not report only its own shard; and `sampler.set_epoch` is called so shuffling differs
  across epochs. The previous version had the rank *guards* but no DDP wrapping and no
  launcher -- it advertised multi-GPU and silently ran on one. Launch with
  `scripts/launch_ddp.sh` or `torchrun`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from msp.math.bottleneck import BetaSchedule, VIBTerms, vib_objective
from msp.models.nets import BeliefEncoder, OutcomeHead
from msp.types import Outcome

__all__ = ["TrainConfig", "Trainer", "setup_distributed", "cleanup_distributed",
           "is_distributed", "rank", "world_size"]


# ======================================================================================
# Distributed helpers
# ======================================================================================


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def setup_distributed() -> torch.device:
    """Initialize the process group from torchrun's environment. Returns this rank's device.

    Falls back to single-device cleanly when not launched under torchrun, so the same
    `scripts/train.py` works both ways.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def _all_reduce_mean(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """Average metrics across ranks. Without this, rank 0 reports only its own shard, which
    is a quietly wrong validation curve on a multi-GPU run."""
    if not is_distributed() or not metrics:
        return metrics
    keys = sorted(metrics)
    t = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world_size()
    return dict(zip(keys, t.tolist(), strict=True))


# ======================================================================================


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
    seed: int = 0
    compile: bool = False

    #: Posterior samples used to estimate the distortion  D = E_{z~q(z|o)}[ -log p(y|z,a) ]  at
    #: TRAINING time. It must be > 1, and here is why.
    #:
    #: The distortion is an EXPECTATION over z (Eq 13/14). Estimating it from a SINGLE draw is
    #: unbiased but the variance is ruinous, because the posterior is noise-dominated: measured on
    #: LIBERO, |mu| ~ 0.37 while sigma ~ 0.71, so one sample of z is mostly noise.
    #:
    #: What that costs is not slower convergence -- it is a DIFFERENT MODEL. Whether a grasp works
    #: depends on the angle between the jaws and the object, i.e. on an INTERACTION between the
    #: action and the object's pose in z. Through a single noisy draw the head cannot resolve the
    #: pose, so it cannot form that interaction, and it converges to the only thing that survives
    #: the noise: the per-object base success rate. The action is then ignored outright --
    #: prediction std across the 8 candidate grasps of a scene was 0.0027, against 0.4148 in the
    #: ground truth. Within-scene AUC 0.524 (chance) while the POOLED AUC read a healthy 0.716,
    #: because a pooled AUC over objects whose base rates range from 0.04 to 0.999 rewards object
    #: RECOGNITION. The model looked like it worked and could not rank two grasps.
    #:
    #: Raising beta does not fix it (beta=300 lifts the rate 3.9 -> 11.6 nats and within-scene AUC
    #: stays at 0.524); the information is already in the latent. Fitting a head to the FROZEN
    #: encoder's mean recovers within-scene AUC 0.615 against an oracle ceiling of 0.685, which is
    #: what proves the encoder was never the problem.
    train_samples: int = 8


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
        self.device = device
        self.cfg = cfg

        encoder, head = encoder.to(device), head.to(device)
        if cfg.compile and device.type == "cuda":
            encoder = torch.compile(encoder)  # type: ignore[assignment]
            head = torch.compile(head)  # type: ignore[assignment]

        # Parameters must be collected BEFORE the DDP wrap, so the optimizer and the
        # clipper see the real leaves rather than DDP's replicas.
        self.params = list(encoder.parameters()) + list(head.parameters())

        if is_distributed():
            ids = [device.index] if device.type == "cuda" else None
            encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=ids)
            head = nn.parallel.DistributedDataParallel(head, device_ids=ids)

        self.encoder, self.head = encoder, head
        self.train_loader, self.val_loader = train_loader, val_loader

        self.opt = torch.optim.AdamW(
            self.params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            fused=(device.type == "cuda"),
        )

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)
        self.use_amp = self.amp_dtype is not None and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16
        )

        self.out_dir = Path(cfg.out_dir)
        if rank() == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_val = math.inf
        self.epoch = 0

    # -- schedule ------------------------------------------------------------

    def _lr_at(self, epoch: int) -> float:
        """Linear warmup, then cosine decay to 1% of peak."""
        w = self.cfg.warmup_epochs
        if epoch < w:
            return self.cfg.lr * (epoch + 1) / max(1, w)
        t = (epoch - w) / max(1, self.cfg.epochs - w)
        return self.cfg.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    # -- one step ------------------------------------------------------------

    def _forward(self, batch: dict[str, Any]) -> VIBTerms:
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

        # --- network: reduced precision is fine here ---
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            belief = self.encoder(obs)
            k = self.cfg.train_samples
            z = belief.rsample(k)  # (B, K, d) -- K > 1; see TrainConfig.train_samples
            pred = self.head(z, actions)  # (B, K, Na, 6)

        # --- loss: fp32, always. These numbers get PLOTTED. ---
        with torch.autocast("cuda", enabled=False):
            return vib_objective(
                pred.float(),
                y.expand_to(k),
                belief.mu.float(),
                belief.logvar.float(),
                self.cfg.beta,
                weights=weights,
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
            terms = self._forward(batch)

            self.scaler.scale(terms.loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)  # JOINT norm
            self.scaler.step(self.opt)
            self.scaler.update()

            for k, v in terms.to_metrics().items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        out = {k: v / max(1, n) for k, v in totals.items()}
        out = _all_reduce_mean(out, self.device)
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
            terms = self._forward(batch)
            for k, v in terms.to_metrics().items():
                totals[f"val/{k}"] = totals.get(f"val/{k}", 0.0) + v
            n += 1
        return _all_reduce_mean({k: v / max(1, n) for k, v in totals.items()}, self.device)

    # -- checkpoints ---------------------------------------------------------

    def save(self, name: str, extra: dict[str, Any] | None = None) -> Path | None:
        if rank() != 0:
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
                # RNG state, so a resumed run is bit-exact rather than merely similar.
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "git_sha": _git_sha(),
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
            if ck.get("rng_cpu") is not None:
                torch.set_rng_state(ck["rng_cpu"].cpu())
            if ck.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])

    # -- driver --------------------------------------------------------------

    def fit(self, logger: Any = None) -> dict[str, float]:
        metrics: dict[str, float] = {}
        while self.epoch < self.cfg.epochs:
            metrics = self.train_epoch() | self.validate()

            if rank() == 0:
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
    m = getattr(m, "module", m)  # DDP
    return getattr(m, "_orig_mod", m)  # torch.compile


def _git_sha() -> str:
    """Record the commit that produced a checkpoint. A number in a paper that cannot be
    traced back to a commit is not reproducible."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"
