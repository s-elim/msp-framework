"""Deterministic seeding."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int = 0, *, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    `deterministic` is OPT-IN. The audited seeder forced cudnn.deterministic=True and
    benchmark=False globally with no way to turn it off -- a permanent throughput tax on
    every run, including the many that do not need bit-exactness.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
