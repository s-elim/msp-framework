"""Batch collation shared by every learned scorer."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["collate"]


def collate(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Stack per-scene dicts into a batch; importance weights are renormalised per batch.

    Multi-view observations (V, ...) with different view counts are padded to the largest count
    by repeating views already present. Repetition is safe because the encoder aggregates views
    with a permutation-invariant mean, so a duplicate view adds nothing new.
    """
    first = items[0]["observation"]
    if first.dim() in (3, 4) and any(it["observation"].shape[0] != first.shape[0] for it in items):
        vmax = max(it["observation"].shape[0] for it in items)
        for it in items:
            o = it["observation"]
            if o.shape[0] < vmax:
                pad = o[torch.randint(0, o.shape[0], (vmax - o.shape[0],))]
                it["observation"] = torch.cat([o, pad], dim=0)

    out = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    w = out["weights"]
    out["weights"] = w / w.sum().clamp_min(1e-12)
    return out
