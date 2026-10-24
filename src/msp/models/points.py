"""Point-cloud branch of the LM-Grasp outcome scorer (study 6): RGB-D crop + 4D point cloud.

Point layout (B, N, 7): [x, y, z, t, r, g, b]
    x, y, z  current camera frame, minus the visible centroid c, in DECIMETRES (the frame and
             units of the action translation, so a TCP and a point can be compared directly)
    t        signed frame offset of the view the point came from, / 10 (0 = current frame)
    r, g, b  in [0, 1]
Actions (B, Na, A >= 9): [p (3, decimetres, minus c), TCP x axis (3), TCP y axis (3), ...];
the TCP approach is its local -z (msp.data.linemod.tcp_pose).

Two features come out of the points:
    global   PointNet (shared MLP + max pool) over the whole cloud, fused with the ResNet-18
             RGB-D feature before the latent code
    local    per candidate: the k points nearest the TCP, expressed in the TCP frame, through a
             second PointNet; appended to that candidate's action features (grasp-centric point
             features as in PointNetGPD, Liang et al., ICRA 2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from msp.belief import Belief, DiagonalGaussianBelief
from msp.models.nets import BeliefEncoder, ResNetBackbone

__all__ = ["GraspLocalPointNet", "PointBeliefEncoder", "PointNetSet", "tcp_frames"]

POINT_DIM = 7


class PointNetSet(nn.Module):
    """Shared per-point MLP, then max pool over the point axis. (..., N, C) -> (..., F)."""

    def __init__(self, in_dim: int, dims: tuple[int, ...] = (64, 128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for i, h in enumerate(dims):
            layers.append(nn.Linear(d, h))
            if i < len(dims) - 1:
                layers.append(nn.GELU())
            d = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = d

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.mlp(x).max(dim=-2).values
        return out


def tcp_frames(actions: Tensor) -> tuple[Tensor, Tensor]:
    """(p (B, Na, 3), R (B, Na, 3, 3)) from the first 9 action coordinates; R's columns are the
    TCP x, y, z axes, re-orthonormalised (Gram-Schmidt) for safety."""
    p, x, y = actions[..., :3], actions[..., 3:6], actions[..., 6:9]
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y = y - (y * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return p, torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


class GraspLocalPointNet(nn.Module):
    """Per-candidate features of the k points nearest each TCP, in that TCP's frame."""

    def __init__(self, k: int = 128, out_dim: int = 32) -> None:
        super().__init__()
        self.k = k
        self.net = PointNetSet(POINT_DIM, (64, 64, out_dim))
        self.output_dim = out_dim

    def forward(self, points: Tensor, actions: Tensor) -> Tensor:
        """points (B, N, 7), actions (B, Na, A) -> (B, Na, out_dim)."""
        if points.dim() != 3 or points.shape[-1] != POINT_DIM:
            raise ValueError(f"points must be (B, N, {POINT_DIM}); got {tuple(points.shape)}")
        if actions.dim() != 3 or actions.shape[-1] < 9:
            raise ValueError(f"actions must be (B, Na, >= 9); got {tuple(actions.shape)}")
        b, n, _ = points.shape
        na = actions.shape[1]
        k = min(self.k, n)
        p, R = tcp_frames(actions.float())
        xyz = points[..., :3].float()
        d = torch.cdist(p, xyz)  # (B, Na, N)
        idx = d.topk(k, dim=-1, largest=False).indices  # (B, Na, k)
        nb = torch.gather(points.float().unsqueeze(1).expand(b, na, n, POINT_DIM), 2,
                          idx.unsqueeze(-1).expand(b, na, k, POINT_DIM))  # (B, Na, k, 7)
        local = ((nb[..., :3] - p.unsqueeze(2)).unsqueeze(-2) @ R.unsqueeze(2)).squeeze(-2)
        out: Tensor = self.net(torch.cat([local, nb[..., 3:]], dim=-1))
        return out


class PointBeliefEncoder(BeliefEncoder):
    """q(z | RGB-D crop, point cloud): ResNet-18 crop feature and PointNet global feature,
    concatenated and projected to the backbone width, then the usual (mu, logvar) head."""

    def __init__(self, latent_dim: int = 32, hidden: int = 256, width: int = 256,
                 pretrained: bool = True, local_k: int = 128, local_dim: int = 32) -> None:
        super().__init__(ResNetBackbone(output_dim=width, pretrained=pretrained),
                         latent_dim=latent_dim, hidden=hidden)
        self.points = PointNetSet(POINT_DIM, (64, 128, 256))
        self.fuse = nn.Sequential(nn.Linear(width + self.points.output_dim, width), nn.GELU())
        self.local = GraspLocalPointNet(k=local_k, out_dim=local_dim)

    def forward(self, obs: Tensor, points: Tensor | None = None) -> Belief:
        """obs (B, 4, S, S); points (B, N, 7)."""
        if points is None:
            raise ValueError("PointBeliefEncoder needs the point cloud")
        if obs.dim() != 4 or points.dim() != 3 or obs.shape[0] != points.shape[0]:
            raise ValueError(f"obs {tuple(obs.shape)} / points {tuple(points.shape)}")
        h = self.fuse(torch.cat([self.backbone(obs), self.points(points.float())], dim=-1))
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return DiagonalGaussianBelief(mu, logvar)

    def grasp_features(self, points: Tensor, actions: Tensor) -> Tensor:
        """Actions with each candidate's local point feature appended.

        (B, Na, A) -> (B, Na, A + local_dim)."""
        return torch.cat([actions, self.local(points, actions).to(actions.dtype)], dim=-1)
