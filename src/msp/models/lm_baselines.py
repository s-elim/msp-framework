"""Baselines for LM-Grasp, matched to the MSP encoder in data, crop and compute.

    PoseRegressor   ResNet-18 on the same RGB-D crop -> 6-D rotation + translation offset from
                    the visible centroid. Its pose feeds a pose-then-plan scorer that re-simulates
                    each candidate under the estimated pose (scripts/sr/plan_with_pose.py).
    GraspPatchCNN   the Dex-Net 2.0 GQ-CNN architecture (Mahler et al., RSS 2017) retrained on
                    LM-Grasp: a depth patch cropped and rotated by each candidate grasp, plus a
                    scalar stream. Deviation: the scalar stream carries the TCP depth relative to
                    the object and the 3-D approach direction, because LM-Grasp approaches are
                    tilted up to 50 degrees from the camera ray rather than top-down.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["GraspPatchCNN", "PoseRegressor", "grasp_patches", "rot6d_to_matrix"]


def rot6d_to_matrix(x: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt (Zhou et al., CVPR 2019)."""
    a, b = x[..., :3], x[..., 3:6]
    e1 = F.normalize(a, dim=-1)
    e2 = F.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


class PoseRegressor(nn.Module):
    """(B, 4, S, S) -> R (B, 3, 3), dt (B, 3) metres from the visible centroid."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from msp.models.nets import ResNetBackbone

        self.backbone = ResNetBackbone(output_dim=256, pretrained=pretrained)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(256, 9))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        p = self.out(self.backbone(obs))
        return rot6d_to_matrix(p[:, :6]), 0.1 * p[:, 6:9]


PATCH = 32


class GraspPatchCNN(nn.Module):
    """GQ-CNN layer stack: conv(64,7) conv(64,5) pool conv(64,3) conv(64,3) pool fc(1024)."""

    def __init__(self, scalar_dim: int = 4) -> None:
        super().__init__()
        self.im = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (PATCH // 4) ** 2, 1024), nn.ReLU(inplace=True),
        )
        self.z = nn.Sequential(nn.Linear(scalar_dim, 16), nn.ReLU(inplace=True))
        self.merge = nn.Sequential(nn.Linear(1024 + 16, 1024), nn.ReLU(inplace=True),
                                   nn.Linear(1024, 1))

    def forward(self, patch: Tensor, scalars: Tensor) -> Tensor:
        return self.merge(torch.cat([self.im(patch), self.z(scalars)], -1)).squeeze(-1)


def grasp_patches(depth: Tensor, T: Tensor, K: Tensor, crop: Tensor, c: Tensor,
                  size: int = PATCH, half_width_m: float = 0.09) -> tuple[Tensor, Tensor]:
    """Grasp-aligned depth patches from the (B,1,S,S) crop depth.

    T (B, Na, 4, 4) camera-frame TCP poses, K (B,3,3), crop (B,3) = (u0, v0, side) in pixels of
    the original image, c (B,3) visible centroid. The patch is centred on the projected TCP,
    rotated so the projected closing axis is horizontal, and spans +-half_width_m along it.
    Returns patches (B*Na, 1, size, size) and scalars (B*Na, 4).
    """
    b, na = T.shape[:2]
    s_img = depth.shape[-1]
    p = T[..., :3, 3]  # (B, Na, 3)
    xax = T[..., :3, 0]
    zax = T[..., :3, 2]

    def project(pt: Tensor) -> Tensor:
        u = K[:, None, 0, 0] * pt[..., 0] / pt[..., 2] + K[:, None, 0, 2]
        v = K[:, None, 1, 1] * pt[..., 1] / pt[..., 2] + K[:, None, 1, 2]
        scale = s_img / crop[:, None, 2]
        return torch.stack([(u - crop[:, None, 0]) * scale, (v - crop[:, None, 1]) * scale], -1)

    uv = project(p)
    uv_tip = project(p + half_width_m * xax)
    d = uv_tip - uv
    theta = torch.atan2(d[..., 1], d[..., 0])
    half_px = d.norm(dim=-1).clamp(2.0, s_img)
    lin = torch.linspace(-1.0, 1.0, size, device=depth.device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    base = torch.stack([gx, gy], -1).reshape(1, size * size, 2)
    cos, sin = theta.cos().reshape(-1, 1), theta.sin().reshape(-1, 1)
    rot = torch.stack([torch.cat([cos, -sin], 1), torch.cat([sin, cos], 1)], 1)
    pts = torch.einsum("bij,bpj->bpi", rot, base * half_px.reshape(-1, 1, 1))
    pts = pts + uv.reshape(-1, 1, 2)
    grid = torch.stack([2 * pts[..., 0] / (s_img - 1) - 1, 2 * pts[..., 1] / (s_img - 1) - 1], -1)
    src = depth.repeat_interleave(na, 0)
    patches = F.grid_sample(src, grid.reshape(-1, size, size, 2), mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    patches = patches - patches.mean(dim=(1, 2, 3), keepdim=True)
    rel_depth = ((p[..., 2] - c[:, None, 2]) / 0.1).reshape(-1, 1)
    scalars = torch.cat([rel_depth, -zax.reshape(-1, 3)], -1)
    return patches, scalars
