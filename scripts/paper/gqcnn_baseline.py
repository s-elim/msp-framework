"""GQ-CNN (Dex-Net 2.0) architecture, retrained on the MSP corpus.

This is a REIMPLEMENTATION of the architecture, not the released Dex-Net model. The
original ships TensorFlow-1 checkpoints that do not run on this stack, and its weights
were trained on a different gripper, object set and depth sensor, so a zero-shot number
would measure domain gap rather than the scoring function. Retraining the same
architecture on the same corpus with the same candidate grasps isolates what the paper
actually compares: how well each scoring function ranks grasps.

Faithful to Mahler et al., RSS 2017, Section 5:

  * input is a depth patch aligned to the grasp -- rotated so the jaw axis is horizontal
    and centred on the grasp point -- plus the gripper depth as a scalar;
  * conv(64,7) conv(64,5) pool conv(64,3) conv(64,3) pool, fc(1024), a 16-unit stream for
    the scalar depth, merged, fc(1024), 2-way output.

Deviations, all forced by the corpus and all recorded here: patches are 32x32 taken from
96x96 renders rather than 32x32 from a 640x480 sensor, so the effective resolution is
lower for every method including ours; and the gripper depth is the grasp centre height
above the table rather than a sensor-frame distance.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from make_qualitative_fig import camera_basis, project, quat_to_matrix  # same camera model

log = logging.getLogger("gqcnn")

PATCH = 32
TABLE_Z = 0.0


class GQCNN(nn.Module):
    """Dex-Net 2.0 grasp-quality CNN."""

    def __init__(self) -> None:
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
        self.z = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True))
        self.merge = nn.Sequential(
            nn.Linear(1024 + 16, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1),
        )

    def features(self, patch: Tensor, depth: Tensor) -> Tensor:
        """The representation an edge device would have to transmit.

        GQ-CNN's is formed AFTER the action is applied -- the patch is cropped and
        rotated by the grasp pose -- so unlike a per-scene sufficient statistic it must
        be sent once per candidate action, which is what the rate comparison measures.
        """
        h = torch.cat([self.im(patch), self.z(depth)], dim=-1)
        return self.merge[1](self.merge[0](h))

    def head_from_features(self, f: Tensor) -> Tensor:
        return self.merge[2](f).squeeze(-1)

    def forward(self, patch: Tensor, depth: Tensor) -> Tensor:
        return self.head_from_features(self.features(patch, depth))


@torch.no_grad()
def aligned_patches(ds, view: int = 0, size: int = PATCH, scale: float = 1.6) -> tuple[Tensor, Tensor]:
    """(N*A, 1, 32, 32) grasp-aligned depth patches and (N*A, 1) gripper depths.

    The patch is centred on the projected grasp point and rotated so the projected jaw
    axis is horizontal, which is what makes the CNN's filters see a canonical grasp.
    """
    depth = ds.all_obs[:, view, 3:4]                     # (N, 1, H, W), the D of RGB-D
    n, _, h, w = depth.shape
    acts = ds.actions                                    # (N, A, 7)
    a = acts.shape[1]

    centre = acts[..., :3].reshape(-1, 3).numpy()
    uv = torch.from_numpy(project(centre, view, w)).float().reshape(n, a, 2)

    # jaw axis in the image, from the rotated local x of the hand frame
    quats = acts[..., 3:7].reshape(-1, 4).numpy()
    axis = torch.stack([
        torch.from_numpy(quat_to_matrix(q) @ [1.0, 0.0, 0.0]).float() for q in quats
    ]).reshape(n, a, 3)
    tip = acts[..., :3] + 0.05 * axis
    uv_tip = torch.from_numpy(project(tip.reshape(-1, 3).numpy(), view, w)).float().reshape(n, a, 2)
    d_uv = uv_tip - uv
    theta = torch.atan2(d_uv[..., 1], d_uv[..., 0])      # (N, A)
    half_px = (d_uv.norm(dim=-1) * scale).clamp(4.0, w / 2)

    # sampling grid in patch coordinates, rotated and placed on the image
    lin = torch.linspace(-1.0, 1.0, size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).reshape(1, size * size, 2)      # (1, P, 2)

    cos, sin = theta.cos().reshape(-1, 1), theta.sin().reshape(-1, 1)
    rot = torch.stack([
        torch.cat([cos, -sin], dim=1), torch.cat([sin, cos], dim=1)
    ], dim=1)                                                            # (N*A, 2, 2)
    pts = base * half_px.reshape(-1, 1, 1)
    pts = torch.einsum("bij,bpj->bpi", rot, pts) + uv.reshape(-1, 1, 2)
    grid = torch.stack([2 * pts[..., 0] / (w - 1) - 1, 2 * pts[..., 1] / (h - 1) - 1], dim=-1)

    src = depth.repeat_interleave(a, dim=0)                              # (N*A, 1, H, W)
    patches = F.grid_sample(src, grid.reshape(-1, size, size, 2),
                            mode="bilinear", padding_mode="border", align_corners=True)

    gripper_depth = (acts[..., 2].reshape(-1, 1) - TABLE_Z)
    return patches, gripper_depth


def train_gqcnn(ds_train, ds_val, device, epochs: int = 30, bs: int = 256,
                lr: float = 1e-3) -> GQCNN:
    net = GQCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    xp, xz = aligned_patches(ds_train)
    y = ds_train.succ.reshape(-1)
    m = ds_train.executable.reshape(-1)
    xp, xz, y = xp[m], xz[m], y[m]
    # Standardise the patch the way Dex-Net does: per-patch mean removal, so absolute
    # camera distance cannot stand in for graspability.
    xp = xp - xp.mean(dim=(1, 2, 3), keepdim=True)
    log.info("  GQ-CNN train patches: %d", len(xp))

    for ep in range(epochs):
        perm = torch.randperm(len(xp))
        tot = 0.0
        net.train()
        for i in range(0, len(perm), bs):
            j = perm[i : i + bs]
            logit = net(xp[j].to(device), xz[j].to(device))
            loss = F.binary_cross_entropy_with_logits(logit, y[j].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        if (ep + 1) % 10 == 0:
            log.info("  GQ-CNN epoch %2d  loss %.4f", ep + 1, tot / len(perm))
    net.eval()
    return net


@torch.no_grad()
def score_gqcnn(net: GQCNN, ds, device, bs: int = 512) -> Tensor:
    """(N, A) success probabilities, laid out like run_libero._score."""
    xp, xz = aligned_patches(ds)
    xp = xp - xp.mean(dim=(1, 2, 3), keepdim=True)
    out = []
    for i in range(0, len(xp), bs):
        out.append(torch.sigmoid(net(xp[i : i + bs].to(device), xz[i : i + bs].to(device))).cpu())
    return torch.cat(out).reshape(ds.actions.shape[0], ds.actions.shape[1])
