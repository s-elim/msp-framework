"""The two learned modules of MSP. Step 5 of the blueprint reduces the framework to exactly
these: everything else -- selection, abstention, active perception, adaptation -- is an
inference procedure over them.

    A: BeliefEncoder   o          -> q_theta(z | o)      (a Belief, not a tuple)
    B: OutcomeHead     (z, a, g)  -> p_psi(y | z, a, g)  (an OutcomeDistribution)

Plus the amortized acquisition network (Eq 18), which is not a third learned module in the
formal sense -- it estimates a functional of A and B -- but does carry its own weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from msp.belief import Belief, DiagonalGaussianBelief
from msp.types import OutcomeDistribution

__all__ = ["Backbone", "MLPBackbone", "ResNetBackbone", "BeliefEncoder", "OutcomeHead"]


# ======================================================================================
# Backbones
# ======================================================================================


class Backbone(nn.Module, ABC):
    """Observation feature extractor. Contract: (B, ...) -> (B, output_dim)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, obs: Tensor) -> Tensor: ...


class MLPBackbone(Backbone):
    """For vector observations: synthetic worlds, ablations, and fast integration tests.

    Its existence is deliberate. It lets the entire training/inference/calibration pipeline
    be exercised end to end in seconds against `SyntheticOracle`, where the right answer is
    known -- so a pipeline bug cannot hide behind "the vision model needs more epochs".
    """

    def __init__(self, input_dim: int, output_dim: int = 128, hidden: int = 256) -> None:
        super().__init__(output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class ResNetBackbone(Backbone):
    """RGB-D backbone. 4-channel stem, ImageNet-initialized on the RGB channels.

    NOTE ON A KNOWN LIMITATION. This pools to a single global vector, so z is one code for
    the WHOLE SCENE. The formalization's physical reading is that "contact regions and
    center-of-mass indicators are preserved to high resolution because dM/dx is large there"
    -- a spatial, per-object claim. Global average pooling destroys exactly that locality,
    and it makes multi-object scenes (reviewer attack 24) unrepresentable.

    This is adequate for single-object benchmarks and it is what the paper's V1 scope
    (rigid, single object) requires. It is NOT the right inductive bias for the full claim,
    and a per-object / spatially-structured encoder is the first thing to build after the
    core result lands. Documented here rather than discovered later.
    """

    def __init__(self, output_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__(output_dim)
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        stem = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                stem.weight[:, :3] = net.conv1.weight
                # Depth channel: initialize from the mean RGB filter rather than the red
                # channel alone. Depth is a luminance-like signal, not a colour channel.
                stem.weight[:, 3] = net.conv1.weight.mean(dim=1)
        net.conv1 = stem
        net.fc = nn.Linear(net.fc.in_features, output_dim)
        self.net = net

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


# ======================================================================================
# Module A -- the belief encoder
# ======================================================================================


class BeliefEncoder(nn.Module):
    """q_theta(z | o). Returns a `Belief`, never a raw (mu, logvar) tuple.

    Returning the abstraction rather than the tuple is what makes the frozen-variance bug
    unwritable downstream: the only way to reach the outcome head is `Belief.rsample`.
    """

    def __init__(self, backbone: Backbone, latent_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.head = nn.Sequential(
            nn.Linear(backbone.output_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim)
        )

    def forward(self, obs: Tensor) -> Belief:
        h = self.backbone(obs)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return DiagonalGaussianBelief(mu, logvar)  # clamps logvar internally


# ======================================================================================
# Module B -- the outcome head
# ======================================================================================


class OutcomeHead(nn.Module):
    """p_psi(y | z, a, g). The module that DEFINES sufficiency: without it, z has no
    objective. Also the action scorer.

    The gripper descriptor `g` (V2, Change 2) makes the estimand sufficiency with respect to
    a gripper FAMILY rather than one gripper, which is the cross-embodiment claim. It is a
    first-class argument here rather than a later bolt-on, because retrofitting it would
    change the meaning of every trained checkpoint.

    SHAPE CONTRACT -- read this. `z` is (B, K, d) and `a` is (B, Na, Ad), and the head
    evaluates the full K x Na CROSS PRODUCT: every posterior sample against every candidate
    action. The audited head accepted (B, K, d) and (B, Na, ...) and silently returned the
    DIAGONAL when K happened to equal Na, quietly destroying the marginalization of Eq 13-14.
    Here the cross product is explicit in the shapes and there is no branch to get wrong.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 7,
        gripper_dim: int = 0,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.gripper_dim = gripper_dim

        in_dim = latent_dim + action_dim + gripper_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # 5 params: succ_logit, margin_mu, margin_logvar, slip_log_mu, slip_log_logvar
        self.out = nn.Linear(hidden, 5)

    def forward(
        self, z: Tensor, actions: Tensor, gripper: Tensor | None = None
    ) -> OutcomeDistribution:
        """z: (B, K, d) or (B, d).  actions: (B, Na, Ad).  gripper: (B, Gd) or None.

        Returns OutcomeDistribution with fields of shape (B, K, Na, 1), or (B, Na, 1) if a
        (B, d) latent was supplied (the training path, one sample per scene).
        """
        squeeze_k = z.dim() == 2
        if squeeze_k:
            z = z.unsqueeze(1)  # (B, 1, d)
        if z.dim() != 3:
            raise ValueError(f"z must be (B, d) or (B, K, d); got {tuple(z.shape)}")
        if actions.dim() != 3:
            raise ValueError(f"actions must be (B, Na, Ad); got {tuple(actions.shape)}")

        b, k, d = z.shape
        _, na, ad = actions.shape
        if d != self.latent_dim:
            raise ValueError(f"latent dim {d} != head's {self.latent_dim}")
        if ad != self.action_dim:
            raise ValueError(f"action dim {ad} != head's {self.action_dim}")

        # Explicit K x Na outer product. No implicit broadcast that can silently diagonalize.
        z_e = z.unsqueeze(2).expand(b, k, na, d)
        a_e = actions.unsqueeze(1).expand(b, k, na, ad)
        parts = [z_e, a_e]

        if self.gripper_dim > 0:
            if gripper is None:
                raise ValueError("head was built with gripper_dim > 0 but gripper=None")
            parts.append(gripper.view(b, 1, 1, self.gripper_dim).expand(b, k, na, -1))
        elif gripper is not None:
            raise ValueError("gripper supplied but head was built with gripper_dim = 0")

        p = self.out(self.trunk(torch.cat(parts, dim=-1)))  # (B, K, Na, 5)
        if squeeze_k:
            p = p.squeeze(1)  # (B, Na, 5)

        return OutcomeDistribution(
            succ_logit=p[..., 0:1],
            margin_mu=p[..., 1:2],
            margin_logvar=p[..., 2:3],
            slip_log_mu=p[..., 3:4],
            slip_log_logvar=p[..., 4:5],
        )

    def success_probs(
        self, z: Tensor, actions: Tensor, gripper: Tensor | None = None
    ) -> Tensor:
        """sigma_psi(z_k, a) for the full cross product. (B, K, Na). Feeds Eq 13/14."""
        dist = self.forward(z, actions, gripper=gripper)
        return dist.success_prob().squeeze(-1)
