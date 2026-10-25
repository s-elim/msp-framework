"""The two learned modules of the outcome scorers. Selection, abstention, multi-view fusion and
test-time adaptation are inference procedures over them.

    A: BeliefEncoder   o          -> q_theta(z | o)      (a Belief, not a tuple)
    B: OutcomeHead     (z, a, g)  -> p_psi(y | z, a, g)  (an OutcomeDistribution)

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from msp.belief import Belief, DiagonalGaussianBelief
from msp.types import OutcomeDistribution

__all__ = [
    "Backbone",
    "BeliefEncoder",
    "MLPBackbone",
    "OutcomeHead",
    "ResNetBackbone",
]


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
    """For vector observations: ablations and fast tests.

    Its existence is deliberate. It lets the training, inference and calibration pipeline be
    exercised end to end in seconds on vector data, so a pipeline bug cannot hide behind "the
    vision model needs more epochs".
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
    the WHOLE SCENE. Contact regions and centre-of-mass indicators matter at high
    resolution, a spatial, per-object property. Global average pooling destroys that locality,
    and it makes multi-object scenes unrepresentable.

    This is adequate for single-object benchmarks such as LM-Grasp. A per-object, spatially
    structured encoder is the natural next step. Documented here rather than discovered later.
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

    MULTI-VIEW, AND WHY THE ENCODER MUST DO THE FUSING ITSELF.

    Fusing views means re-encoding the belief from the fused observation after each new view:
    (mu, logsig2) = Encoder_theta(o'). The ENCODER consumes the set of views. It is not a
    post-hoc combination of separately-encoded beliefs.

    That distinction is not pedantic, and getting it wrong produces a result that looks like
    success. The obvious alternative -- encode each view independently and fuse the Gaussians as a
    product (precision-weighted, the exact Bayesian rule for CONDITIONALLY INDEPENDENT
    observations) -- assumes an independence that camera views of one object flatly do not have.
    Measured on the real RGB-D corpus with that fusion:

        * re-fusing the CURRENT view with itself produced the LARGEST information gain of all
          eight candidates -- you "learn" the most by not moving, because the arithmetic simply
          double-counts the same evidence;
        * the OPPOSITE view, which sees the most genuinely new surface, scored the LOWEST.

    An acquisition network trained on those targets learns to stay exactly where it is. The
    reported 74% ambiguity reduction was precision arithmetic, not information.

    A permutation-invariant set encoder has no such problem: duplicate views collapse under the
    mean and contribute nothing, and correlations between neighbouring views are LEARNED from
    data rather than assumed away. Train it with a random number of views per scene and it
    handles any count at deployment, which is what active perception needs -- views are acquired
    one at a time until the ambiguity is low enough, so the count is not known in advance.
    """

    def __init__(self, backbone: Backbone, latent_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.head = nn.Sequential(
            nn.Linear(backbone.output_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim)
        )

    def encode_views(self, obs: Tensor) -> Tensor:
        """(B, V, C, H, W) -> (B, F). Permutation-invariant aggregation over the view axis."""
        b, v = obs.shape[:2]
        flat = obs.reshape(b * v, *obs.shape[2:])
        feats = self.backbone(flat).reshape(b, v, -1)
        return feats.mean(dim=1)

    def forward(self, obs: Tensor) -> Belief:
        """`obs` may be a single view (B, C, H, W) or a SET of views (B, V, C, H, W).

        For images, a set is 5-D and a single view is 4-D. For vector observations, a set is 3-D
        and a single vector is 2-D. Both are handled, so the pipeline can also run on vector
        data.
        """
        is_image = obs.dim() >= 4
        is_set = obs.dim() == 5 if is_image else obs.dim() == 3

        h = self.encode_views(obs) if is_set else self.backbone(obs)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return DiagonalGaussianBelief(mu, logvar)  # clamps logvar internally


# ======================================================================================
# Module B -- the outcome head
# ======================================================================================


class OutcomeHead(nn.Module):
    """p_psi(y | z, a, g): the outcome model and the action scorer. Without it z has no
    training signal.

    The gripper descriptor `g` conditions the prediction on the gripper, so one model can cover
    a gripper FAMILY rather than one gripper. It is a first-class argument here rather than a
    later bolt-on, because retrofitting it would change the meaning of every trained checkpoint.

    SHAPE CONTRACT -- read this. `z` is (B, K, d) and `a` is (B, Na, Ad), and the head
    evaluates the full K x Na CROSS PRODUCT: every posterior sample against every candidate
    action. An earlier head accepted (B, K, d) and (B, Na, ...) and silently returned the
    DIAGONAL when K happened to equal Na, quietly destroying the marginalization over z.
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

        # THE ACTION MUST INTERACT MULTIPLICATIVELY WITH THE BELIEF, AND A CONCAT-MLP WILL NOT DO IT.
        #
        # Whether a grasp holds depends on the ANGLE BETWEEN THE JAWS AND THE OBJECT -- a product of
        # something in `a` (the grasp's orientation) and something in `z` (the object's pose). The
        # head used to be a 2x256 MLP on cat[z, a] and had to discover that interaction from scratch.
        # It did not. It converged instead to the one signal available without any interaction at
        # all -- the per-object base success rate -- and IGNORED THE ACTION OUTRIGHT: across the 8
        # candidate grasps of a scene its prediction had std 0.0027, against 0.4148 in the truth.
        # Within-scene AUC was 0.524, i.e. chance: it could not say which of an object's own grasps
        # would work.
        #
        # It looked healthy anyway, because the POOLED AUC read 0.716. On a corpus whose per-object
        # base rates run from 0.043 (macaroni) to 0.999 (bbq_sauce), a pooled AUC rewards a model
        # that merely RECOGNISES THE OBJECT. The pooled score even exceeded both strata it was
        # computed from (0.691 box / 0.400 curved) -- Simpson's paradox, and the tell.
        #
        # Neither more information nor a better estimator rescued it: raising beta lifted the rate
        # from 3.9 to 11.6 nats (within-scene AUC still 0.524), and estimating the distortion with
        # K=8 posterior samples instead of 1 changed nothing (0.528). Fitting a WIDER, DEEPER head to
        # the same frozen encoder reached 0.615 against an oracle ceiling of 0.685 -- which localised
        # the fault to the head's form, not to the belief.
        #
        # So the interaction is now built in. FiLM: `z` emits a per-feature scale and shift that
        # modulate the action embedding. The product is structural, not something to be learned.
        self.act_embed = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU())
        self.film = nn.Linear(latent_dim + gripper_dim, 2 * hidden)

        in_dim = hidden + latent_dim + gripper_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # 6 params: succ_logit, margin_mu, margin_logvar,
        #           slip_zero_logit, slip_log_mu, slip_log_logvar
        # Slip is ZERO-INFLATED (a held grasp slips exactly zero), so it needs a
        # Bernoulli "did it slip" head on top of the log-normal magnitude.
        self.out = nn.Linear(hidden, 6)

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
        z_e = z.unsqueeze(2).expand(b, k, na, d)  # (B, K, Na, d)
        cond = [z_e]
        if self.gripper_dim > 0:
            if gripper is None:
                raise ValueError("head was built with gripper_dim > 0 but gripper=None")
            cond.append(gripper.view(b, 1, 1, self.gripper_dim).expand(b, k, na, -1))
        elif gripper is not None:
            raise ValueError("gripper supplied but head was built with gripper_dim = 0")
        c = torch.cat(cond, dim=-1)  # (B, K, Na, d + Gd)

        # FiLM: the belief modulates the action embedding, so a * z enters the network as a PRODUCT.
        # `1 + gamma` keeps the map near identity at init, so the action is attended to from step 0
        # rather than having to fight its way in.
        a_e = self.act_embed(actions).unsqueeze(1).expand(b, k, na, -1)  # (B, K, Na, H)
        gamma, shift = self.film(c).chunk(2, dim=-1)
        h = torch.nn.functional.gelu(a_e * (1.0 + gamma) + shift)

        p = self.out(self.trunk(torch.cat([h, c], dim=-1)))  # (B, K, Na, 6)
        if squeeze_k:
            p = p.squeeze(1)  # (B, Na, 6)

        return OutcomeDistribution(
            succ_logit=p[..., 0:1],
            margin_mu=p[..., 1:2],
            margin_logvar=p[..., 2:3],
            slip_zero_logit=p[..., 3:4],
            slip_log_mu=p[..., 4:5],
            slip_log_logvar=p[..., 5:6],
        )

    def success_probs(
        self, z: Tensor, actions: Tensor, gripper: Tensor | None = None
    ) -> Tensor:
        """sigma_psi(z_k, a) for the full cross product. (B, K, Na)."""
        dist = self.forward(z, actions, gripper=gripper)
        return dist.success_prob().squeeze(-1)
