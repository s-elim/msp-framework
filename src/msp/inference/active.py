"""Active perception: choose the measurement that most reduces outcome ambiguity. Section 5.

    if U(o) > tau_U and the sensing budget remains:
        b* = argmax_b alpha_omega(o, b)
        move the camera, fuse the new view into the belief, re-score
    else:
        certify and act, or abstain

This is contribution C3's first half, and it is the thing that turns MSP's 90% abstention rate from
an apology into a result. From a single photograph you cannot see friction, mass or the centre of
mass, so on most scenes no action can be certified at 90% confidence and the framework correctly
declines. The way out is not a better classifier -- the information is genuinely not in the pixels.
The way out is to LOOK AGAIN, at the viewpoint that resolves the specific ambiguity that is blocking
this specific grasp. That is what Eq 17 computes and what alpha_omega amortizes.

THREE FAILURES THE AUDITED VERSION SHIPPED, ALL OF THEM SILENT:

  * The acquisition network was never trained. No loss, no target, no look-ahead -- an argmax over
    randomly-initialized weights steered the camera. `AcquisitionNet` here refuses to run until it
    has been fitted, and `fitted` is a buffer set by the training loop, so a randomly-initialized
    net cannot be used by accident.
  * `argmax(dim=-1)` was taken over a trailing size-1 axis, so the selected view was the constant 0.
  * The sensing trigger meaned the ambiguity over the BATCH, so a single ambiguous scene could not
    trigger a look of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.belief import Belief, DiagonalGaussianBelief
from msp.math.decision import ambiguity, success_stats
from msp.math.voi import information_gain
from msp.models.nets import AcquisitionNet, BeliefEncoder, OutcomeHead

__all__ = ["ActiveConfig", "ActivePerception", "compute_true_information_gain"]


@dataclass(frozen=True)
class ActiveConfig:
    tau_ambiguity: float = 0.02  # tau_U: sense when U(o) exceeds this
    sensing_budget: int = 3  # Alg 2: "and sensing budget remains"
    num_samples: int = 32  # K, for the ambiguity estimate


# ======================================================================================
# Eq 17 -- the ground-truth information gain, by rendered look-ahead
# ======================================================================================


@torch.no_grad()
def compute_true_information_gain(
    encoder: BeliefEncoder,
    head: OutcomeHead,
    current_views: list[Tensor],
    candidate_views: Tensor,
    actions: Tensor,
    num_samples: int = 32,
) -> tuple[Tensor, Tensor]:
    """IG_true(x, o, b) for every candidate viewpoint. Formalization Eq 17.

    Computable ONLY in simulation, and that is the point: because x is known, the future
    observation o_b can simply be RENDERED rather than predicted, so no generative observation
    model -- no world model -- is needed anywhere. Eq 18 then amortizes these exact values into
    `alpha_omega`, which is what runs at deployment where nothing can be rendered.

    Args:
        current_views: the observations already taken, each (B, 4, H, W).
        candidate_views: RENDERED look-aheads, (B, Nb, 4, H, W). One per candidate b.
        actions: (B, Na, Ad), the action set over which ambiguity is averaged (Eq 16's rho).

    Returns:
        (ig, u_before) with shapes (B, Nb) and (B,).
    """
    # The ENCODER fuses the view set. See `BeliefEncoder`'s docstring for why a
    # product-of-Gaussians fusion of separately-encoded views is wrong here, and why it fails in a
    # way that looks like success: it double-counts correlated evidence, so re-fusing the CURRENT
    # view scored the HIGHEST information gain of all eight candidates and an acquisition net
    # trained on it would learn never to move.
    now = torch.stack(current_views, dim=1)  # (B, k, 4, H, W)
    belief_now = encoder(now)

    probs = head.success_probs(belief_now.rsample(num_samples), actions)
    u_before = ambiguity(success_stats(probs))  # (B,)

    b, nb = candidate_views.shape[:2]
    u_after = torch.empty(b, nb, device=u_before.device, dtype=u_before.dtype)

    for j in range(nb):
        fused = encoder(torch.cat([now, candidate_views[:, j : j + 1]], dim=1))
        p = head.success_probs(fused.rsample(num_samples), actions)
        u_after[:, j] = ambiguity(success_stats(p))

    return information_gain(u_before, u_after), u_before


# ======================================================================================
# Deployment
# ======================================================================================


class ActivePerception:
    """Selects the next viewpoint and fuses it into the belief.

    Args:
        acquisition: the FITTED alpha_omega. An unfitted net is rejected -- see the module docstring.
    """

    def __init__(
        self,
        encoder: BeliefEncoder,
        acquisition: AcquisitionNet,
        config: ActiveConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.acquisition = acquisition
        self.cfg = config or ActiveConfig()

    def should_sense(self, u: Tensor, views_taken: int) -> Tensor:
        """U(o) > tau_U, PER SCENE, and only while the sensing budget lasts. (B,) bool."""
        if u.dim() != 1:
            raise ValueError(f"U must be per-scene (B,); got {tuple(u.shape)}")
        if views_taken >= self.cfg.sensing_budget:
            return torch.zeros_like(u, dtype=torch.bool)
        return u > self.cfg.tau_ambiguity

    @torch.no_grad()
    def select_view(self, obs: Tensor, view_ids: Tensor) -> Tensor:
        """b* = argmax_b alpha_omega(o, b). Formalization Eq 18, at deployment.

        Args:
            obs: the current observation, (B, 4, H, W).
            view_ids: candidate viewpoint indices, (B, Nb).

        Returns:
            (B,) the chosen viewpoint index per scene. NOT (B, Nb) -- the audited version argmaxed
            over a trailing size-1 axis and returned a constant zero of the wrong shape.
        """
        if not bool(self.acquisition.fitted):
            raise RuntimeError(
                "AcquisitionNet has not been fitted. Eq 18 regresses onto IG_true from a rendered "
                "look-ahead; without that training, an argmax over its weights is an argmax over "
                "noise. Train it (see scripts/train_acquisition.py) or disable active perception."
            )
        feats = self.encoder.backbone(obs)  # (B, F)
        scores = self.acquisition(feats, view_ids)  # (B, Nb)
        best = scores.argmax(dim=-1)  # (B,)
        return torch.gather(view_ids, 1, best.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def fuse(self, beliefs: list[Belief]) -> Belief:
        """Fold a newly acquired view into the belief. Eq 17's `o ∪ o_b`."""
        return DiagonalGaussianBelief.fuse(beliefs)  # type: ignore[arg-type]
