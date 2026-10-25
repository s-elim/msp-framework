"""The posterior over the image code: q(z | o), used by the learned outcome scorers.

`Belief` is an abstract posterior whose ONLY route to the outcome head is `rsample(K)`, so a
point-estimate shortcut is not expressible, and whose family is swappable.

Two defects of an earlier implementation motivated it. The belief was a bare `(mu, logvar)`
tuple, so test-time adaptation evaluated the outcome head at the MEAN instead of on samples:
the gradient w.r.t. logvar was identically zero and the variance never moved (measured:
||dlogvar|| = 0.0 after 20 steps). And the Gaussian family was hard-coded, although the
belief for a symmetric object is multi-modal and a diagonal Gaussian collapses it to one
mean that may not be a physically possible pose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = ["Belief"]


class Belief(ABC):
    """A distribution over the image code z in R^d.

    Implementations must be differentiable through `rsample` (the reparameterization
    trick), because the VIB distortion term backpropagates into the encoder through the
    sampled z.

    Shape contract:
        batch_shape: (B,)
        event_shape: (d,)
        rsample(K) -> (B, K, d)
    """

    @property
    @abstractmethod
    def batch_shape(self) -> torch.Size:
        """(B,)"""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """d"""

    @abstractmethod
    def rsample(self, num_samples: int) -> Tensor:
        """Draw `num_samples` REPARAMETERIZED samples: (B, K, d).

        Reparameterized means gradients flow to the distribution parameters through the
        returned tensor. This is the only sanctioned way to feed a belief into the outcome
        head. There is deliberately no `.mean()` shortcut on this interface: evaluating the
        head at the mean is the bug this class exists to prevent, and any function that
        needs a point estimate can call `point_estimate()` and be visible in review.
        """

    @abstractmethod
    def kl_to_prior(self) -> Tensor:
        """KL( q(z|o) || r(z) ) with r = N(0, I_d): the rate term.

        Returns (B,) -- one rate per scene, unreduced.
        """

    @abstractmethod
    def kl_to(self, other: Belief) -> Tensor:
        """KL( self || other ): the test-time-adaptation trust region. Returns (B,)."""

    @abstractmethod
    def detach(self) -> Belief:
        """A copy with parameters detached from the autograd graph.

        TTA needs this. An earlier TTA read the encoder's live `mu_0`/`logvar_0` inside its
        KL term, so `backward()` on step 2 tried to traverse an already-freed encoder graph
        and raised `RuntimeError: Trying to backward through the graph a second time`. The
        function had never been run. Detaching at the boundary is the fix, and making it an
        explicit method means the boundary is visible.
        """

    @abstractmethod
    def point_estimate(self) -> Tensor:
        """A single representative z, (B, d). For logging and for the optional readout ONLY.

        NEVER use this to score actions: it discards the epistemic spread that the outcome
        probability, its variance and multi-view fusion are all computed from.
        """

    def expected_success(self, head, actions: Tensor, num_samples: int, gripper=None) -> Tensor:
        """Convenience: push K samples through an outcome head and return the success
        probabilities sigma_psi(z_k, a), shape (B, K, Na).

        Kept on the base class so that every caller marginalizes the same way and no call
        site can quietly substitute the mean.
        """
        z = self.rsample(num_samples)  # (B, K, d)
        return head.success_probs(z, actions, gripper=gripper)
