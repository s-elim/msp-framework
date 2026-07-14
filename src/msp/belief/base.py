"""The belief over the manipulation-sufficient statistic: q(z | o).

WHY THIS ABSTRACTION EXISTS.

In the audited implementation the belief was a bare `(mu, logvar)` tuple passed between six
call sites. Two consequences followed directly, and both were scientific errors rather than
software ones:

  1. Test-time adaptation evaluated the outcome head at the belief's MEAN instead of on
     samples from it. The gradient of the likelihood term w.r.t. logvar was therefore
     identically zero, and the belief's variance never moved (measured: ||dlogvar|| = 0.0
     after 20 steps). Since the epistemic variance v is what active perception consumes,
     TTA could not reduce the ambiguity U that contribution C3 claims it reduces.

  2. The Gaussian family was hard-coded everywhere. But Theorem 5 says the belief on a
     symmetric object must be supported on a *group orbit* -- inherently multi-modal. A
     diagonal Gaussian is unimodal and collapses to the mean, which is precisely the
     "physically impossible mean pose" that the blueprint's wrong-assumption #10 condemns.
     The reference implementation committed the error the paper was written to refute.

`Belief` fixes both structurally. It is an abstract posterior whose ONLY route to the
outcome head is `rsample(K)` -- so a point-estimate shortcut is not expressible -- and whose
family is swappable, so `MixtureBelief` can carry an orbit where `DiagonalGaussianBelief`
cannot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = ["Belief"]


class Belief(ABC):
    """A distribution over the sufficient statistic z in R^d.

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
        """KL( q(z|o) || r(z) ) with r = N(0, I_d). The rate term, Eq 8 / Eq 11.

        Returns (B,) -- one rate per scene, unreduced.
        """

    @abstractmethod
    def kl_to(self, other: Belief) -> Tensor:
        """KL( self || other ). The TTA trust region, Eq 20. Returns (B,)."""

    @abstractmethod
    def detach(self) -> Belief:
        """A copy with parameters detached from the autograd graph.

        TTA needs this. The audited TTA read the encoder's live `mu_0`/`logvar_0` inside its
        KL term, so `backward()` on step 2 tried to traverse an already-freed encoder graph
        and raised `RuntimeError: Trying to backward through the graph a second time`. The
        function had never been run. Detaching at the boundary is the fix, and making it an
        explicit method means the boundary is visible.
        """

    @abstractmethod
    def point_estimate(self) -> Tensor:
        """A single representative z, (B, d). For logging and for the optional readout ONLY.

        NEVER use this to score actions: it discards the epistemic spread that Eq 14, Eq 15,
        Eq 16 and Eq 17 are all defined in terms of.
        """

    def expected_success(self, head, actions: Tensor, num_samples: int, gripper=None) -> Tensor:
        """Convenience: push K samples through an outcome head and return the success
        probabilities sigma_psi(z_k, a), shape (B, K, Na).

        Kept on the base class so that every caller marginalizes the same way and no call
        site can quietly substitute the mean.
        """
        z = self.rsample(num_samples)  # (B, K, d)
        return head.success_probs(z, actions, gripper=gripper)
