"""Diagonal-Gaussian belief: q(z|o) = N(mu(o), diag sigma(o)^2).

The default posterior family and the one Section 11 prescribes. It is adequate for
asymmetric objects and it is the family the rate term's closed form (Eq 11) assumes.

IT IS NOT ADEQUATE FOR THEOREM 5. A diagonal Gaussian is unimodal; the belief on a
symmetric object should be supported on the symmetry orbit, which is multi-modal. Use
`MixtureBelief` there. This limitation is stated in the class docstring rather than buried,
because using this family on symmetric objects silently reproduces the exact failure the
paper condemns, and it will not announce itself -- the loss will look fine.
"""

from __future__ import annotations

import torch
from torch import Tensor

from msp.belief.base import Belief
from msp.math.divergences import diagonal_gaussian_kl, kl_to_standard_normal

__all__ = ["DiagonalGaussianBelief"]


class DiagonalGaussianBelief(Belief):
    """N(mu, diag exp(logvar)).

    Args:
        mu:     (B, d)
        logvar: (B, d)   log sigma^2. Clamped on construction for numerical safety:
            exp(0.5 * logvar) must not overflow, and exp(-logvar) must not overflow in any
            downstream Gaussian NLL.
    """

    LOGVAR_MIN: float = -20.0
    LOGVAR_MAX: float = 5.0

    def __init__(self, mu: Tensor, logvar: Tensor) -> None:
        if mu.shape != logvar.shape:
            raise ValueError(
                f"mu {tuple(mu.shape)} and logvar {tuple(logvar.shape)} must have equal shape."
            )
        if mu.dim() != 2:
            raise ValueError(f"expected (B, d); got {tuple(mu.shape)}.")
        self.mu = mu
        self.logvar = logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)

    # -- shape ---------------------------------------------------------------

    @property
    def batch_shape(self) -> torch.Size:
        return self.mu.shape[:1]

    @property
    def latent_dim(self) -> int:
        return self.mu.shape[-1]

    @property
    def stddev(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)

    # -- sampling ------------------------------------------------------------

    def rsample(self, num_samples: int) -> Tensor:
        """z = mu + sigma * eps,  eps ~ N(0, I).  Formalization Section 3.

        Returns (B, K, d). Gradients flow to mu and logvar; the noise is detached by
        construction (torch.randn does not require grad).
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples}.")
        b, d = self.mu.shape
        eps = torch.randn(b, num_samples, d, device=self.mu.device, dtype=self.mu.dtype)
        return self.mu.unsqueeze(1) + self.stddev.unsqueeze(1) * eps

    # -- divergences ---------------------------------------------------------

    def kl_to_prior(self) -> Tensor:
        """Eq 11. Returns (B,)."""
        return kl_to_standard_normal(self.mu, self.logvar)

    def kl_to(self, other: Belief) -> Tensor:
        """KL(self || other). Eq 20's trust region. Returns (B,)."""
        if not isinstance(other, DiagonalGaussianBelief):
            raise TypeError(
                "Closed-form KL is only available between two DiagonalGaussianBelief "
                f"instances; got {type(other).__name__}. For mixed families, use a Monte "
                "Carlo estimate via MixtureBelief.kl_to."
            )
        return diagonal_gaussian_kl(self.mu, self.logvar, other.mu, other.logvar)

    # -- graph management ----------------------------------------------------

    def detach(self) -> DiagonalGaussianBelief:
        return DiagonalGaussianBelief(self.mu.detach(), self.logvar.detach())

    def point_estimate(self) -> Tensor:
        return self.mu

    # -- TTA support ---------------------------------------------------------

    def as_free_parameters(self) -> tuple[Tensor, Tensor]:
        """Leaf tensors (mu, logvar) with requires_grad=True, detached from any encoder
        graph, for TTA to optimize directly (Eq 20).

        Returning *leaves* is what prevents the audited crash: the TTA loop calls
        `backward()` once per step, and if the parameters still carried encoder graph
        history the second call would try to traverse a freed graph.
        """
        mu = self.mu.detach().clone().requires_grad_(True)
        logvar = self.logvar.detach().clone().requires_grad_(True)
        return mu, logvar

    def to(self, device: torch.device | str) -> DiagonalGaussianBelief:
        return DiagonalGaussianBelief(self.mu.to(device), self.logvar.to(device))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        b, d = self.mu.shape
        return f"DiagonalGaussianBelief(B={b}, d={d})"
