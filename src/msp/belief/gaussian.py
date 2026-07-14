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

    # -- multi-view fusion ---------------------------------------------------

    @staticmethod
    def fuse(beliefs: list[DiagonalGaussianBelief]) -> DiagonalGaussianBelief:
        """Fuse per-view beliefs into a belief over the fused observation. Formalization Eq 17,
        where `o ∪ o_b` denotes "the belief re-encoded from the fused observation".

        THE MATHEMATICS, because the obvious alternatives are wrong.

        Views are conditionally independent given the state (each camera has its own noise, and
        Eq 1's graphical model makes O_i ⟂ O_j | X). So the posterior over z given several views
        is the normalized product of the per-view posteriors, divided by the prior counted once
        too often::

            q(z | o_1..o_k)  ∝  prod_i q(z | o_i)  /  r(z)^(k-1)

        For diagonal Gaussians with r = N(0, I), that is exact and closed-form in PRECISION space::

            lambda_fused = sum_i lambda_i  -  (k-1) * 1        [prior precision is 1]
            mu_fused     = ( sum_i lambda_i mu_i ) / lambda_fused

        The two tempting shortcuts are both wrong and both hide it well:

          * Averaging the means. This does not sharpen the belief at all -- two independent views
            of the same object should make you MORE certain, and an average leaves the variance
            where it was. Since epistemic variance is the entire signal that drives active
            perception (Eq 16), a fusion that cannot reduce it makes looking again pointless, and
            the information gain of every viewpoint would come out at zero.

          * Concatenating the images and re-encoding. Defensible, but it forces a fixed number of
            views into the architecture, and Algorithm 2 acquires views one at a time until the
            ambiguity drops below tau_U. The number is not known in advance.

        The prior subtraction is what makes this a fusion rather than a pile-up: without it, k
        views of a completely uninformative scene would still collapse the belief to a point.
        """
        if not beliefs:
            raise ValueError("cannot fuse an empty list of beliefs")
        if len(beliefs) == 1:
            return beliefs[0]

        k = len(beliefs)
        precisions = [torch.exp(-b.logvar) for b in beliefs]  # lambda_i
        # Subtract the (k-1) copies of the unit prior precision, and floor so the fused belief can
        # never be sharper than the information actually supports.
        lam = torch.stack(precisions).sum(0) - float(k - 1)
        lam = lam.clamp_min(1e-3)

        weighted = torch.stack([p * b.mu for p, b in zip(precisions, beliefs, strict=True)]).sum(0)
        mu = weighted / lam
        logvar = -torch.log(lam)
        return DiagonalGaussianBelief(mu, logvar)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        b, d = self.mu.shape
        return f"DiagonalGaussianBelief(B={b}, d={d})"
