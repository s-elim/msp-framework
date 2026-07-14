"""Closed-form divergences between diagonal Gaussians.

Pure functions. No modules, no state, no device assumptions.

The previous implementation wrote the diagonal-Gaussian KL out by hand in two separate
places (the VIB rate term and the TTA trust region), which is how the two ended up with
different conventions. There is exactly one implementation here and both callers use it.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["diagonal_gaussian_kl", "kl_to_standard_normal", "gaussian_nll", "log_normal_nll"]


def diagonal_gaussian_kl(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor,
) -> Tensor:
    """KL( N(mu_q, diag exp(logvar_q)) || N(mu_p, diag exp(logvar_p)) ), summed over the
    last (latent) axis.

    Used by:
      * the TTA trust region, Eq 20:  KL( q'(z) || q_theta(z|o) )
      * `kl_to_standard_normal`, which is the special case p = N(0, I) of Eq 11.

    Closed form, per coordinate::

        0.5 * [ logvar_p - logvar_q + (var_q + (mu_q - mu_p)^2) / var_p - 1 ]

    Args:
        mu_q, logvar_q: parameters of q. Shape (..., d).
        mu_p, logvar_p: parameters of p. Shape (..., d), broadcastable against q.

    Returns:
        Tensor of shape (...) -- the latent axis is summed out, batch axes are preserved.
        The caller decides how to reduce over the batch; this function never means over it.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    per_coord = 0.5 * (
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    )
    return per_coord.sum(dim=-1)


def kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL( N(mu, diag exp(logvar)) || N(0, I_d) ). Formalization Eq 11::

        KL = 0.5 * sum_{j=1..d} ( sigma_j^2 + mu_j^2 - 1 - log sigma_j^2 )

    This is the *rate* of the information bottleneck (Eq 8) under the reference prior
    r(z) = N(0, I_d) prescribed in Section 11.

    Args:
        mu:     (..., d)
        logvar: (..., d)   log sigma^2, NOT log sigma.

    Returns:
        (...) -- latent axis summed out.
    """
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)


_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll(y: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Negative log-likelihood of y under N(mu, exp(logvar)), including the normalizing
    constant.

        -log N(y; mu, sigma^2) = 0.5 * [ log(2 pi) + logvar + (y - mu)^2 * exp(-logvar) ]

    The constant 0.5*log(2pi) does not affect gradients, but it *does* affect the reported
    distortion, which the paper compares across models and plots on the rate-distortion
    frontier. A frontier missing a constant is not a likelihood. We keep it.

    `logvar` is assumed pre-clamped (see `OutcomeDistribution`), which is what keeps
    `exp(-logvar)` from overflowing.
    """
    return 0.5 * (_LOG_2PI + logvar + (y - mu) ** 2 * torch.exp(-logvar))


def log_normal_nll(y: Tensor, log_mu: Tensor, log_logvar: Tensor, eps: float = 1e-6) -> Tensor:
    """Negative log-likelihood of y > 0 under LogNormal(log_mu, exp(log_logvar)).

        -log LogNormal(y) = log(y) + 0.5*[log(2pi) + log_logvar
                                          + (log(y) - log_mu)^2 * exp(-log_logvar)]

    Only valid for STRICTLY POSITIVE y. For slip, use `zero_inflated_lognormal_nll`.
    """
    y_safe = y.clamp_min(eps)
    log_y = torch.log(y_safe)
    return log_y + gaussian_nll(log_y, log_mu, log_logvar)


def zero_inflated_lognormal_nll(
    y: Tensor,
    zero_logit: Tensor,
    log_mu: Tensor,
    log_logvar: Tensor,
    threshold: float = 1e-3,
) -> Tensor:
    """Negative log-likelihood of slip under a ZERO-INFLATED log-normal.

        p(y) =      pi          if y ~= 0            (the grasp held; it did not slip at all)
               (1 - pi) * LogNormal(y; mu, sigma^2)  if y > 0

    with pi = sigmoid(zero_logit).

    WHY THIS AND NOT A PLAIN LOG-NORMAL. Slip is bimodal, and the physics is the reason: a grasp
    that holds slips EXACTLY zero, a grasp that fails slips centimetres, and almost nothing lands
    in between. A unimodal log-normal cannot represent that, so it splits the difference -- and
    because it is also fitting a variance, it becomes confidently wrong, at which point
    exp(-logvar) * (log y - mu)^2 detonates.

    That is not hypothetical. On the RGB-D corpus with a plain log-normal, the slip distortion was
    -1.66 on train and +965.2 on validation, and the validation total was 966 -- i.e. the *entire*
    reported distortion, and therefore the entire rate-distortion frontier, was the slip term
    exploding on held-out data.

    Decomposing into "did it slip?" (a Bernoulli) and "by how much, given that it did?" (a
    log-normal on the positive part) matches the physics exactly and is bounded by construction:
    the Bernoulli term cannot exceed a few nats, and the log-normal term is only ever evaluated on
    points where slip is genuinely positive.

    Args:
        y: observed slip, >= 0.
        zero_logit: logit of P(no slip).
        threshold: slip below this counts as "no slip". 1 mm.
    """
    is_zero = (y < threshold).to(y.dtype)

    # "Did it slip at all?" -- always evaluated, always bounded.
    bernoulli = torch.nn.functional.binary_cross_entropy_with_logits(
        zero_logit, is_zero, reduction="none"
    )

    # "By how much?" -- only where it actually slipped. The clamp keeps log(y) finite at the
    # zero points, whose contribution is masked out anyway.
    magnitude = log_normal_nll(y.clamp_min(threshold), log_mu, log_logvar)

    return bernoulli + (1.0 - is_zero) * magnitude
