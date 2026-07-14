"""The manipulation-sufficiency objective: a policy-free, action-conditioned information
bottleneck whose distortion lives in outcome space. Contribution C1.

THE BETA CONVENTION -- read this before touching anything here.

The formalization gives the Lagrangian (Eq 7) as::

    min_{q_theta, p_psi}   I(Z; O)  -  beta * I(Z; Y | A),        beta > 0

Dividing by beta > 0 (which does not change the argmin) and substituting the variational
bounds (Eq 8, Eq 9) gives the scalar training loss (Eq 10)::

    L  =  D  +  (1/beta) * R

where D is the outcome distortion and R is the rate. Generalizing beta per outcome
dimension -- which is the formal handle for philosophy P5, "language sets the sufficiency
budget" -- means putting a separate multiplier on each relevance term in Eq 7::

    min  I(Z;O)  -  sum_j beta_j * I(Z; Y_j | A)
    =>   L  =  sum_j beta_j * D_j  +  R                          (per-dimension form)

So: **beta MULTIPLIES the distortion and the rate carries unit weight.**

    beta -> infinity  ==>  relevance dominates  ==>  sufficiency        (Theorem 3)
    beta -> 0         ==>  rate dominates       ==>  total compression

The audited implementation computed `L = sum_j D_j / beta_j + R`, i.e. beta *divided* the
distortion. That is the same one-parameter family with beta_code = 1/beta_spec, so the
frontier it traces is the same *set* of points -- but every semantic statement about beta
is reversed. Theorem 3's limit (beta -> inf implies sufficiency) became beta -> inf implies
latent collapse, and a per-dimension budget allocated capacity to exactly the wrong
outcomes. The direction of the knob is the scientific claim, so we fix it here and assert
it in `tests/math/test_bottleneck.py::test_beta_direction_matches_theorem_3`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msp.math.divergences import (
    gaussian_nll,
    kl_to_standard_normal,
    zero_inflated_lognormal_nll,
)
from msp.types import Outcome, OutcomeDistribution

__all__ = ["BetaSchedule", "DistortionTerms", "VIBTerms", "rate", "distortion", "vib_objective"]


@dataclass(frozen=True)
class BetaSchedule:
    """Per-outcome-dimension sufficiency budget (beta_succ, beta_margin, beta_slip).

    Larger beta_j => outcome dimension j must be resolved more precisely => less
    compression of the information z retains about it. This is the quantity a language
    instruction would modulate under philosophy P5.
    """

    succ: float = 1.0
    margin: float = 1.0
    slip: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("succ", self.succ), ("margin", self.margin), ("slip", self.slip)):
            if value <= 0.0:
                raise ValueError(f"beta_{name} must be > 0 (Eq 7 requires beta > 0); got {value}.")

    @classmethod
    def uniform(cls, beta: float) -> BetaSchedule:
        """The scalar-beta case of Eq 10, applied to every outcome dimension."""
        return cls(succ=beta, margin=beta, slip=beta)


@dataclass(frozen=True)
class DistortionTerms:
    """Per-dimension outcome distortion, before beta weighting. All scalars."""

    succ: Tensor
    margin: Tensor
    slip: Tensor

    def weighted_sum(self, beta: BetaSchedule) -> Tensor:
        """sum_j beta_j * D_j -- the relevance side of Eq 7."""
        return beta.succ * self.succ + beta.margin * self.margin + beta.slip * self.slip


@dataclass(frozen=True)
class VIBTerms:
    """Everything a caller needs to log a rate-distortion frontier point."""

    loss: Tensor  # the scalar to backward()
    rate: Tensor  # R(theta), Eq 8, in nats
    distortion: DistortionTerms  # D_j, Eq 9, in nats
    total_distortion: Tensor  # sum_j D_j (UNweighted -- the frontier's y-axis)

    def to_metrics(self) -> dict[str, float]:
        return {
            "loss/total": self.loss.item(),
            "loss/rate_kl": self.rate.item(),
            "loss/distortion_total": self.total_distortion.item(),
            "loss/distortion_succ": self.distortion.succ.item(),
            "loss/distortion_margin": self.distortion.margin.item(),
            "loss/distortion_slip": self.distortion.slip.item(),
        }


def rate(mu: Tensor, logvar: Tensor, weights: Tensor | None = None) -> Tensor:
    """R(theta) = E_o[ KL( q_theta(z|o) || r(z) ) ], the variational upper bound on
    I(Z; O). Formalization Eq 8, with r(z) = N(0, I_d) and the closed form of Eq 11.

    Args:
        mu, logvar: belief parameters, (B, d).
        weights: optional per-sample importance weights, (B,), for a non-uniform action or
            scene proposal. Must be normalized by the caller. See `distortion`.

    Returns:
        Scalar. Nats.
    """
    per_sample = kl_to_standard_normal(mu, logvar)  # (B,)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum()


def distortion(
    pred: OutcomeDistribution,
    target: Outcome,
    weights: Tensor | None = None,
) -> DistortionTerms:
    """D(theta, psi) = E[ -log p_psi(y | z, a) ], decomposed per outcome dimension.
    Formalization Eq 9, Eq 10 (first term).

    The factorization p_psi(y|z,a) = Bern(succ) * N(margin) * LogNormal(slip) makes the
    negative log-likelihood a sum of three independent terms, which is what allows a
    per-dimension beta to be meaningful at all.

    IMPORTANCE WEIGHTS. The action measure rho is meant to be boundary-focused (Section 11,
    reviewer attack 27), which makes the training distribution a *biased* sample of the
    action space. An unweighted mean over such a sample estimates the wrong expectation.
    `weights` are the self-normalized importance weights w_i = (drho_target/drho_proposal)
    supplied by the sampler; passing them is what makes the V4 debiasing analysis possible.
    They must sum to 1 over the flattened (B, Na) axis.

    Args:
        pred:   p_psi parameters, each (B, Na, 1).
        target: realized outcomes from M, each (B, Na, 1).
        weights: optional (B, Na) self-normalized importance weights.

    Returns:
        DistortionTerms of scalars, in nats.
    """
    # Bernoulli NLL on the success coordinate. binary_cross_entropy_with_logits is the
    # log-sum-exp-stable form; never take sigmoid then log.
    nll_succ = torch.nn.functional.binary_cross_entropy_with_logits(
        pred.succ_logit, target.succ, reduction="none"
    )
    nll_margin = gaussian_nll(target.margin, pred.margin_mu, pred.margin_logvar)
    nll_slip = zero_inflated_lognormal_nll(
        target.slip, pred.slip_zero_logit, pred.slip_log_mu, pred.slip_log_logvar
    )

    def reduce(x: Tensor) -> Tensor:
        # x: (B, Na, 1) -> scalar
        x = x.squeeze(-1)
        if weights is None:
            return x.mean()
        return (x * weights).sum()

    return DistortionTerms(
        succ=reduce(nll_succ), margin=reduce(nll_margin), slip=reduce(nll_slip)
    )


def vib_objective(
    pred: OutcomeDistribution,
    target: Outcome,
    mu: Tensor,
    logvar: Tensor,
    beta: BetaSchedule,
    weights: Tensor | None = None,
) -> VIBTerms:
    """The full training loss. Formalization Eq 7 (per-dimension form), Eq 10, Eq 11::

        L(theta, psi)  =  sum_j beta_j * D_j  +  R

    Minimizing L over (theta, psi) is the manipulation-sufficiency objective, C1.

    There is no reconstruction term, no pose term, and no geometry regularizer anywhere in
    this function, and none may be added. The entire claim of the paper is that supervision
    comes from what actions *do* (`target`, drawn from the physics kernel M) and never from
    how the object *looks*. A pose loss here would silently reintroduce the estimand the
    paper exists to refute.

    Returns:
        VIBTerms. `.loss` is the scalar to backward(); `.rate` and `.total_distortion` are
        the coordinates of one point on the rate-distortion frontier.
    """
    d = distortion(pred, target, weights=weights)
    r = rate(mu, logvar, weights=None)  # rate is per-scene; action weights do not apply

    loss = d.weighted_sum(beta) + r
    total_d = d.succ + d.margin + d.slip

    return VIBTerms(loss=loss, rate=r, distortion=d, total_distortion=total_d)
