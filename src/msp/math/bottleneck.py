"""Variational information bottleneck for the learned outcome scorers (an ablation).

The training loss puts one multiplier per outcome dimension on the outcome distortion and unit
weight on the rate::

    L  =  sum_j beta_j * D_j  +  R

D_j is the negative log-likelihood of outcome dimension j (success, margin, slip) and R is the
KL from the image-code posterior to N(0, I), a variational bound on I(Z; O). L follows from the
Lagrangian  min I(Z; O) - sum_j beta_j * I(Z; Y_j | A)  after substituting the bounds.

So: **beta MULTIPLIES the distortion and the rate carries unit weight.**

    beta -> infinity  ==>  the outcome terms dominate  ==>  z keeps what predicts outcomes
    beta -> 0         ==>  the rate dominates          ==>  total compression

An earlier implementation computed `L = sum_j D_j / beta_j + R`: the same family of solutions
with every statement about beta reversed. The direction is asserted in
`tests/math/test_bottleneck.py::test_regression_beta_direction`.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

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
    """Per-outcome-dimension budget (beta_succ, beta_margin, beta_slip).

    Larger beta_j => outcome dimension j must be resolved more precisely => less
    compression of the information z retains about it.
    """

    succ: float = 1.0
    margin: float = 1.0
    slip: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("succ", self.succ), ("margin", self.margin), ("slip", self.slip)):
            if value <= 0.0:
                raise ValueError(
                    f"beta_{name} must be > 0 (the objective requires beta > 0); got {value}."
                )

    @classmethod
    def uniform(cls, beta: float) -> BetaSchedule:
        """The scalar-beta case, applied to every outcome dimension.

        WARNING -- THIS IS ALMOST NEVER WHAT YOU WANT, AND IT SILENTLY BREAKS THE MODEL.

        The three D_j are negative log-likelihoods on INCOMMENSURATE SCALES: D_succ is a Bernoulli
        BCE (~0.6 nats), D_slip a zero-inflated log-normal (~3.9), D_margin a Gaussian NLL of a
        continuous quantity. An equal beta is therefore NOT an equal budget -- capacity is handed out
        in proportion to the accidental magnitude of each likelihood, and `succ`, the only outcome
        the decision rule ever reads, is the smallest of the three.

        What that did, measured in an earlier study: the shared trunk optimised for slip and
        margin, the head stopped attending to the action, and within-scene ranking fell to
        chance. Re-weighting to beta_succ = 300, beta_margin = beta_slip = 0.01 restored the
        action response; nothing else tried moved it (beta's magnitude, K, a FiLM head, the
        marginal-likelihood loss, KL annealing, a deterministic warmup).

        Use `success_first` unless you have a reason not to, and if you do reach for a
        genuinely uniform budget, normalize the D_j first.
        """
        return cls(succ=beta, margin=beta, slip=beta)

    @classmethod
    def success_first(cls, beta: float = 300.0) -> BetaSchedule:
        """The sensible default: spend the budget on the outcome the DECISION reads.

        `succ` is what the planner ranks and the certificate tests. `margin` and `slip` are
        auxiliary regression targets kept at a small weight, enough to keep the outcome family
        three-dimensional without letting their much larger NLLs drown the Bernoulli term. See
        `uniform` for what happens when they do.
        """
        return cls(succ=beta, margin=0.01, slip=0.01)


@dataclass(frozen=True)
class DistortionTerms:
    """Per-dimension outcome distortion, before beta weighting. All scalars."""

    succ: Tensor
    margin: Tensor
    slip: Tensor

    def weighted_sum(self, beta: BetaSchedule) -> Tensor:
        """sum_j beta_j * D_j -- the outcome side of the objective."""
        return beta.succ * self.succ + beta.margin * self.margin + beta.slip * self.slip


@dataclass(frozen=True)
class VIBTerms:
    """Everything a caller needs to log one step's rate and distortion."""

    loss: Tensor  # the scalar to backward()
    rate: Tensor  # R(theta), in nats
    distortion: DistortionTerms  # D_j, in nats
    total_distortion: Tensor  # sum_j D_j, UNweighted

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
    I(Z; O), with r(z) = N(0, I_d), in closed form.

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

    The factorization p_psi(y|z,a) = Bern(succ) * N(margin) * LogNormal(slip) makes the
    negative log-likelihood a sum of three independent terms, which is what allows a
    per-dimension beta to be meaningful at all.

    IMPORTANCE WEIGHTS. A boundary-focused action sampler makes the training distribution a
    *biased* sample of the action space, and an unweighted mean over it estimates the wrong
    expectation. `weights` are the self-normalized importance weights
    w_i = (drho_target/drho_proposal) supplied by the sampler; they debias the estimate.
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
        # x: (B, Na, 1) or (B, K, Na, 1) NLLs -> scalar.
        #
        # THE K AXIS IS REDUCED WITH A LOG-MEAN-EXP, NOT A MEAN, AND THE DIFFERENCE IS THE MODEL.
        #
        # The decision rule reads out the MARGINAL probability:
        #
        #       s(o, a)  =  E_{z~q(z|o)} [ p(succ | z, a) ]
        #
        # so the loss that matches it is the MARGINAL negative log-likelihood, -log E_z[p]. The
        # obvious alternative, E_z[-log p], is a DIFFERENT objective -- by Jensen it is an upper
        # bound (it is the ELBO's reconstruction term; -log E_z[p] is IWAE's) -- and here the gap
        # between them decides whether the model works at all.
        #
        # E_z[-log p] asks the head to be right for EVERY draw of z. The posterior is
        # noise-dominated (the noise was about twice the signal when measured), so the only way
        # to be right for every draw is to be ROBUST TO z -- i.e. to ignore its fine structure.
        # But the object's POSE lives in exactly that fine structure, and whether a grasp holds
        # depends on the angle between the jaws and the object, an interaction between the
        # action and the pose. So the head threw the action away and predicted the per-object
        # base rate: prediction std across a scene's 8 candidate grasps 0.003, against 0.415 in
        # the truth, and a within-scene AUC of 0.52 -- chance -- while the POOLED AUC read 0.72
        # by simply recognising the object.
        #
        # -log E_z[p] asks only that the head be right on AVERAGE over z, so a good draw can carry a
        # bad one. That permits the model to USE the pose in z, which is the whole point of the
        # belief. Nothing else fixed it: not beta (30 -> 300 raises the rate 3.9 -> 11.6 nats, AUC
        # unmoved), not K (1 -> 8, unmoved), not the head's form (FiLM, unmoved).
        #
        # Weights are per (scene, action) and do not depend on z, so they broadcast across K.
        x = x.squeeze(-1)
        if x.dim() == 3:  # (B, K, Na): x = -log p(y | z_k, a)
            # -log( (1/K) sum_k p )  =  -logsumexp_k(-x) + log K
            x = -torch.logsumexp(-x, dim=1) + math.log(x.shape[1])
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
    rate_weight: float = 1.0,
) -> VIBTerms:
    """The full training loss::

        L(theta, psi)  =  sum_j beta_j * D_j  +  R

    There is no reconstruction, pose or geometry term: supervision comes only from the
    simulated outcomes of actions (`target`).

    Returns:
        VIBTerms. `.loss` is the scalar to backward(); `.rate` and `.total_distortion` are
        the logged rate and distortion.
    """
    d = distortion(pred, target, weights=weights)
    r = rate(mu, logvar, weights=None)  # rate is per-scene; action weights do not apply

    # `rate_weight` is a TRAINING-SCHEDULE device and must reach 1.0, at which point this is exactly
    # the objective above. It exists to escape POSTERIOR COLLAPSE.
    #
    # The rate term R = KL(q||N(0,I)) pulls sigma -> 1 and mu -> 0. The distortion is supposed
    # to pull back -- but it only can if the head USES the fine structure of z, and early in
    # training it cannot: z is noise. So the head learns to be robust to z, dD/dsigma goes to
    # ~0, nothing opposes R, and sigma parks at the prior. Measured in an earlier study: the
    # noise was about three times the signal, and this is a FIXED POINT -- raising beta tenfold
    # left sigma where it was, because beta multiplies a distortion whose gradient w.r.t. sigma
    # is already zero.
    #
    # The damage is specific: grasp success depends on the ANGLE between the jaws and the
    # object, an interaction between the action and the POSE inside z. Through a noise-dominated
    # z that interaction cannot form, so the head discards the action and predicts the
    # per-object base rate: on the same frozen encoder, z = mu ranked grasps well above chance,
    # sampled z and end-to-end training near chance.
    #
    # Annealing R in from zero lets the head learn the interaction while sigma is still free to be
    # small. Once dD/dsigma is nonzero, the two terms can actually negotiate, which is what the
    # objective assumes.
    loss = d.weighted_sum(beta) + rate_weight * r
    total_d = d.succ + d.margin + d.slip

    return VIBTerms(loss=loss, rate=r, distortion=d, total_distortion=total_d)
