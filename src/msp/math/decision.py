"""The inference decision rule and the ambiguity functional.

Formalization Section 4 (Eq 13-15) and Section 5 (Eq 16).

Pure functions over already-computed success probabilities. Nothing here knows about
networks, devices, or beliefs -- which is what lets the equations be tested against
hand-computed values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["SuccessStats", "success_stats", "risk_averse_score", "ambiguity"]


@dataclass(frozen=True)
class SuccessStats:
    """The two statistics that drive every downstream decision in MSP.

    `s` is what the certified set thresholds (Eq 24) and what the score maximizes (Eq 15).
    `v` is what the risk penalty subtracts (Eq 15), what triggers sensing (Eq 16), and what
    the value of information reduces (Eq 17). Every capability of the framework is a
    function of these two numbers.
    """

    s: Tensor  # (B, Na)  mean success probability,      Eq 13
    v: Tensor  # (B, Na)  epistemic variance,            Eq 14


def success_stats(success_probs: Tensor) -> SuccessStats:
    """Marginalize the belief's epistemic spread through the outcome head.

    Formalization Eq 13 and Eq 14::

        s(o, a) = E_{z~q}[ sigma_psi(z, a) ]   ~=  (1/K) sum_k sigma_psi(z_k, a)
        v(o, a) = Var_{z~q}[ sigma_psi(z, a) ] ~=  (1/K) sum_k (sigma_psi(z_k,a) - s)^2

    Note the estimator of Eq 14 is the BIASED (maximum-likelihood) variance: it divides by
    K, not K-1. `torch.var` defaults to `correction=1` (Bessel), which the audited code
    used, inflating v by K/(K-1) -- 14% at K=8. Because v is compared against an absolute
    threshold tau_U (Eq 16) and scaled by an absolute lambda (Eq 15), that inflation is not
    a harmless reparameterization: it changes how often the robot decides to look again.

    We follow the formalization exactly and pass `correction=0`.

    Args:
        success_probs: sigma_psi(z_k, a) for every posterior sample and candidate action.
            Shape (B, K, Na). Must already be probabilities in [0, 1].

    Returns:
        SuccessStats with s, v of shape (B, Na).
    """
    if success_probs.dim() != 3:
        raise ValueError(
            f"success_probs must be (B, K, Na); got shape {tuple(success_probs.shape)}. "
            "The K axis is required -- a point estimate has no epistemic variance."
        )
    if success_probs.shape[1] < 2:
        raise ValueError(
            f"K must be >= 2 to estimate a variance; got K={success_probs.shape[1]}. "
            "Section 11 prescribes K in [8, 32]."
        )
    s = success_probs.mean(dim=1)
    v = success_probs.var(dim=1, correction=0)  # Eq 14: 1/K, not 1/(K-1)
    return SuccessStats(s=s, v=v)


def risk_averse_score(stats: SuccessStats, lambda_risk: float) -> Tensor:
    """The selection objective. Formalization Eq 15::

        a*(o) = argmax_{a in A_cand} [ s(o, a) - lambda * v(o, a) ],   lambda >= 0

    lambda trades expected success against epistemic risk. This function returns the score;
    the argmax is taken by the inference engine *over the certified set only* (Eq 24), which
    is why selection and certification are deliberately separated into different modules.

    Returns:
        (B, Na) scores.
    """
    if lambda_risk < 0.0:
        raise ValueError(f"lambda must be >= 0 (Eq 15); got {lambda_risk}.")
    return stats.s - lambda_risk * stats.v


def ambiguity(stats: SuccessStats) -> Tensor:
    """The epistemic ambiguity functional. Formalization Eq 16::

        U(o) = E_{a~rho}[ v(o, a) ]  ~=  (1/|A_cand|) sum_{a in A_cand} v(o, a)

    U is the quantity that BOTH active perception (by choosing a measurement, Eq 17) and
    test-time adaptation (by assimilating an outcome, Eq 20) exist to reduce. That they
    reduce the same functional is contribution C3.

    Returned PER SCENE, shape (B,). The audited implementation meaned over the batch as
    well, collapsing B scenes to a single scalar, so one ambiguous scene in a batch could
    not trigger sensing on its own. Sensing is a per-scene decision; never reduce this
    further inside the library.
    """
    return stats.v.mean(dim=-1)
