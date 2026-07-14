"""Value of information: what is a measurement worth? Formalization Section 5, Eq 16-18.

This is the first half of contribution C3. The second half is test-time adaptation (Eq 19-21), and
the claim C3 makes is that they are the SAME operation in two guises: one chooses a measurement,
the other assimilates the returned one, and both reduce the same functional -- the outcome-class
ambiguity U.

    U(o)     = E_{a~rho}[ Var_{z~q(.|o)}[ sigma_psi(z, a) ] ]                          (Eq 16)
    IG(b)    = U(o)  -  E_{o_b ~ p~(.|o,b)}[ U(o ∪ o_b) ]                              (Eq 17)
    b*       = argmax_b IG(b)
    L_acq(w) = E[ ( alpha_w(o, b) - IG_true(x, o, b) )^2 ]                             (Eq 18)

WHY Eq 17 NEEDS A SIMULATOR AND Eq 18 EXISTS TO AVOID ONE.

IG(b) is an expectation over the observation you have not taken yet. Computing it honestly requires
a generative model of the future observation p~(o_b | o, b) -- which is exactly the world model this
framework is built to avoid.

In simulation you do not need one: the true state x is known, so o_b can simply be RENDERED, U(o ∪
o_b) evaluated exactly, and IG_true obtained without any predictive model at all. Eq 18 then trains
an amortized estimator alpha_omega to regress onto those exact values, and at deployment -- where x
is unknown and nothing can be rendered -- a single forward pass of alpha_omega replaces the entire
look-ahead.

That is the whole trick, and it is why Section 5 is implementable without a world model.

WHY THIS MODULE DOES NOT REDUCE OVER THE BATCH. Sensing is a PER-SCENE decision. The audited
implementation meaned the ambiguity over the batch before comparing it to tau_U, so one highly
ambiguous scene among fifteen certain ones could not trigger a second look.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["information_gain", "acquisition_loss", "expected_ambiguity_after"]


def expected_ambiguity_after(fused_ambiguity: Tensor) -> Tensor:
    """E_{o_b}[ U(o ∪ o_b) ] -- the ambiguity remaining after taking measurement b.

    In simulation this expectation is degenerate: the renderer is deterministic given x, so a
    single rendered o_b IS the expectation, up to sensor noise. The averaging that remains is over
    sensor noise only, which the caller supplies by rendering more than once if it cares.

    Args:
        fused_ambiguity: (B, Nb) -- U of the belief fused from the current view and candidate view b.

    Returns:
        (B, Nb), unchanged. Present so the equation reads as it does on the page, and so a caller
        that later adds a stochastic sensor model has an obvious place to put the expectation.
    """
    return fused_ambiguity


def information_gain(ambiguity_before: Tensor, ambiguity_after: Tensor) -> Tensor:
    """IG(b) = U(o) - E[ U(o ∪ o_b) ]. Formalization Eq 17.

    Positive means the viewpoint is worth taking: fusing it into the belief REDUCES the epistemic
    variance of the outcome predictions, which is the only currency this framework recognizes.
    Geometric coverage, surface visibility and reconstruction completeness do not appear anywhere,
    and that is the point -- Eq 17 measures information in OUTCOME space, so it prefers the view
    that disambiguates a grasp over the view that reveals the most surface.

    A NEGATIVE information gain is meaningful and must not be clipped away: fusing a bad view can
    make the belief WORSE, because a view that sees the object badly still contributes precision to
    the fusion. An acquisition net that never learns this will happily recommend such views.

    Args:
        ambiguity_before: (B,)     U(o)
        ambiguity_after:  (B, Nb)  U(o ∪ o_b) for each candidate b

    Returns:
        (B, Nb) information gain per candidate viewpoint.
    """
    if ambiguity_before.dim() != 1:
        raise ValueError(
            f"U(o) must be per-scene, shape (B,); got {tuple(ambiguity_before.shape)}. Sensing is "
            "a per-scene decision -- never reduce the ambiguity over the batch."
        )
    return ambiguity_before.unsqueeze(-1) - expected_ambiguity_after(ambiguity_after)


def acquisition_loss(predicted_ig: Tensor, true_ig: Tensor) -> Tensor:
    """L_acq(omega) = E[ (alpha_omega(o, b) - IG_true(x, o, b))^2 ]. Formalization Eq 18.

    A plain regression, and deliberately so. The audited repository shipped an `AcquisitionNet` with
    NO loss, NO target and NO look-ahead, and then took an argmax over its randomly-initialized
    output to steer the camera. This is the loss that was missing.

    Args:
        predicted_ig: (B, Nb) from the acquisition network.
        true_ig:      (B, Nb) from the rendered look-ahead. Detached by the caller.
    """
    return torch.nn.functional.mse_loss(predicted_ig, true_ig)
