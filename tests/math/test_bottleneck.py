"""The manipulation-sufficiency objective: Eq 7, 8, 9, 10, 11 and Theorem 3.

The central test here is `test_regression_beta_direction_matches_theorem_3`. It is the one
that would have caught the inverted-beta defect, and it encodes the paper's claim directly:
raising beta must buy sufficiency, not compression.
"""

from __future__ import annotations

import pytest
import torch

from msp.math.bottleneck import BetaSchedule, distortion, rate, vib_objective
from msp.math.divergences import kl_to_standard_normal
from msp.types import Outcome, OutcomeDistribution


def _make_pair(b: int = 4, na: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    pred = OutcomeDistribution(
        succ_logit=torch.randn(b, na, 1),
        margin_mu=torch.randn(b, na, 1),
        margin_logvar=torch.randn(b, na, 1) * 0.1,
        slip_log_mu=torch.randn(b, na, 1),
        slip_log_logvar=torch.randn(b, na, 1) * 0.1,
    )
    target = Outcome(
        succ=(torch.rand(b, na, 1) < 0.5).float(),
        margin=torch.randn(b, na, 1),
        slip=torch.rand(b, na, 1),
    )
    return pred, target


# --------------------------------------------------------------------------------------
# Eq 11 / Eq 8 -- the rate
# --------------------------------------------------------------------------------------


def test_eq11_kl_is_zero_at_the_prior() -> None:
    """KL(N(0,I) || N(0,I)) = 0. mu=0, logvar=0 <=> sigma=1."""
    mu = torch.zeros(3, 8)
    logvar = torch.zeros(3, 8)
    torch.testing.assert_close(kl_to_standard_normal(mu, logvar), torch.zeros(3))


def test_eq11_kl_matches_hand_computed_value() -> None:
    """0.5 * sum(sigma^2 + mu^2 - 1 - log sigma^2), computed by hand for one coordinate."""
    mu = torch.tensor([[2.0]])
    logvar = torch.tensor([[math_log_4 := float(torch.log(torch.tensor(4.0)))]])  # sigma^2=4
    # 0.5 * (4 + 4 - 1 - log 4) = 0.5 * (7 - 1.3863) = 2.8069
    expected = 0.5 * (4.0 + 4.0 - 1.0 - math_log_4)
    torch.testing.assert_close(
        kl_to_standard_normal(mu, logvar), torch.tensor([expected]), atol=1e-5, rtol=1e-5
    )


def test_eq11_kl_is_nonnegative() -> None:
    """A KL divergence is non-negative. Cheap invariant; catches sign slips."""
    torch.manual_seed(0)
    for _ in range(20):
        mu = torch.randn(16, 12) * 3
        logvar = torch.randn(16, 12) * 2
        assert torch.all(kl_to_standard_normal(mu, logvar) >= -1e-5)


def test_eq8_rate_matches_the_closed_form() -> None:
    mu, logvar = torch.randn(5, 8), torch.randn(5, 8) * 0.3
    torch.testing.assert_close(rate(mu, logvar), kl_to_standard_normal(mu, logvar).mean())


# --------------------------------------------------------------------------------------
# Eq 9 -- the distortion
# --------------------------------------------------------------------------------------


def test_eq9_distortion_is_a_proper_negative_log_likelihood() -> None:
    """A perfect predictor must attain a LOWER distortion than a bad one, on every term."""
    b, na = 4, 6
    torch.manual_seed(0)
    succ = (torch.rand(b, na, 1) < 0.5).float()
    margin = torch.randn(b, na, 1)
    slip = torch.rand(b, na, 1) + 0.1
    target = Outcome(succ=succ, margin=margin, slip=slip)

    good = OutcomeDistribution(
        succ_logit=torch.where(succ > 0.5, 8.0, -8.0),
        margin_mu=margin.clone(),
        margin_logvar=torch.full_like(margin, -2.0),
        slip_log_mu=torch.log(slip),
        slip_log_logvar=torch.full_like(slip, -2.0),
    )
    bad = OutcomeDistribution(
        succ_logit=torch.where(succ > 0.5, -8.0, 8.0),  # confidently wrong
        margin_mu=margin + 5.0,
        margin_logvar=torch.full_like(margin, -2.0),
        slip_log_mu=torch.log(slip) + 5.0,
        slip_log_logvar=torch.full_like(slip, -2.0),
    )
    d_good, d_bad = distortion(good, target), distortion(bad, target)
    assert d_good.succ < d_bad.succ
    assert d_good.margin < d_bad.margin
    assert d_good.slip < d_bad.slip


def test_regression_gaussian_nll_does_not_overflow_on_a_confident_head() -> None:
    """REGRESSION. The audited head left margin/slip logvar UNCLAMPED, so exp(-logvar)
    overflowed to inf once the head grew confident, producing NaN gradients.
    `OutcomeDistribution` now clamps on construction."""
    b, na = 2, 3
    pred = OutcomeDistribution(
        succ_logit=torch.zeros(b, na, 1),
        margin_mu=torch.zeros(b, na, 1),
        margin_logvar=torch.full((b, na, 1), -500.0),  # absurdly confident
        slip_log_mu=torch.zeros(b, na, 1),
        slip_log_logvar=torch.full((b, na, 1), -500.0),
    )
    target = Outcome(
        succ=torch.ones(b, na, 1), margin=torch.ones(b, na, 1), slip=torch.ones(b, na, 1)
    )
    d = distortion(pred, target)
    assert torch.isfinite(d.succ) and torch.isfinite(d.margin) and torch.isfinite(d.slip)


def test_slip_is_modeled_on_the_nonnegative_halfline() -> None:
    """Y constrains slip >= 0. A log-normal puts no mass on negative slip; the audited
    unconstrained Gaussian did."""
    target = Outcome(
        succ=torch.ones(1, 1, 1),
        margin=torch.zeros(1, 1, 1),
        slip=torch.zeros(1, 1, 1),  # a clean, non-slipping grasp: the common case
    )
    pred = OutcomeDistribution(
        succ_logit=torch.zeros(1, 1, 1),
        margin_mu=torch.zeros(1, 1, 1),
        margin_logvar=torch.zeros(1, 1, 1),
        slip_log_mu=torch.zeros(1, 1, 1),
        slip_log_logvar=torch.zeros(1, 1, 1),
    )
    d = distortion(pred, target)
    assert torch.isfinite(d.slip), "slip=0 must not produce -inf/NaN (log(0))"


# --------------------------------------------------------------------------------------
# Eq 7 / Eq 10 / Theorem 3 -- THE BETA DIRECTION
# --------------------------------------------------------------------------------------


def test_regression_beta_direction_matches_theorem_3() -> None:
    """REGRESSION -- THE INVERTED-BETA DEFECT.

    Eq 7:  min I(Z;O) - beta * I(Z;Y|A).   Theorem 3: beta -> inf  ==>  SUFFICIENCY.

    So raising beta must increase the weight the loss places on DISTORTION (relevance)
    relative to RATE (compression). The audited code computed `L = sum_j D_j/beta_j + R`,
    which does the exact opposite: raising beta shrank the distortion term and let the rate
    dominate, driving the latent to total collapse.

    We assert the derivative of the loss w.r.t. the distortion grows with beta, and that the
    ratio of distortion weight to rate weight is beta itself.
    """
    pred, target = _make_pair()
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8) * 0.2

    d = distortion(pred, target)
    r = rate(mu, logvar)
    total_d = d.succ + d.margin + d.slip

    losses = {}
    for beta in (0.1, 1.0, 10.0):
        terms = vib_objective(pred, target, mu, logvar, BetaSchedule.uniform(beta))
        losses[beta] = terms.loss
        # L = beta * D + R  (per-dimension form of Eq 7 with uniform beta)
        torch.testing.assert_close(terms.loss, beta * total_d + r, atol=1e-5, rtol=1e-5)

    # The distortion's share of the loss must RISE with beta.
    share = {b: (b * total_d / losses[b]).item() for b in (0.1, 1.0, 10.0)}
    assert share[0.1] < share[1.0] < share[10.0], (
        "raising beta must increase the weight on sufficiency (Theorem 3), not decrease it"
    )


def test_beta_zero_limit_is_pure_compression() -> None:
    """beta -> 0: the objective reduces to the rate alone -- maximal compression, zero
    sufficiency. The mirror image of Theorem 3."""
    pred, target = _make_pair()
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8) * 0.2
    terms = vib_objective(pred, target, mu, logvar, BetaSchedule.uniform(1e-8))
    torch.testing.assert_close(terms.loss, rate(mu, logvar), atol=1e-4, rtol=1e-4)


def test_per_dimension_beta_targets_the_right_outcome() -> None:
    """Philosophy P5: a language instruction scales beta per outcome dimension, so the
    budget must land on the dimension it names. Raising beta_slip must increase the loss's
    sensitivity to the SLIP term and leave the others alone."""
    pred, target = _make_pair()
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8) * 0.2

    base = vib_objective(pred, target, mu, logvar, BetaSchedule(1.0, 1.0, 1.0))
    slip_heavy = vib_objective(pred, target, mu, logvar, BetaSchedule(1.0, 1.0, 5.0))

    delta = slip_heavy.loss - base.loss
    expected = 4.0 * base.distortion.slip  # (5 - 1) * D_slip
    torch.testing.assert_close(delta, expected, atol=1e-5, rtol=1e-5)


def test_beta_must_be_positive() -> None:
    """Eq 7 requires beta > 0."""
    with pytest.raises(ValueError, match="must be > 0"):
        BetaSchedule(succ=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        BetaSchedule(slip=-1.0)


# --------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------


def test_objective_is_differentiable_wrt_both_belief_parameters() -> None:
    """The rate term must produce gradient for BOTH mu and logvar; if logvar were detached,
    the bottleneck could not compress."""
    pred, target = _make_pair()
    mu = torch.randn(4, 8, requires_grad=True)
    logvar = (torch.randn(4, 8) * 0.2).requires_grad_(True)

    vib_objective(pred, target, mu, logvar, BetaSchedule.uniform(1.0)).loss.backward()

    assert mu.grad is not None and torch.any(mu.grad != 0)
    assert logvar.grad is not None and torch.any(logvar.grad != 0)


def test_no_geometry_supervision_leaks_into_the_objective() -> None:
    """The entire thesis is that supervision comes from outcomes, never from pose or shape.
    `vib_objective` must depend ONLY on (pred, target, mu, logvar, beta). This test pins the
    signature so a pose loss cannot be quietly added later."""
    import inspect

    params = set(inspect.signature(vib_objective).parameters)
    assert params == {"pred", "target", "mu", "logvar", "beta", "weights"}, (
        f"vib_objective signature changed to {params}. If a geometry/pose term was added, "
        "it contradicts the manipulation-sufficiency estimand (Section 3)."
    )
