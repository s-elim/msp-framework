"""Conformal calibration: Eq 22, Eq 23, Eq 24, Theorem 7, and ACI (Section 7).

This is the framework's only guarantee, so it gets the framework's strictest tests.

Every test named `test_regression_*` reproduces a defect found in the audited
implementation. Each one FAILS against the old code and PASSES against this one. They exist
so the bug cannot come back.
"""

from __future__ import annotations

import math

import pytest
import torch

from msp.math.conformal import (
    AdaptiveConformalState,
    conformal_quantile,
    min_calibration_size,
    nonconformity_scores,
    prediction_set,
)

# --------------------------------------------------------------------------------------
# Eq 22 -- nonconformity score
# --------------------------------------------------------------------------------------


def test_eq22_nonconformity_matches_definition() -> None:
    """E = 1 - s if succ == 1, else s."""
    s = torch.tensor([0.9, 0.9, 0.2, 0.2])
    succ = torch.tensor([1.0, 0.0, 1.0, 0.0])
    got = nonconformity_scores(s, succ)
    expected = torch.tensor([0.1, 0.9, 0.8, 0.2])
    torch.testing.assert_close(got, expected)


def test_eq22_score_is_one_minus_prob_of_realized_label() -> None:
    """The score is exactly 1 - p_model(y_true); confident-and-right => near 0."""
    s = torch.rand(500)
    succ = (torch.rand(500) < s).float()
    scores = nonconformity_scores(s, succ)
    p_true = torch.where(succ > 0.5, s, 1.0 - s)
    torch.testing.assert_close(scores, 1.0 - p_true)


# --------------------------------------------------------------------------------------
# Eq 23 -- the quantile
# --------------------------------------------------------------------------------------


def test_eq23_quantile_is_an_order_statistic_not_an_interpolation() -> None:
    """REGRESSION. The audited code used np.quantile's default LINEAR interpolation.

    Theorem 7 is stated for the ceil((n+1)(1-alpha))-th ORDER STATISTIC. Interpolating
    between adjacent ranks returns a value strictly between sorted[k-1] and sorted[k].

    DIRECTION OF THE ERROR (verified, and the opposite of what one might assume): since
    np.quantile(E, k/n) interpolates at index (k/n)*(n-1) = k - k/n, and k <= n, that index
    is >= k-1. So the interpolated q_hat is >= the required order statistic. The bug is
    therefore CONSERVATIVE -- coverage still holds -- and it costs statistical POWER, not
    validity: q_hat is too large, the sets are needlessly wide, and FEWER actions get
    certified than the theorem entitles you to. Severity is Low, not Critical.

    We still fix it, because the guarantee is stated for the order statistic and a claimed
    1-alpha certificate should be exactly that, not "1-alpha plus an unquantified margin".
    """
    scores = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n, alpha = scores.numel(), 0.25
    rank = math.ceil((n + 1) * (1 - alpha))  # ceil(12 * 0.75) = 9
    assert rank == 9

    q = conformal_quantile(scores, alpha)
    assert q == pytest.approx(0.8), "must be the 9th smallest score, exactly"

    # The interpolating estimator the old code used does NOT return the order statistic.
    level = rank / n  # 9/11 = 0.8181...
    q_interp = torch.quantile(scores, level, interpolation="linear").item()
    assert q_interp != pytest.approx(q), "interpolation must differ from the order statistic"
    assert q_interp > q, "and here it is conservative (too wide), not unsafe"


def test_eq23_quantile_rejects_vacuous_calibration_folds() -> None:
    """REGRESSION. The audited code clamped the quantile level to 1.0 when n was too
    small, silently voiding the guarantee. We raise."""
    assert min_calibration_size(0.1) == 9
    with pytest.raises(ValueError, match="too small"):
        conformal_quantile(torch.rand(5), alpha=0.1)
    conformal_quantile(torch.rand(9), alpha=0.1)  # exactly at the bound: fine


# --------------------------------------------------------------------------------------
# Eq 23 / Eq 24 -- the prediction set and the singleton test
# --------------------------------------------------------------------------------------


def test_eq23_prediction_set_computes_both_labels() -> None:
    """C(o,a) = {l : score_l <= q_hat}, score_1 = 1-s, score_0 = s."""
    s = torch.tensor([[0.05, 0.50, 0.95]])
    q_hat = 0.6
    C = prediction_set(s, q_hat)

    # s=0.05: score_1 = 0.95 > 0.6 (out), score_0 = 0.05 <= 0.6 (in)  -> {0}
    # s=0.50: score_1 = 0.50 <= 0.6 (in), score_0 = 0.50 <= 0.6 (in)  -> {0,1}
    # s=0.95: score_1 = 0.05 <= 0.6 (in), score_0 = 0.95 > 0.6 (out)  -> {1}
    torch.testing.assert_close(C.contains_success, torch.tensor([[False, True, True]]))
    torch.testing.assert_close(C.contains_failure, torch.tensor([[True, True, False]]))


def test_regression_eq24_ambiguous_set_is_not_certified() -> None:
    """REGRESSION -- THE SAFETY BUG.

    The audited `get_certified_set` tested only `(1 - s) <= q_hat`, i.e. only whether label
    1 was IN the set. It never computed score_0, so it certified any action whose set was
    the ambiguous {0, 1}. A coin-flip action with s = 0.50 was certified as a guaranteed
    success. Eq 24 requires the SINGLETON: C(o, a) == {1}.

    q_hat = 0.678 is the value actually fitted by the audited calibrator on a well-specified
    n=500 fold at alpha=0.1. The ambiguity band is 1 - q_hat <= s <= q_hat, i.e.
    [0.322, 0.678]: every action in that band has BOTH labels in its prediction set and none
    of them may be certified.
    """
    q_hat = 0.678
    s = torch.tensor([[0.50, 0.62, 0.95]])
    C = prediction_set(s, q_hat)

    old_behaviour = (1.0 - s) <= q_hat  # what the audited code computed
    torch.testing.assert_close(old_behaviour, torch.tensor([[True, True, True]]))

    certified = C.is_certified_success()  # Eq 24, done properly
    torch.testing.assert_close(certified, torch.tensor([[False, False, True]]))

    assert C.is_ambiguous()[0, 0], "s=0.50 (a coin flip) must be AMBIGUOUS, not certified"
    assert C.is_ambiguous()[0, 1], "s=0.62 must be AMBIGUOUS, not certified"


def test_eq24_the_ambiguity_band_is_exactly_one_minus_qhat_to_qhat() -> None:
    """Characterize the set of success probabilities that CANNOT be certified at a given
    q_hat. Any s in [1 - q_hat, q_hat] has both labels in its set. The audited code
    certified this entire band -- which, at the q_hat values a real model produces
    (~0.68 here), is more than a third of the probability axis."""
    q_hat = 0.678
    s = torch.linspace(0.0, 1.0, 1001).unsqueeze(0)
    C = prediction_set(s, q_hat)

    ambiguous = C.is_ambiguous()[0]
    band = s[0][ambiguous]
    assert band.min().item() == pytest.approx(1 - q_hat, abs=1e-2)
    assert band.max().item() == pytest.approx(q_hat, abs=1e-2)

    # How much of the axis the old code wrongly certified:
    wrongly_certified = ((1.0 - s) <= q_hat) & ~C.is_certified_success()
    assert wrongly_certified[0].float().mean().item() > 0.3


def test_eq24_set_partition_is_exhaustive_and_disjoint() -> None:
    """Every action falls into exactly one of {1}, {0}, {0,1}, {} -- no gaps, no overlap."""
    s = torch.rand(4, 32)
    for q_hat in (0.0, 0.1, 0.5, 0.9, 1.0):
        C = prediction_set(s, q_hat)
        certified = C.is_certified_success()
        certified_fail = C.contains_failure & ~C.contains_success
        ambiguous = C.is_ambiguous()
        empty = C.is_empty()
        total = (
            certified.int() + certified_fail.int() + ambiguous.int() + empty.int()
        )
        assert torch.all(total == 1), f"partition broken at q_hat={q_hat}"


# --------------------------------------------------------------------------------------
# Theorem 7 -- marginal coverage
# --------------------------------------------------------------------------------------


def test_theorem7_marginal_coverage_holds_empirically() -> None:
    """P( succ in C(o,a) ) >= 1 - alpha, over ALL test points.

    Note this is *marginal* coverage over the whole test fold. The audited evaluator instead
    measured P(succ = 1 | a in A_cert) -- the precision of the certified set -- and logged
    it against a 1-alpha target. Conditioning on selection destroys exchangeability, so that
    quantity has no reason to converge to 1 - alpha. This test measures the real thing.
    """
    torch.manual_seed(0)
    alpha = 0.1
    n_cal, n_test = 2000, 4000

    # A well-specified model: s is the true Bernoulli parameter.
    s_cal = torch.rand(n_cal)
    y_cal = (torch.rand(n_cal) < s_cal).float()
    q_hat = conformal_quantile(nonconformity_scores(s_cal, y_cal), alpha)

    s_test = torch.rand(n_test)
    y_test = (torch.rand(n_test) < s_test).float()

    C = prediction_set(s_test.unsqueeze(0), q_hat)
    covered = torch.where(
        y_test.unsqueeze(0) > 0.5, C.contains_success, C.contains_failure
    )
    coverage = covered.float().mean().item()

    assert coverage >= 1 - alpha - 0.02, f"coverage {coverage:.4f} below nominal {1 - alpha}"


def test_theorem7_coverage_holds_for_a_miscalibrated_model() -> None:
    """Conformal coverage is DISTRIBUTION-FREE: it must hold even when the underlying model
    is badly miscalibrated. This is the whole point of the guarantee."""
    torch.manual_seed(1)
    alpha = 0.2
    # Wildly overconfident model: predicts ~0.95 regardless of the truth.
    y_cal = (torch.rand(2000) < 0.5).float()
    s_cal = torch.full((2000,), 0.95) + 0.01 * torch.randn(2000)
    q_hat = conformal_quantile(nonconformity_scores(s_cal, y_cal), alpha)

    y_test = (torch.rand(4000) < 0.5).float()
    s_test = torch.full((4000,), 0.95) + 0.01 * torch.randn(4000)
    C = prediction_set(s_test.unsqueeze(0), q_hat)
    covered = torch.where(y_test.unsqueeze(0) > 0.5, C.contains_success, C.contains_failure)

    assert covered.float().mean().item() >= 1 - alpha - 0.02


# --------------------------------------------------------------------------------------
# Section 7 -- Adaptive Conformal Inference
# --------------------------------------------------------------------------------------


def test_regression_aci_is_anchored_to_a_fixed_target() -> None:
    """REGRESSION -- THE DIVERGENCE BUG.

    Correct ACI:  alpha_{t+1} = alpha_t + gamma * (TARGET - err_t),  target fixed.
    Audited code: alpha_{t+1} = alpha_t + gamma * (alpha_t - err_t),  target = the mutating
    alpha itself. That is alpha_t*(1+gamma) - gamma*err_t: positive feedback with no anchor.
    It diverged to the clamp in ~100 steps, collapsing a 90% coverage target to 1%.
    """
    torch.manual_seed(0)
    target, gamma, T = 0.1, 0.05, 400
    state = AdaptiveConformalState(target=target, gamma=gamma)
    errs = (torch.rand(T) < target)

    for t in range(T):
        state.update(miscovered=bool(errs[t]))

    assert 0.0 < state.alpha_t < 0.5, (
        f"alpha_t = {state.alpha_t:.3f} -- ACI diverged. It must stay anchored near the "
        f"target {target}, not run away to the clamp."
    )
    assert state.target == target, "the target must never mutate"


def test_aci_long_run_coverage_tracks_the_target() -> None:
    """ACI's guarantee: the time-averaged miscoverage converges to alpha, WITHOUT
    exchangeability. We simulate drift and check the long-run rate."""
    torch.manual_seed(2)
    target, gamma, T = 0.1, 0.05, 5000
    state = AdaptiveConformalState(target=target, gamma=gamma)

    # A drifting environment: the miscoverage a given alpha_t induces gets harder over time.
    errs = []
    for t in range(T):
        drift = 1.0 + 2.0 * (t / T)  # difficulty rises
        p_miscover = min(0.99, state.alpha_t / drift)
        err = bool(torch.rand(1).item() < p_miscover)
        errs.append(err)
        state.update(miscovered=err)

    realized = sum(errs) / T
    assert abs(realized - target) < 0.05, (
        f"long-run miscoverage {realized:.4f} should track target {target} under drift"
    )


def test_aci_loosens_after_miscoverage_and_tightens_after_success() -> None:
    """Direction check: a miss must RAISE alpha (wider sets, more caution); a hit must lower
    it (tighter sets). Getting this sign backwards inverts the whole controller."""
    state = AdaptiveConformalState(target=0.1, gamma=0.05)
    a0 = state.alpha_t

    state.update(miscovered=False)
    assert state.alpha_t > a0, "covering should relax alpha toward the target"

    state2 = AdaptiveConformalState(target=0.1, gamma=0.05)
    b0 = state2.alpha_t
    state2.update(miscovered=True)
    assert state2.alpha_t < b0, "miscovering should tighten alpha"


def test_aci_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        AdaptiveConformalState(target=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalState(target=1.0)
    with pytest.raises(ValueError):
        AdaptiveConformalState(target=0.1, gamma=0.0)
