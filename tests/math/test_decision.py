"""The decision rule and ambiguity functional: Eq 13, 14, 15, 16."""

from __future__ import annotations

import pytest
import torch

from msp.math.decision import SuccessStats, ambiguity, risk_averse_score, success_stats


def test_eq13_s_is_the_mean_over_posterior_samples() -> None:
    probs = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]]])  # (B=1, K=3, Na=2)
    stats = success_stats(probs)
    torch.testing.assert_close(stats.s, torch.tensor([[0.4, 0.6]]))


def test_regression_eq14_v_uses_the_biased_estimator() -> None:
    """REGRESSION -- THE VARIANCE-ESTIMATOR DEFECT.

    Eq 14 defines  v = (1/K) sum_k (sigma_k - s)^2  -- the BIASED (maximum-likelihood)
    variance, dividing by K.

    `torch.var` defaults to correction=1 (Bessel), dividing by K-1. The audited code used
    the default, inflating v by K/(K-1): +14% at K=8, +3.2% at K=32. Because v is compared
    against an ABSOLUTE threshold tau_U (Eq 16) and scaled by an ABSOLUTE lambda (Eq 15),
    that inflation is not a harmless reparameterization -- it changes how often the robot
    decides to stop and look again.
    """
    probs = torch.tensor([[[0.1], [0.4], [0.9], [0.6]]])  # (1, K=4, 1)
    stats = success_stats(probs)

    p = probs[0, :, 0]
    expected_biased = ((p - p.mean()) ** 2).mean()  # 1/K
    unbiased = p.var(correction=1)  # 1/(K-1)  -- what the old code returned

    torch.testing.assert_close(stats.v, expected_biased.reshape(1, 1))
    assert not torch.allclose(stats.v, unbiased.reshape(1, 1)), (
        "v must use correction=0 (Eq 14), not torch's Bessel-corrected default"
    )
    torch.testing.assert_close(unbiased / expected_biased, torch.tensor(4.0 / 3.0))


def test_eq14_variance_is_zero_for_a_collapsed_belief() -> None:
    """No epistemic spread => no epistemic variance => no reason to sense (Eq 16)."""
    probs = torch.full((2, 16, 5), 0.7)
    stats = success_stats(probs)
    torch.testing.assert_close(stats.v, torch.zeros(2, 5))
    torch.testing.assert_close(ambiguity(stats), torch.zeros(2))


def test_success_stats_requires_a_sample_axis() -> None:
    """A point estimate has no epistemic variance. Passing (B, Na) must be a hard error,
    not a silently-wrong broadcast."""
    with pytest.raises(ValueError, match=r"\(B, K, Na\)"):
        success_stats(torch.rand(4, 6))
    with pytest.raises(ValueError, match="K must be >= 2"):
        success_stats(torch.rand(4, 1, 6))


def test_eq15_risk_penalty_demotes_uncertain_actions() -> None:
    """a* = argmax [s - lambda*v]. With lambda large, a slightly-worse but confident action
    must beat a slightly-better but uncertain one."""
    stats = SuccessStats(
        s=torch.tensor([[0.80, 0.75]]),  # action 0 has higher mean success
        v=torch.tensor([[0.10, 0.00]]),  # but action 0 is far more uncertain
    )
    torch.testing.assert_close(
        risk_averse_score(stats, 0.0).argmax(-1), torch.tensor([0])
    )  # risk-neutral: pick the higher mean
    torch.testing.assert_close(
        risk_averse_score(stats, 1.0).argmax(-1), torch.tensor([1])
    )  # risk-averse: 0.80-0.10=0.70 < 0.75-0.00=0.75


def test_eq15_lambda_must_be_nonnegative() -> None:
    stats = SuccessStats(s=torch.rand(1, 2), v=torch.rand(1, 2))
    with pytest.raises(ValueError, match="lambda must be >= 0"):
        risk_averse_score(stats, -0.5)


def test_regression_eq16_ambiguity_is_per_scene_not_per_batch() -> None:
    """REGRESSION. The audited `should_sense` did `ambiguity.mean().item()`, collapsing B
    scenes to a single bool -- so one highly ambiguous scene in a batch could not trigger
    sensing on its own. Sensing is a per-scene decision."""
    stats = SuccessStats(
        s=torch.rand(3, 4),
        v=torch.tensor([[0.0, 0.0, 0.0, 0.0],  # scene 0: certain
                        [0.5, 0.5, 0.5, 0.5],  # scene 1: very ambiguous
                        [0.0, 0.0, 0.0, 0.0]]),  # scene 2: certain
    )
    U = ambiguity(stats)
    assert U.shape == (3,), "U must be one scalar PER SCENE"
    torch.testing.assert_close(U, torch.tensor([0.0, 0.5, 0.0]))

    tau_U = 0.1
    should_sense = U > tau_U
    torch.testing.assert_close(should_sense, torch.tensor([False, True, False]))
    assert should_sense[1], "the ambiguous scene must trigger sensing independently"
