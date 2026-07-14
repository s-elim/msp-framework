"""Selective prediction, and the degeneracy it exists to expose.

The load-bearing test is `test_a_system_that_never_acts_is_not_rewarded`. The conformal certificate
of Eq 24 abstained on 100% of scenes while reporting coverage 0.906 against a 0.90 target, and every
number on that table looked like a pass. Coverage cannot tell a calibrated certificate from a
catatonic one; only an act rate can.
"""

from __future__ import annotations

import pytest
import torch

from msp.diagnostics import compare_selective, risk_coverage


def _perfect(n: int = 200, na: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    succ = (torch.rand(n, na, generator=g) < 0.15).float()
    ex = torch.ones(n, na, dtype=torch.bool)
    return succ, ex, g


# --------------------------------------------------------------------------------------
# the degeneracy
# --------------------------------------------------------------------------------------


def test_a_system_that_never_acts_is_not_rewarded() -> None:
    """THE WHOLE POINT.

    A certificate that abstains everywhere achieves perfect coverage and zero utility. The
    conformal evaluation reported exactly that (coverage 0.9056 >= 0.90, abstention 1.000) and it
    reads as a pass. Selective prediction cannot be fooled that way: an abstaining system simply has
    no act rate to score, and its curve is empty.
    """
    succ, ex, _ = _perfect()
    # A scorer so under-confident that no action clears any sane threshold still HAS a ranking,
    # so risk-coverage still scores it -- there is no hiding behind abstention.
    timid = torch.full_like(succ, 0.01)
    rc = risk_coverage(timid, succ, ex)
    assert rc.n_scenes == succ.shape[0]
    assert rc.act_rate[-1] == pytest.approx(1.0)
    # With a constant score, its "top" pick is arbitrary -> it can do no better than chance.
    assert rc.top1_success == pytest.approx(float(succ[:, 0].mean()), abs=1e-6)


def test_no_executable_grasp_means_no_decision() -> None:
    """A scene where nothing is reachable poses no decision and must not dilute the precision."""
    succ = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    score = torch.tensor([[0.9, 0.1], [0.9, 0.9]])
    ex = torch.tensor([[True, True], [False, False]])
    rc = risk_coverage(score, succ, ex)
    assert rc.n_scenes == 1


def test_unreachable_grasps_are_never_selected() -> None:
    """REGRESSION. The argmax must not be allowed to pick a grasp the gripper cannot reach: a
    deployed system knows its own kinematics and filters those BEFORE choosing. Scoring them would
    measure a collision checker, not a grasp scorer."""
    # The unreachable action has the HIGHEST score and would win a naive argmax -- and it "succeeds",
    # so failing to exclude it would silently INFLATE the result.
    succ = torch.tensor([[0.0, 1.0]])
    score = torch.tensor([[0.1, 99.0]])
    ex = torch.tensor([[True, False]])
    rc = risk_coverage(score, succ, ex)
    assert rc.top1_success == 0.0, "an unreachable grasp was selected"


# --------------------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------------------


def test_an_oracle_scorer_is_perfectly_precise_where_it_can_be() -> None:
    """A scorer that equals the truth must succeed whenever a successful grasp exists at all."""
    succ, ex, _ = _perfect()
    rc = risk_coverage(succ.clone(), succ, ex)
    has_win = (succ.sum(dim=1) > 0).float().mean()
    assert rc.top1_success == pytest.approx(float(has_win), abs=1e-6)
    # Most selective first: the scenes it is most confident about are the ones it wins.
    assert rc.precision_at(0.1) == pytest.approx(1.0)


def test_precision_of_an_informative_scorer_falls_as_it_acts_more() -> None:
    """The defining shape of a risk-coverage curve: being choosier must pay. A scorer whose
    precision is FLAT in the act rate -- which is what the Ferrari-Canny proxy does, 0.235 at a 10%
    act rate and 0.161 at 100% -- is not ranking, it is guessing."""
    succ, ex, _ = _perfect()
    informative = succ + 0.3 * torch.randn(succ.shape, generator=torch.Generator().manual_seed(1))
    rc = risk_coverage(informative, succ, ex)
    assert rc.precision_at(0.1) > rc.precision_at(1.0) + 0.1


def test_random_pick_control_is_reported() -> None:
    """Success is rare, so a scorer can look respectable while ranking no better than chance. The
    uniform-pick control is what exposes that, and the comparison must always carry it."""
    succ, ex, g = _perfect()
    noise = torch.rand(succ.shape, generator=g)
    cmp_ = compare_selective(analytic_score=noise, msp_score=succ.clone(), succ=succ,
                             executable=ex, generator=g)
    # An uninformative scorer lands near the random control; the oracle beats both.
    assert abs(cmp_.analytic.top1_success - cmp_.random_pick) < 0.12
    assert cmp_.msp.top1_success > cmp_.analytic.top1_success + 0.2
    assert "random pick" in cmp_.summary()
