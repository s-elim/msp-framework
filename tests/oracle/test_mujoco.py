"""The rigid-body tier of M, and the gap between it and the analytic prior.

STATUS: the simulator RUNS and produces non-degenerate outcomes, but it is NOT yet validated.
`test_slip_is_near_zero_for_a_successful_grasp` currently FAILS (xfail) and that failure is real,
not a tolerance quibble: an object that is genuinely carried by the hand should not translate ~10cm
relative to it. Until that is fixed, `MuJoCoOracle` must not be used to produce a number for the
paper. The test is left in, failing and visible, rather than deleted or loosened.
"""

from __future__ import annotations

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from msp.oracle import AnalyticGraspOracle  # noqa: E402
from msp.oracle.composite import CompositeOracle  # noqa: E402
from msp.oracle.mujoco_sim import MuJoCoOracle, SimConfig  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rollouts():
    torch.manual_seed(0)
    an = AnalyticGraspOracle(shape="box", noise=False)
    sim = MuJoCoOracle(shape="box")
    g = torch.Generator().manual_seed(0)
    x = an.sample_states(24, generator=g)
    a = an.sample_actions(x, 6, generator=g)
    return an, sim, x, a, sim.query(x, a)


def test_simulator_is_version_pinned(rollouts) -> None:
    """A rigid-body result is a function of the solver settings as much as of the physics. The
    reviewers ask for a 'version-pinned' simulator; `provenance()` is what pins it."""
    _, sim, _, _, _ = rollouts
    p = sim.provenance()
    assert "mujoco_version" in p
    assert p["timestep"] == SimConfig().timestep
    assert p["solver_iterations"] == SimConfig().solver_iterations


def test_outcomes_are_valid_and_non_degenerate(rollouts) -> None:
    """The simulator must actually grasp things sometimes.

    REGRESSION. The first MJCF welded the fingers to the hand AND gave them slide joints. A weld
    pins the body completely, so the jaws could never close and the success rate was exactly 0.000
    across every rollout. If a simulator says nothing ever works, suspect the model.
    """
    _, _, _, _, y = rollouts
    y.validate()
    rate = float(y.succ.mean())
    assert 0.05 < rate < 0.95, f"simulated success rate {rate:.3f} is degenerate"


def test_the_simulator_is_not_differentiable_and_says_so(rollouts) -> None:
    """You cannot autodiff through mj_step, and finite-differencing a contact-discontinuous
    simulator does not give a usable J(x) -- it gives noise with the numerical rank of whatever
    the roundoff happened to be. Rather than silently returning that, we raise."""
    _, sim, x, a, _ = rollouts
    assert sim.differentiable is False
    with pytest.raises(NotImplementedError, match="CompositeOracle"):
        sim.outcome_params(x, a)


@pytest.mark.xfail(
    reason=(
        "KNOWN DEFECT, not a tolerance issue. A successfully-lifted object should be carried "
        "rigidly by the hand, so its displacement RELATIVE to the hand should be ~0. We measure "
        "~0.10 m: the object is sliding through the fingers during the lift even when it clears "
        "the success height. Either the grip force, the finger friction, or the weld stiffness is "
        "wrong. Until this passes, MuJoCoOracle must not produce a number for the paper."
    ),
    strict=True,
)
def test_slip_is_near_zero_for_a_successful_grasp(rollouts) -> None:
    _, _, _, _, y = rollouts
    held = y.succ.squeeze(-1) == 1
    assert held.any(), "no successful grasps to check"
    slip_when_held = y.slip.squeeze(-1)[held].mean()
    assert float(slip_when_held) < 0.02


# --------------------------------------------------------------------------------------
# The composition -- and the gap, which is a RESULT
# --------------------------------------------------------------------------------------


def test_composite_takes_phi_from_the_differentiable_tier(rollouts) -> None:
    """Phi -- and therefore J(x), and therefore contribution C2 -- must come from tier 1, because
    tier 2 has no derivative worth having."""
    an, sim, x, a, _ = rollouts
    comp = CompositeOracle(analytic=an, simulator=sim)
    assert comp.differentiable is True
    torch.testing.assert_close(comp.outcome_params(x, a), an.outcome_params(x, a))


def test_composite_takes_success_and_slip_from_the_simulator(rollouts) -> None:
    """The dynamic quantities come from the tier that actually simulates dynamics; the wrench-space
    margin comes from the tier that actually has a wrench-space metric."""
    an, sim, x, a, y_sim = rollouts
    comp = CompositeOracle(analytic=an, simulator=sim)
    y = comp.query(x, a)
    torch.testing.assert_close(y.succ, y_sim.succ)
    torch.testing.assert_close(y.margin, an.query(x, a).margin, atol=1e-2, rtol=1e-1)


def test_tier_gap_is_measurable_and_large(rollouts) -> None:
    """THE RESULT THIS COMPOSITION EXISTS TO PRODUCE.

    The analytic Ferrari-Canny prior is a POOR predictor of whether the object actually gets
    lifted: its false-positive rate against the simulator is ~0.74. It says "force closed, good
    grasp" and the object falls out of the hand three times in four.

    This is the in-simulation analogue of the sim-to-real sufficiency gap, and it is the single
    most important number the oracle produces. It says, quantitatively, that training the encoder
    on the analytic tier alone would teach it to be confidently wrong -- which is exactly why
    Section 11 demands the composition rather than either tier by itself, and exactly the concern
    behind reviewer attacks 3 and 11 ("force closure computed on estimated geometry is not
    grounded in outcome").

    We assert the gap is MEASURED, not that it is small. A small gap would be a finding; a gap we
    never looked at would be negligence.
    """
    an, sim, x, a, _ = rollouts
    comp = CompositeOracle(analytic=an, simulator=sim)
    gap = comp.tier_gap(x, a)

    assert 0.0 <= gap["success_agreement"] <= 1.0
    assert gap["n"] == 24 * 6
    assert gap["analytic_false_positive_rate"] > 0.3, (
        "if the analytic prior suddenly agrees with the simulator, either the simulator broke or "
        "the prior got much better -- both are worth investigating before trusting this number"
    )


def test_tier_gap_requires_a_simulator() -> None:
    comp = CompositeOracle(analytic=AnalyticGraspOracle())
    with pytest.raises(RuntimeError, match="needs a simulator"):
        comp.tier_gap(torch.zeros(1, 14), torch.zeros(1, 1, 7))
